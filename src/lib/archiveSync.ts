/**
 * Keeping the browser's archive and the server's archive the same.
 *
 * The server (server/archive.mjs) holds the records beside the files; this
 * module keeps localStorage as a cache of it. Three movements:
 *
 *   1. On start, pull everything the server has and merge it in (the server
 *      wins by id), then push every local record the server has never seen.
 *      That second step is how an archive that predates the server migrates:
 *      nothing is lost, and nothing has to be exported by hand.
 *   2. Every local commit that did not itself come from the server is queued
 *      and pushed, debounced, in batches. A push that fails stays queued and
 *      is retried with backoff; the reader is told the archive is "this
 *      browser only for now".
 *   3. A server-sent event carries the revision after every commit anywhere,
 *      and this tab pulls what it has not seen. A phone that was asleep pulls
 *      when it wakes, because a suspended EventSource misses things.
 *
 * Without the server (a static host), the mode is `local` and nothing here
 * runs. The archive works exactly as it did before, per browser.
 */
import { useSyncExternalStore } from 'react'
import { serverCapabilities } from './capabilities'
import { history, onCommit, type HistoryEntry } from './history'
import { recoverUnfiled } from './recover'

export type ArchiveSyncMode = 'starting' | 'server' | 'local' | 'offline'

export type ArchiveSyncState = {
  mode: ArchiveSyncMode
  /** Local changes not yet accepted by the server. */
  pending: number
  /** The last server revision this tab has merged. */
  rev: number
  error: string | null
  lastSyncAt: number | null
}

let state: ArchiveSyncState = { mode: 'starting', pending: 0, rev: 0, error: null, lastSyncAt: null }
const listeners = new Set<() => void>()

function set(patch: Partial<ArchiveSyncState>): void {
  state = { ...state, ...patch }
  for (const fn of [...listeners]) {
    try { fn() } catch { /* one broken subscriber must not stop the rest */ }
  }
}

function subscribe(fn: () => void): () => void {
  listeners.add(fn)
  return () => { listeners.delete(fn) }
}

const getState = () => state

export function useArchiveSync(): ArchiveSyncState {
  return useSyncExternalStore(subscribe, getState, getState)
}

export function archiveSyncState(): ArchiveSyncState {
  return state
}

// ------------------------------------------------------------------ wire --

async function api<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, body === undefined
    ? { headers: { Accept: 'application/json' } }
    : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
  const type = res.headers.get('content-type') ?? ''
  if (!type.includes('json')) throw new Error('the archive server is not mounted here')
  const data = (await res.json()) as T & { error?: string }
  if (!res.ok) throw new Error(String(data.error ?? `HTTP ${res.status}`))
  return data
}

type Pull = { rev: number; nextNo: number; full: boolean; records: unknown[]; removed: string[] }
type Pushed = { rev: number; nextNo: number; assigned: { id: string; no: number; rev: number }[] }

const queue = { upsert: new Map<string, HistoryEntry>(), remove: new Set<string>() }
let flushTimer: ReturnType<typeof setTimeout> | null = null
let retryTimer: ReturnType<typeof setTimeout> | null = null
let flushing = false
let backoff = 5000
let pulling: Promise<void> | null = null
const BATCH = 200

function pendingCount(): number {
  return queue.upsert.size + queue.remove.size
}

async function pull(since: number): Promise<void> {
  if (pulling) return pulling
  pulling = (async () => {
    const data = await api<Pull>(`/api/archive?since=${since}`)
    history.mergeFromServer(data.records, data.removed, data.nextNo)
    set({ rev: Math.max(state.rev, data.rev), mode: 'server', error: null, lastSyncAt: Date.now() })
  })().finally(() => { pulling = null })
  return pulling
}

function scheduleFlush(delay = 300): void {
  if (flushTimer) clearTimeout(flushTimer)
  flushTimer = setTimeout(() => { flushTimer = null; void flush() }, delay)
}

async function flush(): Promise<void> {
  if (flushing) return
  flushing = true
  try {
    while (queue.remove.size || queue.upsert.size) {
      if (queue.remove.size) {
        const ids = [...queue.remove]
        const r = await api<{ rev: number }>('/api/archive/remove', { ids })
        for (const id of ids) queue.remove.delete(id)
        set({ rev: Math.max(state.rev, r.rev), pending: pendingCount() })
      }
      if (queue.upsert.size) {
        const batch = [...queue.upsert.values()].slice(0, BATCH)
        const r = await api<Pushed>('/api/archive/upsert', { records: batch })
        // Only drop what was sent as sent: a record edited while the request
        // was in flight is a newer object and stays queued.
        for (const e of batch) if (queue.upsert.get(e.id) === e) queue.upsert.delete(e.id)
        history.applyServerMeta(r.assigned)
        set({ rev: Math.max(state.rev, r.rev), pending: pendingCount() })
      }
    }
    backoff = 5000
    set({ mode: 'server', error: null, pending: 0, lastSyncAt: Date.now() })
  } catch (err) {
    set({ mode: 'offline', error: err instanceof Error ? err.message : String(err), pending: pendingCount() })
    if (retryTimer) clearTimeout(retryTimer)
    retryTimer = setTimeout(() => { retryTimer = null; void flush() }, backoff)
    backoff = Math.min(60000, backoff * 2)
  } finally {
    flushing = false
  }
}

function openStream(): void {
  let es: EventSource
  try {
    es = new EventSource('/api/archive/stream')
  } catch {
    return
  }
  es.onmessage = (ev) => {
    let rev = 0
    try { rev = Number((JSON.parse(ev.data) as { rev?: number }).rev) || 0 } catch { return }
    if (rev > state.rev) void pull(state.rev).catch(() => {})
  }
  // A suspended tab's EventSource misses everything. Catch up on waking.
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
      void pull(state.rev).catch(() => {})
      if (pendingCount()) scheduleFlush(0)
    }
  })
}

let started = false

/**
 * Start once, from the one component mounted for the life of the page.
 * Resolves when the first full sync has landed or been given up on.
 */
export async function startArchiveSync(): Promise<void> {
  if (started) return
  started = true
  const caps = await serverCapabilities()
  if (!caps.archive) {
    set({ mode: 'local', error: caps.reason })
    return
  }

  onCommit((delta) => {
    if (delta.remote) return
    for (const e of delta.upserted) { queue.upsert.set(e.id, e); queue.remove.delete(e.id) }
    for (const id of delta.removed) { queue.remove.add(id); queue.upsert.delete(id) }
    set({ pending: pendingCount() })
    scheduleFlush()
  })

  try {
    await pull(0)
    // Every record the server has never seen: the migration of a browser-only
    // archive, and the ordinary case of a picture made while offline.
    for (const e of history.unsynced()) queue.upsert.set(e.id, e)
    set({ pending: pendingCount() })
    await flush()
    openStream()
    // Files on disk that no record describes are filed on idle, once.
    setTimeout(() => { void recoverUnfiled().catch(() => {}) }, 4000)
  } catch (err) {
    set({ mode: 'offline', error: err instanceof Error ? err.message : String(err) })
    if (retryTimer) clearTimeout(retryTimer)
    retryTimer = setTimeout(() => {
      retryTimer = null
      started = false
      void startArchiveSync()
    }, backoff)
    backoff = Math.min(60000, backoff * 2)
  }
}
