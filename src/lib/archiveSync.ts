/**
 * Keeping the browser's archive and the server's archive the same.
 *
 * The server (server/archive.mjs) holds the records beside the files; this
 * module keeps localStorage as a cache of it. Three movements:
 *
 *   1. On start, pull what the server has changed since this browser last
 *      read its log and merge it in (the server wins by id), then push
 *      everything the server does not yet have from here. That second step is
 *      how an archive that predates the server migrates: nothing is lost, and
 *      nothing has to be exported by hand.
 *   2. Every local change is pushed, debounced, in batches. What is waiting is
 *      not held in a queue here but marked in the archive itself (see
 *      `pending` and `gone` in history.ts) and saved with it, so a change made
 *      while the server was away survives the tab being closed. A push that
 *      fails is retried with backoff; the reader is told the archive is "this
 *      browser only for now".
 *   3. A server-sent event carries the revision after every commit anywhere,
 *      and this tab pulls what it has not seen. A tab that was hidden pulls
 *      when it is shown again, because it gives up its stream while hidden.
 *
 * Without the server (a static host), the mode is `local` and nothing here
 * runs. The archive works exactly as it did before, per browser.
 */
import { useSyncExternalStore } from 'react'
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

function notePending(): void {
  const pending = history.pendingCount()
  if (pending !== state.pending) set({ pending })
}

// ------------------------------------------------------------------ wire --

/**
 * Nothing serves the archive at this address: a static host, which will not
 * change while the page is open. Told apart from a server that is only away
 * (a network error, or a proxy's 502 while Vite restarts), which is retried.
 */
class NotMounted extends Error {}

async function api<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, body === undefined
    ? { headers: { Accept: 'application/json' } }
    : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
  const type = res.headers.get('content-type') ?? ''
  if (!type.includes('json')) {
    if (res.status < 500) throw new NotMounted('the archive server is not mounted here')
    throw new Error(`the archive server answered HTTP ${res.status}`)
  }
  const data = (await res.json()) as T & { error?: string }
  if (!res.ok) throw new Error(String(data.error ?? `HTTP ${res.status}`))
  return data
}

type Pull = {
  rev: number
  nextNo: number
  full: boolean
  records: unknown[]
  removed: string[]
  /** Absent from a server that predates it. */
  epoch?: string
}
type Pushed = {
  rev: number
  nextNo: number
  assigned: { id: string; no: number; rev: number }[]
  /** Absent from a server that predates it. */
  refused?: string[]
}

let flushTimer: ReturnType<typeof setTimeout> | null = null
let retryTimer: ReturnType<typeof setTimeout> | null = null
let flushing = false
let backoff = 5000
let pulling: Promise<void> | null = null
/** The server has answered at least once this page. Nothing is pushed before that. */
let connected = false
const BATCH = 200

async function pull(): Promise<void> {
  if (pulling) return pulling
  pulling = (async () => {
    const { rev: since, epoch } = history.syncCursor()
    let data = await api<Pull>(`/api/archive?since=${since}`)
    // A revision only means something within one log. A different epoch, or a
    // log that is now behind where this browser had read to, is an archive
    // started again, and its tail would skip records this browser never saw.
    if (since > 0 && ((data.epoch != null && data.epoch !== epoch) || data.rev < since)) {
      data = await api<Pull>('/api/archive?since=0')
    }
    history.mergeFromServer(data.records, data.removed, data.nextNo, { rev: data.rev, epoch: data.epoch ?? null })
    set({ rev: data.rev, mode: 'server', error: null, lastSyncAt: Date.now(), pending: history.pendingCount() })
  })().finally(() => { pulling = null })
  return pulling
}

function scheduleFlush(delay = 300): void {
  if (flushTimer) clearTimeout(flushTimer)
  flushTimer = setTimeout(() => { flushTimer = null; void flush() }, delay)
}

/** A record as the server should see it: the browser's own mark stays here. */
function wire(e: HistoryEntry): Omit<HistoryEntry, 'pending'> {
  const { pending: _pending, ...rest } = e
  return rest
}

async function flush(): Promise<void> {
  if (flushing || !connected) return
  flushing = true
  // Each object goes at most once a pass. One the server neither stamps nor
  // refuses keeps its mark and goes again with the next change, rather than
  // spinning here.
  const sent = new Set<HistoryEntry>()
  try {
    for (;;) {
      const ids = history.gone()
      if (ids.length) {
        await api<{ rev: number }>('/api/archive/remove', { ids })
        history.acknowledgeRemoved(ids)
      }
      const batch = history.unsynced().filter((e) => !sent.has(e)).slice(0, BATCH)
      if (!batch.length) break
      for (const e of batch) sent.add(e)
      const r = await api<Pushed>('/api/archive/upsert', { records: batch.map(wire) })
      history.applyServerMeta(r.assigned, batch)
      if (r.refused?.length) history.forget(r.refused)
      notePending()
    }
    backoff = 5000
    set({ mode: 'server', error: null, pending: history.pendingCount(), lastSyncAt: Date.now() })
  } catch (err) {
    fail(err)
  } finally {
    flushing = false
  }
}

/** The server did not answer: say so, and try the whole round again later. */
function fail(err: unknown): void {
  set({ mode: 'offline', error: err instanceof Error ? err.message : String(err), pending: history.pendingCount() })
  if (retryTimer) clearTimeout(retryTimer)
  retryTimer = setTimeout(() => { retryTimer = null; void catchUp() }, backoff)
  backoff = Math.min(60000, backoff * 2)
}

/** Pull what is new, then push what is waiting. */
async function catchUp(): Promise<void> {
  try {
    await pull()
  } catch (err) {
    if (err instanceof NotMounted) settleLocal(err)
    else fail(err)
    return
  }
  if (!connected) {
    connected = true
    if (!hidden()) openStream()
    // Files on disk that no record describes are filed on idle, once.
    setTimeout(() => { void recoverUnfiled().catch(() => {}) }, 4000)
  }
  await flush()
}

// ---------------------------------------------------------------- stream --

let stream: EventSource | null = null

const hidden = () => typeof document !== 'undefined' && document.visibilityState === 'hidden'

function openStream(): void {
  if (stream || typeof EventSource === 'undefined') return
  try {
    stream = new EventSource('/api/archive/stream')
  } catch {
    return
  }
  stream.onmessage = (ev) => {
    let rev = 0
    try { rev = Number((JSON.parse(ev.data) as { rev?: number }).rev) || 0 } catch { return }
    if (rev > history.syncCursor().rev) void pull().catch(() => {})
  }
}

function closeStream(): void {
  stream?.close()
  stream = null
}

/**
 * A hidden tab gives its stream back. Over plain HTTP a browser allows six
 * connections to one host across every tab, and an event stream holds one for
 * as long as it is open, so streams kept by background tabs leave nothing for
 * the pictures and pushes of the tab being used. Shown again, the tab reopens
 * it and catches up on what it missed.
 */
function onVisibility(): void {
  if (!connected) return
  if (hidden()) {
    closeStream()
    return
  }
  openStream()
  void catchUp()
}

let started = false
let unlisten: (() => void) | null = null

/** A static host: nothing to share with, and nothing that will change. */
function settleLocal(err: Error): void {
  unlisten?.()
  unlisten = null
  if (typeof document !== 'undefined') document.removeEventListener('visibilitychange', onVisibility)
  closeStream()
  if (flushTimer) clearTimeout(flushTimer)
  if (retryTimer) clearTimeout(retryTimer)
  history.setServerBacked(false)
  set({ mode: 'local', error: err.message, pending: 0 })
}

/**
 * Start once, from the one component mounted for the life of the page.
 * Resolves when the first sync has landed or been given up on.
 *
 * Changes are listened for before the server has answered, so one made while
 * it is away is pushed when it comes back. The page does not settle for
 * "this browser only" because a first request failed: only a server that
 * answers with something other than the archive is taken to be absent.
 */
export async function startArchiveSync(): Promise<void> {
  if (started) return
  started = true
  unlisten = onCommit((delta) => {
    if (!delta.remote) scheduleFlush()
    notePending()
  })
  if (typeof document !== 'undefined') document.addEventListener('visibilitychange', onVisibility)
  notePending()
  await catchUp()
}

/**
 * The whole archive as the server holds it, in the shape "Restore from a file"
 * reads. This browser keeps only a window of it, so a copy saved from here
 * would be missing whatever fell outside.
 */
export async function serverArchiveCopy(): Promise<string> {
  const data = await api<Pull>('/api/archive?since=0')
  return JSON.stringify({ v: 2, nextNo: data.nextNo, entries: data.records }, null, 2)
}
