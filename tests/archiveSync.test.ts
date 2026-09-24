import { copyFileSync, readFileSync, rmSync } from 'node:fs'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'
import type * as History from '../src/lib/history'
import type * as Sync from '../src/lib/archiveSync'
import { call, mounted, tempRoots, type Handler } from './http'

/**
 * The browser's sync against the real archive server, joined by a fetch that
 * calls the middleware directly. A server is "restarted" by loading a fresh
 * copy of its module on a given archive file, which is what a process that
 * read that file from disk would hold. The event stream is stood in for, and
 * a test says when it speaks.
 */
let root = ''
let outputs = ''
let server: Handler
let h: typeof History
let sync: typeof Sync
let loads = 0
const asked: string[] = []

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))
/** Wait for the sync to get somewhere, rather than for a fixed time that a slow runner might not keep to. */
const until = (what: () => boolean | Promise<boolean>) =>
  vi.waitFor(
    async () => {
      if (!(await what())) throw new Error('not yet')
    },
    { timeout: 5000, interval: 25 },
  )
/** How many records the file on disk holds. */
const savedIn = (file: string) => {
  try {
    return Object.keys(JSON.parse(readFileSync(file, 'utf8')).records).length
  } catch {
    return 0
  }
}

async function boot(file: string): Promise<Handler> {
  process.env.SWITCHGEN_ARCHIVE = file
  // A query makes a fresh copy of the module: a new load of the file.
  const spec = `../server/archive.mjs?load=${++loads}`
  const mod = await import(/* @vite-ignore */ spec)
  return mounted(mod.switchgenArchive())
}

const streams: { onmessage: ((ev: { data: string }) => void) | null }[] = []
/** The stream reconnects to whichever server is up, and says where its log is. */
async function streamSpeaks(): Promise<void> {
  const d = (await call(server, { url: '/api/archive' })).json()
  for (const s of streams) s.onmessage?.({ data: JSON.stringify({ rev: d.rev, epoch: d.epoch, boot: d.boot }) })
}
const held = async () => (await call(server, { url: '/api/archive?since=0' })).json()

const entry = (name: string): History.NewEntry => ({
  desk: 'images',
  kind: 'image',
  mode: 't2i',
  file: { filename: name, subfolder: '', type: 'output' },
  familyId: 'x',
  familyLabel: 'X',
  variant: null,
  model: 'm',
  modelLabel: 'M',
  prompt: name,
  negative: null,
  seed: 1,
  steps: 1,
  cfg: 1,
  sampler: 's',
  scheduler: 's',
  width: 1,
  height: 1,
  promptId: '',
  durationMs: 0,
})

beforeAll(async () => {
  const roots = tempRoots()
  root = roots.root
  outputs = roots.outputs
  vi.stubGlobal('fetch', async (url: string, init: { method?: string; body?: string } = {}) => {
    asked.push(url)
    if (!url.startsWith('/api/archive') && url !== '/api/outputs') {
      return new Response(JSON.stringify({ error: 'not here' }), { status: 404, headers: { 'content-type': 'application/json' } })
    }
    const r = await call(server, { method: init.method ?? 'GET', url, body: init.body ? JSON.parse(init.body) : undefined })
    return new Response(r.body, { status: r.status, headers: { 'content-type': String(r.headers['content-type'] ?? 'application/json') } })
  })
  vi.stubGlobal(
    'EventSource',
    class {
      onmessage: ((ev: { data: string }) => void) | null = null
      constructor() {
        streams.push(this)
      }
      close() {}
    },
  )
  server = await boot(process.env.SWITCHGEN_ARCHIVE!)
  h = await import('../src/lib/history')
  sync = await import('../src/lib/archiveSync')
  await sync.startArchiveSync()
})

afterAll(async () => {
  // Let every server's debounced save land before its folder goes.
  await sleep(500)
  vi.unstubAllGlobals()
  rmSync(root, { recursive: true, force: true })
})

describe('a server that restarts behind what it answered', () => {
  const archiveFile = () => path.join(outputs, '.switchgen', 'archive.json')
  let persisted = ''
  let ids: string[] = []

  it('gets back the records it lost, and this browser gets what was written since', async () => {
    ids = ['a.png', 'b.png', 'c.png'].map((n) => h.add(entry(n)).id)
    await until(async () => (await held()).records.length === 3)
    // The server's save lands, then it answers two more it never saves.
    await until(() => savedIn(archiveFile()) === 3)
    persisted = path.join(root, 'persisted.json')
    copyFileSync(archiveFile(), persisted)
    ids.push(...['d.png', 'e.png'].map((n) => h.add(entry(n)).id))
    await until(() => h.get(ids[4]!)?.rev !== undefined && h.pendingCount() === 0)

    // Killed before saving d and e, it reads the saved file on the way back up.
    server = await boot(persisted)
    expect((await held()).records).toHaveLength(3)
    // Another device files f at once, reusing a rev this browser holds.
    await call(server, { method: 'POST', url: '/api/archive/upsert', body: { records: [{ id: 'f', at: 1, file: { filename: 'f.png', subfolder: '', type: 'output' } }] } })
    await streamSpeaks()
    await until(async () => (await held()).records.length === 6 && h.pendingCount() === 0)

    const now = (await held()).records.map((r: { id: string }) => r.id)
    expect(now).toEqual(expect.arrayContaining([ids[3], ids[4]]))
    expect(h.get('f')).toBeDefined()
    expect(h.pendingCount()).toBe(0)
  }, 15_000)

  it('does not read the whole archive again after a restart that lost nothing', async () => {
    // The restarted server's own save lands, and it restarts on that file.
    await until(() => savedIn(persisted) === 6)
    server = await boot(persisted)
    const { boot: load } = (await held()) as { boot: string }
    asked.length = 0
    await streamSpeaks()
    // The pull is over once this browser has taken the new load as its own.
    await until(() => h.syncCursor().boot === load)
    expect(asked.some((u) => u.startsWith('/api/archive?since='))).toBe(true)
    expect(asked.some((u) => u.endsWith('since=0'))).toBe(false)
  }, 15_000)

  it('sends every record to a server whose archive file was lost', async () => {
    await until(() => savedIn(persisted) === 6)
    rmSync(path.join(outputs, '.switchgen'), { recursive: true, force: true })
    server = await boot(archiveFile())
    expect((await held()).records).toHaveLength(0)
    await streamSpeaks()
    await until(async () => (await held()).records.length === 6 && h.pendingCount() === 0)
    const back = (await held()).records.map((r: { id: string }) => r.id).sort()
    expect(back).toEqual(h.all().map((e) => e.id).sort())
    expect(h.pendingCount()).toBe(0)
  }, 15_000)
})

describe('an undo after the removal reached the server', () => {
  it('brings the record back for every device, though another device\'s stale edit of it was refused', async () => {
    const r = h.add(entry('u.png'))
    await until(() => h.get(r.id)?.rev !== undefined)
    // Another device holds the stamped copy and has not heard of the removal.
    const theirs = (await held()).records.find((x: { id: string }) => x.id === r.id)
    expect(theirs?.rev).toBeDefined()
    const [removed] = h.removeMany([r.id])
    await until(async () => (await held()).removed.includes(r.id) && h.pendingCount() === 0)
    const stale = await call(server, {
      method: 'POST',
      url: '/api/archive/upsert',
      body: { records: [{ ...theirs, prompt: 'edited on the phone' }] },
    })
    expect(stale.json().refused).toEqual([r.id])

    // The reader undoes the removal here.
    h.restore(removed!)
    await until(() => h.pendingCount() === 0)
    await until(async () => (await held()).records.some((x: { id: string }) => x.id === r.id))
    const back = (await held()).records.find((x: { id: string }) => x.id === r.id)
    expect(back?.prompt).toBe('u.png')
    expect(h.get(r.id)?.rev).toBe(back?.rev)
    expect(h.pendingCount()).toBe(0)
  }, 20_000)
})

describe('a clip filed after the fact on one device and by its desk on another', () => {
  it('shows the note the reader wrote on the other device, and keeps it through an edit here', async () => {
    // Another device's recovery pass files the clip, and the reader stars it
    // there and writes a note. This browser has not pulled since.
    const file = { filename: 'fold_00001_.webm', subfolder: 'video', type: 'output' }
    const put = (records: unknown[]) => call(server, { method: 'POST', url: '/api/archive/upsert', body: { records } })
    const filed = (await put([{ id: 'elsewhere', at: 1, file, recovered: true }])).json()
    await put([{ id: 'elsewhere', at: 1, file, recovered: true, rev: filed.assigned[0].rev, starred: true, note: 'n' }])
    expect(h.get('elsewhere')).toBeUndefined()

    // The desk here files the clip it made, and its push takes that one's place.
    const mine = h.add({ ...entry('fold_00001_.webm'), desk: 'video', kind: 'video', mode: 't2v', file })
    await until(() => h.get(mine.id)?.rev !== undefined && h.pendingCount() === 0)
    expect(h.get(mine.id)?.note).toBe('n')
    expect(h.get(mine.id)?.starred).toBe(true)

    // Its next edit is sent whole, and must not take the note away everywhere.
    h.update(mine.id, { tags: ['tram'] })
    await until(() => h.pendingCount() === 0)
    const there = (await held()).records.find((r: { id: string }) => r.id === mine.id)
    expect(there).toMatchObject({ note: 'n', starred: true, tags: ['tram'] })
    expect((await held()).removed).toContain('elsewhere')
  }, 15_000)
})
