import { spawnSync } from 'node:child_process'
import { existsSync, mkdirSync, readFileSync, rmSync, utimesSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest'
import { call, mounted, open, tempRoots, type Handler } from './http'

// Every server module reads its roots from the environment when it loads, so
// they are pointed at empty temporary folders first and imported after.
let root = ''
let outputs = ''
const handlers: Record<string, Handler> = {}

beforeAll(async () => {
  const roots = tempRoots()
  root = roots.root
  outputs = roots.outputs
  const [api, archive, thumbs, downloads, reel, vision] = await Promise.all([
    import('../server/api.mjs'),
    import('../server/archive.mjs'),
    import('../server/thumbs.mjs'),
    import('../server/downloads.mjs'),
    import('../server/reel.mjs'),
    import('../server/vision.mjs'),
  ])
  handlers.api = mounted(api.switchgenApi())
  handlers.archive = mounted(archive.switchgenArchive())
  handlers.thumbs = mounted(thumbs.switchgenThumbs())
  handlers.downloads = mounted(downloads.switchgenDownloads())
  handlers.reel = mounted(reel.switchgenReel())
  handlers.vision = mounted(vision.switchgenVision())
})

afterAll(async () => {
  // Let the archive's debounced write land before its folder goes.
  await new Promise((resolve) => setTimeout(resolve, 400))
  rmSync(root, { recursive: true, force: true })
})

describe('a request path that is not a URL', () => {
  // `new URL('//', base)` throws. Parsed outside a handler's own try, that
  // throw was an unhandled rejection, and Node ends the process on those: one
  // GET for // stopped the app for every device.
  it('is answered 400 by every middleware, which does not reject', async () => {
    for (const name of ['api', 'archive', 'thumbs', 'downloads', 'reel', 'vision']) {
      const r = await call(handlers[name]!, { url: '//' })
      expect(r.passed, name).toBe(false)
      expect(r.status, name).toBe(400)
      expect(r.json().error, name).toMatch(/not a valid URL/)
    }
  })

  it('leaves paths outside a middleware\'s own namespace to the next one', async () => {
    for (const name of ['archive', 'thumbs', 'downloads', 'reel', 'vision']) {
      expect((await call(handlers[name]!, { url: '/api/somewhere-else' })).passed, name).toBe(true)
    }
    expect((await call(handlers.api!, { url: '/index.html' })).passed).toBe(true)
  })
})

describe('the archive server', () => {
  const record = (id: string, extra: Record<string, unknown> = {}) => ({
    id,
    at: 1_700_000_000_000,
    file: { filename: `${id}.png`, subfolder: '', type: 'output' },
    prompt: 'a lighthouse at dusk',
    ...extra,
  })
  const post = (url: string, body: unknown) => call(handlers.archive!, { method: 'POST', url, body })
  const pull = async () => (await call(handlers.archive!, { url: '/api/archive' })).json()

  it('writes to the temporary archive, never the real one', async () => {
    expect(String((await pull()).file).startsWith(root)).toBe(true)
  })

  it('refuses to bring back a record removed after the writer last saw it', async () => {
    const put = (await post('/api/archive/upsert', { records: [record('r1')] })).json()
    expect(put.refused).toEqual([])
    const seen = put.assigned[0].rev as number

    // Another device removes it.
    expect((await post('/api/archive/remove', { ids: ['r1'] })).json().removed).toBe(1)

    // A tab that last saw the record at `seen` sends its edit afterwards.
    const stale = (await post('/api/archive/upsert', { records: [record('r1', { rev: seen, prompt: 'edited' })] })).json()
    expect(stale.refused).toEqual(['r1'])
    expect(stale.assigned).toEqual([])
    const after = await pull()
    expect(after.records.map((r: { id: string }) => r.id)).not.toContain('r1')
    expect(after.removed).toContain('r1')
  })

  it('lets an undo, which carries no stamp, and the restore route bring it back', async () => {
    await post('/api/archive/upsert', { records: [record('r2')] })
    const stamped = (await pull()).records.find((r: { id: string }) => r.id === 'r2')
    expect((await post('/api/archive/remove', { ids: ['r2'] })).json().removed).toBe(1)

    // Undo sends the record as it was before the server ever stamped it.
    const undo = (await post('/api/archive/upsert', { records: [record('r2')] })).json()
    expect(undo.assigned.map((a: { id: string }) => a.id)).toEqual(['r2'])

    expect((await post('/api/archive/remove', { ids: ['r2'] })).json().removed).toBe(1)
    const restored = (await post('/api/archive/restore', { records: [{ ...stamped }] })).json()
    expect(restored.assigned.map((a: { id: string }) => a.id)).toEqual(['r2'])
    expect((await pull()).records.map((r: { id: string }) => r.id)).toContain('r2')
  })

  it('refuses a write from another origin before reading it', async () => {
    const r = await call(handlers.archive!, {
      method: 'POST',
      url: '/api/archive/upsert',
      headers: { origin: 'http://evil.example', host: '127.0.0.1:5273' },
      body: { records: [record('r3')] },
    })
    expect(r.status).toBe(403)
    expect((await pull()).records.map((x: { id: string }) => x.id)).not.toContain('r3')
  })
})

describe('a removed record\'s file', () => {
  // The files stay on disk when a record goes, and the recovery pass must not
  // file them again. The outputs listing is cached for 4 s, so the clock is
  // moved past that rather than waited out.
  afterEach(() => {
    vi.useRealTimers()
  })
  const post = (url: string, body: unknown) => call(handlers.archive!, { method: 'POST', url, body })
  // Only ever forward: the cache remembers the time it was filled.
  let clock = 0
  const listed = async (rel: string) => {
    clock = Math.max(clock, Date.now()) + 5000
    vi.setSystemTime(clock)
    return (await call(handlers.archive!, { url: '/api/outputs' })).json().files.find((f: { rel: string }) => f.rel === rel)
  }
  const aged = (rel: string, secondsAgo: number) => {
    const full = path.join(outputs, rel)
    writeFileSync(full, 'x')
    const t = Date.now() / 1000 - secondsAgo
    utimesSync(full, t, t)
  }
  const recordFor = (id: string, filename: string) => ({ id, at: 1, file: { filename, subfolder: '', type: 'output' } })

  it('is marked dismissed, the mark is saved, and an undo takes it back', async () => {
    vi.useFakeTimers({ toFake: ['Date'] })
    aged('dismiss-a.png', 60)
    await post('/api/archive/upsert', { records: [recordFor('DA', 'dismiss-a.png')] })
    await post('/api/archive/remove', { ids: ['DA'] })
    expect((await listed('dismiss-a.png'))?.dismissed).toBe(true)

    vi.useRealTimers()
    await new Promise((resolve) => setTimeout(resolve, 400))
    const saved = JSON.parse(readFileSync(process.env.SWITCHGEN_ARCHIVE!, 'utf8'))
    expect(typeof saved.dismissed?.['dismiss-a.png']).toBe('number')

    // The undo sends the record back unstamped.
    vi.useFakeTimers({ toFake: ['Date'] })
    expect((await post('/api/archive/upsert', { records: [recordFor('DA', 'dismiss-a.png')] })).json().assigned).toHaveLength(1)
    const back = await listed('dismiss-a.png')
    expect(back?.filed).toBe(true)
    expect(back?.dismissed).toBeUndefined()
  })

  it('is not dismissed once a new file is written at its path', async () => {
    vi.useFakeTimers({ toFake: ['Date'] })
    aged('dismiss-b.png', 60)
    await post('/api/archive/upsert', { records: [recordFor('DB', 'dismiss-b.png')] })
    await post('/api/archive/remove', { ids: ['DB'] })
    expect((await listed('dismiss-b.png'))?.dismissed).toBe(true)
    // A render that took the name afterwards: written after the removal.
    const later = Date.now() / 1000 + 2
    writeFileSync(path.join(outputs, 'dismiss-b.png'), 'y')
    utimesSync(path.join(outputs, 'dismiss-b.png'), later, later)
    expect((await listed('dismiss-b.png'))?.dismissed).toBeUndefined()
  })
})

describe('where the archive\'s log stands', () => {
  it('tells a full pull which load of the log it is, what it read from disk, and each removal\'s rev', async () => {
    const post = (url: string, body: unknown) => call(handlers.archive!, { method: 'POST', url, body })
    await post('/api/archive/upsert', { records: [{ id: 'L1', at: 1, file: { filename: 'l1.png', subfolder: '', type: 'output' } }] })
    const gone = (await post('/api/archive/remove', { ids: ['L1'] })).json()
    const pull = (await call(handlers.archive!, { url: '/api/archive?since=0' })).json()
    expect(typeof pull.boot).toBe('string')
    // Nothing was on disk when this log was made.
    expect(pull.base).toBe(0)
    expect(pull.removedRev.L1).toBe(gone.rev)
    expect(pull.removed).toContain('L1')
  })

  it('opens the stream with the rev, the log and the load', async () => {
    const s = open(handlers.archive!, { url: '/api/archive/stream' })
    await vi.waitFor(() => {
      if (!s.reply.body.includes('\n\n')) throw new Error('nothing yet')
    })
    s.hangUp()
    const first = JSON.parse(/^data: (.*)$/m.exec(s.reply.body)![1]!)
    const pull = (await call(handlers.archive!, { url: '/api/archive' })).json()
    expect(first).toEqual({ rev: pull.rev, epoch: pull.epoch, boot: pull.boot })
  })
})

describe('the archive when the process ends', () => {
  // A child process files one record and ends at once, before the debounced
  // save. Vite ends on SIGTERM with process.exit(); Ctrl-C is SIGINT.
  const ARCHIVE_MJS = pathToFileURL(path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'server', 'archive.mjs')).href
  const CHILD = `
import { EventEmitter } from 'node:events'
import { Readable } from 'node:stream'
const [how, dir] = process.argv.slice(2)
process.env.SWITCHGEN_OUTPUTS = dir
process.env.SWITCHGEN_ARCHIVE = dir + '/.switchgen/archive.json'
const { switchgenArchive } = await import(${JSON.stringify(ARCHIVE_MJS)})
let handler
switchgenArchive().configurePreviewServer({ middlewares: { use: (fn) => { handler = fn } } })
const body = JSON.stringify({ records: [{ id: 'kept', at: 1, file: { filename: 'k.png', subfolder: '', type: 'output' } }] })
const req = Object.assign(Readable.from([Buffer.from(body)]), { method: 'POST', url: '/api/archive/upsert', headers: { 'content-type': 'application/json' } })
await new Promise((resolve) => {
  const res = Object.assign(new EventEmitter(), { req, statusCode: 200, headersSent: false, setHeader() {}, writeHead() {}, write() {}, end: resolve })
  handler(req, res, resolve)
})
if (how === 'exit') process.exit(0)
if (how === 'SIGINT') process.kill(process.pid, 'SIGINT')
setTimeout(() => process.exit(9), 3000)
`
  let dir = ''
  let script = ''
  beforeAll(() => {
    dir = path.join(root, 'exit')
    mkdirSync(dir, { recursive: true })
    script = path.join(dir, 'child.mjs')
    writeFileSync(script, CHILD)
  })
  const endWith = (how: string) => {
    const out = path.join(dir, how)
    mkdirSync(out, { recursive: true })
    const r = spawnSync(process.execPath, [script, how, out], { timeout: 15_000, encoding: 'utf8' })
    const file = path.join(out, '.switchgen', 'archive.json')
    return { status: r.status, saved: existsSync(file) ? JSON.parse(readFileSync(file, 'utf8')) : null }
  }

  it('writes what it answered when the process exits straight after', () => {
    const r = endWith('exit')
    expect(r.status).toBe(0)
    expect(Object.keys(r.saved?.records ?? {})).toEqual(['kept'])
  })

  it('writes and ends on Ctrl-C with the code the default would give', () => {
    const r = endWith('SIGINT')
    expect(r.status).toBe(130)
    expect(Object.keys(r.saved?.records ?? {})).toEqual(['kept'])
  })
})
