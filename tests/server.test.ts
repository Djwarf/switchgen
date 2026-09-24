import { spawn, spawnSync, type ChildProcess } from 'node:child_process'
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

    // Saved after a debounce and an asynchronous write, so asked until it lands.
    vi.useRealTimers()
    await vi.waitFor(
      () => {
        const saved = JSON.parse(readFileSync(process.env.SWITCHGEN_ARCHIVE!, 'utf8'))
        expect(typeof saved.dismissed?.['dismiss-a.png']).toBe('number')
      },
      { timeout: 5000, interval: 50 },
    )

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

  // The mark is enforced here too, not only by a client that reads it: an
  // older bundle still in a browser's cache files every unnamed file it finds.
  const recovered = (id: string, filename: string, extra: Record<string, unknown> = {}) => ({ ...recordFor(id, filename), recovered: true, ...extra })
  const removedOnce = async (id: string, filename: string) => {
    aged(filename, 60)
    await post('/api/archive/upsert', { records: [recordFor(id, filename)] })
    await post('/api/archive/remove', { ids: [id] })
  }
  const ids = (a: { id: string }[]) => a.map((x) => x.id)

  it('refuses a record filed after the fact for it, and the removal stands', async () => {
    vi.useFakeTimers({ toFake: ['Date'] })
    await removedOnce('RF1', 'refile-1.png')
    const r = (await post('/api/archive/upsert', { records: [recovered('OLD1', 'refile-1.png')] })).json()
    expect(r.refused).toEqual(['OLD1'])
    expect(r.assigned).toEqual([])
    expect((await listed('refile-1.png'))?.dismissed).toBe(true)
  })

  it('takes one the reader asked for, marked as refiled', async () => {
    vi.useFakeTimers({ toFake: ['Date'] })
    await removedOnce('RF2', 'refile-2.png')
    const r = (await post('/api/archive/upsert', { records: [recovered('NEW2', 'refile-2.png', { refiled: true })] })).json()
    expect(ids(r.assigned)).toEqual(['NEW2'])
    expect(r.refused).toEqual([])
    expect((await listed('refile-2.png'))?.filed).toBe(true)
  })

  it('takes one sent through the restore route, which is always asked for', async () => {
    vi.useFakeTimers({ toFake: ['Date'] })
    await removedOnce('RF3', 'refile-3.png')
    const r = (await post('/api/archive/restore', { records: [recovered('NEW3', 'refile-3.png')] })).json()
    expect(ids(r.assigned)).toEqual(['NEW3'])
  })

  it('takes back a recovered record removed here, sent back by an undo', async () => {
    vi.useFakeTimers({ toFake: ['Date'] })
    aged('refile-4.png', 60)
    await post('/api/archive/upsert', { records: [recovered('R4', 'refile-4.png')] })
    await post('/api/archive/remove', { ids: ['R4'] })
    // Its tombstone is still here, so this is the removed record coming back.
    const r = (await post('/api/archive/upsert', { records: [recovered('R4', 'refile-4.png')] })).json()
    expect(ids(r.assigned)).toEqual(['R4'])
  })

  it('takes one for a file written at that path since the removal', async () => {
    vi.useFakeTimers({ toFake: ['Date'] })
    await removedOnce('RF5', 'refile-5.png')
    const later = Date.now() / 1000 + 2
    writeFileSync(path.join(outputs, 'refile-5.png'), 'y')
    utimesSync(path.join(outputs, 'refile-5.png'), later, later)
    const r = (await post('/api/archive/upsert', { records: [recovered('NEW5', 'refile-5.png')] })).json()
    expect(ids(r.assigned)).toEqual(['NEW5'])
  })

  it('lets a desk\'s record take the place of one filed after the fact for the same file', async () => {
    vi.useFakeTimers({ toFake: ['Date'] })
    mkdirSync(path.join(outputs, 'video'), { recursive: true })
    aged('video/wan_00001_.webm', 60)
    const clip = { filename: 'wan_00001_.webm', subfolder: 'video', type: 'output' }
    const first = (await post('/api/archive/upsert', { records: [{ id: 'rec1', at: 1, file: clip, recovered: true, starred: true, note: 'n' }] })).json()
    const no = first.assigned[0].no
    const desk = (await post('/api/archive/upsert', { records: [{ id: 'desk1', at: 2, file: clip, promptId: 'p1', no: 7 }] })).json()
    expect(ids(desk.assigned)).toEqual(['desk1'])

    const pull = (await call(handlers.archive!, { url: '/api/archive?since=0' })).json()
    const forClip = pull.records.filter((r: { file: { filename: string } }) => r.file.filename === 'wan_00001_.webm')
    expect(ids(forClip)).toEqual(['desk1'])
    expect(forClip[0]).toMatchObject({ no, starred: true, note: 'n', promptId: 'p1' })
    expect(forClip[0].recovered).toBeUndefined()
    expect(pull.removed).toContain('rec1')

    // A device that still holds the old record, as stamped, cannot bring it back.
    const stale = (await post('/api/archive/upsert', { records: [{ id: 'rec1', rev: 1, at: 1, file: clip, recovered: true }] })).json()
    expect(stale.refused).toEqual(['rec1'])
    const listing = await listed('video/wan_00001_.webm')
    expect(listing?.filed).toBe(true)
    expect(listing?.dismissed).toBeUndefined()
  })

  it('tells the desk that filed in its place what it kept from the record it replaced', async () => {
    vi.useFakeTimers({ toFake: ['Date'] })
    mkdirSync(path.join(outputs, 'video'), { recursive: true })
    aged('video/wan_00002_.webm', 60)
    const clip = { filename: 'wan_00002_.webm', subfolder: 'video', type: 'output' }
    await post('/api/archive/upsert', {
      records: [{ id: 'rec3', at: 1, file: clip, recovered: true, starred: true, note: 'n', tags: ['a'], rating: 'general' }],
    })
    // The desk's copy has tags of its own, read on the device that made it.
    const desk = (await post('/api/archive/upsert', { records: [{ id: 'desk3', at: 2, file: clip, promptId: 'p3', tags: ['own'] }] })).json()
    expect(ids(desk.assigned)).toEqual(['desk3'])
    // Without these the desk's device, stamped at this rev, never pulls them,
    // and its next edit, sent whole, would put its copy over them everywhere.
    expect(desk.assigned[0].kept).toEqual({ starred: true, note: 'n', rating: 'general' })

    // The same record again, now that it is here, took no one's place.
    const stamped = (await call(handlers.archive!, { url: '/api/archive?since=0' })).json().records.find((r: { id: string }) => r.id === 'desk3')
    expect(stamped).toMatchObject({ starred: true, note: 'n', tags: ['own'], rating: 'general' })
    const again = (await post('/api/archive/upsert', { records: [stamped] })).json()
    expect(ids(again.assigned)).toEqual(['desk3'])
    expect(again.assigned[0].kept).toBeUndefined()
  })

  it('still refuses a record filed after the fact for a file a desk has filed', async () => {
    vi.useFakeTimers({ toFake: ['Date'] })
    aged('desk-first.png', 60)
    const file = { filename: 'desk-first.png', subfolder: '', type: 'output' }
    await post('/api/archive/upsert', { records: [{ id: 'desk2', at: 1, file, promptId: 'p2' }] })
    const r = (await post('/api/archive/upsert', { records: [{ id: 'rec2', at: 1, file, recovered: true }] })).json()
    expect(r.refused).toEqual(['rec2'])
    const pull = (await call(handlers.archive!, { url: '/api/archive?since=0' })).json()
    expect(ids(pull.records.filter((x: { file: { filename: string } }) => x.file.filename === 'desk-first.png'))).toEqual(['desk2'])
  })

  it('refuses one for a file that has gone from disk since, so the removal stands', async () => {
    vi.useFakeTimers({ toFake: ['Date'] })
    await removedOnce('RF6', 'refile-6.png')
    rmSync(path.join(outputs, 'refile-6.png'))
    const r = (await post('/api/archive/upsert', { records: [recovered('NEW6', 'refile-6.png')] })).json()
    expect(r.refused).toEqual(['NEW6'])
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
    }, { timeout: 5000 })
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
  }, 20_000)

  it('writes and ends on Ctrl-C with the code the default would give', () => {
    const r = endWith('SIGINT')
    expect(r.status).toBe(130)
    expect(Object.keys(r.saved?.records ?? {})).toEqual(['kept'])
  }, 20_000)
})

describe('the archive loaded again in one process', () => {
  // Vite loads server/archive.mjs afresh each time its config reloads, in the
  // same process. Each load keeps its own copy of the archive, and only the
  // newest may write it at exit: an older load holds the archive as it stood
  // before, and every load kept on the list was one more copy in memory.
  const ARCHIVE_MJS = pathToFileURL(path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'server', 'archive.mjs')).href
  const CHILD = `
import { EventEmitter } from 'node:events'
import { chmodSync, mkdirSync, readFileSync } from 'node:fs'
import { Readable } from 'node:stream'
const [how, dir] = process.argv.slice(2)
process.env.SWITCHGEN_OUTPUTS = dir
process.env.SWITCHGEN_ARCHIVE = dir + '/.switchgen/archive.json'
const KEY = Symbol.for('switchgen.archive.flush')
const report = (o) => console.log('REPORT ' + JSON.stringify(o))
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms))
const loadNo = async (n) => {
  const { switchgenArchive } = await import(${JSON.stringify(ARCHIVE_MJS)} + '?load=' + n)
  let handler
  switchgenArchive().configurePreviewServer({ middlewares: { use: (fn) => { handler = fn } } })
  return (id) => {
    const body = JSON.stringify({ records: [{ id, at: 1, file: { filename: id + '.png', subfolder: '', type: 'output' } }] })
    const req = Object.assign(Readable.from([Buffer.from(body)]), { method: 'POST', url: '/api/archive/upsert', headers: { 'content-type': 'application/json' } })
    return new Promise((resolve) => {
      const res = Object.assign(new EventEmitter(), { req, statusCode: 200, headersSent: false, setHeader() {}, writeHead() {}, write() {}, end: resolve })
      handler(req, res, resolve)
    })
  }
}

if (how === 'twice') {
  await loadNo(1)
  await loadNo(2)
  const list = globalThis[KEY]
  report({ map: list instanceof Map, size: list?.size })
  process.exit(0)
}

if (how === 'handover') {
  // What a build before this one left behind: a Set of flushes, and exit
  // hooks that call everything in it.
  const old = new Set([() => console.log('OLD FLUSH CALLED')])
  globalThis[KEY] = old
  process.once('exit', () => { for (const fn of old) fn() })
  await loadNo(1)
  report({ oldSize: old.size, map: globalThis[KEY] instanceof Map })
  process.exit(0)
}

// 'exit' or 'SIGINT': an older load whose last write failed, then a newer one.
// Each step waits for what it needs to have happened, not for a set time.
const until = async (what, done) => {
  for (let i = 0; i < 500 && !done(); i++) await wait(20)
  if (!done()) { report({ timedOut: what }); process.exit(8) }
}
const onDisk = () => {
  try { return Object.keys(JSON.parse(readFileSync(dir + '/.switchgen/archive.json', 'utf8')).records ?? {}) } catch { return [] }
}
let writeFailed = false
const warn = console.warn
console.warn = (...args) => {
  if (String(args[0]).includes('could not write')) writeFailed = true
  warn(...args)
}
const one = await loadNo(1)
await one('x')
await until('x on disk', () => onDisk().includes('x'))
const folder = dir + '/.switchgen'
chmodSync(folder, 0o500)
await one('stale')
await until('the failed write', () => writeFailed)
chmodSync(folder, 0o700)
const two = await loadNo(2)
await two('fresh')
if (how === 'exit') process.exit(0)
if (how === 'SIGINT') process.kill(process.pid, 'SIGINT')
setTimeout(() => process.exit(9), 3000)
`
  let dir = ''
  let script = ''
  beforeAll(() => {
    dir = path.join(root, 'reload')
    mkdirSync(dir, { recursive: true })
    script = path.join(dir, 'child.mjs')
    writeFileSync(script, CHILD)
  })
  const runChild = (how: string) => {
    const out = path.join(dir, how)
    mkdirSync(out, { recursive: true })
    const r = spawnSync(process.execPath, [script, how, out], { timeout: 15_000, encoding: 'utf8' })
    const line = /^REPORT (.*)$/m.exec(r.stdout ?? '')?.[1]
    const file = path.join(out, '.switchgen', 'archive.json')
    return {
      status: r.status,
      stdout: r.stdout ?? '',
      report: line ? JSON.parse(line) : null,
      saved: existsSync(file) ? Object.keys(JSON.parse(readFileSync(file, 'utf8')).records ?? {}).sort() : null,
    }
  }

  it('keeps one flush per archive file, however many times it is loaded', () => {
    const r = runChild('twice')
    expect(r.status).toBe(0)
    expect(r.report).toEqual({ map: true, size: 1 })
  }, 20_000)

  // A folder made read-only does not stop root, so the failed write this
  // needs cannot be staged when the suite runs as root.
  const asRoot = process.getuid?.() === 0
  it.skipIf(asRoot)('lets only the newest load write at exit, not an older one whose last write failed', () => {
    for (const how of ['exit', 'SIGINT']) {
      const r = runChild(how)
      expect(r.report, how).toBeNull()
      expect(r.status, how).toBe(how === 'exit' ? 0 : 130)
      expect(r.saved, how).toEqual(['fresh', 'x'])
    }
  }, 40_000)

  it('takes over from the list an older build left, which then calls nothing', () => {
    const r = runChild('handover')
    expect(r.status).toBe(0)
    expect(r.report).toEqual({ oldSize: 0, map: true })
    expect(r.stdout).not.toContain('OLD FLUSH CALLED')
  }, 20_000)
})

describe('one server to an archive', () => {
  // Two servers on one archive each hold it whole and write it whole, so the
  // one that saves last undoes the other's records. Each server here is a
  // process of its own, driven by lines on its stdin.
  const ARCHIVE_MJS = pathToFileURL(path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'server', 'archive.mjs')).href
  const SERVER = `
import { EventEmitter } from 'node:events'
import readline from 'node:readline'
import { Readable } from 'node:stream'
const [dir] = process.argv.slice(2)
process.env.SWITCHGEN_OUTPUTS = dir
process.env.SWITCHGEN_ARCHIVE = dir + '/.switchgen/archive.json'
const { switchgenArchive } = await import(${JSON.stringify(ARCHIVE_MJS)})
let handler
switchgenArchive().configurePreviewServer({ middlewares: { use: (fn) => { handler = fn } } })
const ask = (method, url, body) => new Promise((resolve) => {
  const raw = body === undefined ? [] : [Buffer.from(JSON.stringify(body))]
  const req = Object.assign(Readable.from(raw), { method, url, headers: body === undefined ? {} : { 'content-type': 'application/json' } })
  let status = 200
  let text = ''
  const res = Object.assign(new EventEmitter(), {
    req, headersSent: false, destroyed: false, setHeader() {}, getHeader() {},
    writeHead(code) { status = code; return res },
    write(chunk) { text += chunk; return true },
    end(chunk) { if (chunk !== undefined) text += chunk; resolve({ status, body: text }) },
  })
  Object.defineProperty(res, 'statusCode', { get: () => status, set: (v) => { status = v } })
  handler(req, res, () => resolve({ status: 0, body: 'passed on' }))
})
console.log('READY ' + process.pid)
for await (const line of readline.createInterface({ input: process.stdin })) {
  const [what, arg] = line.split(' ')
  if (what === 'exit') process.exit(0)
  const record = { id: arg, at: 1, file: { filename: arg + '.png', subfolder: '', type: 'output' } }
  const r = what === 'upsert' ? await ask('POST', '/api/archive/upsert', { records: [record] }) : await ask('GET', arg)
  console.log('OUT ' + JSON.stringify(r))
}
`
  let dir = ''
  let script = ''
  const running: ChildProcess[] = []
  beforeAll(() => {
    dir = path.join(root, 'lock')
    mkdirSync(dir, { recursive: true })
    script = path.join(dir, 'server.mjs')
    writeFileSync(script, SERVER)
  })
  afterAll(() => {
    for (const child of running) child.kill()
  })

  /** A server process on the archive in `folder`, and a way to ask it things. */
  function server(folder: string) {
    mkdirSync(folder, { recursive: true })
    const child = spawn(process.execPath, [script, folder], { stdio: ['pipe', 'pipe', 'ignore'] })
    running.push(child)
    const lines: string[] = []
    const waiting: (() => void)[] = []
    let rest = ''
    child.stdout!.on('data', (d: Buffer) => {
      rest += d.toString()
      const parts = rest.split('\n')
      rest = parts.pop()!
      lines.push(...parts)
      for (const wake of waiting.splice(0)) wake()
    })
    const exited = new Promise<number | null>((resolve) => child.on('exit', resolve))
    const next = async (prefix: string): Promise<string> => {
      for (;;) {
        const i = lines.findIndex((l) => l.startsWith(prefix))
        if (i >= 0) return lines.splice(i, 1)[0]!.slice(prefix.length)
        await new Promise<void>((resolve) => waiting.push(resolve))
      }
    }
    const ready = next('READY ')
    return {
      pid: child.pid!,
      ready,
      exited,
      async ask(line: string): Promise<{ status: number; json: any }> {
        child.stdin!.write(`${line}\n`)
        const r = JSON.parse(await next('OUT ')) as { status: number; body: string }
        return { status: r.status, json: JSON.parse(r.body) }
      },
      exit() {
        child.stdin!.write('exit\n')
        return exited
      },
    }
  }
  const lockOf = (folder: string) => path.join(folder, '.switchgen', 'archive.json.lock')

  it('lets the second server stand back, naming the first, and take over once it stops', async () => {
    const folder = path.join(dir, 'two')
    const a = server(folder)
    await a.ready
    expect((await a.ask('upsert fromA')).status).toBe(200)
    const b = server(folder)
    await b.ready

    const write = await b.ask('upsert fromB')
    expect(write.status).toBe(503)
    expect(write.json.busy).toBe('archive')
    expect(write.json.error).toContain(`process ${a.pid}`)
    const files = await b.ask('get /api/outputs')
    expect(files.status).toBe(503)
    expect(files.json.busy).toBe('archive')

    expect(await a.exit()).toBe(0)
    expect(existsSync(lockOf(folder))).toBe(false)
    const pull = await b.ask('get /api/archive')
    expect(pull.status).toBe(200)
    expect(pull.json.records.map((r: { id: string }) => r.id)).toEqual(['fromA'])
    await b.exit()
  }, 30_000)

  const onLinux = existsSync('/proc/self/stat') && existsSync('/proc/sys/kernel/random/boot_id')
  const boot = () => readFileSync('/proc/sys/kernel/random/boot_id', 'utf8').trim()
  const startOf = (pid: number) => {
    const stat = readFileSync(`/proc/${pid}/stat`, 'utf8')
    return stat.slice(stat.lastIndexOf(')') + 2).split(' ')[19]
  }
  /** What a server answers for the archive when it starts with `holder` in the lock. */
  const withLock = async (name: string, holder: Record<string, unknown>) => {
    const folder = path.join(dir, name)
    mkdirSync(path.join(folder, '.switchgen'), { recursive: true })
    writeFileSync(lockOf(folder), JSON.stringify(holder))
    const s = server(folder)
    await s.ready
    const r = await s.ask('get /api/archive')
    await s.exit()
    return r
  }

  it('takes a lock left by a process that has gone', async () => {
    const gone = spawnSync(process.execPath, ['-e', '']).pid
    expect((await withLock('dead', { pid: gone })).status).toBe(200)
  }, 20_000)

  it.skipIf(!onLinux)('takes a lock whose number now names another process, or one from before the machine restarted', async () => {
    expect((await withLock('reused', { pid: 1, boot: boot(), start: '999' })).status).toBe(200)
    expect((await withLock('rebooted', { pid: 1, boot: 'old' })).status).toBe(200)
  }, 20_000)

  it.skipIf(!onLinux)('leaves a lock held by a server that still runs', async () => {
    const r = await withLock('live', { pid: process.pid, boot: boot(), start: startOf(process.pid) })
    expect(r.status).toBe(503)
    expect(r.json.busy).toBe('archive')
    expect(r.json.error).toContain(`process ${process.pid}`)
  }, 20_000)
})

describe('the queue on the server in /api/capabilities', () => {
  // The queue's word comes from the registry its plugin keeps; none is
  // mounted here, so each case puts its own word there and takes it away.
  const KEY = Symbol.for('switchgen.runner')
  const registry = globalThis as Record<symbol, unknown>
  const caps = async () => {
    const r = await call(handlers.api!, { url: '/api/capabilities' })
    expect(r.status).toBe(200)
    return r.json()
  }
  afterEach(() => {
    delete registry[KEY]
    vi.restoreAllMocks()
  })

  it('says there is no queue when none is mounted, and every other field as before', async () => {
    expect(await caps()).toMatchObject({ server: 'switchgen', deleteFiles: true, archive: true, runner: false, runnerDesks: [], runnerReason: 'The queue on the server is not running.' })
  })

  it('passes on the queue\'s own word, desks only while it runs and a reason only while it does not', async () => {
    registry[KEY] = { current: { status: () => ({ active: true, desks: ['video', 'reel'], reason: null }) } }
    expect(await caps()).toMatchObject({ runner: true, runnerDesks: ['video', 'reel'], runnerReason: null })
    registry[KEY] = { current: { status: () => ({ active: false, desks: ['video'], reason: 'Turned off with SWITCHGEN_RUNNER=off.' }) } }
    expect(await caps()).toMatchObject({ runner: false, runnerDesks: [], runnerReason: 'Turned off with SWITCHGEN_RUNNER=off.' })
    registry[KEY] = { current: { status: () => ({ active: false, desks: [], reason: '' }) } }
    expect((await caps()).runnerReason).toBe('The queue on the server is not running.')
  })

  it('still answers, with no queue, when the queue cannot say how it stands', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {})
    registry[KEY] = { current: { status: () => { throw new Error('broken') } } }
    expect(await caps()).toMatchObject({ deleteFiles: true, runner: false, runnerDesks: [], runnerReason: 'The queue on the server is not running.' })
  })
})

describe('files the queue on the server is filing', () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  it('are listed as filed while it files them, and not once it lets them go', async () => {
    const { archiveApi } = await import('../server/archive.mjs')
    mkdirSync(path.join(outputs, 'queue'), { recursive: true })
    writeFileSync(path.join(outputs, 'queue', 'filing.webm'), 'x')
    vi.useFakeTimers({ toFake: ['Date'] })
    // The listing is kept for a few seconds; each look here is a later one,
    // after any the tests above made on a clock of their own.
    let clock = Date.now() + 3_600_000
    const filed = async () => {
      clock += 5000
      vi.setSystemTime(clock)
      return (await call(handlers.archive!, { url: '/api/outputs' })).json().files.find((f: { rel: string }) => f.rel === 'queue/filing.webm')?.filed
    }
    const listed = async () => {
      clock += 5000
      vi.setSystemTime(clock)
      return (await call(handlers.archive!, { url: '/api/outputs' })).json().files.map((f: { rel: string }) => f.rel)
    }
    expect(await listed()).toContain('queue/filing.webm')
    expect(await filed()).toBeUndefined()
    archiveApi.claim(['queue/filing.webm'])
    expect(await filed()).toBe(true)
    archiveApi.unclaim(['queue/filing.webm'])
    expect(await filed()).toBeUndefined()
  })
})
