import { chmodSync, existsSync, mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'
import { call, events, mounted, open, tempRoots, type Handler, type Reply } from './http'

/**
 * The downloader against a catalogue of its own, a fake aria2c and a stubbed
 * size probe. Nothing is fetched from anywhere: the fake writes the bytes
 * itself, slowly enough for a second plan to meet it on the same file, and
 * logs every file it is started on.
 *
 * `df` and `nvidia-smi` are stood in for too, so the fit check reads the same
 * disk and card on every machine, and free RAM reads as all of it: the check
 * wants 2 GB of working set free, which a busy machine running the suite (or
 * rendering beside it) does not always have.
 */
let root = ''
let models = ''
let log = ''
let h: Handler
/** The route itself, not wrapped by `safely`, so a test can hear it finish. */
let bare: Handler
let publicPlans: (list: Iterable<Record<string, unknown>>, now?: number) => any[]

const url = (name: string, size: number, ms = 0) => `https://example.invalid/${name}?size=${size}&ms=${ms}`
const dep = (filename: string, size: number, ms = 0) => ({
  filename,
  kind: 'test',
  dest: `test/${filename}`,
  enumValue: filename,
  url: url(filename, size, ms),
  httpStatus: 200,
  sizeBytes: size,
})
/** A family whose graph loads every file it lists, so each is required. */
const family = (id: string, deps: string[]) => ({
  id,
  label: id,
  mode: 'test',
  group: 'test',
  models: [],
  deps,
  graph: { '1': { class_type: 'Loader', inputs: Object.fromEntries(deps.map((d, i) => [`w${i}`, d])) } },
})

const FAKE_ARIA2C = (logFile: string) => `#!${process.execPath}
// Reads the input file the way aria2c does, logs the file it was started on,
// and writes it a slice at a time with a control file beside it, taking up
// where a partial left off. SIGTERM stops it with both left in place.
const fs = require('node:fs')
const args = process.argv.slice(2)
const lines = fs.readFileSync(args[args.indexOf('-i') + 1], 'utf8').split('\\n').map((l) => l.trim())
const src = new URL(lines[0])
const dir = lines.find((l) => l.startsWith('dir=')).slice(4)
const out = lines.find((l) => l.startsWith('out=')).slice(4)
const size = Number(src.searchParams.get('size'))
const ms = Number(src.searchParams.get('ms'))
fs.appendFileSync(${JSON.stringify(logFile)}, out + '\\n')
const full = dir + '/' + out
fs.writeFileSync(full + '.aria2', 'control')
if (!fs.existsSync(full)) fs.writeFileSync(full, '')
let have = fs.statSync(full).size
const steps = Math.max(1, Math.round(ms / 100))
const slice = Math.ceil(size / steps)
process.on('SIGTERM', () => process.exit(7))
const step = () => {
  if (have >= size) {
    fs.unlinkSync(full + '.aria2')
    process.exit(0)
  }
  const n = Math.min(slice, size - have)
  fs.appendFileSync(full, Buffer.alloc(n, 120))
  have += n
  setTimeout(step, ms / steps)
}
step()
`

beforeAll(async () => {
  const roots = tempRoots()
  root = roots.root
  models = roots.models
  log = path.join(root, 'aria2c.log')
  writeFileSync(log, '')
  const bin = path.join(root, 'bin')
  mkdirSync(bin)
  const tool = (name: string, body: string) => {
    writeFileSync(path.join(bin, name), body)
    chmodSync(path.join(bin, name), 0o755)
  }
  tool('aria2c', FAKE_ARIA2C(log))
  tool('df', '#!/bin/sh\necho "Avail Size"\necho "1099511627776 2199023255552"\n')
  tool('nvidia-smi', '#!/bin/sh\nexit 1\n')
  process.env.PATH = `${bin}${path.delimiter}${process.env.PATH}`
  process.env.SWITCHGEN_ARIA2C = path.join(bin, 'aria2c')

  const catalogue = {
    generatedAt: 'the test suite',
    deps: [
      dep('c.bin', 1000),
      dep('d.bin', 300),
      // Each plan's own file takes long enough for the other plan to have
      // passed its checks and started on the shared one, even on a slow runner,
      // and the shared file long enough for the first plan to meet it there.
      dep('a1.bin', 400, 1500),
      dep('s1.bin', 3000, 4000),
      dep('a2.bin', 400, 1500),
      dep('s2.bin', 3000, 4000),
      dep('a3.bin', 400, 1500),
      dep('s3.bin', 3000, 4000),
      dep('e.bin', 10),
      { ...dep('n.bin', 0), sizeBytes: null },
      dep('h1.bin', 400, 800),
      dep('h2.bin', 300, 300),
      dep('k1.bin', 3000, 4000),
      { ...dep('w1.bin', 0), url: `${url('w1.bin', 50)}&hang=1`, sizeBytes: null },
    ],
    families: [
      family('C', ['c.bin', 'd.bin']),
      family('A1', ['a1.bin', 's1.bin']),
      family('B1', ['s1.bin']),
      family('A2', ['a2.bin', 's2.bin']),
      family('B2', ['s2.bin']),
      family('A3', ['a3.bin', 's3.bin']),
      family('B3', ['s3.bin']),
      family('E', ['e.bin']),
      family('N', ['n.bin']),
      family('H', ['h1.bin', 'h2.bin']),
      family('K', ['k1.bin']),
      family('W', ['w1.bin']),
    ],
  }
  process.env.SWITCHGEN_CATALOG = path.join(root, 'catalog.json')
  writeFileSync(process.env.SWITCHGEN_CATALOG, JSON.stringify(catalogue))

  // The size probe: a one-byte ranged GET, answered with the size in the URL.
  // One marked `hang` never answers, and gives up only when it is called off.
  vi.stubGlobal('fetch', async (u: string, init?: RequestInit) => {
    const q = new URL(u).searchParams
    if (q.has('hang')) {
      probesHanging += 1
      return new Promise((_, reject) => {
        init?.signal?.addEventListener('abort', () => reject(new DOMException('called off', 'AbortError')))
      })
    }
    return new Response('x', { status: 206, headers: { 'content-range': `bytes 0-0/${q.get('size')}` } })
  })

  mkdirSync(path.join(models, 'test'), { recursive: true })
  // Another file under c.bin's name, at its destination, with no control file.
  writeFileSync(path.join(models, 'test', 'c.bin'), Buffer.alloc(500, 99))
  // The catalogue's own file, at the size it lists.
  writeFileSync(path.join(models, 'test', 'e.bin'), Buffer.alloc(10, 101))

  const downloads = await import('../server/downloads.mjs')
  publicPlans = downloads.publicPlans
  bare = (downloads as unknown as { downloadsMiddleware: Handler }).downloadsMiddleware
  h = mounted(downloads.switchgenDownloads())
  vi.spyOn(os, 'freemem').mockReturnValue(os.totalmem())
})

let probesHanging = 0

afterAll(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  rmSync(root, { recursive: true, force: true })
})

const file = (name: string) => path.join(models, 'test', name)
const started = () => readFileSync(log, 'utf8').split('\n').filter(Boolean)
const fetchOf = (family: string) => open(h, { method: 'POST', url: '/api/download', body: { family } })
const startOf = (r: Reply, filename: string) =>
  events(r).find((e) => e.event === 'start' && e.data.filename === filename)?.data as { id: string } | undefined
const until = (what: () => unknown) =>
  vi.waitFor(async () => { if (!(await what())) throw new Error('not yet') }, { timeout: 10_000, interval: 20 })

describe('a file under the catalogue\'s name that is not the catalogue\'s', () => {
  it('counts as installed, so the family fits and only the missing file is planned', async () => {
    const plan = (await call(h, { url: '/api/catalog/plan?family=C' })).json()
    expect(plan.blockers).toEqual([])
    expect(plan.fits).toBe(true)
    expect(plan.download.map((f: { filename: string }) => f.filename)).toEqual(['d.bin'])
    expect(plan.reasons.join(' ')).toContain('kept and used as it is')

    const cat = (await call(h, { url: '/api/catalog' })).json()
    expect(cat.families.find((f: { id: string }) => f.id === 'C').installed.missing).toEqual(['d.bin'])
  })

  it('is marked as a conflict in the slim catalogue the panel reads, and counted as installed', async () => {
    // CatalogFile.conflict in src/lib/catalog.ts: the panel tells such a file
    // apart from the catalogue's own by this flag.
    const cat = (await call(h, { url: '/api/catalog?slim=1' })).json()
    const installed = cat.families.find((f: { id: string }) => f.id === 'C').installed
    const c = installed.files.find((f: { filename: string }) => f.filename === 'c.bin')
    expect(c).toMatchObject({ conflict: true, installed: true, partial: false })
    expect(installed.missing).not.toContain('c.bin')
    const d = installed.files.find((f: { filename: string }) => f.filename === 'd.bin')
    expect(d).toMatchObject({ conflict: false, installed: false })
  })

  it('is left alone by a fetch, which fetches the rest', async () => {
    const r = await call(h, { method: 'POST', url: '/api/download', body: { family: 'C', force: true } })
    const sse = events(r)
    expect(sse.find((e) => e.event === 'plan')?.data.files.map((f: { filename: string }) => f.filename)).toEqual(['d.bin'])
    expect(sse.at(-1)?.event).toBe('done')
    expect(readFileSync(file('c.bin'))).toEqual(Buffer.alloc(500, 99))
    expect(statSync(file('d.bin')).size).toBe(300)
  })

  it('leaves nothing to fetch, but is not called ready to run', async () => {
    // d.bin has landed, so c.bin is all that stands between the family and
    // complete, and it is not the catalogue's file. Nothing has loaded it.
    const plan = (await call(h, { url: '/api/catalog/plan?family=C' })).json()
    expect(plan.download).toEqual([])
    expect(plan.asIs).toEqual(['c.bin'])
    expect(plan.verdict.startsWith('Every file is on disk, but 1 of them is not the size the catalogue lists.')).toBe(true)
    expect(plan.verdict).not.toContain('Ready to run')
    expect(plan.reasons).not.toContain('Every file this family needs is already on disk.')

    const cat = (await call(h, { url: '/api/catalog' })).json()
    const installed = cat.families.find((f: { id: string }) => f.id === 'C').installed
    expect(installed.asIs).toEqual(['c.bin'])
    expect(installed.ready).toBe(true)
  })
})

describe('the verdict on a family with nothing to fetch', () => {
  it('says ready to run when every file is the catalogue\'s own', async () => {
    const plan = (await call(h, { url: '/api/catalog/plan?family=E' })).json()
    expect(plan.asIs).toEqual([])
    expect(plan.verdict.startsWith('Ready to run: every file is already installed.')).toBe(true)
    expect(plan.reasons).toContain('Every file this family needs is already on disk.')
  })

  it('does not count a missing file the catalogue has no size for as on disk, or as 0 B', async () => {
    const plan = (await call(h, { url: '/api/catalog/plan?family=N' })).json()
    expect(plan.download.map((f: { filename: string }) => f.filename)).toEqual(['n.bin'])
    expect(plan.reasons).not.toContain('Every file this family needs is already on disk.')
    expect(plan.verdict.startsWith('Fits this machine. 1 file to fetch, size not listed.')).toBe(true)
    expect(plan.verdict).not.toContain('0 B')
  })
})

describe('two plans that share a file', () => {
  it('fetch it once: the second waits for the first, passing its progress on', async () => {
    // A is on its own file when B starts on the shared one, so B's request
    // passes the check and both plans reach s1.bin.
    const a = fetchOf('A1')
    await until(() => startOf(a.reply, 'a1.bin'))
    const b = fetchOf('B1')
    await until(() => startOf(b.reply, 's1.bin'))
    const [ra, rb] = await Promise.all([a.done, b.done])

    expect(started().filter((f) => f === 's1.bin')).toHaveLength(1)
    const onS = events(ra).filter((e) => e.data?.filename === 's1.bin').map((e) => e.event)
    expect(onS.at(-1)).toBe('skip')
    expect(onS.slice(1, -1).length).toBeGreaterThan(0)
    expect(onS.slice(1, -1).every((e) => e === 'progress')).toBe(true)
    expect(events(rb).at(-1)?.event).toBe('done')
    expect(statSync(file('s1.bin')).size).toBe(3000)
    expect(existsSync(file('s1.bin.aria2'))).toBe(false)
  }, 20_000)

  it('keep the file when the plan waiting for it is cancelled', async () => {
    const a = fetchOf('A2')
    await until(() => startOf(a.reply, 'a2.bin'))
    const b = fetchOf('B2')
    await until(() => startOf(b.reply, 's2.bin'))
    await until(() => startOf(a.reply, 's2.bin'))
    const waiting = startOf(a.reply, 's2.bin')!.id
    const cancel = (await call(h, { method: 'POST', url: '/api/download/cancel', body: { id: waiting, keepPartial: false } })).json()
    expect(cancel.removed).toEqual([])
    expect(cancel.keptPartial).toBe(true)

    const rb = await b.done
    await a.done
    expect(events(rb).at(-1)?.event).toBe('done')
    expect(statSync(file('s2.bin')).size).toBe(3000)
    expect(existsSync(file('s2.bin.aria2'))).toBe(false)
  }, 20_000)

  it('hand the file to the waiting plan when the one fetching it is cancelled', async () => {
    const a = fetchOf('A3')
    await until(() => startOf(a.reply, 'a3.bin'))
    const b = fetchOf('B3')
    await until(() => startOf(b.reply, 's3.bin'))
    await until(() => startOf(a.reply, 's3.bin'))
    const fetching = startOf(b.reply, 's3.bin')!.id
    const cancel = (await call(h, { method: 'POST', url: '/api/download/cancel', body: { id: fetching, keepPartial: false } })).json()
    expect(cancel.removed).toEqual([])

    const ra = await a.done
    await b.done
    // B's aria2c stopped part way, and A's took up where it left off.
    expect(started().filter((f) => f === 's3.bin')).toHaveLength(2)
    expect(events(ra).at(-1)?.event).toBe('done')
    expect(statSync(file('s3.bin')).size).toBe(3000)
    expect(existsSync(file('s3.bin.aria2'))).toBe(false)
  }, 20_000)
})

describe('a fetch whose page goes', () => {
  const status = async () => (await call(h, { url: '/api/download/status' })).json()
  const planOf = async (family: string) => (await status()).plans.find((p: { family: string }) => p.family === family)

  it('goes on to the end, and the status lists it for a page that comes back', async () => {
    const page = fetchOf('H')
    await until(() => startOf(page.reply, 'h1.bin'))
    page.hangUp()

    const st = await status()
    const plan = st.plans.find((p: { family: string }) => p.family === 'H')
    expect(plan.state).toBe('running')
    expect(st.downloads.map((d: { id: string }) => d.id)).toContain(plan.current.jobId)

    await until(async () => (await planOf('H'))?.state !== 'running')
    const ended = await planOf('H')
    expect(ended.state).toBe('done')
    expect(ended.finished).toEqual(['h1.bin', 'h2.bin'])
    expect(statSync(file('h1.bin')).size).toBe(400)
    expect(statSync(file('h2.bin')).size).toBe(300)
    expect(existsSync(file('h1.bin.aria2'))).toBe(false)
    expect(existsSync(file('h2.bin.aria2'))).toBe(false)
  }, 20_000)

  it('is still stopped by Stop, which names the file it is on', async () => {
    const page = fetchOf('K')
    await until(() => startOf(page.reply, 'k1.bin'))
    page.hangUp()
    await until(async () => (await planOf('K'))?.current?.jobId)
    const { jobId } = (await planOf('K')).current
    const cancel = await call(h, { method: 'POST', url: '/api/download/cancel', body: { id: jobId, keepPartial: false } })
    expect(cancel.status).toBe(200)

    await until(async () => (await planOf('K'))?.state !== 'running')
    const ended = await planOf('K')
    expect(ended.state).toBe('cancelled')
    expect(ended.current).toBeNull()
    expect(ended.error).toBe('cancelled')
  }, 20_000)

  it('is called off when the page goes before the plan starts', async () => {
    const before = started().length
    const page = open(bare, { method: 'POST', url: '/api/download', body: { family: 'W' } })
    page.hangUp()
    await page.ran
    expect(events(page.reply).some((e) => e.event === 'plan')).toBe(false)
    expect(started()).toHaveLength(before)
    expect(await planOf('W')).toBeUndefined()
  })

  it('calls off the size probe of a file fetched by address when the page goes during it', async () => {
    const before = probesHanging
    const page = open(bare, {
      method: 'POST',
      url: '/api/download',
      body: { url: `${url('by-address.bin', 50)}&hang=1`, dest: 'test/by-address.bin' },
    })
    await until(() => probesHanging > before)
    page.hangUp()
    // The probe never answers by itself (the server's own limit on it is 30
    // s), so the route finishes in time only if the hang-up reached it.
    const finished = await Promise.race([
      page.ran.then(() => true),
      new Promise((resolve) => setTimeout(resolve, 5000, false)),
    ])
    expect(finished).toBe(true)
    expect(events(page.reply).some((e) => e.event === 'plan')).toBe(false)
    expect(started()).not.toContain('by-address.bin')
    expect(existsSync(file('by-address.bin'))).toBe(false)
  }, 10_000)
})

describe('the plans the status lists', () => {
  const job = (state: string) => ({
    id: 'j1', filename: 'a.bin', fileIndex: 1, fileCount: 2, state, done: 5, total: 10, speed: 1, etaSec: 5,
  })
  const plan = (over: Record<string, unknown>) => ({
    family: 'F', state: 'done', files: [], current: null, finished: [], error: null, startedAt: 1000, endedAt: 2000, ...over,
  })
  const NOW = 3000

  it('shows the running plan of a family over one that ended', () => {
    const out = publicPlans([plan({ state: 'done', startedAt: 5 }), plan({ state: 'running', startedAt: 1, endedAt: null })], NOW)
    expect(out).toHaveLength(1)
    expect(out[0].state).toBe('running')
  })

  it('shows the one started last when every plan of a family has ended', () => {
    const out = publicPlans([plan({ state: 'error', startedAt: 900 }), plan({ state: 'cancelled', startedAt: 1500 })], NOW)
    expect(out.map((p) => p.state)).toEqual(['cancelled'])
  })

  it('leaves out a fetch that has no family, and one that ended over ten minutes ago', () => {
    expect(publicPlans([plan({ family: null })], NOW)).toEqual([])
    expect(publicPlans([plan({ endedAt: 0 })], 10 * 60 * 1000 + 1)).toEqual([])
    expect(publicPlans([plan({ endedAt: 0 })], 10 * 60 * 1000)).toHaveLength(1)
  })

  it('names the file in hand only while the plan runs and the file is being fetched', () => {
    const [live] = publicPlans([plan({ state: 'running', endedAt: null, current: job('downloading') })], NOW)
    expect(live.current).toMatchObject({ jobId: 'j1', filename: 'a.bin', index: 1, count: 2, pct: 0.5 })
    expect(publicPlans([plan({ state: 'running', endedAt: null, current: job('starting') })], NOW)[0].current).not.toBeNull()
    expect(publicPlans([plan({ state: 'running', endedAt: null, current: job('done') })], NOW)[0].current).toBeNull()
    expect(publicPlans([plan({ state: 'cancelled', current: job('downloading') })], NOW)[0].current).toBeNull()
  })
})
