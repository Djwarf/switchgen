import { chmodSync, existsSync, mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs'
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
 * disk and card on every machine. RAM is still this machine's; every family
 * here needs a few kilobytes of it.
 */
let root = ''
let models = ''
let log = ''
let h: Handler

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
      dep('a1.bin', 400, 1000),
      dep('s1.bin', 3000, 2500),
      dep('a2.bin', 400, 600),
      dep('s2.bin', 3000, 2500),
      dep('a3.bin', 400, 600),
      dep('s3.bin', 3000, 3000),
      dep('e.bin', 10),
      { ...dep('n.bin', 0), sizeBytes: null },
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
    ],
  }
  process.env.SWITCHGEN_CATALOG = path.join(root, 'catalog.json')
  writeFileSync(process.env.SWITCHGEN_CATALOG, JSON.stringify(catalogue))

  // The size probe: a one-byte ranged GET, answered with the size in the URL.
  vi.stubGlobal('fetch', async (u: string) => {
    const size = new URL(u).searchParams.get('size')
    return new Response('x', { status: 206, headers: { 'content-range': `bytes 0-0/${size}` } })
  })

  mkdirSync(path.join(models, 'test'), { recursive: true })
  // Another file under c.bin's name, at its destination, with no control file.
  writeFileSync(path.join(models, 'test', 'c.bin'), Buffer.alloc(500, 99))
  // The catalogue's own file, at the size it lists.
  writeFileSync(path.join(models, 'test', 'e.bin'), Buffer.alloc(10, 101))

  h = mounted((await import('../server/downloads.mjs')).switchgenDownloads())
})

afterAll(() => {
  vi.unstubAllGlobals()
  rmSync(root, { recursive: true, force: true })
})

const file = (name: string) => path.join(models, 'test', name)
const started = () => readFileSync(log, 'utf8').split('\n').filter(Boolean)
const fetchOf = (family: string) => open(h, { method: 'POST', url: '/api/download', body: { family } })
const startOf = (r: Reply, filename: string) =>
  events(r).find((e) => e.event === 'start' && e.data.filename === filename)?.data as { id: string } | undefined
const until = (what: () => unknown) => vi.waitFor(() => { if (!what()) throw new Error('not yet') }, { timeout: 10_000, interval: 20 })

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
