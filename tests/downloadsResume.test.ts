import { chmodSync, mkdirSync, readFileSync, rmSync, writeFileSync, existsSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'
import { call, mounted, open, tempRoots, type Handler } from './http'

/**
 * A page reloaded while a model fetch runs, against the real downloader with
 * a fake aria2c: the new page takes the run up from the server's status,
 * follows it to the end and says so once, and its Stop still reaches it.
 */
let root = ''
let models = ''
let log = ''
let h: Handler
/** The same route, not wrapped, so a test can hear when its work is over (see `ran` in http.ts). */
let bare: Handler
const url = (name: string, size: number, ms = 0) => `https://example.invalid/${name}?size=${size}&ms=${ms}`
const dep = (filename: string, size: number, ms = 0) => ({ filename, kind: 'test', dest: `test/${filename}`, enumValue: filename, url: url(filename, size, ms), httpStatus: 200, sizeBytes: size })
const family = (id: string, deps: string[]) => ({ id, label: id, mode: 'test', group: 'test', models: [], deps, graph: { '1': { class_type: 'Loader', inputs: Object.fromEntries(deps.map((d, i) => [`w${i}`, d])) } } })
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
  const tool = (name: string, body: string) => { writeFileSync(path.join(bin, name), body); chmodSync(path.join(bin, name), 0o755) }
  tool('aria2c', FAKE_ARIA2C(log))
  tool('df', '#!/bin/sh\necho "Avail Size"\necho "1099511627776 2199023255552"\n')
  tool('nvidia-smi', '#!/bin/sh\nexit 1\n')
  process.env.PATH = `${bin}${path.delimiter}${process.env.PATH}`
  process.env.SWITCHGEN_ARIA2C = path.join(bin, 'aria2c')
  process.env.SWITCHGEN_CATALOG = path.join(root, 'catalog.json')
  writeFileSync(process.env.SWITCHGEN_CATALOG, JSON.stringify({ generatedAt: 't', deps: [dep('p1.bin', 3000, 3000), dep('p2.bin', 300, 600), dep('q1.bin', 3000, 6000), dep('r1.bin', 300, 600)], families: [family('P', ['p1.bin', 'p2.bin']), family('Q', ['q1.bin']), family('R', ['r1.bin'])] }))
  vi.stubGlobal('fetch', async (u: string, init?: RequestInit) => {
    if (u.startsWith('https://')) {
      const size = new URL(u).searchParams.get('size')
      return new Response('x', { status: 206, headers: { 'content-range': `bytes 0-0/${size}` } })
    }
    const r = await call(h, { method: init?.method ?? 'GET', url: u, body: init?.body ? JSON.parse(String(init.body)) : undefined, headers: { origin: 'http://localhost', host: 'localhost' } })
    return new Response(r.body, { status: r.status, headers: { 'content-type': String(r.headers['content-type'] ?? 'application/json') } })
  })
  mkdirSync(path.join(models, 'test'), { recursive: true })
  const downloads = await import('../server/downloads.mjs')
  h = mounted(downloads.switchgenDownloads())
  // Exported for this, though its declarations name only the plugin.
  bare = (downloads as unknown as { downloadsMiddleware: Handler }).downloadsMiddleware
  // The fit check wants 2 GB of working set free, which a busy machine does not always have.
  vi.spyOn(os, 'freemem').mockReturnValue(os.totalmem())
})
afterAll(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  rmSync(root, { recursive: true, force: true })
})

describe('a reloaded page', () => {
  it('takes up the plan the last page started, follows it to the end, and announces it once', async () => {
    const first = open(h, { method: 'POST', url: '/api/download', body: { family: 'P' }, headers: { origin: 'http://localhost', host: 'localhost' } })
    await vi.waitFor(() => { if (!first.reply.body.includes('event: start')) throw new Error('x') }, { timeout: 5000 })
    first.hangUp()
    vi.resetModules()
    const dl = await import('../src/lib/downloads')
    const landed: string[] = []
    dl.onPlanLanded((f) => landed.push(f))
    await dl.resumeRuns()
    const r = dl.downloadRuns().get('P')!
    expect(r).toBeTruthy()
    expect(r.state).toBe('running')
    expect(r.followed).toBe(true)
    expect(r.current?.filename).toBe('p1.bin')
    expect(r.current?.jobId).toBeTruthy()
    expect(r.files.map((f) => f.filename)).toEqual(['p1.bin', 'p2.bin'])
    await vi.waitFor(() => { if (dl.downloadRuns().get('P')!.state !== 'done') throw new Error('x') }, { timeout: 15000, interval: 200 })
    expect(landed).toEqual(['P'])
    expect(existsSync(path.join(models, 'test', 'p2.bin'))).toBe(true)
  }, 20000)

  it('stops a taken-up plan at the server', async () => {
    const first = open(h, { method: 'POST', url: '/api/download', body: { family: 'Q' }, headers: { origin: 'http://localhost', host: 'localhost' } })
    await vi.waitFor(() => { if (!first.reply.body.includes('event: start')) throw new Error('x') }, { timeout: 5000 })
    first.hangUp()
    vi.resetModules()
    const dl = await import('../src/lib/downloads')
    await dl.resumeRuns()
    expect(dl.downloadRuns().get('Q')!.state).toBe('running')
    await dl.cancelPlan('Q', false)
    const st = (await call(h, { url: '/api/download/status' })).json()
    const q = st.plans.find((p: { family: string }) => p.family === 'Q')
    expect(q.state).toBe('cancelled')
    expect(existsSync(path.join(models, 'test', 'q1.bin'))).toBe(false)
  }, 20000)
})

describe('a fetch whose page goes before it has begun', () => {
  // The server calls a fetch off when its page hangs up during the checks
  // before the plan begins, and lets it run once it has. The catalogue's line
  // under a fetch rests on that: only one that has begun is said to carry on
  // without the page.
  const H = { origin: 'http://localhost', host: 'localhost' }
  const planOf = async (fam: string) =>
    (await call(h, { url: '/api/download/status' })).json().plans.find((p: { family: string }) => p.family === fam)

  it('is called off, and one whose page goes once it has begun runs on', async () => {
    const started = () => readFileSync(log, 'utf8').split('\n').filter((l) => l === 'r1.bin').length
    const early = open(bare, { method: 'POST', url: '/api/download', body: { family: 'R' }, headers: H })
    // Gone while the server still reads the catalogue, the disk and the card.
    early.hangUp()
    await early.ran
    expect(early.reply.body).not.toContain('event: plan')
    expect(started()).toBe(0)
    expect(await planOf('R')).toBeUndefined()

    const again = open(h, { method: 'POST', url: '/api/download', body: { family: 'R' }, headers: H })
    await vi.waitFor(() => { if (!again.reply.body.includes('event: start')) throw new Error(`no start: ${again.reply.status} ${again.reply.body}`) }, { timeout: 5000 })
    again.hangUp()
    expect((await planOf('R'))?.state).toBe('running')
    await vi.waitFor(async () => expect((await planOf('R'))?.state).toBe('done'), { timeout: 10_000, interval: 100 })
    expect(started()).toBe(1)
    expect(existsSync(path.join(models, 'test', 'r1.bin'))).toBe(true)
  }, 15_000)

  it('is not promised to carry on while it is still starting, in the catalogue\'s words', () => {
    const panel = readFileSync(path.resolve(import.meta.dirname, '..', 'src', 'components', 'advanced', 'CataloguePanel.tsx'), 'utf8')
    const said = /run\.state === 'running'\s*\?\s*'([^']*)'\s*:\s*'([^']*)'/.exec(panel)
    expect(said?.[1]).toBe('The server carries on with it if this page is closed or the phone is locked.')
    expect(said?.[2]).toContain('Keep this page open until the fetch begins.')
    expect(panel.match(/The server carries on with it if this page is closed/g)).toHaveLength(1)
  })
})
