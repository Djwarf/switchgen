import { chmodSync, existsSync, mkdirSync, rmSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest'
import { call, events, mounted, open, tempRoots, type Handler } from './http'

/**
 * A single file fetched by address (an add-on from the picker, the picture
 * reader's tagger) goes on when its page goes, and nothing lists it for the
 * page that comes back. That page shows plain fetch, and pressing it asks for
 * the same file again. These check that the ask takes up the fetch still
 * running, with its id, progress and Stop, against the real downloader and a
 * fake aria2c; before, it was refused as already downloading, with no way to
 * stop it from the page until it finished.
 */
let root = ''
let models = ''
let h: Handler
const url = (name: string, size: number, ms: number) => `https://example.invalid/${name}?size=${size}&ms=${ms}`
/** Writes the file a slice at a time with a control file beside it, as downloadsResume.test.ts's does. */
const FAKE_ARIA2C = `#!${process.execPath}
const fs = require('node:fs')
const args = process.argv.slice(2)
const lines = fs.readFileSync(args[args.indexOf('-i') + 1], 'utf8').split('\\n').map((l) => l.trim())
const src = new URL(lines[0])
const dir = lines.find((l) => l.startsWith('dir=')).slice(4)
const out = lines.find((l) => l.startsWith('out=')).slice(4)
const size = Number(src.searchParams.get('size'))
const ms = Number(src.searchParams.get('ms'))
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
  const bin = path.join(root, 'bin')
  mkdirSync(bin)
  const tool = (name: string, body: string) => {
    writeFileSync(path.join(bin, name), body)
    chmodSync(path.join(bin, name), 0o755)
  }
  tool('aria2c', FAKE_ARIA2C)
  tool('df', '#!/bin/sh\necho "Avail Size"\necho "1099511627776 2199023255552"\n')
  tool('nvidia-smi', '#!/bin/sh\nexit 1\n')
  process.env.PATH = `${bin}${path.delimiter}${process.env.PATH}`
  process.env.SWITCHGEN_ARIA2C = path.join(bin, 'aria2c')
  process.env.SWITCHGEN_CATALOG = path.join(root, 'catalog.json')
  writeFileSync(process.env.SWITCHGEN_CATALOG, JSON.stringify({ generatedAt: 't', deps: [], families: [] }))
  // The size probe a bare address gets before its disk check.
  vi.stubGlobal('fetch', async (u: string) => {
    const size = new URL(u).searchParams.get('size')
    return new Response('x', { status: 206, headers: { 'content-range': `bytes 0-0/${size}` } })
  })
  mkdirSync(path.join(models, 'loras'), { recursive: true })
  h = mounted((await import('../server/downloads.mjs')).switchgenDownloads())
})

afterAll(() => {
  vi.unstubAllGlobals()
  rmSync(root, { recursive: true, force: true })
})

const H = { origin: 'http://localhost', host: 'localhost' }
const fetchOf = (name: string, size: number, ms: number) => ({ url: url(name, size, ms), filename: name, dest: `loras/${name}` })
const post = (body: unknown) => open(h, { method: 'POST', url: '/api/download', body, headers: H })
/** The job a stream said it started, once it has said so. */
const startOf = async (s: ReturnType<typeof post>) => {
  await vi.waitFor(
    () => {
      if (!events(s.reply).some((e) => e.event === 'start')) throw new Error(`no start yet: ${s.reply.status} ${s.reply.body}`)
    },
    { timeout: 5000, interval: 20 },
  )
  return events(s.reply).find((e) => e.event === 'start')!.data as { id: string; filename: string }
}
const cancel = (id: string, keepPartial = false) =>
  call(h, { method: 'POST', url: '/api/download/cancel', body: { id, keepPartial }, headers: H })
const listed = async () => (await call(h, { url: '/api/download/status', headers: H })).json().downloads as { id: string }[]

describe('a page that asks again for a file it was fetching', () => {
  // Each test ends what it started, so one that fails leaves no fetch
  // holding a slot or a file for the next.
  afterEach(async () => {
    for (const d of await listed()) await cancel(d.id)
  })

  it('takes up the fetch still running and follows it to the end', async () => {
    const body = fetchOf('a.safetensors', 4000, 2500)
    const first = post(body)
    const job = await startOf(first)
    first.hangUp()
    const again = post(body)
    const joined = await startOf(again)
    expect(joined.id).toBe(job.id)
    const done = await again.done
    const names = events(done).map((e) => e.event)
    expect(names).toContain('progress')
    expect(names.slice(-2)).toEqual(['file', 'done'])
    expect(existsSync(path.join(models, 'loras', 'a.safetensors'))).toBe(true)
    expect(existsSync(path.join(models, 'loras', 'a.safetensors.aria2'))).toBe(false)
  }, 15_000)

  it('stops it from the page that took it up', async () => {
    const body = fetchOf('b.safetensors', 4000, 8000)
    const first = post(body)
    const job = await startOf(first)
    first.hangUp()
    const again = post(body)
    const joined = await startOf(again)
    const stopped = await cancel(joined.id, true)
    expect(stopped.status).toBe(200)
    const done = await again.done
    expect(events(done).at(-1)!.event).toBe('error')
    // Let go of once its plan has settled, which is a moment after it stops.
    await vi.waitFor(async () => expect((await listed()).find((d) => d.id === job.id)).toBeUndefined(), { timeout: 5000, interval: 20 })
  }, 15_000)

  it('still refuses the same file asked for from another address', async () => {
    const body = fetchOf('c.safetensors', 4000, 8000)
    const first = post(body)
    const job = await startOf(first)
    const other = await call(h, { method: 'POST', url: '/api/download', body: { ...body, url: url('c.safetensors', 4001, 8000) }, headers: H })
    expect(other.status).toBe(409)
    expect(other.json().error).toContain('already downloading')
    await cancel(job.id)
    await first.done
  }, 15_000)

  it('takes it up though every slot is taken, since it starts nothing', async () => {
    const one = fetchOf('d1.safetensors', 4000, 8000)
    const two = fetchOf('d2.safetensors', 4000, 8000)
    const s1 = post(one)
    const s2 = post(two)
    const [j1, j2] = await Promise.all([startOf(s1), startOf(s2)])
    // A third fetch has no slot to take.
    const third = await call(h, { method: 'POST', url: '/api/download', body: fetchOf('d3.safetensors', 4000, 8000), headers: H })
    expect(third.status).toBe(429)
    s1.hangUp()
    const again = post(one)
    const joined = await startOf(again)
    expect(joined.id).toBe(j1.id)
    // Hanging up on the second telling stops only the telling.
    again.hangUp()
    expect((await listed()).map((d) => d.id).sort()).toEqual([j1.id, j2.id].sort())
    await cancel(j1.id)
    await cancel(j2.id)
    await s2.done
  }, 15_000)
})
