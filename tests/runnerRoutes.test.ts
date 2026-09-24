import { spawn, type ChildProcess } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { existsSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { Readable } from 'node:stream'
import { EventEmitter } from 'node:events'
import { pathToFileURL } from 'node:url'
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest'
import { call, open, tempRoots, type Handler } from './http'
import { Unanswered } from '../server/runner/comfy.mjs'
import {
  CHAINED,
  CHAIN_TOKEN,
  FakeComfy,
  HEAVY,
  clearRegistry,
  groupBody,
  harness,
  mountPlugin,
  revEvents,
  runnerEnv,
  standIn,
  tempRoot,
  type Harness,
} from './runnerFake'

/**
 * The queue's routes under /api/runner, its event stream, and its word in
 * /api/capabilities. Runners here are built against the scripted ComfyUI and
 * stepped by hand, except where the plugin itself is under test, which talks
 * to a stand-in ComfyUI on a local port.
 */

// What vitest.config.ts gives every test, before this file changes anything.
const START_ENV = { ...process.env }

// A disk that fills up, for the list of work only, when a test says so.
const disk = vi.hoisted(() => ({ full: false, after: 0 }))
vi.mock('node:fs', async (importOriginal) => {
  const real = await importOriginal<typeof import('node:fs')>()
  const nospace = () => Object.assign(new Error('ENOSPC: no space left on device, write'), { code: 'ENOSPC' })
  const writeFileSync = ((...args: Parameters<typeof real.writeFileSync>) => {
    const where = String(args[0])
    // The list of work writes through a descriptor; its payloads by path under jobs/.
    if (disk.full && (typeof args[0] === 'number' || where.includes(`${path.sep}jobs${path.sep}`))) {
      // Room for `after` more writes first, then none.
      if (disk.after > 0) disk.after--
      else throw nospace()
    }
    return real.writeFileSync(...args)
  }) as typeof real.writeFileSync
  return { ...real, writeFileSync, default: { ...real, writeFileSync } }
})

let restoreEnv = () => {}
let roots = ''
beforeAll(() => {
  // Every root a server module reads, then the queue's own.
  roots = tempRoots().root
  restoreEnv = runnerEnv()
})
afterAll(() => {
  restoreEnv()
  // tempRoots' folder; runnerFake removes its own, and every tempRoot().
  if (roots) rmSync(roots, { recursive: true, force: true })
})
afterEach(() => {
  disk.full = false
  disk.after = 0
})

const asRoot = process.getuid?.() === 0

/** A request whose body records whether anything read it. */
function guarded(handler: Handler, url: string, headers: Record<string, string>) {
  const req = Object.assign(Readable.from([Buffer.from('{}')]), { method: 'POST', url, headers })
  let status = 0
  let body = ''
  return new Promise<{ status: number; body: string; read: boolean }>((resolve) => {
    const res = Object.assign(new EventEmitter(), {
      req,
      headersSent: false,
      destroyed: false,
      setHeader() {},
      getHeader() {},
      writeHead(code: number) {
        status = code
        return res
      },
      end(chunk?: string) {
        body += chunk ?? ''
        resolve({ status, body, read: req.readableFlowing !== null || req.readableEnded })
      },
    })
    Object.defineProperty(res, 'statusCode', { get: () => status, set: (v: number) => { status = v } })
    void handler(req, res, () => resolve({ status: -1, body: 'passed on', read: false }))
  })
}

describe('the test environment', () => {
  it('points ComfyUI at a port nothing may reach, keeps the queue off, and keeps the roots in the temp folder', () => {
    expect(START_ENV.COMFY_URL).toBe('http://127.0.0.1:9')
    expect(START_ENV.SWITCHGEN_RUNNER).toBe('off')
    for (const k of ['SWITCHGEN_OUTPUTS', 'SWITCHGEN_MODELS']) expect(path.resolve(START_ENV[k]!).startsWith(os.tmpdir()), k).toBe(true)
  })
})

describe('every write to the queue', () => {
  const POSTS = ['/api/runner/groups', `/api/runner/groups/${randomUUID()}/stop`, `/api/runner/jobs/${randomUUID()}/stop`, '/api/runner/lane', '/api/runner/dismiss']

  it('is refused from another site and with a body that is not JSON, before a byte of it is read, and nothing is sent', async () => {
    const h = await harness()
    await h.submit(groupBody({ desk: 'images', jobs: [{}] }))
    for (const url of POSTS) {
      const cross = await guarded(h.runner.handler, url, { 'sec-fetch-site': 'cross-site', origin: 'http://evil.example', host: '127.0.0.1:5273', 'content-type': 'application/json' })
      expect(cross.status, url).toBe(403)
      expect(cross.read, url).toBe(false)
      const plain = await guarded(h.runner.handler, url, { 'content-type': 'text/plain' })
      expect(plain.status, url).toBe(415)
      expect(plain.read, url).toBe(false)
    }
    expect(h.snap().jobs).toHaveLength(1)
    expect(h.comfy.log).toEqual([])
    await h.runner.retire()
  })

  it('answers a wrong method with 405 JSON naming the right one, and an unknown path with 404 JSON', async () => {
    const h = await harness()
    for (const [url, allow] of [
      ['/api/runner/groups', 'POST'],
      ['/api/runner/lane', 'POST'],
      ['/api/runner/dismiss', 'POST'],
      [`/api/runner/jobs/${randomUUID()}/stop`, 'POST'],
      ['/api/runner/stream', 'GET'],
    ] as const) {
      const r = await call(h.runner.handler, { method: allow === 'POST' ? 'GET' : 'POST', url, body: allow === 'POST' ? undefined : {} })
      expect(r.status, url).toBe(405)
      expect(r.headers.allow, url).toBe(allow)
      expect(r.json().error, url).toMatch(/takes/)
    }
    const r = await h.get('/api/runner/nothing')
    expect(r.status).toBe(404)
    expect(r.json().error).toMatch(/no such endpoint/)
    expect((await h.get('/api/runnerish')).passed).toBe(true)
    expect((await h.get('/api/archive')).passed).toBe(true)
    await h.runner.retire()
  })

  it('answers 404 for ids that are not the queue\'s own, `__proto__` among them', async () => {
    const h = await harness()
    for (const url of ['/api/runner/jobs/__proto__', '/api/runner/jobs/constructor', '/api/runner/jobs/__proto__/preview', '/api/runner/jobs/%E0%A4%A']) {
      expect((await h.get(url)).status, url).toBe(404)
    }
    for (const url of ['/api/runner/jobs/__proto__/stop', '/api/runner/groups/__proto__/stop', '/api/runner/groups/hasOwnProperty/stop']) {
      expect((await h.post(url)).status, url).toBe(404)
    }
    expect((await h.post('/api/runner/dismiss', { jobIds: ['__proto__'] })).json()).toEqual({ dismissed: [] })
    await h.runner.retire()
  })
})

describe('taking a group in', () => {
  const bad: [string, (b: ReturnType<typeof groupBody>) => void][] = [
    ['a group id that is not a lowercase v4 uuid', (b) => void (b.group.id = b.group.id.toUpperCase())],
    ['a group id of another version', (b) => void (b.group.id = '6ba7b810-9dad-11d1-80b4-00c04fd430c8')],
    ['a job id that is not a uuid', (b) => void (b.jobs[0]!.id = 'job-1')],
    ['a job id used twice', (b) => void (b.jobs[1]!.id = b.jobs[0]!.id)],
    ['a desk of the wrong kind', (b) => void ((b.group as any).kind = 'batch')],
    ['a label over 200', (b) => void (b.group.label = 'x'.repeat(201))],
    ['a prompt over 4000', (b) => void (b.jobs[0]!.prompt = 'x'.repeat(4001))],
    ['a graph node with no class', (b) => void ((b.jobs[0]!.graph as any)['9'] = { inputs: {} })],
    ['a graph over 1 MiB', (b) => void ((b.jobs[0]!.graph as any)['9'].inputs.text = 'x'.repeat(1_100_000))],
    ['a record that names no mode', (b) => void delete (b.jobs[0]!.record as any).mode],
    ['a record over 64 KB', (b) => void ((b.jobs[0]!.record as any).note = 'x'.repeat(70_000))],
    ['meta over 16 KB', (b) => void ((b.jobs[0] as any).meta = { x: 'x'.repeat(17_000) })],
    ['the frame\'s place with no chain', (b) => void ((b.jobs[0]!.graph as any)['20'] = { class_type: 'LoadImage', inputs: { image: CHAIN_TOKEN } })],
    ['the frame\'s place inside a longer value', (b) => void ((b.jobs[0]!.graph as any)['9'].inputs.text = `see ${CHAIN_TOKEN}`)],
  ]

  for (const [what, spoil] of bad) {
    it(`refuses ${what}, and keeps nothing`, async () => {
      const h = await harness()
      const b = groupBody({ desk: 'video', jobs: [{}, {}] })
      spoil(b)
      const r = await h.submit(b)
      expect(r.status, r.body).toBe(400)
      expect(typeof r.json().error).toBe('string')
      expect(h.snap().jobs).toEqual([])
      expect(existsSync(path.join(h.dir, 'jobs')) ? readdirSync(path.join(h.dir, 'jobs')) : []).toEqual([])
      await h.runner.retire()
    })
  }

  it('refuses a chained shot whose chain names a later job, itself, or places that do not hold the frame\'s place, or misses one', async () => {
    const h = await harness()
    const ids = [randomUUID(), randomUUID(), randomUUID()]
    const pass = (chain: { after: string; at: [string, string][] }, graph: Record<string, unknown> = CHAINED()) =>
      groupBody({ desk: 'reel', jobs: [{ id: ids[0], heavy: true }, { id: ids[1], heavy: true, graph, chain }, { id: ids[2], heavy: true }] })
    const twoPlaces = { ...CHAINED(), '21': { class_type: 'LoadImage', inputs: { image: CHAIN_TOKEN } } }
    for (const [what, b] of [
      ['a later job', pass({ after: ids[2]!, at: [['20', 'image']] })],
      ['itself', pass({ after: ids[1]!, at: [['20', 'image']] })],
      ['a place without the token', pass({ after: ids[0]!, at: [['20', 'upload']] })],
      ['the same place twice', pass({ after: ids[0]!, at: [['20', 'image'], ['20', 'image']] })],
      ['one of two places', pass({ after: ids[0]!, at: [['20', 'image']] }, twoPlaces)],
      ['no places', pass({ after: ids[0]!, at: [] })],
    ] as const) {
      const r = await h.submit(b)
      expect(r.status, what).toBe(400)
    }
    const shotless = groupBody({ desk: 'reel', jobs: [{}] })
    delete (shotless.jobs[0] as any).meta
    expect((await h.submit(shotless)).status).toBe(400)
    expect((await h.submit(pass({ after: ids[0]!, at: [['20', 'image']] }))).status).toBe(200)
    await h.runner.retire()
  })

  it('refuses 201 jobs and none, and takes 200', async () => {
    const h = await harness()
    expect((await h.submit(groupBody({ desk: 'video', jobs: Array.from({ length: 201 }, () => ({})) }))).status).toBe(400)
    expect((await h.submit(groupBody({ desk: 'video', jobs: [] }))).status).toBe(400)
    const r = await h.submit(groupBody({ desk: 'video', jobs: Array.from({ length: 200 }, () => ({})) }))
    expect(r.status).toBe(200)
    expect(r.json().jobs.map((j: { index: number; total: number }) => [j.index, j.total]).slice(0, 2)).toEqual([[1, 200], [2, 200]])
    await h.runner.retire()
  })

  it('takes the same group sent again as the same work, and refuses different work under its ids', async () => {
    const h = await harness()
    const b = groupBody({ desk: 'video', jobs: [{ heavy: true }, {}] })
    const first = await h.submit(b)
    expect(first.status).toBe(200)
    expect(first.json()).toMatchObject({ replayed: false, rev: h.snap().rev, group: { id: b.group.id, state: 'active', jobIds: b.jobs.map((j) => j.id) } })
    const rev = h.snap().rev
    const again = await h.submit(b)
    expect(again.status).toBe(200)
    expect(again.json()).toMatchObject({ replayed: true, rev, group: { id: b.group.id } })
    expect(again.json().jobs.map((j: { id: string }) => j.id)).toEqual(b.jobs.map((j) => j.id))
    expect(h.snap().rev).toBe(rev)
    expect(h.snap().jobs).toHaveLength(2)

    const changed = structuredClone(b)
    changed.jobs[1]!.prompt = 'something else'
    expect((await h.submit(changed)).json()).toMatchObject({ conflict: 'id' })
    expect((await h.submit(changed)).status).toBe(409)
    const reordered = structuredClone(b)
    reordered.jobs.reverse()
    expect((await h.submit(reordered)).status).toBe(409)
    // A job id already here under another group.
    const stolen = groupBody({ desk: 'video', jobs: [{ id: b.jobs[0]!.id, graph: { '7': { class_type: 'Other', inputs: {} } } }] })
    expect((await h.submit(stolen)).json()).toMatchObject({ conflict: 'id' })
    expect(h.snap().jobs).toHaveLength(2)
    // The job already here keeps the graph it was taken with.
    expect((await h.get(`/api/runner/jobs/${b.jobs[0]!.id}`)).json().graph).toEqual(b.jobs[0]!.graph)
    await h.runner.retire()
  })

  it('takes one batch of pictures at a time, from any device', async () => {
    const h = await harness()
    const first = groupBody({ desk: 'images', jobs: [{}, {}], device: 'phone' })
    expect((await h.submit(first)).status).toBe(200)
    const r = await h.submit(groupBody({ desk: 'images', jobs: [{}], device: 'laptop' }))
    expect(r.status).toBe(409)
    expect(r.json()).toEqual({ error: 'A batch of pictures is already being made, from this page or another.', busy: 'images' })
    // Clips are not limited.
    expect((await h.submit(groupBody({ desk: 'video', jobs: [{}] }))).status).toBe(200)
    expect((await h.submit(groupBody({ desk: 'video', jobs: [{}] }))).status).toBe(200)
    await h.post(`/api/runner/groups/${first.group.id}/stop`)
    expect((await h.submit(groupBody({ desk: 'images', jobs: [{}] }))).status).toBe(200)
    await h.runner.retire()
  })

  it('refuses a group for a desk this server does not send through the queue', async () => {
    const h = await harness({ desks: ['video'] })
    const r = await h.submit(groupBody({ desk: 'images', jobs: [{}] }))
    expect(r.status).toBe(400)
    expect(r.json().error).toBe('the images desk does not send its work through the queue on this server')
    expect(h.runner.status().desks).toEqual(['video'])
    await h.runner.retire()
  })

  it('answers 507 in so many words when the disk is full, and keeps nothing', async () => {
    const h = await harness()
    await h.submit(groupBody({ desk: 'video', jobs: [{}] }))
    const rev = h.snap().rev
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    disk.full = true
    const r = await h.submit(groupBody({ desk: 'video', jobs: [{}, {}] }))
    disk.full = false
    warn.mockRestore()
    expect(r.status).toBe(507)
    expect(r.json()).toEqual({ error: 'The server’s disk is full, so it cannot save this work. Nothing was taken.' })
    expect(h.snap().rev).toBe(rev)
    expect(h.snap().jobs).toHaveLength(1)
    expect(readdirSync(path.join(h.dir, 'jobs'))).toHaveLength(1)
    await h.runner.retire()
  })

  it('takes back the graphs it saved when the disk fills part way through a group', async () => {
    const h = await harness()
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    disk.full = true
    disk.after = 2
    const r = await h.submit(groupBody({ desk: 'video', jobs: [{}, {}, {}] }))
    disk.full = false
    warn.mockRestore()
    expect(r.status).toBe(507)
    expect(readdirSync(path.join(h.dir, 'jobs'))).toEqual([])
    await h.runner.retire()
  })

  it.skipIf(asRoot)('answers 507 without claiming a full disk when its folder cannot be written', async () => {
    const h = await harness()
    await h.submit(groupBody({ desk: 'video', jobs: [{}] }))
    const { chmodSync } = await import('node:fs')
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    chmodSync(path.join(h.dir, 'jobs'), 0o500)
    let r
    try {
      r = await h.submit(groupBody({ desk: 'video', jobs: [{}] }))
    } finally {
      chmodSync(path.join(h.dir, 'jobs'), 0o700)
      warn.mockRestore()
    }
    expect(r.status).toBe(507)
    expect(r.json().error).toMatch(/^The server cannot save this work: .+\. Nothing was taken\.$/)
    expect(h.snap().jobs).toHaveLength(1)
    await h.runner.retire()
  })
})

describe('a write to the list of work that fails later', () => {
  it('holds that job back, says it waits for room at the current rev, sends nothing, and goes on once there is room', async () => {
    const h = await harness()
    const A = (await h.submit(groupBody({ desk: 'images', jobs: [{}] }))).json().jobs[0].id
    const s = open(h.runner.handler, { url: '/api/runner/stream' })
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    disk.full = true
    await h.tick(3)
    const rev = h.snap().rev
    expect(h.comfy.log).toEqual([])
    expect(h.job(A)).toMatchObject({ status: 'waiting', wait: { for: 'disk' } })
    const said = revEvents(s.reply).filter((e) => e.event === 'job')
    expect(said.at(-1)!.data).toMatchObject({ rev, job: { id: A, wait: { for: 'disk' } } })
    disk.full = false
    await h.tick()
    warn.mockRestore()
    expect(h.job(A).status).toBe('queued')
    expect(h.comfy.prompts()).toHaveLength(1)
    s.hangUp()
    await h.runner.retire()
  })
})

describe('the stream', () => {
  it('opens with the whole state, then one rev per commit on every event of it, with no gap', async () => {
    const h = await harness()
    const s = open(h.runner.handler, { url: '/api/runner/stream' })
    expect(s.reply.headers['content-type']).toBe('text/event-stream')
    const [first] = revEvents(s.reply)
    expect(first).toMatchObject({ event: 'state', data: { v: 1, available: true, rev: h.snap().rev, boot: h.snap().boot, jobs: [] } })
    const start = first!.data.rev as number

    // Work that goes through most kinds of commit: intake, sends, a loss
    // (with misses that change nothing a page shows), a hold, a filing.
    const [A, B] = (await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }, { heavy: true }] }))).json().jobs.map((j: { id: string }) => j.id)
    await h.tick()
    h.comfy.restart()
    await h.tick(3, 21_000)
    await h.post('/api/runner/lane', { action: 'send' })
    await h.tick(1, 2000)
    h.comfy.finish(h.job(B).promptId!, { files: [{ filename: 'b.webm', video: true }] })
    await h.tick(1, 2000)
    await h.post('/api/runner/dismiss', { jobIds: [A, B] })
    expect(h.job(A).status).toBe('lost')

    const withRev = revEvents(s.reply).slice(1).filter((e) => typeof e.data.rev === 'number')
    const revs = [...new Set(withRev.map((e) => e.data.rev as number))]
    const end = h.snap().rev
    expect(revs).toEqual(Array.from({ length: end - start }, (_, i) => start + 1 + i))
    expect(new Set(withRev.map((e) => e.event))).toEqual(new Set(['job', 'group', 'lane']))
    // A page that applies them in order ends where the server is.
    const last = new Map<string, any>()
    for (const e of withRev) if (e.event === 'job') last.set(e.data.job.id, e.data.job)
    expect(last.get(B)).toEqual(h.job(B))
    expect(last.get(A)).toEqual(h.job(A))
    s.hangUp()
    await h.runner.retire()
  })

  it('says when ComfyUI stops and starts answering, with no rev', async () => {
    const h = await harness()
    const s = open(h.runner.handler, { url: '/api/runner/stream' })
    await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }] }))
    h.comfy.unanswered = 1
    await h.tick()
    await h.tick(1, 2000)
    const comfy = revEvents(s.reply).filter((e) => e.event === 'comfy')
    expect(comfy.map((e) => e.data.answering)).toEqual([false, true])
    expect(comfy.every((e) => e.data.rev === undefined && typeof e.data.since === 'number')).toBe(true)
    s.hangUp()
    await h.runner.retire()
  })

  it('pings every 25 s, and forgets a watcher that hung up', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] })
    try {
      const h = await harness()
      const s = open(h.runner.handler, { url: '/api/runner/stream' })
      vi.advanceTimersByTime(25_000)
      expect(s.reply.body).toContain(': ping\n\n')
      s.hangUp()
      const before = s.reply.body
      await h.submit(groupBody({ desk: 'images', jobs: [{}] }))
      vi.advanceTimersByTime(25_000)
      expect(s.reply.body).toBe(before)
      await h.runner.retire()
    } finally {
      vi.useRealTimers()
    }
  })

  it('carries a new boot once the queue is started again', async () => {
    const h = await harness()
    const one = revEvents(open(h.runner.handler, { url: '/api/runner/stream' }).reply)[0]!
    const h2 = await h.restart()
    const two = revEvents(open(h2.runner.handler, { url: '/api/runner/stream' }).reply)[0]!
    expect(two.data.boot).not.toBe(one.data.boot)
    expect(two.data.rev).toBeGreaterThanOrEqual(one.data.rev)
    await h2.runner.retire()
  })
})

describe('progress and previews from ComfyUI\'s socket', () => {
  async function queuedJob(h: Harness) {
    const id = (await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }] }))).json().jobs[0].id as string
    await h.tick()
    return { id, pid: h.job(id).promptId! }
  }

  it('reports steps with the node\'s class and sampling pass, and no rev', async () => {
    const h = await harness()
    const { id, pid } = await queuedJob(h)
    const s = open(h.runner.handler, { url: '/api/runner/stream' })
    h.comfy.emit({ type: 'progress', data: { prompt_id: pid, node: '2', value: 3, max: 10 } })
    const p = revEvents(s.reply).find((e) => e.event === 'progress')!.data
    expect(p).toMatchObject({ id, value: 3, max: 10, node: '2', classType: 'KSamplerAdvanced', pass: { index: 2, count: 2 }, previewN: 0 })
    expect(p.rev).toBeUndefined()
    expect(h.snap().progress[id]).toMatchObject({ value: 3, max: 10 })
    // Work that is not ours is not reported.
    h.comfy.emit({ type: 'progress', data: { prompt_id: 'someone-else', node: '1', value: 1, max: 2 } })
    expect(revEvents(s.reply).filter((e) => e.event === 'progress')).toHaveLength(1)
    s.hangUp()
    await h.runner.retire()
  })

  it('sends at most four a second for a job, and always the last', async () => {
    vi.useFakeTimers({ toFake: ['Date', 'setTimeout', 'clearTimeout'] })
    try {
      const h = await harness()
      const { pid } = await queuedJob(h)
      const s = open(h.runner.handler, { url: '/api/runner/stream' })
      const said = () => revEvents(s.reply).filter((e) => e.event === 'progress').map((e) => e.data.value)
      vi.advanceTimersByTime(1000)
      for (let v = 1; v <= 6; v++) h.comfy.emit({ type: 'progress', data: { prompt_id: pid, node: '1', value: v, max: 10 } })
      expect(said()).toEqual([1])
      vi.advanceTimersByTime(249)
      expect(said()).toEqual([1])
      vi.advanceTimersByTime(1)
      expect(said()).toEqual([1, 6])
      vi.advanceTimersByTime(300)
      h.comfy.emit({ type: 'progress', data: { prompt_id: pid, node: '1', value: 7, max: 10 } })
      expect(said()).toEqual([1, 6, 7])
      s.hangUp()
      await h.runner.retire()
    } finally {
      vi.useRealTimers()
    }
  })

  it('serves the newest preview with no-store and same-origin, and none before there is one', async () => {
    const h = await harness()
    const { id, pid } = await queuedJob(h)
    expect((await h.get(`/api/runner/jobs/${id}/preview`)).status).toBe(404)
    const s = open(h.runner.handler, { url: '/api/runner/stream' })
    h.comfy.emit({ type: 'preview', promptId: pid, mime: 'image/png', bytes: Buffer.from('first') })
    // The old shape names no prompt: it is the one ComfyUI said it is executing.
    h.comfy.emit({ type: 'executing', data: { prompt_id: pid, node: '1' } })
    h.comfy.emit({ type: 'preview', promptId: null, mime: 'image/jpeg', bytes: Buffer.from('second') })
    expect(revEvents(s.reply).filter((e) => e.event === 'preview').map((e) => e.data)).toEqual([{ id, n: 1 }, { id, n: 2 }])
    const r = await h.get(`/api/runner/jobs/${id}/preview`)
    expect(r.status).toBe(200)
    expect(r.body).toBe('second')
    expect(r.headers).toMatchObject({ 'content-type': 'image/jpeg', 'cache-control': 'no-store', 'cross-origin-resource-policy': 'same-origin' })
    expect(h.snap().progress[id]!.previewN).toBe(2)
    s.hangUp()
    await h.runner.retire()
  })
})

describe('reading one job', () => {
  it('gives its view, graph and record, and 404 for one it never had', async () => {
    const h = await harness()
    const b = groupBody({ desk: 'video', jobs: [{ heavy: true }] })
    await h.submit(b)
    const r = await h.get(`/api/runner/jobs/${b.jobs[0]!.id}`)
    expect(r.status).toBe(200)
    expect(r.json()).toEqual({ job: h.job(b.jobs[0]!.id), graph: HEAVY(), record: b.jobs[0]!.record })
    // What only the queue keeps is not shown.
    for (const k of ['promptIdInternal', 'sighted', 'misses', 'specSha', 'graphSha', 'chain', 'noFile']) expect(r.json().job, k).not.toHaveProperty(k)
    expect((await h.get(`/api/runner/jobs/${randomUUID()}`)).json()).toEqual({ error: 'no such job' })
    await h.runner.retire()
  })
})

describe('words to the queue', () => {
  it('lane: 400 for no word it knows, 409 with nothing held', async () => {
    const h = await harness()
    expect((await h.post('/api/runner/lane', { action: 'hold' })).status).toBe(400)
    expect((await h.post('/api/runner/lane', { action: 'send' })).json()).toEqual({ error: 'Nothing is held.' })
    await h.runner.retire()
  })

  it('dismiss: only ended jobs, and the group once all its jobs are', async () => {
    const h = await harness()
    const g = (await h.submit(groupBody({ desk: 'video', jobs: [{}, {}] }))).json()
    const [A, B] = g.jobs.map((j: { id: string }) => j.id)
    await h.post(`/api/runner/jobs/${A}/stop`)
    expect((await h.post('/api/runner/dismiss', { jobIds: [A, B, 'nothing'] })).json()).toEqual({ dismissed: [A] })
    expect(h.job(A).dismissed).toBe(true)
    expect(h.job(B).dismissed).toBe(false)
    await h.post(`/api/runner/jobs/${B}/stop`)
    await h.post('/api/runner/dismiss', { jobIds: [B] })
    expect(h.group(g.group.id)).toMatchObject({ state: 'ended', dismissed: true })
    expect((await h.post('/api/runner/dismiss', { jobIds: 'all' })).status).toBe(400)
    await h.runner.retire()
  })
})

describe('letting ended work go', () => {
  it('drops jobs 48 hours after they ended with their saved graphs, keeping a group still going whole', async () => {
    const h = await harness()
    const s = open(h.runner.handler, { url: '/api/runner/stream' })
    const old = (await h.submit(groupBody({ desk: 'video', jobs: [{}] }))).json()
    const oldId = old.jobs[0].id
    await h.post(`/api/runner/jobs/${oldId}/stop`)
    const batch = (await h.submit(groupBody({ desk: 'images', jobs: [{}, {}] }))).json()
    const [P1, P2] = batch.jobs.map((j: { id: string }) => j.id)
    await h.tick()
    h.comfy.finish(h.job(P1).promptId!, { files: [{ filename: 'p1.png' }] })
    await h.tick()
    expect(h.job(P1).status).toBe('done')
    await h.tick(1, 49 * 3_600_000)
    expect(h.snap().jobs.map((j) => j.id).sort()).toEqual([P1, P2].sort())
    expect(h.group(old.group.id)).toBeUndefined()
    expect(existsSync(path.join(h.dir, 'jobs', `${oldId}.json`))).toBe(false)
    expect(existsSync(path.join(h.dir, 'jobs', `${P1}.json`))).toBe(true)
    const gone = revEvents(s.reply).find((e) => e.event === 'gone')!
    expect(gone.data).toEqual({ rev: gone.data.rev, jobs: [oldId], groups: [old.group.id] })
    s.hangUp()
    await h.runner.retire()
  })
})

describe('a reel pass no page has taken in', () => {
  const H = 3_600_000
  const D = 24 * H

  it('outlives the 48 hours other ended work gets, and goes as other work does once a page dismissed it', async () => {
    const h = await harness()
    const S = (await h.submit(groupBody({ desk: 'reel', jobs: [{}] }))).json().jobs[0].id
    await h.tick()
    h.comfy.finish(h.job(S).promptId!, { files: [{ filename: 's.webm', video: true }] })
    await h.tick()
    expect(h.job(S).status).toBe('done')
    const clip = (await h.submit(groupBody({ desk: 'video', jobs: [{}] }))).json().jobs[0].id
    await h.post(`/api/runner/jobs/${clip}/stop`)
    await h.tick(1, 49 * H)
    const ids = h.snap().jobs.map((j) => j.id)
    expect(ids).toContain(S)
    expect(ids).not.toContain(clip)
    expect(existsSync(path.join(h.dir, 'jobs', `${S}.json`))).toBe(true)
    await h.post('/api/runner/dismiss', { jobIds: [S] })
    await h.tick(1, 61_000)
    expect(h.snap().jobs.map((j) => j.id)).not.toContain(S)
    await h.runner.retire()
  })

  /** `sizes` groups of `desk`, each stopped before any of it was sent, a second apart: ended jobs cheaply. */
  async function stoppedGroups(h: Harness, desk: 'video' | 'reel', sizes: number[]) {
    const out: { id: string; jobs: string[] }[] = []
    for (const n of sizes) {
      h.clock.t += 1000
      const g = (await h.submit(groupBody({ desk, jobs: Array.from({ length: n }, () => ({})) }))).json()
      expect((await h.post(`/api/runner/groups/${g.group.id}/stop`, {})).status).toBe(200)
      out.push({ id: g.group.id, jobs: g.jobs.map((j: { id: string }) => j.id) })
    }
    return out
  }

  it('outlives the cap of 500 on other ended work too, which loses its oldest', async () => {
    const h = await harness()
    const S = (await h.submit(groupBody({ desk: 'reel', jobs: [{}] }))).json().jobs[0].id
    await h.tick()
    h.comfy.finish(h.job(S).promptId!, { files: [{ filename: 's.webm', video: true }] })
    await h.tick()
    expect(h.job(S).status).toBe('done')
    // 600 clips that ended after the pass did, the first 100 of them oldest.
    const [oldest, ...rest] = await stoppedGroups(h, 'video', [100, 200, 200, 100])
    await h.tick(1, 61_000)
    const ids = new Set(h.snap().jobs.map((j) => j.id))
    expect(ids.has(S)).toBe(true)
    expect(existsSync(path.join(h.dir, 'jobs', `${S}.json`))).toBe(true)
    expect(ids.size).toBe(501)
    expect(oldest!.jobs.filter((id) => ids.has(id))).toEqual([])
    expect(h.group(oldest!.id)).toBeUndefined()
    for (const g of rest) expect(g.jobs.filter((id) => !ids.has(id))).toEqual([])
    await h.runner.retire()
  })

  it('is kept to the newest 2000 jobs of such passes, apart from other ended work', async () => {
    const h = await harness()
    const clips = await stoppedGroups(h, 'video', [10])
    const [oldest, ...rest] = await stoppedGroups(h, 'reel', [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200])
    await h.tick(1, 61_000)
    const ids = new Set(h.snap().jobs.map((j) => j.id))
    expect(ids.size).toBe(2010)
    expect(clips[0]!.jobs.filter((id) => !ids.has(id))).toEqual([])
    expect(oldest!.jobs.filter((id) => ids.has(id))).toEqual([])
    for (const g of rest) expect(g.jobs.filter((id) => !ids.has(id))).toEqual([])
    await h.runner.retire()
  })

  it('goes after 30 days, dismissed or not', async () => {
    const h = await harness()
    const S = (await h.submit(groupBody({ desk: 'reel', jobs: [{}] }))).json().jobs[0].id
    await h.post(`/api/runner/jobs/${S}/stop`)
    await h.tick(1, 29 * D)
    expect(h.snap().jobs.map((j) => j.id)).toContain(S)
    await h.tick(1, 2 * D)
    expect(h.snap().jobs.map((j) => j.id)).not.toContain(S)
    await h.runner.retire()
  })
})

describe('the queue turned off, or standing back', () => {
  it('answers reads with available false and every write with 503, and never touches its folder', async () => {
    const h = await harness({ enabled: false })
    expect(h.runner.status()).toEqual({ active: false, desks: [], reason: 'Turned off with SWITCHGEN_RUNNER=off.' })
    const snap = (await h.get('/api/runner')).json()
    expect(snap).toMatchObject({ available: false, reason: 'Turned off with SWITCHGEN_RUNNER=off.', jobs: [] })
    const r = await h.submit(groupBody({ desk: 'images', jobs: [{}] }))
    expect(r.status).toBe(503)
    expect(r.json()).toEqual({ error: 'Turned off with SWITCHGEN_RUNNER=off.', busy: 'runner', reason: 'Turned off with SWITCHGEN_RUNNER=off.' })
    expect(existsSync(h.dir)).toBe(false)
    expect(h.comfy.sockets).toBe(0)
    await h.runner.retire()
  })

  it('stays off with SWITCHGEN_RUNNER=off whatever the caller asks', async () => {
    process.env.SWITCHGEN_RUNNER = 'off'
    try {
      const h = await harness({ enabled: true })
      expect(h.runner.status().reason).toBe('Turned off with SWITCHGEN_RUNNER=off.')
      expect(existsSync(h.dir)).toBe(false)
      await h.runner.retire()
    } finally {
      process.env.SWITCHGEN_RUNNER = 'on'
    }
  })
})

describe('a ComfyUI older than the jobs list', () => {
  const OLD = 'This ComfyUI is older than the server queue needs (it has no jobs list); pages send their own work.'

  /** The scripted ComfyUI, saying whether it has the jobs list: yes, no, or nothing at all. */
  class Listing extends FakeComfy {
    has: boolean | 'down' = true
    asked = 0
    async hasJobsList(): Promise<boolean> {
      this.asked++
      if (this.has === 'down') throw new Unanswered('ComfyUI did not answer')
      return this.has
    }
  }

  it('keeps the queue off with the reason, takes no work, and sends nothing', async () => {
    const comfy = new Listing()
    comfy.has = false
    const h = await harness({ comfy })
    await h.tick()
    expect(h.runner.status()).toEqual({ active: false, desks: [], reason: OLD })
    const r = await h.submit(groupBody({ desk: 'images', jobs: [{}] }))
    expect(r.status).toBe(503)
    expect(r.json().reason).toBe(OLD)
    expect(comfy.prompts()).toEqual([])
    await h.runner.retire()
  })

  it('keeps the queue on while ComfyUI does not answer, and sends nothing until the list is seen', async () => {
    const comfy = new Listing()
    comfy.has = 'down'
    const h = await harness({ comfy })
    const A = (await h.submit(groupBody({ desk: 'images', jobs: [{}] }))).json().jobs[0].id
    await h.tick(3, 1000)
    expect(h.runner.status().active).toBe(true)
    expect(h.job(A).wait).toEqual({ for: 'comfy' })
    expect(comfy.prompts()).toEqual([])
    comfy.has = true
    await h.tick()
    expect(h.job(A).promptId).toBeTruthy()
    await h.runner.retire()
  })

  it('stands back when the ComfyUI that comes back has no list, calls no clip it sent lost, and runs again once it has', async () => {
    const comfy = new Listing()
    const h = await harness({ comfy })
    const [X, Y] = (await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }, { heavy: true }] }))).json().jobs.map((j: { id: string }) => j.id)
    await h.tick()
    expect(h.job(X).promptId).toBeTruthy()
    const asked = comfy.asked
    // Down for a read, then back as a ComfyUI older than the list.
    comfy.unanswered = 1
    await h.tick(1, 1000)
    comfy.has = false
    await h.tick(3, 21_000)
    expect(comfy.asked).toBeGreaterThan(asked)
    expect(h.runner.status()).toMatchObject({ active: false, reason: OLD })
    expect(h.onDisk().jobs[X].status).toBe('queued')
    // The list is back: the work that waited through it is held until the reader says.
    comfy.has = true
    await h.tick(1, 10_000)
    expect(h.runner.status().active).toBe(true)
    expect(h.snap().lane.held).toMatchObject({ why: 'paused', scope: 'all' })
    expect(h.job(Y).wait).toEqual({ for: 'held' })
    await h.runner.retire()
  })

  it('asks again when ComfyUI greets a new connection while the queue is idle, and sends nothing to one without the list', async () => {
    const comfy = new Listing()
    const h = await harness({ comfy })
    await h.tick(2, 1000)
    const asked = comfy.asked
    // No read fails: ComfyUI restarts between two idle ticks and comes back
    // as an older build, and all the runner sees is the new connection's greeting.
    comfy.has = false
    comfy.emit({ type: 'status', data: { status: { exec_info: { queue_remaining: 0 } }, sid: 'new-session' } } as never)
    await h.tick(1, 1000)
    expect(comfy.asked).toBe(asked + 1)
    expect(h.runner.status()).toMatchObject({ active: false, reason: OLD })
    const r = await h.submit(groupBody({ desk: 'images', jobs: [{}] }))
    expect(r.status).toBe(503)
    expect(comfy.prompts()).toEqual([])
    await h.runner.retire()
  })

  it('is asked once while ComfyUI goes on answering, not on every pass', async () => {
    const comfy = new Listing()
    const h = await harness({ comfy })
    await h.tick(5, 1000)
    expect(comfy.asked).toBe(1)
    await h.runner.retire()
  })
})

describe('/api/capabilities and the plugin', () => {
  let si: Awaited<ReturnType<typeof standIn>> | null = null
  let child: ChildProcess | null = null
  let apiHandler: Handler
  const caps = async () => (await call(apiHandler, { url: '/api/capabilities' })).json()

  beforeAll(async () => {
    si = await standIn()
    process.env.COMFY_URL = si.url
    const { switchgenApi } = await import('../server/api.mjs')
    apiHandler = await mountPlugin(switchgenApi())
  })
  afterEach(async () => {
    await clearRegistry()
    child?.kill()
    child = null
  })
  afterAll(async () => {
    await si?.close()
  })

  it('says the queue runs and which desks it takes, from the queue\'s own word', async () => {
    const { switchgenRunner } = await import('../server/runner.mjs')
    await mountPlugin(switchgenRunner())
    const c = await caps()
    expect(c).toMatchObject({ server: 'switchgen', deleteFiles: true, archive: true, runner: true, runnerDesks: ['video', 'images', 'reel', 'lab'], runnerReason: null })
  })

  it('takes only the desks SWITCHGEN_RUNNER_DESKS names', async () => {
    process.env.SWITCHGEN_RUNNER_DESKS = 'reel, video,pictures'
    try {
      const { switchgenRunner } = await import('../server/runner.mjs')
      const h = await mountPlugin(switchgenRunner())
      expect((await caps()).runnerDesks).toEqual(['reel', 'video'])
      const r = await call(h, { method: 'POST', url: '/api/runner/groups', body: groupBody({ desk: 'images', jobs: [{}] }) })
      expect(r.status).toBe(400)
    } finally {
      delete process.env.SWITCHGEN_RUNNER_DESKS
    }
  })

  it('says no, and why, with SWITCHGEN_RUNNER=off, and the queue takes nothing', async () => {
    process.env.SWITCHGEN_RUNNER = 'off'
    const dir = tempRoot()
    process.env.SWITCHGEN_RUNNER_DIR = path.join(dir, 'runner')
    try {
      const { switchgenRunner } = await import('../server/runner.mjs')
      const sockets = si!.socketUrls.length
      const h = await mountPlugin(switchgenRunner())
      expect(await caps()).toMatchObject({ runner: false, runnerDesks: [], runnerReason: 'Turned off with SWITCHGEN_RUNNER=off.', deleteFiles: true })
      expect((await call(h, { method: 'POST', url: '/api/runner/groups', body: groupBody({ desk: 'images', jobs: [{}] }) })).status).toBe(503)
      expect(existsSync(path.join(dir, 'runner'))).toBe(false)
      // It never opened a socket to ComfyUI either.
      await new Promise((resolve) => setTimeout(resolve, 50))
      expect(si!.socketUrls).toHaveLength(sockets)
    } finally {
      process.env.SWITCHGEN_RUNNER = 'on'
    }
  })

  it('says no while another server holds the archive, and yes once that server stops', async () => {
    const { archiveApi } = await import('../server/archive.mjs')
    // Taken from here only once the other server has it.
    await clearRegistry()
    const script = path.join(tempRoot(), 'holder.mjs')
    const archiveUrl = pathToFileURL(path.resolve(import.meta.dirname, '..', 'server', 'archive.mjs')).href
    writeFileSync(
      script,
      `process.env.SWITCHGEN_OUTPUTS = ${JSON.stringify(archiveApi.outputsRoot)}
process.env.SWITCHGEN_ARCHIVE = ${JSON.stringify(archiveApi.archiveFile)}
const { switchgenArchive } = await import(${JSON.stringify(archiveUrl)})
switchgenArchive().configurePreviewServer({ middlewares: { use() {} } })
console.log('HOLDING')
process.stdin.on('data', () => process.exit(0))
`,
    )
    // Let go of the lock this process may hold, so the other server can take it.
    const { unlinkSync } = await import('node:fs')
    try { unlinkSync(`${archiveApi.archiveFile}.lock`) } catch { /* not held */ }
    child = spawn(process.execPath, [script], { stdio: ['pipe', 'pipe', 'ignore'] })
    // Its ending, taken as it starts, so an early one is not waited for again.
    const exited = new Promise((resolve) => child!.once('exit', resolve))
    child.stdin!.on('error', () => {})
    await new Promise<void>((resolve, reject) => {
      child!.stdout!.on('data', (d: Buffer) => d.toString().includes('HOLDING') && resolve())
      child!.on('exit', () => reject(new Error('the holder ended early')))
    })
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const { switchgenRunner } = await import('../server/runner.mjs')
    const h = await mountPlugin(switchgenRunner())
    const held = 'Another SwitchGen server holds the archive, and the queue with it.'
    expect(await caps()).toMatchObject({ runner: false, runnerDesks: [], runnerReason: held })
    const r = await call(h, { method: 'POST', url: '/api/runner/groups', body: groupBody({ desk: 'images', jobs: [{}] }) })
    expect(r.status).toBe(503)
    expect(r.json()).toEqual({ error: held, busy: 'runner', reason: held })
    expect((await call(h, { url: '/api/runner' })).json()).toMatchObject({ available: false, reason: held })

    if (child.exitCode === null && child.signalCode === null) child.stdin!.write('exit\n')
    await exited
    child = null
    expect(await caps()).toMatchObject({ runner: true, runnerReason: null })
    expect((await call(h, { url: '/api/runner' })).json().available).toBe(true)
    warn.mockRestore()
  }, 30_000)

  it('keeps a folder of work for each archive: two archives in one folder never share one', async () => {
    const { runnerDirFor } = await import('../server/runner.mjs')
    expect(runnerDirFor('/o/.switchgen/archive.json')).toBe('/o/.switchgen/runner')
    expect(runnerDirFor('/o/.switchgen/archive-b.json')).toBe('/o/.switchgen/archive-b.json.runner')

    // Two servers on one outputs root, the second given an archive of its
    // own beside the first's, with no SWITCHGEN_RUNNER_DIR: both run a queue,
    // each over its own folder, each holding that folder's lock.
    const root = tempRoot()
    const outputs = path.join(root, 'outputs')
    const script = path.join(root, 'server.mjs')
    const runnerUrl = pathToFileURL(path.resolve(import.meta.dirname, '..', 'server', 'runner.mjs')).href
    writeFileSync(
      script,
      `const { switchgenRunner, runnerStatus } = await import(${JSON.stringify(runnerUrl)})
await switchgenRunner().configurePreviewServer({ middlewares: { use() {} } })
console.log('STATUS ' + JSON.stringify(runnerStatus()))
process.stdin.on('data', () => process.exit(0))
`,
    )
    // Each server with the promise of its ending, taken as it starts, so one
    // that ended early is neither written to nor waited for again.
    const servers: { c: ChildProcess; ended: Promise<unknown> }[] = []
    const start = (archiveName: string) =>
      new Promise<{ active: boolean; reason: string | null }>((resolve, reject) => {
        const env: NodeJS.ProcessEnv = { ...process.env, SWITCHGEN_OUTPUTS: outputs, SWITCHGEN_ARCHIVE: path.join(outputs, '.switchgen', archiveName), SWITCHGEN_MODELS: path.join(root, 'models'), SWITCHGEN_RUNNER: 'on', COMFY_URL: si!.url }
        // Every other root follows the outputs root, in the temporary folder.
        for (const k of ['SWITCHGEN_RUNNER_DIR', 'SWITCHGEN_RUNNER_DESKS', 'SWITCHGEN_THUMBS']) delete env[k]
        const c = spawn(process.execPath, [script], { env, stdio: ['pipe', 'pipe', 'ignore'] })
        servers.push({ c, ended: new Promise((done) => c.once('exit', done)) })
        // A server that ends as it is told to may close its input first.
        c.stdin!.on('error', () => {})
        let out = ''
        c.stdout!.on('data', (d: Buffer) => {
          out += d.toString()
          const line = /STATUS (.*)/.exec(out)?.[1]
          if (line) resolve(JSON.parse(line))
        })
        c.on('exit', () => reject(new Error(`the server on ${archiveName} ended early`)))
      })
    try {
      expect(await start('archive.json')).toMatchObject({ active: true, reason: null })
      expect(await start('archive-b.json')).toMatchObject({ active: true, reason: null })
      const lockOf = (folder: string) => JSON.parse(readFileSync(path.join(outputs, '.switchgen', folder, 'lock'), 'utf8')).pid
      expect(lockOf('runner')).toBe(servers[0]!.c.pid)
      expect(lockOf('archive-b.json.runner')).toBe(servers[1]!.c.pid)
    } finally {
      for (const { c, ended } of servers) {
        if (c.exitCode === null && c.signalCode === null) c.stdin!.write('exit\n')
        await ended
      }
    }
    // Each let its folder go as it ended.
    expect(existsSync(path.join(outputs, '.switchgen', 'runner', 'lock'))).toBe(false)
    expect(existsSync(path.join(outputs, '.switchgen', 'archive-b.json.runner', 'lock'))).toBe(false)
  }, 30_000)

  it('still answers, with no queue, when the queue cannot say how it stands', async () => {
    const key = Symbol.for('switchgen.runner')
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    ;(globalThis as Record<symbol, unknown>)[key] = { current: { status: () => { throw new Error('broken') } } }
    try {
      const r = await call(apiHandler, { url: '/api/capabilities' })
      expect(r.status).toBe(200)
      expect(r.json()).toMatchObject({ deleteFiles: true, archive: true, runner: false, runnerDesks: [], runnerReason: 'The queue on the server is not running.' })
    } finally {
      delete (globalThis as Record<symbol, unknown>)[key]
      warn.mockRestore()
    }
    // And with no queue mounted at all.
    expect(await caps()).toMatchObject({ runner: false, runnerDesks: [], runnerReason: 'The queue on the server is not running.' })
  })
})

describe('the Vite config', () => {
  it('mounts the queue straight after the archive whose lock it runs under', async () => {
    type Plugin = { name?: string }
    const config = (await import('../vite.config')).default as unknown as { plugins: unknown[] }
    const names = (config.plugins.flat(Infinity) as (Plugin | null | false)[]).filter((p): p is Plugin => !!p).map((p) => p.name)
    expect(names[names.indexOf('switchgen-archive') + 1]).toBe('switchgen-runner')
  })
})
