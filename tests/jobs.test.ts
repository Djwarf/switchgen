import { createElement } from 'react'
import { renderToString } from 'react-dom/server'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type * as Comfy from '../src/lib/comfy'
import type * as Faults from '../src/lib/faults'
import type * as Jobs from '../src/components/shell/jobs'
import type * as Mirror from '../src/components/shell/mirror'
import type * as Notice from '../src/components/shell/Notice'

/**
 * The transport, the fault sentences and the press ledger against a fake
 * ComfyUI: fetch answers from a route table, and a fake socket delivers
 * whatever a test tells it to. Every module is loaded fresh for each test,
 * because each holds its own socket, queue and ledger.
 */
type Route = (url: string, init?: RequestInit) => Response | undefined
let routes: Route[] = []
let calls: string[] = []
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })

class FakeSocket {
  static OPEN = 1
  static CONNECTING = 0
  static last: FakeSocket | null = null
  readyState = 0
  binaryType = ''
  onopen: (() => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  onmessage: ((ev: { data: unknown }) => void) | null = null
  url: string
  constructor(url: string) {
    this.url = url
    FakeSocket.last = this
    queueMicrotask(() => {
      this.readyState = 1
      this.onopen?.()
    })
  }
  send() {}
  close() {
    this.readyState = 3
    this.onclose?.()
  }
  deliver(type: string, data: unknown) {
    this.onmessage?.({ data: JSON.stringify({ type, data }) })
  }
}

let comfy: typeof Comfy
let faults: typeof Faults
let jobs: typeof Jobs.jobs
let notice: typeof Notice
let mirror: typeof Mirror.mirror
let prompts = 0

const graph = { '8': { class_type: 'VAEDecode', inputs: {} } } as unknown as Comfy.ApiWorkflow
const tick = (ms = 0) => new Promise((resolve) => setTimeout(resolve, ms))
/** Queue a run and wait until the socket is up and the prompt accepted. */
async function started(): Promise<{ id: string; result: Promise<unknown> }> {
  const result = comfy.run(graph, () => {}).catch((e: unknown) => e)
  await vi.waitFor(() => {
    if (!calls.includes('POST /comfy/prompt') || FakeSocket.last?.readyState !== 1) throw new Error('not yet')
  })
  await tick(5)
  return { id: `p${prompts}`, result }
}

beforeEach(async () => {
  routes = [
    (url, init) => (url === '/comfy/prompt' && init?.method === 'POST' ? json({ prompt_id: `p${++prompts}` }) : undefined),
  ]
  calls = []
  prompts = 0
  vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
    calls.push(`${init?.method ?? 'GET'} ${url}`)
    for (const r of routes) {
      const res = r(String(url), init)
      if (res) return res
    }
    return json({ error: 'no route' }, 599)
  })
  vi.stubGlobal('WebSocket', FakeSocket)
  vi.stubGlobal('location', { protocol: 'http:', host: 'harness', hash: '' })
  vi.resetModules()
  comfy = await import('../src/lib/comfy')
  faults = await import('../src/lib/faults')
  jobs = (await import('../src/components/shell/jobs')).jobs
  notice = await import('../src/components/shell/Notice')
  mirror = (await import('../src/components/shell/mirror')).mirror
})

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe('a job that fails while it runs', () => {
  it('names the node class ComfyUI reported, and still reads a memory failure as one', async () => {
    const { id, result } = await started()
    FakeSocket.last!.deliver('execution_error', {
      prompt_id: id,
      node_id: '8',
      node_type: 'VAEDecode',
      exception_message: 'Allocation on device: This error means you ran out of memory on your GPU.',
    })
    const err = (await result) as Comfy.ComfyError
    expect(err.nodeType).toBe('VAEDecode')
    const f = faults.faultOf(err)
    expect(faults.faultWhere(f)).toBe('The trouble is in VAEDecode (node 8).')
    expect(faults.faultTitle(f)).toBe('The card ran out of memory')
  })

  it('is not called rejected: the queue took it and it broke part way', async () => {
    const { id, result } = await started()
    FakeSocket.last!.deliver('execution_error', { prompt_id: id, node_id: '11', node_type: 'LoadImage', exception_message: 'bad frame' })
    const f = faults.faultOf(await result)
    expect(faults.faultTitle(f)).toBe('That job did not finish')
    expect(faults.faultWhere(f)).toBe('The trouble is in LoadImage (node 11).')
  })

  it('names no trouble for a job that was stopped, though ComfyUI names the node', async () => {
    const { id, result } = await started()
    FakeSocket.last!.deliver('execution_interrupted', { prompt_id: id, node_id: '3', node_type: 'KSampler' })
    const f = faults.faultOf(await result)
    expect(f.cancelled).toBe(true)
    expect(faults.faultWhere(f)).toBeNull()
  })
})

describe('a job the queue refuses', () => {
  const refuse = (nodeErrors: Record<string, unknown>) =>
    routes.unshift((url, init) =>
      url === '/comfy/prompt' && init?.method === 'POST'
        ? json({ error: { message: 'Prompt outputs failed validation' }, node_errors: nodeErrors }, 400)
        : undefined,
    )

  it('names the one node it refused', async () => {
    refuse({ '11': { errors: [{ message: 'Invalid image file', details: 'x.png' }], class_type: 'LoadImage' } })
    const err = (await comfy.submit(graph).catch((e: unknown) => e)) as Comfy.ComfyError
    expect(err.node).toBe('11')
    expect(err.nodeType).toBe('LoadImage')
    expect(faults.faultTitle(faults.faultOf(err))).toBe('That job was rejected')
  })

  it('names none when it refused several', async () => {
    refuse({
      '11': { errors: [{ message: 'Invalid image file' }], class_type: 'LoadImage' },
      '3': { errors: [{ message: 'Value out of range' }], class_type: 'KSampler' },
    })
    const err = (await comfy.submit(graph).catch((e: unknown) => e)) as Comfy.ComfyError
    expect(err.node).toBeNull()
    expect(err.nodeType).toBeNull()
  })
})

describe('a failure read back from /history', () => {
  const failedRecord = (id: string) => ({
    [id]: {
      prompt: [0, id, graph, {}, []],
      outputs: {},
      status: {
        status_str: 'error',
        completed: false,
        messages: [
          ['execution_start', { prompt_id: id, timestamp: 1 }],
          [
            'execution_error',
            {
              prompt_id: id,
              node_id: '8',
              node_type: 'VAEDecode',
              exception_message: 'CUDA out of memory. Tried to allocate 2 GiB',
              exception_type: 'torch.OutOfMemoryError',
              timestamp: 2,
            },
          ],
        ],
      },
    },
  })

  it('keeps ComfyUI\'s own words when the socket missed the failure', async () => {
    routes.unshift((url) => (url === '/comfy/history/p1' ? json(failedRecord('p1')) : undefined))
    const { result } = await started()
    // The socket drops while the job fails, so the execution_error never arrives.
    FakeSocket.last!.close()
    const err = await Promise.race([result, tick(3000).then(() => 'timeout')])
    const f = faults.faultOf(err)
    expect(f.message).toMatch(/CUDA out of memory/)
    expect(faults.faultTitle(f)).toBe('The card ran out of memory')
    expect(faults.faultWhere(f)).toBe('The trouble is in VAEDecode (node 8).')
  })

  it('reads the error off a past run, and none off a run that succeeded', async () => {
    routes.unshift((url) => (url === '/comfy/history/p1' ? json(failedRecord('p1')) : undefined))
    routes.unshift((url) =>
      url === '/comfy/history/ok'
        ? json({ ok: { prompt: [0, 'ok', graph, {}, []], outputs: {}, status: { status_str: 'success', completed: true, messages: [] } } })
        : undefined,
    )
    expect((await comfy.fetchPastRun('p1'))?.error).toEqual({
      message: 'CUDA out of memory. Tried to allocate 2 GiB',
      node: '8',
      nodeType: 'VAEDecode',
    })
    expect((await comfy.fetchPastRun('ok'))?.error).toBeNull()
  })
})

describe('stopping a job', () => {
  let reply: () => Response
  beforeEach(() => {
    reply = () => json({ cancelled: true })
    routes.unshift((url, init) => {
      if (/\/comfy\/api\/jobs\/[^/]+\/cancel$/.test(url) && init?.method === 'POST') return reply()
      if (/\/comfy\/api\/jobs\/[^/?]+$/.test(url)) return json({ id: 'x', status: 'in_progress' })
      return undefined
    })
  })

  it('leaves an accepted stop live and stopping until its desk says how it ended', async () => {
    const id = jobs.start({ desk: 'video', kind: 'video', label: 'Wan', prompt: '', promptId: 'r1' })
    jobs.apply(id, { phase: 'running', node: null, value: 3, max: 20 })
    await jobs.cancel(id)
    expect(jobs.get(id)).toMatchObject({ status: 'running', cancelling: true, finishedAt: null })
    jobs.apply(id, { phase: 'running', node: null, value: 4, max: 20 })
    expect(jobs.get(id)).toMatchObject({ status: 'running', cancelling: true })
    // The clip finished before the stop landed, and was filed.
    jobs.succeed(id, { entryId: 'e1' })
    expect(jobs.get(id)).toMatchObject({ status: 'done', entryId: 'e1', cancelling: false })
    jobs.apply(id, { phase: 'running', node: null, value: 20, max: 20 })
    expect(jobs.get(id)?.status).toBe('done')
  })

  it('gives Stop back when there was nothing left to stop', async () => {
    reply = () => json({ cancelled: false })
    const id = jobs.start({ desk: 'images', kind: 'image', label: 'Krea', prompt: '', promptId: 'r2' })
    await jobs.cancel(id)
    expect(jobs.get(id)).toMatchObject({ status: 'queued', cancelling: false })
  })

  it('gives Stop back and says so when ComfyUI refuses', async () => {
    reply = () => json({ error: 'boom' }, 500)
    const id = jobs.start({ desk: 'video', kind: 'video', label: 'Wan', prompt: '', promptId: 'r3' })
    await jobs.cancel(id)
    expect(jobs.get(id)).toMatchObject({ status: 'queued', cancelling: false })
    const html = renderToString(createElement(notice.NoticeRail))
    expect(html).toContain('Could not stop that job')
    expect(html).toContain('HTTP 500')
  })

  it('holds a stop asked for before the queue answered, and sends it once it does', async () => {
    const id = jobs.start({ desk: 'video', kind: 'video', label: 'Wan', prompt: '' })
    await jobs.cancel(id)
    expect(calls).toEqual([])
    expect(jobs.get(id)).toMatchObject({ status: 'submitting', cancelling: true })
    jobs.attach(id, 'r4')
    await vi.waitFor(() => expect(calls).toContain('POST /comfy/api/jobs/r4/cancel'))
    jobs.fail(id, 'Stopped.', { cancelled: true })
    expect(jobs.get(id)?.status).toBe('cancelled')
  })

  it('keeps the later verdict and its time when a stopped job turns out to have finished', () => {
    vi.useFakeTimers()
    const id = jobs.start({ desk: 'images', kind: 'image', label: 'Krea', prompt: '', promptId: 'r5' })
    jobs.fail(id, 'x', { cancelled: true })
    // The slug shows news for a few seconds from when it came.
    vi.advanceTimersByTime(5000)
    const at = Date.now()
    jobs.succeed(id)
    const j = jobs.get(id)!
    expect(j.status).toBe('done')
    expect(j.error).toBeNull()
    expect(j.finishedAt).toBeGreaterThanOrEqual(at)
  })

  const rail = () => renderToString(createElement(notice.NoticeRail))

  it('gives Stop back and says so when steps keep coming well after an accepted stop', async () => {
    vi.useFakeTimers()
    const id = jobs.start({ desk: 'images', kind: 'image', label: 'Krea', prompt: '', promptId: 'r1' })
    jobs.apply(id, { phase: 'running', node: null, value: 3, max: 20 })
    await jobs.cancel(id)
    await vi.advanceTimersByTimeAsync(5000)
    // Early steps are the stop still on its way.
    jobs.apply(id, { phase: 'running', node: null, value: 4, max: 20 })
    expect(jobs.get(id)?.cancelling).toBe(true)
    await vi.advanceTimersByTimeAsync(11_000)
    // A node starting proves nothing: ComfyUI announces it before it notices the stop.
    jobs.apply(id, { phase: 'running', node: '9', value: 0, max: 1 })
    expect(jobs.get(id)?.cancelling).toBe(true)
    jobs.apply(id, { phase: 'running', node: null, value: 5, max: 20 })
    expect(jobs.get(id)?.cancelling).toBe(false)
    expect(rail()).toContain('That job has not stopped yet')
    expect(rail()).toContain('ComfyUI is still working on it')

    // Stop can be held again, and asking again takes the notice away.
    const before = calls.filter((c) => c === 'POST /comfy/api/jobs/r1/cancel').length
    await jobs.cancel(id)
    expect(calls.filter((c) => c === 'POST /comfy/api/jobs/r1/cancel').length).toBe(before + 1)
    expect(rail()).not.toContain('That job has not stopped yet')
  })

  it('keeps a running job that shows nothing either way as stopping', async () => {
    vi.useFakeTimers()
    const id = jobs.start({ desk: 'images', kind: 'image', label: 'Krea', prompt: '', promptId: 'r6' })
    jobs.apply(id, { phase: 'running', node: null, value: 3, max: 20 })
    await jobs.cancel(id)
    await vi.advanceTimersByTimeAsync(60_000)
    expect(jobs.get(id)?.cancelling).toBe(true)
  })

  it('gives Stop back when a stop did not take a job ComfyUI still has waiting', async () => {
    vi.useFakeTimers()
    routes.unshift((url) => (url === '/comfy/api/jobs/q1' ? json({ id: 'q1', status: 'pending' }) : undefined))
    // A desk whose own stop went nowhere.
    const id = jobs.start({ desk: 'video', kind: 'video', label: 'Wan', prompt: '', promptId: 'q1', stop: () => {} })
    await jobs.cancel(id)
    await vi.advanceTimersByTimeAsync(14_000)
    expect(jobs.get(id)?.cancelling).toBe(true)
    await vi.advanceTimersByTimeAsync(2000)
    expect(jobs.get(id)?.cancelling).toBe(false)
    expect(rail()).toContain('ComfyUI still has it waiting in its queue')
  })

  it('keeps it stopping when ComfyUI no longer lists it: the stop took', async () => {
    vi.useFakeTimers()
    routes.unshift((url) => (url === '/comfy/api/jobs/q2' ? json({ error: 'no such job' }, 404) : undefined))
    const id = jobs.start({ desk: 'video', kind: 'video', label: 'Wan', prompt: '', promptId: 'q2', stop: () => {} })
    await jobs.cancel(id)
    await vi.advanceTimersByTimeAsync(20_000)
    expect(jobs.get(id)?.cancelling).toBe(true)
    expect(rail()).not.toContain('That job has not stopped yet')
  })

  it('takes a failed stop\'s notice away once the job ends, however it ends', async () => {
    reply = () => json({ error: 'boom' }, 500)
    const endings: [string, (id: string) => void][] = [
      ['succeeded', (id) => jobs.succeed(id)],
      ['failed', (id) => jobs.fail(id, 'broke')],
      ['stopped', (id) => jobs.fail(id, 'Stopped.', { cancelled: true })],
      ['finished on the socket', (id) => jobs.apply(id, { phase: 'done', files: [] })],
      ['dismissed', (id) => jobs.dismiss(id)],
    ]
    for (const [how, end] of endings) {
      const id = jobs.start({ desk: 'images', kind: 'image', label: 'Krea', prompt: '', promptId: `e-${how}` })
      await jobs.cancel(id)
      expect(rail(), how).toContain('Could not stop that job')
      end(id)
      expect(rail(), how).not.toContain('Could not stop that job')
    }
  })
})

describe('the mirror between a desk and the ledger', () => {
  type Row = Mirror.Reported
  const row = (over: Partial<Row> & { key: string }): Row => ({
    status: 'running',
    promptId: null,
    label: 'Wan',
    prompt: 'a shot',
    value: 0,
    max: 0,
    entryId: null,
    error: null,
    ...over,
  })
  /** A desk whose rows a test sets, and whose own stop records what it was asked. */
  const desk = () => {
    let rows: Row[] = []
    const heard = new Set<() => void>()
    const stopped: string[] = []
    const bridge: Mirror.Bridge = {
      desk: 'reel',
      kind: 'video',
      subscribe: (fn) => {
        heard.add(fn)
        return () => void heard.delete(fn)
      },
      read: () => rows,
      stop: (key) => void stopped.push(key),
      seen: new Map(),
    }
    const report = (...next: Row[]) => {
      rows = next
      for (const fn of heard) fn()
    }
    return { bridge, report, stopped }
  }

  it('keeps a stopped shot stopped when the reel puts its earlier clip back as done', () => {
    const d = desk()
    const stop = mirror(d.bridge)
    d.report(row({ key: 'k1', status: 'running', promptId: 'p-k1', value: 2, max: 8 }))
    const id = d.bridge.seen.get('k1')!
    expect(jobs.get(id)?.status).toBe('running')
    d.report(row({ key: 'k1', status: 'cancelled', promptId: 'p-k1' }))
    expect(jobs.get(id)?.status).toBe('cancelled')
    // The pass ends and the shot shows the clip it had before.
    d.report(row({ key: 'k1', status: 'done', promptId: 'p-k1', entryId: 'old-clip' }))
    expect(jobs.get(id)).toMatchObject({ status: 'cancelled', entryId: null })
    stop()
  })

  it('still takes a later verdict over a failure, and a record named after the ending', () => {
    const d = desk()
    const stop = mirror(d.bridge)
    d.report(row({ key: 'k2', status: 'running', promptId: 'p-k2' }))
    const id = d.bridge.seen.get('k2')!
    d.report(row({ key: 'k2', status: 'error', promptId: 'p-k2', error: 'lost' }))
    expect(jobs.get(id)?.status).toBe('error')
    d.report(row({ key: 'k2', status: 'done', promptId: 'p-k2' }))
    expect(jobs.get(id)?.status).toBe('done')
    d.report(row({ key: 'k2', status: 'done', promptId: 'p-k2', entryId: 'e2' }))
    expect(jobs.get(id)).toMatchObject({ status: 'done', entryId: 'e2' })
    stop()
  })

  it('sends Stop to the desk\'s own stop, even before the job has a prompt', async () => {
    const d = desk()
    const stop = mirror(d.bridge)
    d.report(row({ key: 'clip-7', status: 'submitting' }))
    const id = d.bridge.seen.get('clip-7')!
    await jobs.cancel(id)
    expect(d.stopped).toEqual(['clip-7'])
    expect(calls.some((c) => c.endsWith('/cancel'))).toBe(false)
    expect(jobs.get(id)?.cancelling).toBe(true)
    stop()
  })
})

describe('the options a node offers', () => {
  it('reads both shapes ComfyUI sends a list in, and nothing else', () => {
    expect(comfy.optionsFor({ A: { input: { required: { f: [['a', 'b'], {}] } } } }, 'A', 'f')).toEqual(['a', 'b'])
    // UpscaleModelLoader already sends the newer shape.
    expect(comfy.optionsFor({ B: { input: { required: { model_name: ['COMBO', { options: ['4x.pth'] }] } } } }, 'B', 'model_name')).toEqual(['4x.pth'])
    expect(comfy.optionsFor({ C: { input: { optional: { o: ['COMBO', { options: ['x', 3, 'y'] }] } } } }, 'C', 'o')).toEqual(['x', 'y'])
    expect(comfy.optionsFor({ D: { input: { required: { n: ['INT', { default: 1 }] } } } }, 'D', 'n')).toEqual([])
    expect(comfy.optionsFor({}, 'Nothing', 'f')).toEqual([])
  })
})

describe('counting ComfyUI\'s queue', () => {
  const page = (status: string, total: number) =>
    json({ jobs: [{ id: `${status}-1`, status }], pagination: { offset: 0, limit: 1, total, has_more: total > 1 } })

  it('counts the whole queue from its totals, not the rows on one page', async () => {
    routes.unshift((url) => {
      if (url === '/comfy/api/jobs?status=in_progress&limit=1') return page('in_progress', 1)
      if (url === '/comfy/api/jobs?status=pending&limit=1') return page('pending', 60)
      return undefined
    })
    const id = jobs.start({ desk: 'video', kind: 'video', label: 'Wan', prompt: '', promptId: 'mine' })
    jobs.apply(id, { phase: 'running', node: null, value: 1, max: 20 })
    const stop = jobs.subscribe(() => {})
    await vi.waitFor(() => expect(jobs.snapshot().server.known).toBe(true))
    expect(jobs.snapshot().server).toEqual({ running: 1, pending: 60, foreign: 60, known: true })
    stop()
  })

  it('never calls a job of ours lost because a page of newer rows did not list it', async () => {
    vi.useFakeTimers()
    // The page the ledger used to read: fifty newer jobs, none of them ours.
    const rows = Array.from({ length: 50 }, (_, i) => ({ id: `other-${i}`, status: 'pending' }))
    routes.unshift((url) =>
      url.startsWith('/comfy/api/jobs?status=pending%2Cin_progress') || url.startsWith('/comfy/api/jobs?status=pending,in_progress')
        ? json({ jobs: rows, pagination: { offset: 0, limit: 50, total: 80, has_more: true } })
        : url.startsWith('/comfy/api/jobs?status=')
          ? page(url.includes('in_progress') ? 'in_progress' : 'pending', url.includes('in_progress') ? 1 : 80)
          : undefined,
    )
    const id = jobs.start({ desk: 'video', kind: 'video', label: 'Wan', prompt: '', promptId: 'mine' })
    jobs.apply(id, { phase: 'running', node: null, value: 1, max: 20 })
    const stop = jobs.subscribe(() => {})
    for (let i = 0; i < 3; i++) await vi.advanceTimersByTimeAsync(4000)
    expect(calls.filter((c) => c.startsWith('GET /comfy/api/jobs?')).length).toBeGreaterThanOrEqual(3)
    expect(jobs.get(id)?.status).toBe('running')
    stop()
  })
})

describe('the running slug', () => {
  beforeEach(() => {
    vi.stubGlobal('window', { addEventListener() {}, removeEventListener() {}, location: { hash: '' }, history: {} })
  })

  it('says Stopping, and takes the button out of reach, while a stop is on its way', async () => {
    const { RunningSlug } = await import('../src/components/shell/RunningSlug')
    routes.unshift((url) => (url.endsWith('/cancel') ? json({ cancelled: true }) : undefined))
    const id = jobs.start({ desk: 'images', kind: 'image', label: 'Krea', prompt: 'p', promptId: 'x1' })
    jobs.apply(id, { phase: 'running', node: null, value: 3, max: 20 })
    await jobs.cancel(id)
    const html = renderToString(createElement(RunningSlug))
    expect(html).toContain('Stopping')
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*aria-label="Stopping this picture"/)
    expect(html).toContain('If it is one of a batch, the rest of the batch is not made.')
  })

  it('promises nothing about the rest of the queue when a clip is stopped', async () => {
    const { RunningSlug } = await import('../src/components/shell/RunningSlug')
    jobs.start({ desk: 'video', kind: 'video', label: 'Wan', prompt: 'q', promptId: 'x2' })
    const html = renderToString(createElement(RunningSlug))
    expect(html).toContain('Hold to stop this clip')
    expect(html).not.toContain('Nothing else is affected')
  })
})
