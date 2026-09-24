import { readFileSync } from 'node:fs'
import path from 'node:path'
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
  }, { timeout: 5000 })
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
    await vi.waitFor(() => expect(calls).toContain('POST /comfy/api/jobs/r4/cancel'), { timeout: 5000 })
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

  describe('a desk\'s stop that says whether it got through', () => {
    // As the queue on the server's bridge stops: a job waiting there has no
    // prompt id and sends no steps, so only the word's own answer can say it
    // did not reach the server.
    const rail = () => renderToString(createElement(notice.NoticeRail))
    const answering = (stop: (key: string) => void | Promise<boolean>) => {
      const bridge: Mirror.Bridge = {
        desk: 'images',
        kind: 'image',
        subscribe: () => () => {},
        read: () => [row({ key: 'k', status: 'submitting', label: 'A picture', stage: 'Held' })],
        stop,
        seen: new Map(),
      }
      const off = mirror(bridge)
      return { id: bridge.seen.get('k')!, off }
    }

    it('gives Stop back and says so when the word did not get through', async () => {
      const { id, off } = answering(() => Promise.resolve(false))
      await jobs.cancel(id)
      expect(jobs.get(id)).toMatchObject({ status: 'submitting', cancelling: false })
      expect(rail()).toContain('Could not stop that job')
      expect(rail()).toContain('Hold Stop again')
      off()
    })

    it('takes a word that failed outright as one that did not get through', async () => {
      const { id, off } = answering(() => Promise.reject(new Error('offline')))
      await jobs.cancel(id)
      expect(jobs.get(id)?.cancelling).toBe(false)
      expect(rail()).toContain('Could not stop that job')
      off()
    })

    it('keeps the job stopping once a word gets through, and takes the earlier refusal away', async () => {
      let answer = false
      const { id, off } = answering(() => Promise.resolve(answer))
      await jobs.cancel(id)
      expect(rail()).toContain('Could not stop that job')
      answer = true
      await jobs.cancel(id)
      expect(jobs.get(id)?.cancelling).toBe(true)
      expect(rail()).not.toContain('Could not stop that job')
      off()
    })

    it('leaves a stop that says nothing stopping, as the desks\' own stops always have', async () => {
      const { id, off } = answering(() => undefined)
      await jobs.cancel(id)
      expect(jobs.get(id)?.cancelling).toBe(true)
      expect(rail()).not.toContain('Could not stop that job')
      off()
    })

    it('says nothing of a late no for a job that ended meanwhile', async () => {
      let settle: (v: boolean) => void = () => {}
      const { id, off } = answering(() => new Promise<boolean>((res) => (settle = res)))
      const asked = jobs.cancel(id)
      jobs.fail(id, 'Stopped.', { cancelled: true })
      settle(false)
      await asked
      expect(rail()).not.toContain('Could not stop that job')
      off()
    })
  })

  describe('and what the slug says of it', () => {
    beforeEach(() => {
      vi.stubGlobal('window', { addEventListener() {}, removeEventListener() {}, location: { hash: '' }, history: {} })
    })
    const slug = async () => {
      const { RunningSlug } = await import('../src/components/shell/RunningSlug')
      return renderToString(createElement(RunningSlug))
    }

    it('times a job taken up after a reload from when its desk started it, not from the reload', async () => {
      const d = desk()
      const stop = mirror(d.bridge)
      const startedAt = Date.now() - 600_000
      d.report(row({ key: 'c1', status: 'running', promptId: 'p1', value: 3, max: 20, startedAt }))
      expect(jobs.get(d.bridge.seen.get('c1')!)?.startedAt).toBe(startedAt)
      // Ten minutes in, where the ledger's own clock said it had just begun.
      expect(await slug()).toMatch(/>10:0\d</)
      stop()
    })

    it('opens a job with the start and stage it is given, and its own only when it is given none', () => {
      const open = (over: Partial<Jobs.JobInit>) => jobs.get(jobs.start({ desk: 'video', kind: 'video', label: 'Wan', prompt: '', ...over }))!
      expect(open({ startedAt: 5 }).startedAt).toBe(5)
      const before = Date.now()
      const at = open({}).startedAt
      expect(at).toBeGreaterThanOrEqual(before)
      expect(at).toBeLessThanOrEqual(Date.now())
      expect(open({ stage: 'Held' }).stage).toBe('Held')
      expect(open({}).stage).toBe('Sending it over')
      // Sent already: the queue's word, whatever the desk called it.
      expect(open({ stage: 'Held', promptId: 'p9' }).stage).toBe('Queued')
    })

    it('says what the desk says a job not sent yet is doing, and that it is being sent only when the desk says nothing', async () => {
      const Ledger = await import('../src/components/shell/jobs')
      const d = desk()
      const stop = mirror(d.bridge)
      // A clip held behind a lost one until the reader answers.
      d.report(row({ key: 'c2', status: 'submitting', stage: 'Held' }))
      let html = await slug()
      expect(html).toContain('>Held<')
      expect(html).not.toContain(Ledger.SENDING)
      d.report(row({ key: 'c2', status: 'submitting', stage: 'Waiting for ComfyUI' }))
      expect(await slug()).toContain('>Waiting for ComfyUI<')

      d.report(row({ key: 'c2', status: 'cancelled' }), row({ key: 'c3', status: 'submitting' }))
      const id = d.bridge.seen.get('c3')!
      expect(jobs.get(id)?.stage).toBe(Ledger.SENDING)
      html = await slug()
      expect(html).toContain(`>${Ledger.SENDING}<`)
      stop()
    })

    it('leaves the stages to the ledger once the queue has the job', () => {
      const d = desk()
      const stop = mirror(d.bridge)
      d.report(row({ key: 'c4', status: 'queued', promptId: 'p4', stage: 'Sending the job' }))
      const id = d.bridge.seen.get('c4')!
      expect(jobs.get(id)?.stage).toBe('Queued')
      d.report(row({ key: 'c4', status: 'queued', promptId: 'p4', stage: 'Something of its own' }))
      expect(jobs.get(id)?.stage).toBe('Queued')
      stop()
    })

    it('shows a reel shot that waits for ComfyUI\'s memory before it is sent as doing that', async () => {
      const d = desk()
      const stop = mirror(d.bridge)
      // The reel counts a shot as queued from when its turn comes, before it has a prompt.
      d.report(row({ key: 'r1', status: 'queued', promptId: null, stage: 'Freeing memory first' }))
      expect(jobs.get(d.bridge.seen.get('r1')!)?.status).toBe('submitting')
      expect(await slug()).toContain('>Freeing memory first<')
      stop()
    })

    // A desk may hold the id it made for a send still on its way. The ledger
    // takes an id as the queue's word that it has the job, so the bridges in
    // App.tsx pass one on only once the job is past sending. App.tsx loads
    // the rooms, which a test without a DOM cannot, so a bridge that does
    // the same stands in for them, and the page is read for the line itself.
    it('is not told a job is queued while its desk is still sending it under an id of its own', async () => {
      const desks = [{ id: 'k5', status: 'submitting' as const, promptId: 'minted-before-the-send', stage: 'Sending the job' }]
      const heard = new Set<() => void>()
      const bridge: Mirror.Bridge = {
        desk: 'video',
        kind: 'video',
        subscribe: (fn) => {
          heard.add(fn)
          return () => void heard.delete(fn)
        },
        read: () =>
          desks.map((job) => row({ key: job.id, status: job.status, promptId: job.status === 'submitting' ? null : job.promptId, stage: job.stage })),
        seen: new Map(),
      }
      const stop = mirror(bridge)
      expect(jobs.get(bridge.seen.get('k5')!)).toMatchObject({ status: 'submitting', promptId: null, stage: 'Sending the job' })
      expect(await slug()).toContain('>Sending the job<')
      stop()

      const app = readFileSync(path.resolve(import.meta.dirname, '..', 'src', 'App.tsx'), 'utf8')
      expect(app.split("promptId: job.status === 'submitting' ? null : job.promptId,")).toHaveLength(3)
    })
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
    await vi.waitFor(() => expect(jobs.snapshot().server.known).toBe(true), { timeout: 5000 })
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

// ---------------------------------------------------------------------------
// Round four: a ComfyUI that restarts, sends that lose their answer, jobs
// followed without a socket, and what the ledger shows of a two-pass clip.
// ---------------------------------------------------------------------------

const empty502 = () => new Response('', { status: 502, headers: { 'content-type': 'text/plain' } })
/** Answers `url` from `answers` in turn, the last one for good; counts the asks in `asked`. */
function inTurn(url: string, answers: (() => Response)[], asked = { n: 0 }) {
  routes.unshift((u) => {
    if (u !== url) return undefined
    const answer = answers[Math.min(asked.n, answers.length - 1)]!
    asked.n += 1
    return answer()
  })
  return asked
}
const V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/

describe('a send that ComfyUI does not answer', () => {
  it('reads an empty 502 as ComfyUI not answering, not as a refusal', async () => {
    routes.unshift((url, init) => (url === '/comfy/prompt' && init?.method === 'POST' ? empty502() : undefined))
    const err = (await comfy.submit(graph).catch((e: unknown) => e)) as Comfy.ComfyError
    expect(err.message).toBe(comfy.COMFY_NOT_ANSWERING)
    expect(err.unreachable).toBe(true)
    expect(faults.faultTitle(faults.faultOf(err))).toBe('ComfyUI is not answering')
  })

  it('reads a page of HTML the same way, whatever its status', async () => {
    routes.unshift((url, init) =>
      url === '/comfy/prompt' && init?.method === 'POST'
        ? new Response('<html><body>Bad Gateway</body></html>', { status: 500, headers: { 'content-type': 'text/html' } })
        : undefined,
    )
    const err = (await comfy.submit(graph).catch((e: unknown) => e)) as Comfy.ComfyError
    expect(err.message).toBe(comfy.COMFY_NOT_ANSWERING)
    expect(err.unreachable).toBe(true)
  })

  it('still reads ComfyUI\'s own refusal as one', async () => {
    routes.unshift((url, init) =>
      url === '/comfy/prompt' && init?.method === 'POST'
        ? json({ error: { message: 'Prompt outputs failed validation' }, node_errors: { '3': { errors: [{ message: 'Value out of range' }], class_type: 'KSampler' } } }, 400)
        : undefined,
    )
    const err = (await comfy.submit(graph).catch((e: unknown) => e)) as Comfy.ComfyError
    expect(err.unreachable).toBe(false)
    expect(err.nodeType).toBe('KSampler')
    expect(faults.faultTitle(faults.faultOf(err))).toBe('That job was rejected')
  })
})

describe('the prompt id a send carries', () => {
  it('is a lowercase version 4 UUID, even where crypto.randomUUID is missing (plain http)', async () => {
    const node = globalThis.crypto
    vi.stubGlobal('crypto', { getRandomValues: (b: Uint8Array<ArrayBuffer>) => node.getRandomValues(b) })
    const ids = Array.from({ length: 50 }, () => comfy.newPromptId())
    for (const id of ids) expect(id).toMatch(V4)
    expect(new Set(ids).size).toBe(50)

    let sent: unknown = null
    routes.unshift((url, init) => {
      if (url !== '/comfy/prompt' || init?.method !== 'POST') return undefined
      sent = JSON.parse(String(init.body)).prompt_id
      return json({ prompt_id: sent })
    })
    expect(await comfy.submit(graph)).toBe(sent)
    expect(sent).toMatch(V4)
  })
})

describe('a send under an id its desk made', () => {
  /** ComfyUI takes the id it is sent, as it does; every body sent is kept. */
  const echo = () => {
    const bodies: { prompt_id?: string }[] = []
    routes.unshift((url, init) => {
      if (url !== '/comfy/prompt' || init?.method !== 'POST') return undefined
      const body = JSON.parse(String(init.body)) as { prompt_id?: string }
      bodies.push(body)
      return json({ prompt_id: body.prompt_id })
    })
    return bodies
  }

  it('goes under that id, and under a fresh one of its own when it is given none', async () => {
    const bodies = echo()
    const id = comfy.newPromptId()
    expect(await comfy.submit(graph, { promptId: id })).toBe(id)
    expect(bodies[0]?.prompt_id).toBe(id)
    const own = await comfy.submit(graph)
    expect(own).toMatch(V4)
    expect(own).not.toBe(id)
    expect(bodies[1]?.prompt_id).toBe(own)
  })

  it('is passed on by run(), whose queued event names it', async () => {
    const bodies = echo()
    const id = comfy.newPromptId()
    const queuedAs: string[] = []
    const result = comfy
      .run(graph, (e) => {
        if (e.phase === 'queued') queuedAs.push(e.promptId)
      }, { promptId: id })
      .catch((e: unknown) => e)
    await vi.waitFor(() => expect(queuedAs).toEqual([id]), { timeout: 5000 })
    expect(bodies.map((b) => b.prompt_id)).toEqual([id])
    // Settled, so nothing is left watching it after the test.
    FakeSocket.last!.deliver('execution_interrupted', { prompt_id: id, node_id: '8', node_type: 'VAEDecode' })
    expect((await result as Comfy.ComfyError).cancelled).toBe(true)
  })
})

describe('a send whose answer was lost on the way back', () => {
  let posted = ''
  beforeEach(() => {
    vi.useFakeTimers()
    posted = ''
    routes.unshift((url, init) => {
      if (url !== '/comfy/prompt' || init?.method !== 'POST') return undefined
      posted = JSON.parse(String(init.body)).prompt_id
      throw new TypeError('Failed to fetch')
    })
  })
  const outcome = async () => {
    const out = comfy.submit(graph).then(
      (id) => ({ id, err: null }),
      (err: Comfy.ComfyError) => ({ id: null, err }),
    )
    await vi.advanceTimersByTimeAsync(10_000)
    return out
  }

  it('follows the job when ComfyUI has it after all', async () => {
    routes.unshift((url) => (posted && url === `/comfy/api/jobs/${posted}` ? json({ id: posted, status: 'pending' }) : undefined))
    const r = await outcome()
    expect(r.err).toBeNull()
    expect(r.id).toBe(posted)
  })

  it('says nothing was queued when ComfyUI says twice it has no such job', async () => {
    routes.unshift((url) => (posted && url === `/comfy/api/jobs/${posted}` ? json({ error: 'not found' }, 404) : undefined))
    const r = await outcome()
    expect(r.err?.message).toMatch(/did not receive it/)
    expect(r.err?.unreachable).toBe(true)
  })

  it('says it could not tell when ComfyUI cannot be asked either', async () => {
    routes.unshift((url) => {
      if (posted && url === `/comfy/api/jobs/${posted}`) throw new TypeError('Failed to fetch')
      return undefined
    })
    const r = await outcome()
    expect(r.err?.message).toMatch(/could not be asked/)
    expect(r.err?.unreachable).toBe(true)
  })
})

describe('a stop sent while ComfyUI is restarting', () => {
  beforeEach(() => {
    routes.unshift((url, init) => (/\/cancel$/.test(url) && init?.method === 'POST' ? empty502() : undefined))
  })

  it('is not ComfyUI declining to stop it', async () => {
    const err = (await comfy.cancelJob('x').catch((e: unknown) => e)) as Comfy.ComfyError
    expect(err.unreachable).toBe(true)
  })

  it('says a restart ends the job, and to hold Stop again if it comes back', async () => {
    const id = jobs.start({ desk: 'video', kind: 'video', label: 'Wan', prompt: '', promptId: 'r9' })
    await jobs.cancel(id)
    const html = renderToString(createElement(notice.NoticeRail))
    expect(html).toContain('Could not stop that job')
    expect(html).toContain('hold Stop again')
    expect(html).not.toContain('It may still be running')
  })
})

describe('following a prompt without its socket', () => {
  const record = (id: string, extra: Record<string, unknown>) => ({ [id]: { prompt: [0, id, graph, {}, []], outputs: {}, ...extra } })
  const follow = (over: { signal?: AbortSignal; onState?: (s: 'queued' | 'running') => void } = {}) =>
    comfy.followPrompt('f1', { intervalMs: 5, ...over })

  it('follows it through the queue to its files, marking those ComfyUI took from its cache', async () => {
    inTurn('/comfy/api/jobs/f1', [
      () => json({ id: 'f1', status: 'pending' }),
      () => json({ id: 'f1', status: 'in_progress' }),
      () => json({ id: 'f1', status: 'completed' }),
    ])
    routes.unshift((url) =>
      url === '/comfy/history/f1'
        ? json(
            record('f1', {
              outputs: {
                '9': { images: [{ filename: 'kept.png', subfolder: '', type: 'output' }] },
                '12': { images: [{ filename: 'new.webm', subfolder: 'video', type: 'output' }], animated: [true] },
              },
              status: { status_str: 'success', completed: true, messages: [['execution_cached', { nodes: ['9'], prompt_id: 'f1' }]] },
            }),
          )
        : undefined,
    )
    const heard: string[] = []
    const r = await follow({ onState: (s) => heard.push(s) })
    expect(r.status).toBe('done')
    const files = (r as { files: Comfy.OutputFile[] }).files
    expect(files.find((f) => f.filename === 'kept.png')?.cached).toBe(true)
    const fresh = files.find((f) => f.filename === 'new.webm')!
    expect(fresh.kind).toBe('video')
    expect('cached' in fresh).toBe(false)
    expect(heard).toEqual(['queued', 'running'])
  })

  it('calls it lost only when ComfyUI answers twice that it has no such job, not when it does not answer', async () => {
    const asked = inTurn('/comfy/api/jobs/f1', [empty502, () => json({ error: 'nf' }, 404), () => json({ error: 'nf' }, 404)])
    expect(await follow()).toEqual({ status: 'lost' })
    expect(asked.n).toBe(3)
  })

  it('names the node a failed job broke in', async () => {
    inTurn('/comfy/api/jobs/f1', [() => json({ id: 'f1', status: 'failed' })])
    routes.unshift((url) =>
      url === '/comfy/history/f1'
        ? json(
            record('f1', {
              status: {
                status_str: 'error',
                messages: [['execution_error', { prompt_id: 'f1', node_id: '8', node_type: 'VAEDecode', exception_message: 'boom' }]],
              },
            }),
          )
        : undefined,
    )
    expect(await follow()).toEqual({ status: 'error', message: 'boom', node: '8', nodeType: 'VAEDecode' })
  })

  it('says a stopped job was stopped', async () => {
    inTurn('/comfy/api/jobs/f1', [() => json({ id: 'f1', status: 'cancelled' })])
    expect(await follow()).toEqual({ status: 'cancelled' })
  })

  it('names the Archive\'s control when an ended job left no record', async () => {
    inTurn('/comfy/api/jobs/f1', [() => json({ id: 'f1', status: 'completed' })])
    routes.unshift((url) => (url === '/comfy/history/f1' ? json({}) : undefined))
    const r = (await follow()) as { status: string; message: string }
    expect(r.status).toBe('error')
    expect(r.message).toContain('Look for files with no record')
  })

  it('stops following when told to', async () => {
    inTurn('/comfy/api/jobs/f1', [() => json({ id: 'f1', status: 'in_progress' })])
    const stop = new AbortController()
    const r = follow({ signal: stop.signal })
    stop.abort()
    await expect(r).rejects.toBeDefined()
  })
})

describe('a re-run ComfyUI answers from its cache', () => {
  it('marks the files of the nodes it named as cached, and no others', async () => {
    const { id, result } = await started()
    const sock = FakeSocket.last!
    sock.deliver('execution_cached', { prompt_id: id, nodes: ['9'] })
    sock.deliver('executed', { prompt_id: id, node: '9', output: { images: [{ filename: 'old.png', subfolder: '', type: 'output' }] } })
    sock.deliver('executed', { prompt_id: id, node: '10', output: { images: [{ filename: 'new.png', subfolder: '', type: 'output' }] } })
    sock.deliver('executing', { prompt_id: id, node: null })
    const files = (await result) as Comfy.OutputFile[]
    expect(files.find((f) => f.filename === 'old.png')?.cached).toBe(true)
    expect('cached' in files.find((f) => f.filename === 'new.png')!).toBe(false)
  })
})

describe('the catalogue of nodes', () => {
  const info = () => calls.filter((c) => c === 'GET /comfy/object_info').length

  it('is fetched once for everyone who asks at the same time', async () => {
    routes.unshift((url) => (url === '/comfy/object_info' ? json({ KSampler: {} }) : undefined))
    const [a, b] = await Promise.all([comfy.objectInfo(), comfy.objectInfo()])
    expect(info()).toBe(1)
    expect(a).toBe(b)
    await comfy.objectInfo()
    expect(info()).toBe(1)
    comfy.forgetObjectInfo()
    await comfy.objectInfo()
    expect(info()).toBe(2)
  })

  it('is asked for again after a fetch that failed', async () => {
    let fail = true
    routes.unshift((url) => {
      if (url !== '/comfy/object_info') return undefined
      if (fail) throw new TypeError('Failed to fetch')
      return json({ KSampler: {} })
    })
    await expect(comfy.objectInfo()).rejects.toBeDefined()
    fail = false
    expect(await comfy.objectInfo()).toEqual({ KSampler: {} })
    expect(info()).toBe(2)
  })
})

describe('a read ComfyUI never answers', () => {
  beforeEach(() => {
    routes.unshift((url, init) => {
      if (url !== '/comfy/api/jobs/slow') return undefined
      // Stands in for a ComfyUI that keeps the socket open and says nothing.
      return new Promise<Response>((_, reject) => {
        init?.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')))
      }) as unknown as Response
    })
  })

  it('gives up after ten seconds, so the polls cannot fill the browser\'s connections', async () => {
    vi.useFakeTimers()
    let settled = false
    const read = comfy.getJob('slow').catch(() => {
      settled = true
    })
    await vi.advanceTimersByTimeAsync(9000)
    expect(settled).toBe(false)
    await vi.advanceTimersByTimeAsync(1500)
    expect(settled).toBe(true)
    await read
  })

  it('gives up at once when its caller does', async () => {
    const stop = new AbortController()
    const read = comfy.getJob('slow', { signal: stop.signal })
    stop.abort()
    await expect(read).rejects.toBeDefined()
  })
})

describe('a job that may have finished without its result', () => {
  it('asks the reader to look before running it again', () => {
    const f = faults.faultOf(new comfy.LostJob('Ended without its result.', 'p1', { mayExist: true }))
    expect(faults.faultBody(f)).toContain('look there before you run it again')
    const plain = faults.faultOf(new comfy.LostJob('We lost track of this job.', 'p2'))
    expect(faults.faultBody(plain)).toContain('so you can try again')
  })
})

describe('a run whose socket drops, or whose ComfyUI restarts', () => {
  const clip = { images: [{ filename: 'wan_00001_.webm', subfolder: 'video', type: 'output' }], animated: [true] }
  const frame = { images: [{ filename: 'wan_00001_.png', subfolder: 'video', type: 'output' }] }

  it('resolves with every file, a clip as a clip and a picture as a picture', async () => {
    const { id, result } = await started()
    const sock = FakeSocket.last!
    sock.deliver('executed', { prompt_id: id, node: '9', output: clip })
    sock.deliver('executed', { prompt_id: id, node: '10', output: frame })
    sock.deliver('execution_success', { prompt_id: id })
    const files = (await result) as Comfy.OutputFile[]
    expect(files.map((f) => [f.filename, f.kind])).toEqual([
      ['wan_00001_.webm', 'video'],
      ['wan_00001_.png', 'image'],
    ])
  })

  it('reads every file from /history once the socket missed some, and not before the record is written', async () => {
    let written = false
    routes.unshift((url) =>
      url === '/comfy/history/p1'
        ? json(
            written
              ? { p1: { prompt: [0, 'p1', graph, {}, []], outputs: { '9': clip, '10': frame }, status: { status_str: 'success', completed: true, messages: [] } } }
              : {},
          )
        : undefined,
    )
    const { id, result } = await started()
    const first = FakeSocket.last!
    first.deliver('executed', { prompt_id: id, node: '9', output: clip })
    first.close()
    await vi.waitFor(() => {
      if (FakeSocket.last === first || FakeSocket.last?.readyState !== 1) throw new Error('not back yet')
    }, { timeout: 5000 })
    let settled = false
    void result.then(() => {
      settled = true
    })
    // ComfyUI sends this before it writes the record, so it cannot settle yet.
    FakeSocket.last!.deliver('execution_success', { prompt_id: id })
    written = true
    await tick(20)
    expect(settled).toBe(false)
    FakeSocket.last!.deliver('executing', { prompt_id: id, node: null })
    const files = (await result) as Comfy.OutputFile[]
    expect(files.map((f) => f.filename)).toEqual(['wan_00001_.webm', 'wan_00001_.png'])
  })

  describe('a ComfyUI that has forgotten the job', () => {
    // The watch's interval is made inside run(), so the clock is faked first.
    beforeEach(() => {
      vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'Date'] })
      routes.unshift((url) => (url.startsWith('/comfy/api/jobs?') ? json({ jobs: [], pagination: { offset: 0, limit: 100, total: 0, has_more: false } }) : undefined))
    })
    const outcome = (result: Promise<unknown>) => {
      let got: unknown = null
      void result.then((r) => {
        got = r
      })
      return () => got
    }

    it('gives the job up as lost once it has no record of it', async () => {
      routes.unshift((url) => (url === '/comfy/api/jobs/p1' ? json({ error: 'nf' }, 404) : undefined))
      const { result } = await started()
      FakeSocket.last!.close()
      const got = outcome(result)
      await vi.advanceTimersByTimeAsync(40_000)
      const err = got() as Comfy.LostJob
      expect(err?.name).toBe('LostJob')
      expect(err.message).toMatch(/lost track/)
      expect(err.mayExist).toBe(false)
    })

    it('never gives it up while ComfyUI does not answer about it', async () => {
      routes.unshift((url) => (url === '/comfy/api/jobs/p1' ? empty502() : undefined))
      const { result } = await started()
      const got = outcome(result)
      await vi.advanceTimersByTimeAsync(60_000)
      expect(got()).toBeNull()
    })

    it('waits for the record of a job it says has ended, then names the Archive\'s control', async () => {
      routes.unshift((url) => (url === '/comfy/api/jobs/p1' ? json({ id: 'p1', status: 'completed' }) : undefined))
      routes.unshift((url) => (url === '/comfy/history/p1' ? json({}) : undefined))
      const { result } = await started()
      const got = outcome(result)
      // Two looks at the record by 40 s, each after two ticks without the job.
      await vi.advanceTimersByTimeAsync(40_000)
      expect(got()).toBeNull()
      await vi.advanceTimersByTimeAsync(10_000)
      const err = got() as Comfy.LostJob
      expect(err?.name).toBe('LostJob')
      expect(err.message).toMatch(/never sent the result/)
      expect(err.message).toContain('Look for files with no record')
      expect(err.mayExist).toBe(true)
    })
  })
})

describe('a clip that samples in two passes', () => {
  type J = Parameters<typeof Jobs.progressOf>[0]
  const at = (value: number, index: number): J => ({ value, max: 10, pass: { index, count: 2 } })

  it('moves the rule through both passes, each counted from one', async () => {
    const { progressOf } = await import('../src/components/shell/jobs')
    expect(progressOf(at(5, 2))).toBe(0.75)
    expect(progressOf(at(5, 1))).toBe(0.25)
  })

  it('times what is left from this pass\'s own pace, and adds the pass to come', async () => {
    const { remainingOf } = await import('../src/components/shell/jobs')
    const job = (value: number, index: number) => ({
      status: 'running' as const,
      value,
      max: 10,
      pass: { index, count: 2 },
      passAt: 100_000,
      passFrom: 1,
      startedAt: 0,
    })
    // Four steps in 40 s: 10 s a step.
    expect(remainingOf(job(5, 2), 140_000)).toBe(50_000)
    expect(remainingOf(job(5, 1), 140_000)).toBe(150_000)
    expect(remainingOf(job(2, 2), 140_000)).toBeNull()
  })

  it('reaches the ledger from a desk that reports its pass', () => {
    const rows: Mirror.Reported[] = []
    const heard = new Set<() => void>()
    const bridge: Mirror.Bridge = {
      desk: 'video',
      kind: 'video',
      subscribe: (fn) => {
        heard.add(fn)
        return () => void heard.delete(fn)
      },
      read: () => rows,
      stop: () => {},
      seen: new Map(),
    }
    const report = (value: number, max: number, index: number) => {
      rows.splice(0, rows.length, {
        key: 'c1', status: 'running', promptId: 'p-c1', label: 'Wan', prompt: '', value, max, pass: { index, count: 2 }, entryId: null, error: null,
      })
      for (const fn of heard) fn()
    }
    const stop = mirror(bridge)
    report(10, 10, 1)
    const id = bridge.seen.get('c1')!
    expect(jobs.get(id)?.pass).toEqual({ index: 1, count: 2 })
    // The same step count, in the second pass: only the pass moved.
    report(10, 10, 2)
    expect(jobs.get(id)?.pass?.index).toBe(2)
    expect(jobs.get(id)?.stage).toBe('Drawing, pass 2 of 2')
    report(0, 1, 2)
    expect(jobs.get(id)?.stage).toBe('Working')
    stop()
  })
})

describe('the slug\'s clock', () => {
  it('ticks while something is live or still news, and stops after', async () => {
    const { needsClock, newsUntil, RECENT_MS } = await import('../src/components/shell/jobs')
    const finished = (ago: number) => ({ active: [], recent: [{ finishedAt: 1_000_000 - ago }] }) as unknown as Parameters<typeof needsClock>[0]
    expect(needsClock({ active: [{}], recent: [] } as unknown as Parameters<typeof needsClock>[0], 1_000_000)).toBe(true)
    expect(needsClock(finished(RECENT_MS - 1), 1_000_000)).toBe(true)
    expect(needsClock(finished(RECENT_MS + 1), 1_000_000)).toBe(false)
    expect(needsClock({ active: [], recent: [] }, 1_000_000)).toBe(false)
    expect(newsUntil(finished(0))).toBe(1_000_000 + RECENT_MS)
    expect(newsUntil({ active: [], recent: [] })).toBeNull()
  })
})

describe('the card busy with work this page is not following', () => {
  beforeEach(() => {
    vi.stubGlobal('window', { addEventListener() {}, removeEventListener() {}, location: { hash: '' }, history: {} })
  })

  it('does not call a job sent before a reload someone else\'s', async () => {
    const { RunningSlug } = await import('../src/components/shell/RunningSlug')
    const page = (total: number) => json({ jobs: [], pagination: { offset: 0, limit: 1, total, has_more: false } })
    routes.unshift((url) => {
      if (url === '/comfy/api/jobs?status=in_progress&limit=1') return page(1)
      if (url === '/comfy/api/jobs?status=pending&limit=1') return page(0)
      return undefined
    })
    const stop = jobs.subscribe(() => {})
    await vi.waitFor(() => expect(jobs.snapshot().server.known).toBe(true), { timeout: 5000 })
    const html = renderToString(createElement(RunningSlug))
    expect(html).toContain('this page is not following')
    expect(html).not.toContain('outside SwitchGen')
    stop()
  })
})

describe('the section bar on a phone', () => {
  beforeEach(() => {
    vi.stubGlobal('window', { addEventListener() {}, removeEventListener() {}, location: { hash: '' }, history: {}, matchMedia: () => ({ matches: false, addEventListener() {}, removeEventListener() {} }) })
    vi.stubGlobal('document', { documentElement: { style: { setProperty() {}, removeProperty() {} } }, addEventListener() {}, removeEventListener() {} })
    vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} })
  })
  const middle = (html: string) => /<nav[\s\S]*?<\/nav><div class="([^"]*)"/.exec(html)?.[1] ?? ''

  it('gives a running job a row of its own, and none when the card is idle', async () => {
    const { SectionBar } = await import('../src/components/shell/SectionBar')
    expect(middle(renderToString(createElement(SectionBar)))).not.toContain('max-lg:order-last max-lg:basis-full')
    const id = jobs.start({ desk: 'video', kind: 'video', label: 'Wan', prompt: 'a tram', promptId: 'sb1' })
    jobs.apply(id, { phase: 'running', node: null, value: 3, max: 20 })
    expect(middle(renderToString(createElement(SectionBar)))).toContain('max-lg:order-last max-lg:basis-full')
  })

  // On that row a job's line and Hold to stop go to either end. The busy line
  // is one sentence in four pieces, and spread the same way they stood apart
  // across the row: the mark at one edge, a lone dot mid-row, the count at
  // the other.
  const busyLine = (html: string) => /<p class="([^"]*)"><span class="sg-mark sg-mark-live"[^>]*><\/span><span>The card is busy/.exec(html)?.[1]

  it('keeps the busy line together, and spreads only a job and its stop', async () => {
    const { SectionBar } = await import('../src/components/shell/SectionBar')
    const page = (total: number) => json({ jobs: [], pagination: { offset: 0, limit: 1, total, has_more: false } })
    routes.unshift((url) => {
      if (url === '/comfy/api/jobs?status=in_progress&limit=1') return page(1)
      if (url === '/comfy/api/jobs?status=pending&limit=1') return page(0)
      return undefined
    })
    const stop = jobs.subscribe(() => {})
    await vi.waitFor(() => expect(jobs.snapshot().server.foreign).toBe(1), { timeout: 5000 })
    const busy = busyLine(renderToString(createElement(SectionBar)))
    expect(busy).toBeDefined()
    expect(busy).toContain('max-lg:w-full')
    expect(busy).not.toContain('justify-between')

    const id = jobs.start({ desk: 'video', kind: 'video', label: 'Wan', prompt: 'a tram', promptId: 'sb2' })
    jobs.apply(id, { phase: 'running', node: null, value: 3, max: 20 })
    const html = renderToString(createElement(SectionBar))
    expect(busyLine(html)).toBeUndefined()
    const slug = /<div class="(flex min-w-0 items-center gap-3[^"]*)">/.exec(html)?.[1]
    expect(slug).toContain('max-lg:justify-between')
    stop()
  })

  // At 360 to 412 px "The card is busy" broke over two lines, and the count
  // was cut short before "following", the word it turns on. Neither phrase
  // breaks inside itself now, and the count takes a line of its own instead.
  it('keeps each phrase of the busy line whole, and lets the count go to a line of its own', async () => {
    const { SectionBar } = await import('../src/components/shell/SectionBar')
    const page = (total: number) => json({ jobs: [], pagination: { offset: 0, limit: 1, total, has_more: false } })
    routes.unshift((url) => {
      if (url === '/comfy/api/jobs?status=in_progress&limit=1') return page(1)
      if (url === '/comfy/api/jobs?status=pending&limit=1') return page(0)
      return undefined
    })
    const stop = jobs.subscribe(() => {})
    await vi.waitFor(() => expect(jobs.snapshot().server.foreign).toBe(1), { timeout: 5000 })
    const busy = busyLine(renderToString(createElement(SectionBar)))!.split(' ')
    expect(busy).toContain('whitespace-nowrap')
    expect(busy).toContain('max-lg:flex-wrap')
    stop()
  })
})
