import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type * as DL from '../src/lib/downloads'

/**
 * The page's side of a model fetch that outlives the page: runs rebuilt from
 * GET /api/download/status after a reload, followed there while the stream
 * that started them is gone, and ended with the server's own words. The
 * server is stood in for by a status the test sets; the clock is faked.
 */
let dl: typeof DL
let status: unknown
let posts: { url: string; body: unknown }[] = []
/** The fetch stand-in, for a test that answers differently. */
const fetchStub = () => globalThis.fetch as unknown as ReturnType<typeof vi.fn>
/** Drops the stream a startPlan opened, as a phone that loses its connection does. */
let killStream: () => void = () => {}
const json = (b: unknown, s = 200) => new Response(JSON.stringify(b), { status: s, headers: { 'content-type': 'application/json' } })

beforeEach(async () => {
  vi.useFakeTimers()
  posts = []
  status = { downloads: [], plans: [] }
  vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
    if (url === '/api/download/status') {
      if (status === null) throw new TypeError('fetch failed')
      return json(status)
    }
    if (url === '/api/download/cancel') { posts.push({ url, body: JSON.parse(String(init?.body)) }); return json({ cancelled: true }) }
    if (url === '/api/download') {
      return new Promise((_, reject) => {
        init?.signal?.addEventListener('abort', () => reject(new DOMException('a', 'AbortError')))
        killStream = () => reject(new TypeError('network error'))
      })
    }
    throw new Error('unexpected ' + url)
  }))
  vi.resetModules()
  dl = await import('../src/lib/downloads')
})
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals() })

const job = (over: Partial<DL.DownloadJob> = {}): DL.DownloadJob => ({ id: 'j1', family: 'wan', filename: 'a.safetensors', dest: 'x/a.safetensors', state: 'downloading', done: 5, total: 10, pct: 0.5, speed: 3, etaSec: 60, fileIndex: 2, fileCount: 3, error: null, ...over })
const plan = (over: Partial<DL.ServerPlan> = {}): DL.ServerPlan => ({ family: 'wan', state: 'running', current: { filename: 'a.safetensors', index: 2, count: 3, done: 5, total: 10, pct: 0.5, etaSec: 60 }, finished: ['z.safetensors'], error: null, ...over })

describe('a run taken up from the server', () => {
  it('takes up a running plan after a reload, follows it to done and announces once', async () => {
    status = { downloads: [job()], plans: [plan()] }
    const landed: string[] = []
    dl.onPlanLanded((f) => landed.push(f))
    await dl.resumeRuns()
    const r = dl.downloadRuns().get('wan')!
    expect(r.state).toBe('running')
    expect(r.followed).toBe(true)
    expect(r.current?.jobId).toBe('j1')
    expect(r.current?.index).toBe(2)
    expect(r.files.map((f) => f.filename)).toEqual(['z.safetensors', 'a.safetensors'])
    status = { downloads: [], plans: [plan({ state: 'done', current: null, finished: ['z.safetensors', 'a.safetensors', 'b'] })] }
    await vi.advanceTimersByTimeAsync(2000)
    expect(dl.downloadRuns().get('wan')!.state).toBe('done')
    await vi.advanceTimersByTimeAsync(10000)
    expect(landed).toEqual(['wan'])
  })

  it('stops a taken-up run by the job id the server lists', async () => {
    status = { downloads: [job()], plans: [plan()] }
    await dl.resumeRuns()
    await dl.cancelPlan('wan', false)
    expect(posts).toEqual([{ url: '/api/download/cancel', body: { id: 'j1', keepPartial: false } }])
    expect(dl.downloadRuns().get('wan')!.state).toBe('cancelled')
  })

  it('a lost stream is followed, not failed', async () => {
    dl.startPlan({ family: 'wan' })
    expect(dl.downloadRuns().get('wan')!.state).toBe('starting')
    // The stream drops before this page heard the plan start, while the
    // server has it running.
    status = { downloads: [job()], plans: [plan()] }
    ;killStream()
    await vi.advanceTimersByTimeAsync(0)
    expect(dl.downloadRuns().get('wan')!.followed).toBe(true)
    expect(dl.downloadRuns().get('wan')!.state).toBe('starting')
    await vi.advanceTimersByTimeAsync(2000)
    expect(dl.downloadRuns().get('wan')!.state).toBe('running')
  })

  it('a lost stream before start with nothing at the server says so', async () => {
    dl.startPlan({ family: 'wan' })
    status = { downloads: [], plans: [plan({ state: 'error', error: 'old' })] }
    ;killStream()
    await vi.advanceTimersByTimeAsync(2000)
    const r = dl.downloadRuns().get('wan')!
    expect(r.state).toBe('error')
    expect(r.error).toMatch(/before the fetch began/)
  })

  it('no answer keeps following and says out of touch', async () => {
    status = { downloads: [job()], plans: [plan()] }
    await dl.resumeRuns()
    status = null
    await vi.advanceTimersByTimeAsync(2000)
    expect(dl.downloadRuns().get('wan')!.outOfTouch).toBe(true)
    expect(dl.downloadRuns().get('wan')!.state).toBe('running')
    status = { downloads: [], plans: [] }
    await vi.advanceTimersByTimeAsync(2000)
    expect(dl.downloadRuns().get('wan')!.state).toBe('error')
    expect(dl.downloadRuns().get('wan')!.error).toBe(dl.PLAN_GONE)
  })

  it('planFor prefers running', () => {
    expect(dl.planFor([plan({ state: 'error' }), plan({ state: 'running', error: 'x' }), plan({ state: 'done' })], 'wan')?.state).toBe('running')
    expect(dl.planFor([plan({ state: 'error' }), plan({ state: 'done' })], 'wan')?.state).toBe('done')
    expect(dl.planFor([plan({ family: null })], 'wan')).toBeNull()
  })

  it('downloadFile follows a lost stream to its landing', async () => {
    // a stream that sends start then dies
    fetchStub().mockImplementation(async (url: string) => {
      if (url === '/api/download') {
        const enc = new TextEncoder()
        let ctrl!: ReadableStreamDefaultController
        const body = new ReadableStream({ start(c) { ctrl = c } })
        setTimeout(() => { ctrl.enqueue(enc.encode(`event: start\ndata: ${JSON.stringify(job({ id: 'u1', family: null }))}\n\n`)); setTimeout(() => ctrl.error(new TypeError('network')), 10) }, 0)
        return new Response(body, { headers: { 'content-type': 'text/event-stream' } })
      }
      if (url === '/api/download/status') return json(status)
      throw new Error(url)
    })
    status = { downloads: [job({ id: 'u1', family: null })], plans: [] }
    const seen: string[] = []
    const p = dl.downloadFile({ url: 'https://x', filename: 'a.safetensors', dest: 'loras' }, (e) => seen.push(e.state))
    let settled = false
    void p.then(() => { settled = true })
    await vi.advanceTimersByTimeAsync(2100)
    expect(settled).toBe(false)
    status = { downloads: [], plans: [plan({ family: null, state: 'done', current: null, finished: ['a.safetensors'] })] }
    await vi.advanceTimersByTimeAsync(2000)
    expect(settled).toBe(true)
    expect(seen.at(-1)).toBe('done')
  })
})

describe('a single file whose stream was lost', () => {
  const streamThenDie = (jobOver: Partial<DL.DownloadJob>, models: unknown) =>
    fetchStub().mockImplementation(async (url: string) => {
      if (url === '/api/download') {
        const enc = new TextEncoder()
        let ctrl!: ReadableStreamDefaultController
        const body = new ReadableStream({ start(c) { ctrl = c } })
        setTimeout(() => { ctrl.enqueue(enc.encode(`event: start\ndata: ${JSON.stringify(job(jobOver))}\n\n`)); setTimeout(() => ctrl.error(new TypeError('network')), 10) }, 0)
        return new Response(body, { headers: { 'content-type': 'text/event-stream' } })
      }
      if (url === '/api/download/status') return json(status)
      if (url === '/api/models') return json(models)
      throw new Error(url)
    })
  it('a weight file found in the models listing landed', async () => {
    streamThenDie({ id: 'u2', family: null, filename: 'l.safetensors', dest: 'Lora/l.safetensors' }, { files: [{ rel: 'Lora/l.safetensors' }] })
    status = { downloads: [], plans: [] }
    const p = dl.downloadFile({ url: 'https://x', filename: 'l.safetensors', dest: 'Lora' }, () => {})
    let ok = false
    void p.then(() => { ok = true })
    await vi.advanceTimersByTimeAsync(2100)
    expect(ok).toBe(true)
  })
  it('a weight file missing from the listing is said to be missing', async () => {
    streamThenDie({ id: 'u3', family: null, filename: 'l.safetensors', dest: 'Lora/l.safetensors' }, { files: [] })
    status = { downloads: [], plans: [] }
    const p = dl.downloadFile({ url: 'https://x', filename: 'l.safetensors', dest: 'Lora' }, () => {})
    const caught = p.catch((e: Error) => e.message)
    await vi.advanceTimersByTimeAsync(2100)
    expect(await caught).toMatch(/not in the models folder now/)
  })
  it('a file the listing never names is not claimed either way', async () => {
    streamThenDie({ id: 'u4', family: null, filename: 'model.onnx', dest: 'wd14/model.onnx' }, { files: [] })
    status = { downloads: [], plans: [] }
    const p = dl.downloadFile({ url: 'https://x', filename: 'model.onnx', dest: 'wd14' }, () => {})
    const caught = p.catch((e: Error) => e.message)
    await vi.advanceTimersByTimeAsync(2100)
    expect(await caught).toMatch(/not known here/)
  })
  it('a lost stream before the job was named says so', async () => {
    fetchStub().mockImplementation(async (url: string) => {
      if (url === '/api/download') throw new TypeError('network')
      return json(status)
    })
    const caught = dl.downloadFile({ url: 'https://x', filename: 'l.safetensors', dest: 'Lora' }, () => {}).catch((e: Error) => e.message)
    await vi.advanceTimersByTimeAsync(2100)
    expect(await caught).toMatch(/before the fetch began, and nothing is fetching/)
  })
  it('a refusal is the server sentence', async () => {
    fetchStub().mockImplementation(async (url: string) => {
      if (url === '/api/download') return json({ error: 'l.safetensors is already downloading' }, 409)
      return json(status)
    })
    const caught = dl.downloadFile({ url: 'https://x', filename: 'l.safetensors', dest: 'Lora' }, () => {}).catch((e: Error) => e.message)
    await vi.advanceTimersByTimeAsync(10)
    expect(await caught).toBe('l.safetensors is already downloading')
  })
})

describe('a plan the server refuses', () => {
  it('a refused plan fails with the server sentence and is not followed', async () => {
    fetchStub().mockImplementation(async (url: string) => {
      if (url === '/api/download') return json({ error: '2 downloads are already running; wait for one to finish' }, 429)
      return json({ downloads: [], plans: [] })
    })
    dl.startPlan({ family: 'wan' })
    await vi.advanceTimersByTimeAsync(10)
    const r = dl.downloadRuns().get('wan')!
    expect(r.state).toBe('error')
    expect(r.followed).toBe(false)
    expect(r.error).toBe('2 downloads are already running; wait for one to finish')
  })
})
