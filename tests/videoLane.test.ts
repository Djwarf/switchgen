import { readFileSync } from 'node:fs'
import path from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type * as VideoDesk from '../src/routes/Video'

/**
 * The Video desk's memory-release lane, kept for the tab. A heavy clip waits
 * in the lane until ComfyUI's queue is empty, and nothing on the server knows
 * about it until then, so the lane is written to the tab's session storage
 * and the next page in the tab takes it up. ComfyUI is stood in for by a
 * route table: /comfy/queue says busy or idle, and nothing is ever sent
 * anywhere real.
 */
const LANE = 'switchgen.videolane.v1'
const LEFT = 'switchgen.videolane.v1.left'
const SENT = 'switchgen.videosent.v1'
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

let calls: string[] = []
/** What ComfyUI says of a job asked about by id; null answers that it has none. */
let jobStatus: string | null = 'completed'
/** The same, for one prompt id, over jobStatus. */
const statusOf = new Map<string, string | null>()
/** ComfyUI's /history record of a prompt, by id. */
let historyOf: (id: string) => unknown = () => ({})
/** How many jobs ComfyUI reports running. */
let busy = 1
let prompts = 0
/** Whether the saved lane was still there when the prompt went. */
let laneAtPrompt: boolean | null = null
/** ComfyUI answers a send under the id it was sent, as it does, rather than as p1, p2 and on. */
let echo = false
/** What the last send carried, and the clips the tab held as sent at that moment. */
let promptBody: { prompt_id: string } | null = null
let sentAtPrompt: { released: boolean; jobs: { id: string; promptId: string; sending?: boolean }[] } | null = null
/** ComfyUI is away: every ask gets the empty 502 the proxy in front of it answers with. */
let down = false
/** Whether ComfyUI says it stopped a job it is asked to cancel. */
let cancels = true
const session = new Map<string, string>()
const listeners = new Map<string, ((e: unknown) => void)[]>()
const reload = vi.fn()
let v: typeof VideoDesk | null = null

const clip = (id: string, prompt: string) => ({
  id,
  startedAt: 1000,
  familyLabel: 'Wan 2.2 T2V A14B',
  modelLabel: '',
  composition: { desk: 'video', mode: 't2v', familyId: 'wan22-14b-t2v', prompt, seed: 1, length: 81, width: 832, height: 480 },
  graph: { '8': { class_type: 'VAEDecode', inputs: {} } },
})
const saved = () => JSON.parse(session.get(LANE) ?? 'null')
/** Room for a slow runner: vi.waitFor gives up after one second by default. */
const WAIT = { timeout: 5000, interval: 10 }
/** What was sent to ComfyUI's queue or its memory release. */
const posts = () => calls.filter((c) => c === 'POST /comfy/free' || c === 'POST /comfy/prompt')
const fire = (type: string, e: unknown = {}) => {
  for (const fn of listeners.get(type) ?? []) fn(e)
}
/** Leave a lane in the tab as a page before this one did, then load the desk as this page does. */
async function load(lane: unknown): Promise<typeof VideoDesk> {
  session.set(LANE, JSON.stringify(lane))
  v = await import('../src/routes/Video')
  return v
}

beforeEach(() => {
  calls = []
  jobStatus = 'completed'
  statusOf.clear()
  historyOf = () => ({})
  FakeSocket.last = null
  busy = 1
  prompts = 0
  laneAtPrompt = null
  echo = false
  promptBody = null
  sentAtPrompt = null
  down = false
  cancels = true
  session.clear()
  listeners.clear()
  reload.mockReset()
  vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
    calls.push(`${init?.method ?? 'GET'} ${url}`)
    if (down) return new Response('', { status: 502, headers: { 'content-type': 'text/plain' } })
    if (url === '/comfy/queue') return json({ queue_running: Array(busy).fill([]), queue_pending: [] })
    if (url === '/comfy/free') return json({})
    if (url === '/comfy/prompt') {
      laneAtPrompt = session.has(LANE)
      promptBody = JSON.parse(String(init?.body)) as { prompt_id: string }
      sentAtPrompt = JSON.parse(session.get(SENT) ?? 'null') as typeof sentAtPrompt
      return json({ prompt_id: echo ? promptBody.prompt_id : `p${++prompts}` })
    }
    if (/^\/comfy\/api\/jobs\/[^/]+\/cancel$/.test(url)) return json({ cancelled: cancels })
    const job = /^\/comfy\/api\/jobs\/([^/?]+)$/.exec(url)
    if (job) {
      const status = statusOf.has(job[1]!) ? statusOf.get(job[1]!)! : jobStatus
      return status ? json({ id: job[1], status }) : json({ error: 'not found' }, 404)
    }
    if (url.startsWith('/comfy/api/jobs?')) return json({ jobs: [], pagination: { offset: 0, limit: 100, total: 0, has_more: false } })
    const past = /^\/comfy\/history\/([^/?]+)$/.exec(url)
    if (past) return json(historyOf(decodeURIComponent(past[1]!)))
    return json({ error: 'no route' }, 599)
  })
  vi.stubGlobal('WebSocket', FakeSocket)
  vi.stubGlobal('location', { protocol: 'http:', host: 'harness', hash: '' })
  vi.stubGlobal('sessionStorage', {
    getItem: (k: string) => session.get(k) ?? null,
    setItem: (k: string, value: string) => void session.set(k, String(value)),
    removeItem: (k: string) => void session.delete(k),
  })
  vi.stubGlobal('window', {
    addEventListener: (type: string, fn: (e: unknown) => void) => listeners.set(type, [...(listeners.get(type) ?? []), fn]),
    removeEventListener: (type: string, fn: (e: unknown) => void) =>
      listeners.set(type, (listeners.get(type) ?? []).filter((f) => f !== fn)),
    location: { reload },
  })
  vi.resetModules()
})

afterEach(async () => {
  // A loaded desk keeps polling for as long as a clip waits, and it reaches
  // the next test's stand-ins, so every clip it holds is stopped first.
  for (const job of v?.videoJobs.snapshot() ?? []) v?.stopVideoJob(job.id)
  // On whichever clock the test ran: a faked one never reaches a real sleep.
  if (vi.isFakeTimers()) await vi.advanceTimersByTimeAsync(10)
  else await new Promise((resolve) => setTimeout(resolve, 10))
  v = null
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe('a lane the page before handed on', () => {
  it('is taken up in order, kept for this tab as this page\'s own, and holds the page while it waits', async () => {
    const d = await load({ writer: 'old', released: true, clips: [clip('a', 'first'), clip('b', 'second')] })
    const js = d.videoJobs.snapshot()
    expect(js.map((j) => j.id).sort()).toEqual(['a', 'b'])
    expect(js.every((j) => j.status === 'submitting' && j.startedAt === 1000)).toBe(true)
    expect(saved().clips.map((c: { id: string }) => c.id)).toEqual(['a', 'b'])
    expect(saved().writer).not.toBe('old')
    expect(saved().released).toBe(false)
    expect(listeners.get('beforeunload')).toHaveLength(1)
    expect(d.videoJobs.waitingCount()).toBe(2)
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(posts()).toEqual([])
  })

  it('lets Stop from the section bar reach a clip still waiting, and takes it out of the saved lane', async () => {
    const d = await load({ writer: 'old', released: true, clips: [clip('a', 'first'), clip('b', 'second')] })
    d.stopVideoJob('a')
    await vi.waitFor(() => expect(d.videoJobs.snapshot().find((j) => j.id === 'a')?.status).toBe('cancelled'), WAIT)
    expect(saved().clips.map((c: { id: string }) => c.id)).toEqual(['b'])
    d.stopVideoJob('b')
    await vi.waitFor(() => expect(d.videoJobs.snapshot().find((j) => j.id === 'b')?.status).toBe('cancelled'), WAIT)
    expect(session.has(LANE)).toBe(false)
    expect(listeners.get('beforeunload')).toHaveLength(0)
    expect(posts()).toEqual([])
  })

  it('says it is going on pagehide, and leaves the saved lane before its prompt goes', async () => {
    const d = await load({ writer: 'old', released: true, clips: [clip('a', 'first')] })
    fire('pagehide')
    expect(saved().released).toBe(true)
    fire('pageshow', { persisted: true })
    expect(saved().released).toBe(false)
    busy = 0
    await vi.waitFor(() => expect(calls).toContain('POST /comfy/prompt'), { timeout: 5000 })
    expect(calls.indexOf('POST /comfy/free')).toBeLessThan(calls.indexOf('POST /comfy/prompt'))
    // Gone before the prompt went: a page that went away in between would
    // otherwise send it a second time.
    expect(laneAtPrompt).toBe(false)
    expect(d.videoJobs.waitingCount()).toBe(0)
    // The queue is read every two seconds on real time, so this waits one read.
  }, 15_000)

  it('defers to a later page that took the lane while this one was in the browser\'s cache', async () => {
    const d = await load({ writer: 'old', released: true, clips: [clip('a', 'first')] })
    fire('pagehide')
    session.set(LANE, JSON.stringify({ writer: 'later', released: true, clips: [clip('a', 'first')] }))
    fire('pageshow', { persisted: true })
    expect(reload).toHaveBeenCalled()
    expect(d.videoJobs.waitingCount()).toBe(0)
    expect(saved().writer).toBe('later')
    expect(posts()).toEqual([])
  })
})

describe('a lane the page before did not hand on', () => {
  // A copied tab, or a page that crashed: the page that wrote it may still be
  // sending it, so it is never sent on a guess, and never dropped unsaid.
  it('is not sent, but kept for the reader to send from here', async () => {
    const d = await load({ writer: 'alive', released: false, clips: [clip('a', 'first')] })
    expect(d.videoJobs.snapshot()).toEqual([])
    expect(d.videoJobs.leftOver().map((c) => c.id)).toEqual(['a'])
    expect(session.has(LANE)).toBe(false)
    expect(JSON.parse(session.get(LEFT)!).clips.map((c: { id: string }) => c.id)).toEqual(['a'])

    d.videoJobs.sendLeftOver()
    expect(d.videoJobs.leftOver()).toEqual([])
    expect(session.has(LEFT)).toBe(false)
    expect(d.videoJobs.snapshot().map((j) => j.id)).toEqual(['a'])
    expect(saved().clips.map((c: { id: string }) => c.id)).toEqual(['a'])
  })

  it('or forgotten, which starts nothing', async () => {
    const d = await load({ writer: 'alive', released: false, clips: [clip('a', 'first')] })
    d.videoJobs.forgetLeftOver()
    expect(d.videoJobs.leftOver()).toEqual([])
    expect(session.has(LEFT)).toBe(false)
    expect(d.videoJobs.snapshot()).toEqual([])
    expect(session.has(LANE)).toBe(false)
    expect(posts()).toEqual([])
  })
})

const SENT_LEFT = 'switchgen.videosent.v1.left'
/** A clip an earlier page sent, as it kept it for the tab. */
const sent = (id: string, promptId: string, ranAt: number | null = 2000, extra: Record<string, unknown> = {}) => ({
  ...clip(id, 'a tram at night'),
  promptId,
  ranAt,
  release: true,
  ...extra,
})
const theClip = { filename: 'clip_0001.webm', subfolder: 'video', type: 'output' }
/** ComfyUI's record of a prompt that wrote the clip, `cached` naming the nodes it answered from its cache. */
const wrote = (id: string, cached: string[] = []) => ({
  [id]: {
    prompt: [1, id, {}, {}, []],
    outputs: { '9': { images: [theClip], animated: [true] } },
    status: {
      status_str: 'success',
      messages: [
        ['execution_start', { timestamp: 5000 }],
        ...(cached.length ? [['execution_cached', { nodes: cached, prompt_id: id }]] : []),
        ['execution_success', { timestamp: 9000 }],
      ],
    },
  },
})
const jobOf = (d: typeof VideoDesk, id: string) => d.videoJobs.snapshot().find((j) => j.id === id)

describe('a clip an earlier page sent', () => {
  it('is followed as a live job when that page handed it on, and filed as the job it is', async () => {
    historyOf = (id) => wrote(id)
    jobStatus = 'pending'
    session.set(SENT, JSON.stringify({ writer: 'old', released: true, jobs: [sent('s1', 'p-old')] }))
    const d = (v = await import('../src/routes/Video'))
    const { history } = await import('../src/lib/history')
    expect(d.videoJobs.snapshot().map((j) => [j.id, j.promptId, j.status, j.resumed])).toEqual([['s1', 'p-old', 'queued', true]])
    expect(JSON.parse(session.get(SENT)!).writer).not.toBe('old')

    jobStatus = 'completed'
    // ComfyUI is asked again four seconds after it said the clip was waiting.
    await vi.waitFor(() => expect(jobOf(d, 's1')?.status).toBe('done'), { timeout: 12_000, interval: 20 })
    const j = jobOf(d, 's1')!
    const record = history.get(j.entryId!)!
    expect(record.promptId).toBe('p-old')
    // Counted from when it began running, as the page that sent it heard, to
    // when ComfyUI's record says it ended, and dated to that end.
    expect(record.durationMs).toBe(7000)
    expect(record.at).toBe(9000)
    expect(session.has(SENT)).toBe(false)
  }, 20_000)

  it('is not followed when that page did not hand it on, but kept for the reader to follow', async () => {
    session.set(SENT, JSON.stringify({ writer: 'alive', released: false, jobs: [sent('s1', 'p-old')] }))
    const d = (v = await import('../src/routes/Video'))
    expect(d.videoJobs.snapshot()).toEqual([])
    expect(d.videoJobs.leftSent().map((c) => c.id)).toEqual(['s1'])
    expect(JSON.parse(session.get(SENT_LEFT)!).jobs[0].id).toBe('s1')
    await new Promise((resolve) => setTimeout(resolve, 20))
    expect(calls.filter((c) => c.includes('/api/jobs'))).toEqual([])
  })

  it('holds the lane behind it when it is lost, and sends nothing until the reader says', async () => {
    jobStatus = null
    busy = 0
    session.set(SENT, JSON.stringify({ writer: 'old', released: true, jobs: [sent('s1', 'p-old')] }))
    session.set(LANE, JSON.stringify({ writer: 'old', released: true, clips: [clip('w1', 'behind it')] }))
    const d = (v = await import('../src/routes/Video'))
    // Two answers that ComfyUI has no such job, four seconds apart.
    await vi.waitFor(() => expect(jobOf(d, 's1')?.status).toBe('error'), { timeout: 12_000, interval: 50 })
    expect(d.videoJobs.held()).toBe(true)
    expect(posts()).toEqual([])
    d.stopVideoJob('w1')
    await vi.waitFor(() => expect(jobOf(d, 'w1')?.status).toBe('cancelled'), WAIT)
    expect(d.videoJobs.held()).toBe(false)
  }, 20_000)

  it('shows the record that already names a file ComfyUI answered from its cache, and files nothing', async () => {
    historyOf = (id) => wrote(id, ['9'])
    const { history } = await import('../src/lib/history')
    const { recordOf } = await import('../src/lib/session')
    const earlier = history.add(
      recordOf(sent('x', 'y').composition as never, {
        file: theClip,
        kind: 'video',
        promptId: 'p-first',
        durationMs: 1000,
        seed: 1,
        familyLabel: 'Wan',
        modelLabel: '',
        at: 1,
      }),
    )
    session.set(SENT, JSON.stringify({ writer: 'old', released: true, jobs: [sent('s1', 'p-again')] }))
    const d = (v = await import('../src/routes/Video'))
    await vi.waitFor(() => expect(jobOf(d, 's1')?.status).toBe('done'), WAIT)
    expect(history.count()).toBe(1)
    expect(jobOf(d, 's1')?.entryId).toBe(earlier.id)
    expect(jobOf(d, 's1')?.repeatOf).toBe(earlier.id)
  })
})

describe('a clip sent from this page', () => {
  it('is kept for the tab once ComfyUI has it, handed on at pagehide, and releases again if others got in first', async () => {
    jobStatus = 'in_progress'
    busy = 0
    const d = await load({ writer: 'old', released: true, clips: [clip('w1', 'first')] })
    await vi.waitFor(() => expect(jobOf(d, 'w1')?.promptId).not.toBeNull(), WAIT)
    const promptId = jobOf(d, 'w1')!.promptId
    const kept = JSON.parse(session.get(SENT)!)
    expect(kept.jobs.map((j: { id: string; promptId: string }) => [j.id, j.promptId])).toEqual([['w1', promptId]])
    expect(kept.released).toBe(false)
    fire('pagehide')
    expect(JSON.parse(session.get(SENT)!).released).toBe(true)
    // The queue is read again once the prompt is in, for the second release.
    await vi.waitFor(() => expect(calls.slice(calls.indexOf('POST /comfy/prompt') + 1)).toContain('GET /comfy/queue'), WAIT)
  }, 15_000)

  it('files how long it ran from when it began running, not from when it was made', async () => {
    busy = 0
    const d = await load({ writer: 'old', released: true, clips: [clip('w1', 'first')] })
    await vi.waitFor(() => expect(jobOf(d, 'w1')?.promptId).toBe('p1'), { timeout: 10_000, interval: 10 })
    const { history } = await import('../src/lib/history')
    const sock = FakeSocket.last!
    sock.deliver('execution_start', { prompt_id: 'p1' })
    await new Promise((resolve) => setTimeout(resolve, 20))
    sock.deliver('executed', { prompt_id: 'p1', node: '9', output: { images: [theClip], animated: [true] } })
    sock.deliver('executing', { prompt_id: 'p1', node: null })
    await vi.waitFor(() => expect(jobOf(d, 'w1')?.status).toBe('done'), WAIT)
    const j = jobOf(d, 'w1')!
    const took = history.get(j.entryId!)!.durationMs
    expect(took).toBeGreaterThan(0)
    // The clip was made at 1000 ms past the epoch and waited in the lane.
    expect(took).toBeLessThan(j.finishedAt! - 1000)
  }, 15_000)
})

describe('a heavy clip lost on this page', () => {
  it('holds the next one, which frees nothing and sends nothing until the reader says', async () => {
    // The loss watch in run() ticks every five seconds from when the clip was
    // sent; its clock is faked before the desk loads.
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'Date'] })
    jobStatus = null
    busy = 0
    const d = await load({ writer: 'old', released: true, clips: [clip('a', 'first'), clip('b', 'second')] })
    await vi.waitFor(() => expect(jobOf(d, 'a')?.promptId).toBe('p1'), { timeout: 10_000, interval: 10 })
    await vi.advanceTimersByTimeAsync(40_000)
    await vi.waitFor(() => expect(jobOf(d, 'a')?.status).toBe('error'), WAIT)
    await vi.waitFor(() => expect(jobOf(d, 'b')?.stage).toBe('Held'), WAIT)
    expect(d.videoJobs.held()).toBe(true)
    expect(posts()).toEqual(['POST /comfy/free', 'POST /comfy/prompt'])

    d.videoJobs.sendHeld()
    await vi.waitFor(() => expect(posts()).toEqual(['POST /comfy/free', 'POST /comfy/prompt', 'POST /comfy/free', 'POST /comfy/prompt']), {
      timeout: 10_000,
      interval: 10,
    })
  }, 20_000)

  it('is not offered as a second go while the rest is held behind it', () => {
    // The notice is drawn only in the page, which this suite has no DOM to
    // draw, so the page is read: its fault says what is held (faults.test.ts)
    // whenever a heavy clip failed and the lane is held behind it.
    const page = readFileSync(path.resolve(import.meta.dirname, '..', 'src', 'routes', 'Video.tsx'), 'utf8')
    const at = page.indexOf('faultBody(')
    expect(page.indexOf('faultBody(', at + 1)).toBe(-1)
    // The call and its options, up to the end of the call.
    const call = page.slice(at, page.indexOf('})}', at))
    expect(call).toMatch(/held: failed\.release && laneHeld/)
  })
})

/**
 * Every timer the desk's waits run on, faked before the desk loads: the
 * follow of a sent clip asks ComfyUI every four seconds, and a clip in the
 * lane reads the queue every two. Time is then moved on by hand, so a slow
 * runner waits no longer than a fast one.
 */
const fakeClock = () => vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'] })
/** Move the faked clock on, a second at a time, letting what each second starts run. */
async function pass(ms: number): Promise<void> {
  for (let t = 0; t < ms; t += 1000) await vi.advanceTimersByTimeAsync(Math.min(1000, ms - t))
}
/**
 * Wait for `check` on the faked clock, moving it on a tenth of a second
 * between looks, for up to `within` of faked time. Each move lets the answers
 * already due come in, so how long this takes does not depend on the runner.
 */
async function soon(check: () => void, within = 30_000): Promise<void> {
  for (let t = 0; ; t += 100) {
    try {
      check()
      return
    } catch (err) {
      if (t >= within) throw err
    }
    await vi.advanceTimersByTimeAsync(100)
  }
}

describe('the lane behind more than one heavy clip', () => {
  it('holds a clip back until every heavy clip followed after a reload has settled, not the last alone', async () => {
    fakeClock()
    busy = 0
    jobStatus = 'pending'
    session.set(SENT, JSON.stringify({ writer: 'old', released: true, jobs: [sent('s1', 'p-a'), sent('s2', 'p-b')] }))
    const d = await load({ writer: 'old', released: true, clips: [clip('w1', 'behind them')] })
    // The one followed last settles first.
    statusOf.set('p-b', 'cancelled')
    await soon(() => expect(jobOf(d, 's2')?.status).toBe('cancelled'))
    await pass(12_000)
    expect(jobOf(d, 's1')?.status).toBe('queued')
    expect(posts()).toEqual([])
    statusOf.set('p-a', 'cancelled')
    await soon(() => expect(jobOf(d, 's1')?.status).toBe('cancelled'))
    await soon(() => expect(posts()).toEqual(['POST /comfy/free', 'POST /comfy/prompt']))
  })

  it('does not let the clip behind a stopped one go beside the clip ahead of it', async () => {
    fakeClock()
    busy = 1
    jobStatus = 'pending'
    const d = await load({ writer: 'old', released: true, clips: [clip('a', 'first'), clip('b', 'second'), clip('c', 'third')] })
    await pass(100)
    d.stopVideoJob('b')
    await soon(() => expect(jobOf(d, 'b')?.status).toBe('cancelled'))
    busy = 0
    await soon(() => expect(posts()).toContain('POST /comfy/prompt'))
    // Several reads of the queue later, c still waits for a to settle.
    await pass(6000)
    expect(posts()).toEqual(['POST /comfy/free', 'POST /comfy/prompt'])
    expect(jobOf(d, 'c')?.stage).toBe('Waiting its turn')
  })
})

describe('the screen while the lane is held', () => {
  const LANE_HOLD = 'Clips waiting in the Video desk lane'
  beforeEach(() => {
    vi.stubGlobal('navigator', { wakeLock: { request: async () => ({ released: false, release: async () => {}, addEventListener() {} }) } })
  })

  it('is let sleep for a lane held when the page went, and kept on again once the reader sends it', async () => {
    const d = await load({ writer: 'old', released: true, held: true, clips: [clip('w1', 'held')] })
    const { awakeReasons } = await import('../src/lib/wakeLock')
    expect(d.videoJobs.held()).toBe(true)
    expect(awakeReasons()).toEqual([])
    // Nothing is sent while it is held, but a reload would still lose it.
    expect(listeners.get('beforeunload')).toHaveLength(1)
    d.videoJobs.sendHeld()
    expect(awakeReasons()).toEqual([LANE_HOLD])
    expect(listeners.get('beforeunload')).toHaveLength(1)
  })

  it('is let sleep once a heavy clip followed after a reload is lost and the lane is held behind it', async () => {
    fakeClock()
    busy = 0
    statusOf.set('p-old', null)
    session.set(SENT, JSON.stringify({ writer: 'old', released: true, jobs: [sent('s1', 'p-old')] }))
    const d = await load({ writer: 'old', released: true, clips: [clip('w1', 'behind it')] })
    const { awakeReasons } = await import('../src/lib/wakeLock')
    expect(awakeReasons()).toEqual([LANE_HOLD])
    // Two answers that ComfyUI has no such job, four seconds apart.
    await soon(() => expect(jobOf(d, 's1')?.status).toBe('error'))
    expect(d.videoJobs.held()).toBe(true)
    expect(awakeReasons()).toEqual([])
    expect(posts()).toEqual([])
  })
})

describe('a clip taken up after a reload that ComfyUI finished long before', () => {
  // ComfyUI's record says the clip ran from 5000 to 9000; the page comes back
  // forty minutes later, which is when it hears the clip has ended.
  const back = async (ranAt: number | null) => {
    fakeClock()
    vi.setSystemTime(2_400_000)
    session.set(SENT, JSON.stringify({ writer: 'old', released: true, jobs: [sent('s1', 'p-old', ranAt)] }))
    const d = (v = await import('../src/routes/Video'))
    const { history } = await import('../src/lib/history')
    await soon(() => expect(jobOf(d, 's1')?.status).toBe('done'))
    const j = jobOf(d, 's1')!
    return { j, record: history.get(j.entryId!)! }
  }

  it('is filed with ComfyUI\'s end, timed from the start the page that sent it heard', async () => {
    historyOf = (id) => wrote(id)
    const { j, record } = await back(2000)
    expect(record.durationMs).toBe(7000)
    expect(record.at).toBe(9000)
    expect(j.finishedAt).toBe(9000)
    expect(j.tookMs).toBe(7000)
  })

  it('is timed from ComfyUI\'s own start when that page never heard it begin', async () => {
    historyOf = (id) => wrote(id)
    const { record } = await back(null)
    expect(record.durationMs).toBe(4000)
    expect(record.at).toBe(9000)
  })

  it('falls back on the page\'s clock only when ComfyUI\'s record gives no times', async () => {
    historyOf = (id) => ({ [id]: { ...wrote(id)[id], status: { status_str: 'success', messages: [] } } })
    const { j, record } = await back(2000)
    expect(j.finishedAt).toBeGreaterThanOrEqual(2_400_000)
    expect(record.durationMs).toBe(j.finishedAt! - 2000)
  })
})

describe('a stop pressed on a heavy clip ComfyUI lost to a restart', () => {
  it('leaves the loss a loss, and holds the next heavy clip, when the stop never reached a live job', async () => {
    // The loss watch in run() ticks every five seconds from when the clip was
    // sent; its clock is faked before the desk loads.
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'Date'] })
    jobStatus = 'in_progress'
    busy = 0
    const d = await load({ writer: 'old', released: true, clips: [clip('a', 'first'), clip('b', 'second')] })
    await vi.waitFor(() => expect(jobOf(d, 'a')?.promptId).toBe('p1'), { timeout: 10_000, interval: 10 })
    // ComfyUI is killed, and the reader holds Stop on the frozen clip.
    down = true
    d.stopVideoJob('a')
    await vi.waitFor(() => expect(jobOf(d, 'a')?.error).toMatch(/not answering/), WAIT)
    expect(jobOf(d, 'a')?.stopLanded).toBe(false)
    // Back with an empty queue, knowing nothing of the clip, and saying it
    // stopped nothing when asked again.
    down = false
    statusOf.set('p1', null)
    cancels = false
    await vi.advanceTimersByTimeAsync(40_000)
    await vi.waitFor(() => expect(jobOf(d, 'a')?.status).toBe('error'), WAIT)
    expect(jobOf(d, 'a')?.stage).toBe('Lost')
    expect(jobOf(d, 'a')?.fault?.lost).toBe(true)
    await vi.waitFor(() => expect(jobOf(d, 'b')?.stage).toBe('Held'), WAIT)
    expect(d.videoJobs.held()).toBe(true)
    // Nothing more went to the restarted ComfyUI.
    expect(posts()).toEqual(['POST /comfy/free', 'POST /comfy/prompt'])
  }, 20_000)

  it('still reads as stopped when ComfyUI took the stop while the clip waited, and holds nothing', async () => {
    fakeClock()
    busy = 0
    jobStatus = 'pending'
    session.set(SENT, JSON.stringify({ writer: 'old', released: true, jobs: [sent('s1', 'p-a')] }))
    const d = (v = await import('../src/routes/Video'))
    await soon(() => expect(jobOf(d, 's1')?.status).toBe('queued'))
    d.stopVideoJob('s1')
    await soon(() => expect(jobOf(d, 's1')?.stopLanded).toBe(true))
    // A prompt taken out of the queue leaves no record.
    statusOf.set('p-a', null)
    await soon(() => expect(jobOf(d, 's1')?.status).toBe('cancelled'))
    expect(d.videoJobs.held()).toBe(false)
  })
})

describe('a clip on its way to ComfyUI', () => {
  it('is kept for the tab under the id it goes with before it goes, and marked as arrived once ComfyUI has it', async () => {
    fakeClock()
    echo = true
    busy = 0
    jobStatus = 'in_progress'
    const d = await load({ writer: 'old', released: true, clips: [clip('w1', 'first')] })
    await soon(() => expect(promptBody).not.toBeNull())
    // What the tab held when the prompt went: the clip, under that prompt's
    // id, marked as still on its way, and out of the saved lane.
    expect(sentAtPrompt?.jobs.map((j) => [j.id, j.promptId, j.sending])).toEqual([['w1', promptBody!.prompt_id, true]])
    expect(promptBody!.prompt_id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/)
    expect(session.has(LANE)).toBe(false)
    await soon(() => expect(jobOf(d, 'w1')?.promptId).toBe(promptBody!.prompt_id))
    const kept = JSON.parse(session.get(SENT)!)
    expect(kept.jobs[0].promptId).toBe(promptBody!.prompt_id)
    expect(kept.jobs[0].sending).toBeUndefined()
    expect(jobOf(d, 'w1')?.sendingAs).toBeNull()
  })

  it('is handed on at pagehide while ComfyUI has not yet answered', async () => {
    fakeClock()
    busy = 0
    let answer: () => void = () => {}
    const answered = new Promise<void>((resolve) => {
      answer = resolve
    })
    const table = globalThis.fetch
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      if (url !== '/comfy/prompt') return table(url, init)
      calls.push('POST /comfy/prompt')
      await answered
      return json({ prompt_id: JSON.parse(String(init?.body)).prompt_id })
    })
    const d = await load({ writer: 'old', released: true, clips: [clip('w1', 'first')] })
    await soon(() => expect(calls).toContain('POST /comfy/prompt'))
    fire('pagehide')
    const kept = JSON.parse(session.get(SENT)!)
    expect(kept.released).toBe(true)
    expect(kept.jobs.map((j: { id: string; sending?: boolean }) => [j.id, j.sending])).toEqual([['w1', true]])
    expect(jobOf(d, 'w1')?.status).toBe('submitting')
    answer()
  })

  it('is not sent again by the next page when ComfyUI has nothing under its id, and holds nothing behind it', async () => {
    fakeClock()
    echo = true
    busy = 0
    statusOf.set('p-x', null)
    session.set(SENT, JSON.stringify({ writer: 'old', released: true, jobs: [sent('s1', 'p-x', null, { sending: true })] }))
    const d = await load({ writer: 'old', released: true, clips: [clip('w1', 'behind it')] })
    expect(jobOf(d, 's1')).toMatchObject({ status: 'submitting', sendingAs: 'p-x' })
    // Two answers that ComfyUI has no such job, four seconds apart.
    await soon(() => expect(jobOf(d, 's1')?.status).toBe('error'))
    expect(jobOf(d, 's1')?.stage).toBe('Not sent')
    expect(jobOf(d, 's1')?.error).toMatch(/may never have reached ComfyUI/)
    // It says nothing of ComfyUI's memory, so the clip behind it goes.
    expect(d.videoJobs.held()).toBe(false)
    await soon(() => expect(posts()).toEqual(['POST /comfy/free', 'POST /comfy/prompt']))
    expect(promptBody!.prompt_id).toBe(jobOf(d, 'w1')?.promptId)
    expect(promptBody!.prompt_id).not.toBe('p-x')
  })

  it('is followed as ComfyUI\'s once ComfyUI shows it has it, and a later loss is a loss', async () => {
    fakeClock()
    jobStatus = 'pending'
    session.set(SENT, JSON.stringify({ writer: 'old', released: true, jobs: [sent('s1', 'p-x', null, { sending: true })] }))
    const d = (v = await import('../src/routes/Video'))
    await soon(() => expect(jobOf(d, 's1')?.status).toBe('queued'))
    expect(jobOf(d, 's1')?.sendingAs).toBeNull()
    expect(JSON.parse(session.get(SENT)!).jobs[0].sending).toBeUndefined()
    statusOf.set('p-x', null)
    await soon(() => expect(jobOf(d, 's1')?.status).toBe('error'))
    expect(jobOf(d, 's1')?.stage).toBe('Lost')
    expect(jobOf(d, 's1')?.error).not.toMatch(/may never have reached ComfyUI/)
  })
})
