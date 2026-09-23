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
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })

class FakeSocket {
  static OPEN = 1
  static CONNECTING = 0
  readyState = 0
  binaryType = ''
  onopen: (() => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  onmessage: ((ev: { data: unknown }) => void) | null = null
  url: string
  constructor(url: string) {
    this.url = url
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
}

let calls: string[] = []
/** How many jobs ComfyUI reports running. */
let busy = 1
let prompts = 0
/** Whether the saved lane was still there when the prompt went. */
let laneAtPrompt: boolean | null = null
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
  busy = 1
  prompts = 0
  laneAtPrompt = null
  session.clear()
  listeners.clear()
  reload.mockReset()
  vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
    calls.push(`${init?.method ?? 'GET'} ${url}`)
    if (url === '/comfy/queue') return json({ queue_running: Array(busy).fill([]), queue_pending: [] })
    if (url === '/comfy/free') return json({})
    if (url === '/comfy/prompt') {
      laneAtPrompt = session.has(LANE)
      return json({ prompt_id: `p${++prompts}` })
    }
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
  await new Promise((resolve) => setTimeout(resolve, 10))
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
    await vi.waitFor(() => expect(d.videoJobs.snapshot().find((j) => j.id === 'a')?.status).toBe('cancelled'))
    expect(saved().clips.map((c: { id: string }) => c.id)).toEqual(['b'])
    d.stopVideoJob('b')
    await vi.waitFor(() => expect(d.videoJobs.snapshot().find((j) => j.id === 'b')?.status).toBe('cancelled'))
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
  })

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
