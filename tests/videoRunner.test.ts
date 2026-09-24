import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type * as VideoDesk from '../src/routes/Video'
import type * as Runner from '../src/lib/runner'
import type * as History from '../src/lib/history'

/**
 * The Video desk with the queue on the server. fetch answers from a table
 * that stands in for the SwitchGen server's /api/runner routes and for the
 * ComfyUI proxy, the event stream and the socket are stand-ins the test
 * drives, and nothing leaves the test.
 */

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })

class FakeSocket {
  readyState = 0
  binaryType = ''
  onopen: (() => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  onmessage: ((ev: { data: unknown }) => void) | null = null
  constructor(public url: string) {
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

class FakeStream {
  static last: FakeStream | null = null
  readyState = 1
  onerror: (() => void) | null = null
  handlers = new Map<string, ((ev: { data: string }) => void)[]>()
  constructor(public url: string) {
    FakeStream.last = this
  }
  addEventListener(name: string, fn: (ev: { data: string }) => void) {
    this.handlers.set(name, [...(this.handlers.get(name) ?? []), fn])
  }
  close() {
    this.readyState = 2
  }
  emit(name: string, data: unknown) {
    for (const fn of this.handlers.get(name) ?? []) fn({ data: JSON.stringify(data) })
  }
}

let calls: string[] = []
let bodies: Record<string, any[]> = {}
let caps: Record<string, unknown> = {}
let groupsAnswer: (body: any) => Response = () => json({ error: 'x' }, 500)
let rev = 0
let snapshot: any
/** ComfyUI's queue has someone else's job in it, so the page's own lane waits. */
let comfyBusy = false
/** Job ids the stand-in server holds, so a stop for another answers 404 as the route does. */
const known = new Set<string>()
const session = new Map<string, string>()
const listeners = new Map<string, ((e: unknown) => void)[]>()
const WAIT = { timeout: 5000, interval: 10 }

function jobView(over: Record<string, unknown>) {
  return {
    id: 'j', groupId: 'g', desk: 'video', kind: 'video', seq: 1, index: 1, total: 1, label: 'Wan', prompt: 'a tram',
    device: 'other-device', heavy: true, status: 'waiting', wait: { for: 'turn' }, stopRequested: false, stopLanded: false,
    promptId: null, attempt: 0, createdAt: 5000, sentAt: null, ranAt: null, finishedAt: null, endedAt: null, files: [],
    primary: null, frame: null, openedOn: null, entryId: null, entryNo: null, repeatOf: null, durationMs: 0, error: null,
    meta: { frames: 81, fps: 16 }, dismissed: false, ...over,
  }
}

beforeEach(() => {
  calls = []
  bodies = {}
  comfyBusy = false
  rev = 1
  snapshot = { v: 1, available: true, reason: null, boot: 'b1', rev: 1, comfy: { answering: true, since: 0 }, lane: { held: null }, groups: [], jobs: [], progress: {} }
  caps = { runner: true, runnerDesks: ['video', 'images', 'reel'], runnerReason: null }
  groupsAnswer = (body) => {
    for (const j of body.jobs) known.add(j.id)
    const jobs = body.jobs.map((j: any, i: number) => jobView({ id: j.id, groupId: body.group.id, device: body.group.device, seq: 10 + i, label: j.label, prompt: j.prompt, heavy: j.heavy, meta: j.meta, createdAt: Date.now() }))
    rev += 1
    return json({ rev, replayed: false, group: { id: body.group.id, desk: 'video', kind: 'clips', label: body.group.label, device: body.group.device, createdAt: Date.now(), state: 'active', endedBy: null, endedAt: null, jobIds: jobs.map((j: any) => j.id), dismissed: false }, jobs })
  }
  session.clear()
  known.clear()
  listeners.clear()
  FakeStream.last = null
  vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
    const method = init?.method ?? 'GET'
    calls.push(`${method} ${url}`)
    if (init?.body) (bodies[url] ??= []).push(JSON.parse(String(init.body)))
    if (url === '/api/capabilities') return json(caps)
    if (url === '/api/runner' && method === 'GET') return json(snapshot)
    if (url.startsWith('/api/runner/jobs/') && method === 'GET') {
      const id = decodeURIComponent(url.split('/')[4]!)
      const j = snapshot.jobs.find((x: any) => x.id === id)
      return j ? json({ job: j, graph: {}, record: { desk: 'video', mode: 't2v', prompt: 'from the server', familyId: 'wan22-14b-t2v', seed: 42, loras: [{ name: 'grain.safetensors', strength: 0.5 }] } }) : json({ error: 'no such job' }, 404)
    }
    if (url === '/api/runner/groups') return groupsAnswer(JSON.parse(String(init!.body)))
    const stop = /^\/api\/runner\/jobs\/([^/]+)\/stop$/.exec(url)
    if (stop) return known.has(decodeURIComponent(stop[1]!)) || snapshot.jobs.some((j: any) => j.id === decodeURIComponent(stop[1]!)) ? json({ job: {} }) : json({ error: 'no such job' }, 404)
    if (/^\/api\/runner\/(lane|dismiss)$/.test(url)) return json({ ok: true })
    if (url === '/comfy/queue') return json({ queue_running: comfyBusy ? [[0, 'someone-else']] : [], queue_pending: [] })
    if (url.startsWith('/comfy/api/jobs?')) return json({ jobs: [], pagination: { offset: 0, limit: 100, total: 0, has_more: false } })
    return json({ error: 'no route' }, 599)
  })
  vi.stubGlobal('WebSocket', FakeSocket)
  vi.stubGlobal('EventSource', FakeStream)
  vi.stubGlobal('location', { protocol: 'http:', host: 'harness', hash: '' })
  vi.stubGlobal('sessionStorage', {
    getItem: (k: string) => session.get(k) ?? null,
    setItem: (k: string, value: string) => void session.set(k, String(value)),
    removeItem: (k: string) => void session.delete(k),
  })
  vi.stubGlobal('window', {
    addEventListener: (type: string, fn: (e: unknown) => void) => listeners.set(type, [...(listeners.get(type) ?? []), fn]),
    removeEventListener: () => {},
    location: { reload: () => {} },
  })
  vi.resetModules()
})

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

const clip = (release = true) => ({
  composition: { desk: 'video', mode: 't2v', familyId: 'wan22-14b-t2v', model: '', prompt: 'a tram at dusk', negative: null, positivePrefix: null, positive: 'a tram at dusk, words', source: null, width: 832, height: 480, megapixels: null, denoise: null, seed: 7, seedLocked: false, steps: 20, cfg: 4, sampler: 'euler', scheduler: 'simple', length: 81, fps: 16, split: null, shift: null, clipSkip: null, noLora: false, runs: 1, addOnsAccepted: [], addOnsDeclined: [], touched: [] } as any,
  graph: { '1': { class_type: 'KSamplerAdvanced', inputs: { return_with_leftover_noise: 'enable' } }, '2': { class_type: 'KSamplerAdvanced', inputs: {} } } as any,
  familyLabel: 'Wan 2.2 T2V A14B',
  modelLabel: 'Wan 2.2 14B',
  loras: [{ name: 'x.safetensors', strength: 1 }],
  release,
})

async function load() {
  const v = (await import('../src/routes/Video')) as typeof VideoDesk
  const r = (await import('../src/lib/runner')) as typeof Runner
  const h = (await import('../src/lib/history')) as typeof History
  return { v, r, h }
}
const comfyCalls = () => calls.filter((x) => x.includes('/comfy/'))
const find = (v: typeof VideoDesk, id: string) => v.videoJobs.snapshot().find((j) => j.id === id)

describe('Make with the queue on the server', () => {
  it('hands one press over in one request and follows each clip to its filed ending, with nothing sent from the page', async () => {
    const { v, r, h } = await load()
    const sent = await v.sendClips([clip(), clip(false)])
    expect(sent.road).toBe('server')
    expect(bodies['/api/runner/groups']).toHaveLength(1)
    const posted = bodies['/api/runner/groups']![0]
    expect(posted.group).toMatchObject({ desk: 'video', kind: 'clips', device: r.deviceId() })
    expect(posted.jobs).toHaveLength(2)
    expect(posted.jobs[0]).toMatchObject({ kind: 'video', primary: 'video', orFirst: true, noFile: 'done', heavy: true, meta: { frames: 81, fps: 16 }, label: 'Wan 2.2 14B' })
    expect(posted.jobs[1].heavy).toBe(false)
    expect(posted.jobs[0].graph).toEqual(clip().graph)
    const c = clip()
    expect(posted.jobs[0].record).toEqual(JSON.parse(JSON.stringify(r.recordTemplate(c.composition, { seed: 7, familyLabel: c.familyLabel, modelLabel: c.modelLabel, loras: c.loras }))))
    expect(comfyCalls()).toEqual([])
    const ids = (sent as { ids: string[] }).ids
    expect(v.videoJobs.snapshot().map((j) => j.id).sort()).toEqual([...ids].sort())
    expect(v.videoJobs.snapshot().every((j) => j.runner && j.sentHere && j.status === 'submitting')).toBe(true)
    expect(v.videoJobs.waitingCount()).toBe(0)

    await vi.waitFor(() => expect(FakeStream.last).not.toBeNull(), WAIT)
    const s = FakeStream.last!
    s.emit('state', { ...snapshot, rev, jobs: r.runnerStore.snapshot().jobs })
    const first = ids[0]!
    const base = r.runnerStore.snapshot().jobs.find((j) => j.id === first)!
    s.emit('job', { rev: ++rev, job: { ...base, status: 'queued', wait: null, promptId: 'p1' } })
    await vi.waitFor(() => expect(find(v, first)).toMatchObject({ status: 'queued', promptId: 'p1' }), WAIT)
    // A page that goes while the server has the clip keeps nothing of it in the tab.
    for (const fn of listeners.get('pagehide') ?? []) fn({ persisted: false })
    expect(session.has('switchgen.videosent.v1')).toBe(false)
    s.emit('job', { rev: ++rev, job: { ...base, status: 'running', wait: null, promptId: 'p1', ranAt: 9000 } })
    s.emit('progress', { id: first, value: 3, max: 10, node: '2', classType: 'KSamplerAdvanced', pass: { index: 2, count: 2 }, at: 1, previewN: 0 })
    await vi.waitFor(() => expect(find(v, first)?.value).toBe(3), WAIT)
    expect(find(v, first)).toMatchObject({ stage: 'Drawing', pass: { index: 2, count: 2 } })
    const file = { filename: 'c.webm', subfolder: 'video', type: 'output', kind: 'video' }
    s.emit('job', { rev: ++rev, job: { ...base, status: 'done', wait: null, promptId: 'p1', files: [file], primary: file, entryId: first, entryNo: 12, durationMs: 61000, finishedAt: 70000, endedAt: 70001 } })
    await vi.waitFor(() => expect(find(v, first)?.status).toBe('done'), WAIT)
    expect(find(v, first)).toMatchObject({ entryId: first, tookMs: 61000, finishedAt: 70000 })
    // The server filed it; the page files nothing.
    expect(h.history.all()).toHaveLength(0)
    expect(comfyCalls()).toEqual([])
    // A page that goes keeps nothing of the server's clips in the tab.
    for (const fn of listeners.get('pagehide') ?? []) fn({ persisted: false })
    expect(session.has('switchgen.videosent.v1')).toBe(false)
    expect(session.has('switchgen.videolane.v1')).toBe(false)
  })

  it('shows no time for a clip the server could not time', async () => {
    snapshot = { ...snapshot, jobs: [jobView({ id: 'c0', status: 'done', durationMs: 0, finishedAt: 7, endedAt: 7, entryId: 'e0', promptId: 'p0' })] }
    const { v, r } = await load()
    r.runnerStore.start()
    await vi.waitFor(() => expect(v.videoJobs.snapshot()).toHaveLength(1), WAIT)
    expect(v.videoJobs.snapshot()[0]).toMatchObject({ status: 'done', tookMs: null, sentHere: false })
  })

  it('never moves a clip back when a progress report beats the list that says it runs', async () => {
    const { v, r } = await load()
    const sent = (await v.sendClips([clip()])) as { ids: string[] }
    const id = sent.ids[0]!
    await vi.waitFor(() => expect(FakeStream.last).not.toBeNull(), WAIT)
    const s = FakeStream.last!
    s.emit('state', { ...snapshot, rev, jobs: r.runnerStore.snapshot().jobs })
    const base = r.runnerStore.snapshot().jobs.find((j) => j.id === id)!
    s.emit('job', { rev: ++rev, job: { ...base, status: 'running', wait: null, promptId: 'p1' } })
    await vi.waitFor(() => expect(find(v, id)?.status).toBe('running'), WAIT)
    // A late copy of an earlier state.
    s.emit('job', { rev, job: { ...base, status: 'queued', wait: null, promptId: 'p1' } })
    await new Promise((res) => setTimeout(res, 20))
    expect(find(v, id)?.status).toBe('running')
  })
})

describe('a clip the server holds from another browser', () => {
  it('is taken up, stopped on the server, and its hold answered there', async () => {
    const other = jobView({ id: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa', status: 'waiting', wait: { for: 'held' } })
    snapshot = { ...snapshot, lane: { held: { why: 'lost', scope: 'heavy', jobId: 'lostone', since: 1 } }, jobs: [other, jobView({ id: 'reel1', desk: 'reel', status: 'waiting', wait: { for: 'held' } })] }
    const { v, r } = await load()
    r.runnerStore.start()
    await vi.waitFor(() => expect(v.videoJobs.snapshot()).toHaveLength(1), WAIT)
    expect(v.videoJobs.snapshot()[0]).toMatchObject({ runner: true, adopted: true, sentHere: false, status: 'submitting', stage: 'Held', frames: 81, fps: 16 })
    expect(v.videoJobs.snapshot()[0]!.waitNote).toBe('Held until you say, because the heavy clip before it was lost.')
    expect(v.videoJobs.heldOnServer()).toEqual({ why: 'lost', jobId: 'lostone', clips: 1, others: 1, since: 1 })
    // The page's own lane holds nothing.
    expect(v.videoJobs.held()).toBe(false)
    v.videoJobs.sendHeld()
    v.videoJobs.stopHeld()
    await vi.waitFor(() => expect(bodies['/api/runner/lane']).toEqual([{ action: 'send', since: 1 }, { action: 'stop', since: 1 }]), WAIT)
    v.stopVideoJob(other.id)
    await vi.waitFor(() => expect(calls).toContain(`POST /api/runner/jobs/${other.id}/stop`), WAIT)
    expect(v.videoJobs.snapshot()[0]!.stage).toBe('Stopping')
    expect(comfyCalls()).toEqual([])
  })

  it('says a restart hold in the words for one', async () => {
    // Made at 5000, before the machine restarted.
    const held = jobView({ id: 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbc', status: 'waiting', wait: { for: 'held' } })
    snapshot = { ...snapshot, lane: { held: { why: 'restart', scope: 'all', jobId: null, since: 6000 } }, jobs: [held] }
    const { v, r } = await load()
    r.runnerStore.start()
    await vi.waitFor(() => expect(v.videoJobs.snapshot()).toHaveLength(1), WAIT)
    expect(v.videoJobs.snapshot()[0]!.waitNote).toBe(r.HELD_AFTER_RESTART)
    expect(v.videoJobs.heldOnServer()).toMatchObject({ why: 'restart', clips: 1, others: 0 })
  })

  it('is shown lost without holding the page\'s own lane, even with clips of its own waiting there', async () => {
    const lost = jobView({ id: 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb', status: 'running', wait: null, promptId: 'p9' })
    comfyBusy = true
    snapshot = { ...snapshot, available: false, reason: 'The queue on the server is not running.', jobs: [lost] }
    const { v, r } = await load()
    r.runnerStore.start()
    await vi.waitFor(() => expect(v.videoJobs.snapshot()).toHaveLength(1), WAIT)
    // A clip of this page's own, waiting in its lane for ComfyUI's queue to empty.
    expect((await v.sendClips([clip()])).road).toBe('page')
    expect(v.videoJobs.waitingCount()).toBe(1)
    await vi.waitFor(() => expect(FakeStream.last).not.toBeNull(), WAIT)
    FakeStream.last!.emit('state', snapshot)
    FakeStream.last!.emit('job', { rev: 2, job: { ...lost, status: 'lost', error: { code: 'lost', message: null, node: null, nodeType: null, nodeErrors: null, mayExist: false, sent: true, after: null }, endedAt: 1234 } })
    await vi.waitFor(() => expect(find(v, lost.id)!.status).toBe('error'), WAIT)
    expect(find(v, lost.id)).toMatchObject({ stage: 'Lost', finishedAt: 1234, fault: { lost: true } })
    // The server holds its own lane; the page's clip is not held for it.
    expect(v.videoJobs.held()).toBe(false)
    expect(v.videoJobs.waitingCount()).toBe(1)
    for (const job of v.videoJobs.snapshot()) v.stopVideoJob(job.id)
    await new Promise((res) => setTimeout(res, 20))
  })

  it('leaves the desk once another browser put it away', async () => {
    const done = jobView({ id: 'cccccccc-cccc-4ccc-8ccc-cccccccccccc', status: 'done', promptId: 'p', entryId: 'e', finishedAt: 5, endedAt: 5 })
    snapshot = { ...snapshot, jobs: [done] }
    const { v, r } = await load()
    r.runnerStore.start()
    await vi.waitFor(() => expect(v.videoJobs.snapshot()).toHaveLength(1), WAIT)
    await vi.waitFor(() => expect(FakeStream.last).not.toBeNull(), WAIT)
    FakeStream.last!.emit('state', snapshot)
    FakeStream.last!.emit('job', { rev: 2, job: { ...done, dismissed: true } })
    await vi.waitFor(() => expect(v.videoJobs.snapshot()).toHaveLength(0), WAIT)
  })
})

describe('Make without the queue', () => {
  it('sends the clip from the page, unchanged, and says why', async () => {
    caps = { runner: false, runnerDesks: [], runnerReason: 'Turned off with SWITCHGEN_RUNNER=off.' }
    const { v } = await load()
    const sent = await v.sendClips([clip()])
    expect(sent.road).toBe('page')
    expect(calls.some((x) => x.startsWith('POST /api/runner'))).toBe(false)
    expect(v.videoJobs.waitingCount()).toBe(1)
    expect(v.videoJobs.snapshot()[0]!.runner).toBeUndefined()
    expect(v.videoJobs.sendsItself()).toBe('This page sends the work itself: turned off with SWITCHGEN_RUNNER=off.')
    for (const job of v.videoJobs.snapshot()) v.stopVideoJob(job.id)
    await new Promise((res) => setTimeout(res, 20))
  })

  it('falls back to the page under the same ids when the queue answers 503', async () => {
    groupsAnswer = () => json({ error: 'not running', busy: 'runner', reason: 'Another SwitchGen server holds the archive, and the queue with it.' }, 503)
    const { v } = await load()
    const sent = await v.sendClips([clip()])
    expect(sent.road).toBe('page')
    expect((sent as { ids: string[] }).ids).toEqual([bodies['/api/runner/groups']![0].jobs[0].id])
    expect(v.videoJobs.snapshot()[0]!.runner).toBeUndefined()
    expect(v.videoJobs.waitingCount()).toBe(1)
    expect(v.videoJobs.sendsItself()).toBe('This page sends the work itself: another SwitchGen server holds the archive, and the queue with it.')
    for (const job of v.videoJobs.snapshot()) v.stopVideoJob(job.id)
    await new Promise((res) => setTimeout(res, 20))
  })

  it('keeps new clips behind the ones the page is still sending itself, and says so', async () => {
    comfyBusy = true
    snapshot = { ...snapshot, available: false, reason: 'The queue on the server is not running.' }
    const { v, r } = await load()
    r.runnerStore.start()
    await vi.waitFor(() => expect(r.runnerStore.snapshot().boot).toBe('b1'), WAIT)
    expect((await v.sendClips([clip()])).road).toBe('page')
    expect(v.videoJobs.waitingCount()).toBe(1)
    // The queue is back, but a clip still waits in this page's own lane.
    snapshot = { ...snapshot, available: true, reason: null, rev: 2 }
    await r.runnerStore.refresh()
    expect((await r.runnerAvailable('video')).ok).toBe(true)
    expect((await v.sendClips([clip()])).road).toBe('page')
    expect(v.videoJobs.waitingCount()).toBe(2)
    expect(v.videoJobs.sendsItself()).toBe('This page sends the work itself: the clips already waiting in this page go first, and new ones wait behind them here.')
    expect(calls.some((x) => x.startsWith('POST /api/runner'))).toBe(false)
    for (const job of v.videoJobs.snapshot()) v.stopVideoJob(job.id)
    await new Promise((res) => setTimeout(res, 20))
  })

  it('says a refusal and keeps nothing', async () => {
    groupsAnswer = () => json({ error: 'The server’s disk is full, so it cannot save this work. Nothing was taken.' }, 507)
    const { v } = await load()
    expect(await v.sendClips([clip()])).toEqual({ road: 'refused', error: 'The server’s disk is full, so it cannot save this work. Nothing was taken.' })
    expect(v.videoJobs.snapshot()).toEqual([])
  })
})

const OUTBOX = 'switchgen.runner.outbox.v1'
/** The Video desk's stops kept in the tab for clips the server had not listed. */
const STOPS = 'switchgen.videostops.v1'
const outboxGroups = () => JSON.parse(session.get(OUTBOX) ?? '[]') as { body: any; sent: boolean }[]

describe('a hand-over the server does not answer', () => {
  it('keeps the clip as being handed over, then says so on the clip when the replay is refused', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'Date'] })
    groupsAnswer = () => new Response('<html>bad gateway</html>', { status: 502, headers: { 'content-type': 'text/html' } })
    const { v, r } = await load()
    const going = v.sendClips([clip()])
    await vi.advanceTimersByTimeAsync(40_000)
    const sent = await going
    expect(sent).toMatchObject({ road: 'server', pending: true })
    const j = v.videoJobs.snapshot()[0]!
    expect(j).toMatchObject({ status: 'submitting', stage: 'Handing it to the server' })
    expect(j.waitNote).toMatch(/has not answered yet/)
    // The server comes back and refuses the replay.
    groupsAnswer = () => json({ error: 'That is not a clip.' }, 400)
    await vi.waitFor(() => expect(FakeStream.last).not.toBeNull(), WAIT)
    FakeStream.last!.emit('state', { ...snapshot, rev: 50 })
    await vi.advanceTimersByTimeAsync(10)
    await vi.waitFor(() => expect(v.videoJobs.snapshot()[0]!.status).toBe('error'), WAIT)
    expect(v.videoJobs.snapshot()[0]).toMatchObject({ stage: 'Not sent', error: 'That is not a clip.' })
    await vi.advanceTimersByTimeAsync(10)
    expect(r.givenUpBatches()).toEqual([])
    expect(v.videoJobs.givenUp()).toEqual([])
    expect(comfyCalls()).toEqual([])
  })

  it('takes a clip stopped meanwhile out of the hand-over, and keeps the stop in the tab for the next page, which asks it once the server lists the clip', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'Date'] })
    groupsAnswer = () => new Response('<html>bad gateway</html>', { status: 502, headers: { 'content-type': 'text/html' } })
    const { v } = await load()
    const going = v.sendClips([clip()])
    await vi.advanceTimersByTimeAsync(40_000)
    expect(await going).toMatchObject({ road: 'server', pending: true })
    const j = v.videoJobs.snapshot()[0]!
    const group = outboxGroups()[0]!.body.group.id
    v.stopVideoJob(j.id)
    await vi.advanceTimersByTimeAsync(10)
    // An earlier try may have got there, so the card does not say it was never sent.
    expect(find(v, j.id)).toMatchObject({ status: 'cancelled', stage: 'Stopped', error: 'Stopped before the server answered for it.' })
    expect(session.has(OUTBOX)).toBe(false)
    expect(JSON.parse(session.get(STOPS)!)).toEqual([j.id])
    const groupPosts = () => calls.filter((c) => c === 'POST /api/runner/groups').length
    const before = groupPosts()
    await vi.advanceTimersByTimeAsync(60_000)
    expect(groupPosts()).toBe(before)
    vi.useRealTimers()
    // The tab is thrown away; the server had the clip after all.
    known.add(j.id)
    snapshot = { ...snapshot, rev: 7, jobs: [jobView({ id: j.id, groupId: group, device: 'x', status: 'waiting', wait: { for: 'turn' } })] }
    vi.resetModules()
    const again = await load()
    again.r.runnerStore.start()
    await vi.waitFor(() => expect(find(again.v, j.id)).toMatchObject({ cancelRequested: true, stage: 'Stopping' }), WAIT)
    await vi.waitFor(() => expect(calls).toContain(`POST /api/runner/jobs/${j.id}/stop`), WAIT)
    await vi.waitFor(() => expect(session.has(STOPS)).toBe(false), WAIT)
    expect(calls.filter((c) => c === 'POST /api/runner/groups').length).toBe(before)
  })

  it('sends the stop again once the server lists a clip stopped while it was handed over', async () => {
    let answer: (() => void) | null = null
    const { v } = await load()
    const gate = new Promise<void>((res) => (answer = res))
    const f = globalThis.fetch
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      if (url === '/api/runner/groups') await gate
      return f(url, init)
    })
    const going = v.sendClips([clip()])
    await vi.waitFor(() => expect(v.videoJobs.snapshot()).toHaveLength(1), WAIT)
    const id = v.videoJobs.snapshot()[0]!.id
    v.stopVideoJob(id)
    await vi.waitFor(() => expect(calls).toContain(`POST /api/runner/jobs/${id}/stop`), WAIT)
    answer!()
    await going
    await vi.waitFor(() => expect(calls.filter((c) => c === `POST /api/runner/jobs/${id}/stop`)).toHaveLength(2), WAIT)
  })
})

/** What reached ComfyUI to be made, whichever page sent it: nothing, while the queue on the server has the work. */
const sentToComfy = () => calls.filter((c) => c.startsWith('POST /comfy/'))

describe('the server\'s hold, as the Video desk counts and answers it', () => {
  it('counts every waiting job a hold on all the work covers, whatever its wait says', async () => {
    // A batch of ten: only its first picture says held; the rest wait for the one before, held all the same.
    const pictures = Array.from({ length: 10 }, (_, i) => jobView({ id: `p${i}`, desk: 'images', heavy: false, status: 'waiting', wait: { for: i === 0 ? 'held' : 'before' } }))
    const held = jobView({ id: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1', status: 'waiting', wait: { for: 'held' } })
    // Made after the hold began, so it did not wait through the restart and goes as usual.
    const after = jobView({ id: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa3', status: 'waiting', wait: { for: 'turn' }, createdAt: 7000 })
    snapshot = { ...snapshot, lane: { held: { why: 'restart', scope: 'all', jobId: null, since: 6000 } }, jobs: [held, after, ...pictures] }
    const { v, r } = await load()
    r.runnerStore.start()
    await vi.waitFor(() => expect(v.videoJobs.heldOnServer()).toEqual({ why: 'restart', jobId: null, clips: 1, others: 10, since: 6000 }), WAIT)
    expect(find(v, after.id)!.waitNote).not.toBe(r.HELD_AFTER_RESTART)
  })

  it('counts every heavy waiting job under a hold on heavy work, the head and those behind it, and no light one', async () => {
    const shots = Array.from({ length: 4 }, (_, i) => jobView({ id: `s${i}`, desk: 'reel', heavy: true, status: 'waiting', wait: { for: i === 0 ? 'held' : 'before' } }))
    const light = jobView({ id: 'l1', desk: 'images', heavy: false, status: 'waiting', wait: { for: 'turn' } })
    const head = jobView({ id: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa2', status: 'waiting', wait: { for: 'turn' } })
    snapshot = { ...snapshot, lane: { held: { why: 'lost', scope: 'heavy', jobId: 'x', since: 1 } }, jobs: [head, ...shots, light] }
    const { v, r } = await load()
    r.runnerStore.start()
    await vi.waitFor(() => expect(v.videoJobs.heldOnServer()).toEqual({ why: 'lost', jobId: 'x', clips: 1, others: 4, since: 1 }), WAIT)
  })

  it('is answered from the desk when it holds only other desks\' work', async () => {
    const shots = Array.from({ length: 2 }, (_, i) => jobView({ id: `s${i}`, desk: 'reel', heavy: true, status: 'waiting', wait: { for: i === 0 ? 'held' : 'before' } }))
    snapshot = { ...snapshot, lane: { held: { why: 'lost', scope: 'heavy', jobId: 'lostclip', since: 1 } }, jobs: shots }
    const { v, r } = await load()
    r.runnerStore.start()
    await vi.waitFor(() => expect(v.videoJobs.heldOnServer()).toEqual({ why: 'lost', jobId: 'lostclip', clips: 0, others: 2, since: 1 }), WAIT)
    v.videoJobs.sendHeld()
    v.videoJobs.stopHeld()
    await vi.waitFor(() => expect(bodies['/api/runner/lane']).toEqual([{ action: 'send', since: 1 }, { action: 'stop', since: 1 }]), WAIT)
  })
})

describe('the word on the server\'s hold, from the Video desk', () => {
  const held = (since: number, jobId = 'lostone') => ({ held: { why: 'lost', scope: 'heavy', jobId, since } })

  it('names the hold the desk showed, so a word for one that has since changed is not taken as one for the new hold', async () => {
    const other = jobView({ id: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa', status: 'waiting', wait: { for: 'held' } })
    snapshot = { ...snapshot, lane: held(7), jobs: [other] }
    const { v, r } = await load()
    r.runnerStore.start()
    await vi.waitFor(() => expect(v.videoJobs.heldOnServer()?.since).toBe(7), WAIT)
    await vi.waitFor(() => expect(FakeStream.last).not.toBeNull(), WAIT)
    FakeStream.last!.emit('state', { ...snapshot, rev: 41, lane: held(9, 'other') })
    await vi.waitFor(() => expect(v.videoJobs.heldOnServer()?.since).toBe(9), WAIT)
    // A notice drawn for the hold before, answered now; and the one drawn now.
    v.videoJobs.sendHeldOnServer(7)
    v.videoJobs.stopHeld()
    await vi.waitFor(() => expect(bodies['/api/runner/lane']).toEqual([{ action: 'send', since: 7 }, { action: 'stop', since: 9 }]), WAIT)
  })

  it('takes up a new hold that differs from the one before only in when it began', async () => {
    const other = jobView({ id: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaab', status: 'waiting', wait: { for: 'held' } })
    snapshot = { ...snapshot, lane: held(7), jobs: [other] }
    const { v, r } = await load()
    r.runnerStore.start()
    await vi.waitFor(() => expect(v.videoJobs.heldOnServer()?.since).toBe(7), WAIT)
    await vi.waitFor(() => expect(FakeStream.last).not.toBeNull(), WAIT)
    FakeStream.last!.emit('state', { ...snapshot, rev: 41, lane: held(9) })
    await vi.waitFor(() => expect(v.videoJobs.heldOnServer()?.since).toBe(9), WAIT)
    v.videoJobs.sendHeld()
    await vi.waitFor(() => expect(bodies['/api/runner/lane']).toEqual([{ action: 'send', since: 9 }]), WAIT)
  })
})

describe('a clip the server keeps while its queue is off', () => {
  const OFF = 'Turned off with SWITCHGEN_RUNNER=off.'

  it('is shown waiting on the queue, with the server\'s reason, and a stop is asked once, then again once the queue runs', async () => {
    const mine = jobView({ id: 'dddddddd-dddd-4ddd-8ddd-dddddddddddd', status: 'waiting', wait: { for: 'turn' } })
    snapshot = { ...snapshot, available: false, reason: OFF, jobs: [mine] }
    const f = globalThis.fetch
    // The stop route answers 503 while the queue is off, as the server's does.
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      if (url.endsWith('/stop') && !snapshot.available) {
        calls.push(`POST ${url}`)
        return json({ error: OFF, busy: 'runner', reason: OFF }, 503)
      }
      return f(url, init)
    })
    const stops = () => calls.filter((c) => c === `POST /api/runner/jobs/${mine.id}/stop`)
    const { v, r } = await load()
    r.runnerStore.start()
    await vi.waitFor(() => expect(v.videoJobs.snapshot()).toHaveLength(1), WAIT)
    expect(v.videoJobs.snapshot()[0]).toMatchObject({ stage: 'Waiting for the server’s queue', waitNote: OFF })
    v.stopVideoJob(mine.id)
    await vi.waitFor(() => expect(stops()).toHaveLength(1), WAIT)
    await vi.waitFor(() => expect(FakeStream.last).not.toBeNull(), WAIT)
    // Changes while the queue is off ask nothing again: it would only answer no.
    for (let i = 0; i < 3; i++) FakeStream.last!.emit('job', { rev: 50 + i, job: { ...mine, createdAt: 5000 + i } })
    await new Promise((res) => setTimeout(res, 30))
    expect(stops()).toHaveLength(1)
    snapshot = { ...snapshot, available: true, reason: null, rev: 60 }
    FakeStream.last!.emit('state', snapshot)
    await vi.waitFor(() => expect(stops()).toHaveLength(2), WAIT)
  })
})

describe('a second Make while the first hand-over waits for its answer', () => {
  const OUTBOX = 'switchgen.runner.outbox.v1'

  it('is kept in the tab at once, in the order pressed, and sent only in its turn', async () => {
    let answer: () => void = () => {}
    const { v, r } = await load()
    const gate = new Promise<void>((res) => (answer = res))
    const f = globalThis.fetch
    const posted: string[] = []
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      if (url === '/api/runner/groups') {
        posted.push(JSON.parse(String(init!.body)).group.id)
        await gate
      }
      return f(url, init)
    })
    const one = v.sendClips([clip()])
    await vi.waitFor(() => expect(v.videoJobs.snapshot()).toHaveLength(1), WAIT)
    const two = v.sendClips([clip(false)])
    await vi.waitFor(() => expect(v.videoJobs.snapshot()).toHaveLength(2), WAIT)
    const kept = JSON.parse(session.get(OUTBOX) ?? '[]') as { body: Runner.SubmitBody }[]
    expect(kept.map((e) => e.body.jobs[0]!.heavy)).toEqual([true, false])
    expect(kept.map((e) => e.body.group.device)).toEqual([r.deviceId(), r.deviceId()])
    // A whole state read while the first still waits does not send the second out of its turn.
    await vi.waitFor(() => expect(FakeStream.last).not.toBeNull(), WAIT)
    FakeStream.last!.emit('state', { ...snapshot, rev: 40 })
    await new Promise((res) => setTimeout(res, 30))
    expect(posted).toEqual([kept[0]!.body.group.id])
    answer()
    await one
    await two
    expect(posted).toEqual(kept.map((e) => e.body.group.id))
    expect(session.has(OUTBOX)).toBe(false)
  })

  it('is shown, and handed over, by the page loaded again after the tab was thrown away', async () => {
    const { v } = await load()
    const f = globalThis.fetch
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      if (url === '/api/runner/groups') await new Promise(() => {})
      return f(url, init)
    })
    void v.sendClips([clip()])
    await vi.waitFor(() => expect(v.videoJobs.snapshot()).toHaveLength(1), WAIT)
    void v.sendClips([clip(false)])
    await vi.waitFor(() => expect(v.videoJobs.snapshot()).toHaveLength(2), WAIT)
    const ids = v.videoJobs.snapshot().map((j) => j.id).sort()
    // The browser throws the tab away; the next page loads with what the tab kept.
    vi.resetModules()
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      if (url === '/api/runner' && (init?.method ?? 'GET') === 'GET') await new Promise(() => {})
      return f(url, init)
    })
    const again = await load()
    expect(again.v.videoJobs.snapshot().map((j) => j.id).sort()).toEqual(ids)
    expect(again.v.videoJobs.snapshot().every((j) => j.handedBefore && j.stage === 'Handing it to the server')).toBe(true)
  })
})

describe('a clip stopped before its press was handed over', () => {
  /** Hand-overs posted, each held at the door until `open`, if given. */
  const holdPosts = (gate?: Promise<void>) => {
    const posted: any[] = []
    const f = globalThis.fetch
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      if (url === '/api/runner/groups') {
        posted.push(JSON.parse(String(init!.body)))
        await (gate ?? new Promise(() => {}))
      }
      return f(url, init)
    })
    return posted
  }

  it('comes out of the tab\'s outbox, so a page loaded after never hands it over', async () => {
    const { v } = await load()
    const posted = holdPosts()
    void v.sendClips([clip()])
    await vi.waitFor(() => expect(v.videoJobs.snapshot()).toHaveLength(1), WAIT)
    void v.sendClips([clip(false)])
    await vi.waitFor(() => expect(v.videoJobs.snapshot()).toHaveLength(2), WAIT)
    const [a, b] = outboxGroups().map((e) => e.body.group.id)
    const bId = v.videoJobs.snapshot()[0]!.id
    v.stopVideoJob(bId)
    await vi.waitFor(() => expect(find(v, bId)).toMatchObject({ status: 'cancelled', stage: 'Stopped', error: 'Stopped before it was sent.' }), WAIT)
    expect(outboxGroups().map((e) => e.body.group.id)).toEqual([a])
    // Never listed, never sent: there is nothing on the server to stop.
    expect(calls).not.toContain(`POST /api/runner/jobs/${bId}/stop`)
    expect(session.has(STOPS)).toBe(false)
    // The tab is thrown away and the page loaded again.
    vi.resetModules()
    const again = await load()
    expect(again.v.videoJobs.snapshot().map((j) => j.id)).not.toContain(bId)
    again.r.runnerStore.start()
    await vi.waitFor(() => expect(posted.filter((x) => x.group.id === a)).toHaveLength(2), WAIT)
    await new Promise((res) => setTimeout(res, 50))
    expect(posted.map((x) => x.group.id)).not.toContain(b)
  })

  it('is never handed over when every clip of the press was stopped', async () => {
    let open: () => void = () => {}
    const posted = holdPosts(new Promise<void>((res) => (open = res)))
    const { v } = await load()
    const one = v.sendClips([clip()])
    await vi.waitFor(() => expect(v.videoJobs.snapshot()).toHaveLength(1), WAIT)
    const two = v.sendClips([clip(false), clip(false)])
    await vi.waitFor(() => expect(v.videoJobs.snapshot()).toHaveLength(3), WAIT)
    const [a] = outboxGroups().map((e) => e.body.group.id)
    const stopped = v.videoJobs.snapshot().slice(0, 2).map((j) => j.id)
    for (const id of stopped) v.stopVideoJob(id)
    open()
    await one
    expect(await two).toEqual({ road: 'server', ids: [], pending: false })
    expect(posted.map((x) => x.group.id)).toEqual([a])
    expect(stopped.map((id) => find(v, id)!.stage)).toEqual(['Stopped', 'Stopped'])
  })

  it('hands over only the clips of the press left', async () => {
    let open: () => void = () => {}
    const posted = holdPosts(new Promise<void>((res) => (open = res)))
    const { v } = await load()
    const one = v.sendClips([clip()])
    await vi.waitFor(() => expect(v.videoJobs.snapshot()).toHaveLength(1), WAIT)
    const two = v.sendClips([clip(false), clip(false)])
    await vi.waitFor(() => expect(v.videoJobs.snapshot()).toHaveLength(3), WAIT)
    const [stopped, kept] = v.videoJobs.snapshot().slice(0, 2).map((j) => j.id)
    v.stopVideoJob(stopped!)
    open()
    await one
    await two
    expect(posted[1].jobs.map((j: { id: string }) => j.id)).toEqual([kept])
    expect(find(v, stopped!)).toMatchObject({ status: 'cancelled', error: 'Stopped before it was sent.' })
    expect(find(v, kept!)!.status).not.toBe('cancelled')
  })

  it('says of a clip an earlier page pressed but never handed over that it was never sent', async () => {
    const SENT = '99999999-9999-4999-8999-999999999999'
    const J1 = '11111111-1111-4111-8111-111111111111'
    const J2 = '22222222-2222-4222-8222-222222222222'
    const entry = (group: string, id: string, sent: boolean) => ({
      at: Date.now() - 1000,
      sent,
      body: { v: 1, group: { id: group, desk: 'video', kind: 'clips', label: 'Wan', device: 'dev' }, jobs: [{ id, label: 'Wan', prompt: 'x', kind: 'video', primary: 'video', orFirst: true, noFile: 'done', heavy: true, graph: {}, record: {}, meta: {} }] },
    })
    session.set(OUTBOX, JSON.stringify([entry(SENT, J1, true), entry('88888888-8888-4888-8888-888888888888', J2, false)]))
    holdPosts()
    const { v } = await load()
    expect(find(v, J1)!.waitNote).toMatch(/did not hear back/)
    expect(find(v, J2)!.waitNote).toMatch(/^Make was pressed on an earlier page/)
    v.stopVideoJob(J2)
    await vi.waitFor(() => expect(find(v, J2)).toMatchObject({ status: 'cancelled', error: 'Stopped before it was sent.' }), WAIT)
    expect(outboxGroups().map((e) => e.body.group.id)).toEqual([SENT])
  })
})

describe('a clip put away on another device before this page heard back', () => {
  it('goes from a card still handing it over, and does not come back when the answer does', async () => {
    let answer: () => void = () => {}
    const { v, r } = await load()
    const gate = new Promise<void>((res) => (answer = res))
    const f = globalThis.fetch
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      if (url === '/api/runner/groups') await gate
      return f(url, init)
    })
    const going = v.sendClips([clip()])
    await vi.waitFor(() => expect(v.videoJobs.snapshot()).toHaveLength(1), WAIT)
    const id = v.videoJobs.snapshot()[0]!.id
    const group = JSON.parse(session.get('switchgen.runner.outbox.v1')!)[0].body.group.id
    await vi.waitFor(() => expect(FakeStream.last).not.toBeNull(), WAIT)
    FakeStream.last!.emit('state', { ...snapshot, rev: 40, jobs: [jobView({ id, groupId: group, device: r.deviceId(), status: 'done', wait: null, promptId: 'p', entryId: id, finishedAt: 9, endedAt: 9, dismissed: true })] })
    await vi.waitFor(() => expect(v.videoJobs.snapshot()).toEqual([]), WAIT)
    answer()
    await going
    await new Promise((res) => setTimeout(res, 20))
    expect(v.videoJobs.snapshot()).toEqual([])
  })
})

describe('Make while ComfyUI is not answering', () => {
  it('is refused when the clip would go from the page', async () => {
    caps = { runner: false, runnerDesks: [], runnerReason: 'Turned off with SWITCHGEN_RUNNER=off.' }
    const { v } = await load()
    expect(await v.sendClips([clip()], { offline: true })).toEqual({
      road: 'refused',
      error: 'ComfyUI is not answering, and the queue on the server is not taking these clips, so nothing can be queued.',
    })
    expect(v.videoJobs.snapshot()).toEqual([])
    expect(sentToComfy()).toEqual([])
  })

  it('goes to the queue on the server, which sends it once ComfyUI answers', async () => {
    const { v } = await load()
    const sent = await v.sendClips([clip()], { offline: true })
    expect(sent.road).toBe('server')
    expect(sentToComfy()).toEqual([])
  })

  it('is refused, not sent from the page, when the queue turns it back', async () => {
    groupsAnswer = () => json({ error: 'not running', busy: 'runner', reason: 'Another SwitchGen server holds the archive, and the queue with it.' }, 503)
    const { v } = await load()
    expect(await v.sendClips([clip()], { offline: true })).toMatchObject({ road: 'refused' })
    expect(v.videoJobs.snapshot()).toEqual([])
    expect(sentToComfy()).toEqual([])
  })
})

describe('clips this tab was handing over when the page went', () => {
  const GROUP = '99999999-9999-4999-8999-999999999999'
  const J1 = '11111111-1111-4111-8111-111111111111'
  const J2 = '22222222-2222-4222-8222-222222222222'
  const handedAt = (at: number) => {
    const c = clip()
    return {
      at,
      body: {
        v: 1,
        group: { id: GROUP, desk: 'video', kind: 'clips', label: 'Wan 2.2 14B', device: 'dev-here' },
        jobs: [J1, J2].map((id) => ({ id, label: 'Wan 2.2 14B', prompt: 'a tram at dusk', kind: 'video', primary: 'video', orFirst: true, noFile: 'done', heavy: true, graph: c.graph, record: {}, meta: { frames: 81, fps: 16 } })),
      },
    }
  }
  /** The replay the next page sends is held at the door until the test opens it. */
  const gated = () => {
    let open: () => void = () => {}
    const gate = new Promise<void>((res) => (open = res))
    const f = globalThis.fetch
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      if (url === '/api/runner/groups') await gate
      return f(url, init)
    })
    return () => open()
  }

  it('shows each as being handed over until the server lists it, and keeps a stop pressed meanwhile', async () => {
    session.set(OUTBOX, JSON.stringify([handedAt(Date.now() - 1000)]))
    const open = gated()
    const { v, r } = await load()
    expect(v.videoJobs.snapshot().map((j) => j.id)).toEqual([J2, J1])
    const card = v.videoJobs.snapshot()[0]!
    expect(card).toMatchObject({ status: 'submitting', stage: 'Handing it to the server', runner: true, sentHere: true, handedBefore: true, familyLabel: 'Wan 2.2 14B' })
    expect(card.waitNote).toMatch(/earlier page in this tab/)
    v.stopVideoJob(J1)
    await vi.waitFor(() => expect(find(v, J1)!.stage).toBe('Stopping'), WAIT)
    open()
    await vi.waitFor(() => expect(r.runnerStore.snapshot().jobs.map((j) => j.id).sort()).toEqual([J1, J2].sort()), WAIT)
    await vi.waitFor(() => expect(v.videoJobs.snapshot().every((j) => !j.handedBefore && j.adopted && j.prompt === 'a tram at dusk')).toBe(true), WAIT)
    expect(v.videoJobs.snapshot()).toHaveLength(2)
    expect(find(v, J1)).toMatchObject({ cancelRequested: true, stage: 'Stopping' })
    await vi.waitFor(() => expect(calls.filter((c) => c === `POST /api/runner/jobs/${J1}/stop`)).toHaveLength(2), WAIT)
    expect(sentToComfy()).toEqual([])
  })

  it('lets the cards go when the server lists their clips as ended and put away, and sends nothing again', async () => {
    session.set(OUTBOX, JSON.stringify([handedAt(Date.now() - 1000)]))
    snapshot = {
      ...snapshot,
      groups: [{ id: GROUP, desk: 'video', kind: 'clips', label: 'Wan', device: 'dev-here', createdAt: 1, state: 'ended', endedBy: null, endedAt: 9, jobIds: [J1, J2], dismissed: true }],
      jobs: [J1, J2].map((id) => jobView({ id, groupId: GROUP, device: 'dev-here', status: 'done', wait: null, promptId: 'p', entryId: id, finishedAt: 9, endedAt: 9, dismissed: true })),
    }
    // The first read of the server's state comes only once the cards are shown.
    let openRead: () => void = () => {}
    const readGate = new Promise<void>((res) => (openRead = res))
    const f = globalThis.fetch
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      if (url === '/api/runner' && (init?.method ?? 'GET') === 'GET') await readGate
      return f(url, init)
    })
    const { v, r } = await load()
    expect(v.videoJobs.snapshot().map((j) => j.stage)).toEqual(['Handing it to the server', 'Handing it to the server'])
    r.runnerStore.start()
    openRead()
    await vi.waitFor(() => expect(v.videoJobs.snapshot()).toEqual([]), WAIT)
    await vi.waitFor(() => expect(session.has(OUTBOX)).toBe(false), WAIT)
    expect(calls.filter((x) => x === 'POST /api/runner/groups')).toEqual([])
  })

  it('says on each card when the server refuses them', async () => {
    session.set(OUTBOX, JSON.stringify([handedAt(Date.now() - 1000)]))
    groupsAnswer = () => json({ error: 'That is not a clip.' }, 400)
    const open = gated()
    const { v } = await load()
    expect(v.videoJobs.snapshot()).toHaveLength(2)
    open()
    await vi.waitFor(() => expect(v.videoJobs.snapshot().every((j) => j.status === 'error' && j.stage === 'Not sent' && j.error === 'That is not a clip.')).toBe(true), WAIT)
    expect(v.videoJobs.givenUp()).toEqual([])
  })

  it('shows none for a hand-over too old to send again, and says it was given up', async () => {
    session.set(OUTBOX, JSON.stringify([handedAt(Date.now() - 11 * 60_000)]))
    const { v, r } = await load()
    r.runnerStore.start()
    expect(v.videoJobs.snapshot()).toHaveLength(0)
    await vi.waitFor(() => expect(v.videoJobs.givenUp()).toHaveLength(1), WAIT)
    expect(v.videoJobs.snapshot()).toHaveLength(0)
  })
})
