import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type * as RunnerMod from '../src/lib/runner'
import { newComposition, recordOf } from '../src/lib/session'

/**
 * The page's side of the queue on the server (src/lib/runner.ts): the store
 * the stream keeps current, the hand-over with its outbox, following a job,
 * and the words. fetch, EventSource and the tab's storage are stood in; no
 * request leaves the test.
 */

type R = typeof RunnerMod

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })
const html = (status = 200) => new Response('<!doctype html>', { status, headers: { 'content-type': 'text/html' } })

class FakeStorage {
  m = new Map<string, string>()
  getItem(k: string) {
    return this.m.get(k) ?? null
  }
  setItem(k: string, v: string) {
    this.m.set(k, String(v))
  }
  removeItem(k: string) {
    this.m.delete(k)
  }
}

class FakeES {
  static all: FakeES[] = []
  readyState = 1
  handlers = new Map<string, ((ev: { data: string }) => void)[]>()
  onerror: (() => void) | null = null
  closed = false
  constructor(public url: string) {
    FakeES.all.push(this)
  }
  addEventListener(name: string, fn: (ev: { data: string }) => void) {
    this.handlers.set(name, [...(this.handlers.get(name) ?? []), fn])
  }
  emit(name: string, data: unknown) {
    for (const fn of this.handlers.get(name) ?? []) fn({ data: JSON.stringify(data) })
  }
  close() {
    this.closed = true
    this.readyState = 2
  }
}

const job = (over: Partial<RunnerMod.RunnerJob> = {}): RunnerMod.RunnerJob => ({
  id: 'j1',
  groupId: 'g1',
  desk: 'video',
  kind: 'video',
  seq: 1,
  index: 1,
  total: 1,
  label: 'Wan',
  prompt: 'p',
  device: 'd',
  heavy: true,
  status: 'waiting',
  wait: null,
  stopRequested: false,
  stopLanded: false,
  promptId: null,
  attempt: 0,
  createdAt: 1,
  sentAt: null,
  ranAt: null,
  finishedAt: null,
  endedAt: null,
  files: [],
  primary: null,
  frame: null,
  openedOn: null,
  entryId: null,
  entryNo: null,
  repeatOf: null,
  durationMs: 0,
  error: null,
  meta: null,
  dismissed: false,
  ...over,
})
const group = (over: Partial<RunnerMod.RunnerGroup> = {}): RunnerMod.RunnerGroup => ({
  id: 'g1',
  desk: 'video',
  kind: 'clips',
  label: 'x',
  device: 'd',
  createdAt: 1,
  state: 'active',
  endedBy: null,
  endedAt: null,
  jobIds: ['j1'],
  dismissed: false,
  ...over,
})
const snap = (over: Record<string, unknown> = {}) => ({
  v: 1,
  available: true,
  reason: null,
  boot: 'b1',
  rev: 5,
  comfy: { answering: true, since: 0 },
  lane: { held: null },
  groups: [],
  jobs: [],
  progress: {},
  ...over,
})

const G1 = '11111111-1111-4111-8111-111111111111'
const J1 = '22222222-2222-4222-8222-222222222222'
const OUTBOX = 'switchgen.runner.outbox.v1'
const WAIT = { timeout: 5000, interval: 5 }

let session: FakeStorage
let local: FakeStorage
let fetchStub: ReturnType<typeof vi.fn>
let r: R

async function load(): Promise<R> {
  vi.resetModules()
  return (await import('../src/lib/runner')) as R
}

beforeEach(async () => {
  session = new FakeStorage()
  local = new FakeStorage()
  FakeES.all = []
  vi.stubGlobal('sessionStorage', session)
  vi.stubGlobal('localStorage', local)
  vi.stubGlobal('EventSource', FakeES)
  fetchStub = vi.fn(async () => json(snap()))
  vi.stubGlobal('fetch', fetchStub)
  r = await load()
})
afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

const body = (id = G1, device = 'dev'): RunnerMod.SubmitBody => ({
  v: 1,
  group: { id, desk: 'video', kind: 'clips', label: 'x', device },
  jobs: [{ id: J1, label: 'a', prompt: 'p', kind: 'video', primary: 'video', orFirst: true, noFile: 'done', heavy: true, graph: {}, record: {} as never }],
})
const taken = (rev = 6) => json({ rev, replayed: false, group: group({ id: G1, jobIds: [J1] }), jobs: [job({ id: J1, groupId: G1 })] })
const groupsCalls = () => fetchStub.mock.calls.filter(([u]) => u === '/api/runner/groups')

describe('the record a desk hands over', () => {
  it('is recordOf less the six fields the server fills, with no field left undefined', () => {
    const c = newComposition('video', { prompt: 'a cat' })
    const t = r.recordTemplate(c, { seed: 7, familyLabel: 'F', modelLabel: 'M' })
    for (const k of ['file', 'files', 'kind', 'promptId', 'durationMs', 'at']) expect(k in t, k).toBe(false)
    expect(t).toMatchObject({ desk: 'video', mode: c.mode, seed: 7 })
    const full = recordOf(c, { seed: 7, familyLabel: 'F', modelLabel: 'M', file: { filename: 'x', subfolder: '', type: 'output' }, kind: 'video', promptId: 'p', durationMs: 3 })
    const { file: _a, files: _b, kind: _c, promptId: _d, durationMs: _e, at: _f, ...rest } = full
    expect(t).toEqual(JSON.parse(JSON.stringify(rest)))
    expect(Object.values(t)).not.toContain(undefined)
  })
})

describe('handing a group over', () => {
  it('keeps it in the tab before the POST goes, and lets it go on a 200', async () => {
    let inTabAtPost = false
    let headers: Record<string, string> = {}
    fetchStub.mockImplementation(async (url: string, init?: RequestInit) => {
      if (url !== '/api/runner/groups') return json(snap())
      inTabAtPost = (session.getItem(OUTBOX) ?? '').includes(G1)
      headers = init!.headers as Record<string, string>
      return taken()
    })
    const res = await r.submitGroup(body())
    expect(inTabAtPost).toBe(true)
    expect(headers).toMatchObject({ 'Content-Type': 'application/json', 'X-SwitchGen-Device': r.deviceId() })
    expect(res).toMatchObject({ ok: true, replayed: false, group: { id: G1 } })
    expect(session.getItem(OUTBOX)).toBeNull()
    expect(r.outboxPending()).toEqual([])
    // The answer is taken in at its rev.
    expect(r.runnerStore.snapshot()).toMatchObject({ rev: 6, jobs: [{ id: J1 }] })
  })

  it('names this browser when the desk left the device blank', async () => {
    fetchStub.mockImplementation(async (url: string, init?: RequestInit) => (url === '/api/runner/groups' ? (JSON.parse(String(init!.body)).group.device === r.deviceId() ? taken() : json({ error: 'x' }, 400)) : json(snap())))
    expect((await r.submitGroup(body(G1, ''))).ok).toBe(true)
  })

  it('asks again with the same ids while nothing answers, then says pending and keeps it', async () => {
    vi.useFakeTimers()
    fetchStub.mockImplementation(async () => {
      throw new TypeError('fetch failed')
    })
    const p = r.submitGroup(body())
    await vi.advanceTimersByTimeAsync(40_000)
    expect(await p).toEqual({ ok: false, fallback: false, pending: true })
    const bodies = groupsCalls().map(([, init]) => String((init as RequestInit).body))
    expect(bodies.length).toBeGreaterThan(2)
    expect(new Set(bodies).size).toBe(1)
    expect(session.getItem(OUTBOX)).toContain(G1)
    expect(r.outboxPending()).toMatchObject([{ groupId: G1, desk: 'video', jobIds: [J1] }])
  })

  it('never falls back to the page after an attempt with no clear answer', async () => {
    vi.useFakeTimers()
    let n = 0
    fetchStub.mockImplementation(async (url: string) => {
      if (url !== '/api/runner/groups') return json(snap())
      if (n++ === 0) throw new TypeError('fetch failed')
      return json({ error: 'not running', busy: 'runner', reason: 'The queue on the server is not running.' }, 503)
    })
    const p = r.submitGroup(body())
    await vi.advanceTimersByTimeAsync(1500)
    expect(await p).toEqual({ ok: false, fallback: false, pending: true })
    expect(session.getItem(OUTBOX)).toContain(G1)
    expect(r.givenUpBatches()).toEqual([])
    // Sent again on its own a second later, after a read that shows the
    // server does not have it; the queue says no, so it is given up with its
    // line, and the desk, not this library, says what happens to the work.
    await vi.advanceTimersByTimeAsync(5000)
    expect(r.givenUpBatches()).toMatchObject([{ groupId: G1, line: r.OUTBOX_NOT_TAKEN }])
    expect(session.getItem(OUTBOX)).toBeNull()
  })

  it('asks again after a gateway\'s 502 and takes the answer that follows', async () => {
    vi.useFakeTimers()
    let n = 0
    fetchStub.mockImplementation(async (url: string) => (url !== '/api/runner/groups' ? json(snap()) : n++ === 0 ? new Response('bad gateway', { status: 502 }) : taken(9)))
    const p = r.submitGroup(body())
    await vi.advanceTimersByTimeAsync(2000)
    expect(await p).toMatchObject({ ok: true })
    expect(n).toBe(2)
  })

  it('runs in the page where the server has no queue, and is refused where the queue said no', async () => {
    const cases: [() => Response, object][] = [
      [() => json({ error: 'no' }, 404), { ok: false, fallback: true, reason: 'no' }],
      [() => json({ error: 'x' }, 405), { ok: false, fallback: true }],
      [() => json({ error: 'x' }, 501), { ok: false, fallback: true }],
      [() => json({ error: 'x', busy: 'runner', reason: 'Turned off with SWITCHGEN_RUNNER=off.' }, 503), { ok: false, fallback: true, reason: 'Turned off with SWITCHGEN_RUNNER=off.' }],
      [() => html(200), { ok: false, fallback: true, reason: 'This server has no queue of its own.' }],
      [() => html(404), { ok: false, fallback: true }],
      [() => json({ error: 'A batch is running.', busy: 'images' }, 409), { ok: false, fallback: false, status: 409, error: 'A batch is running.', busy: 'images' }],
      [() => json({ error: 'bad uuid' }, 400), { ok: false, fallback: false, status: 400, error: 'bad uuid' }],
      [() => json({}, 422), { ok: false, fallback: false, status: 422, error: 'The server would not take this work (HTTP 422).' }],
      [() => json({ error: 'The server’s disk is full, so it cannot save this work. Nothing was taken.' }, 507), { ok: false, fallback: false, status: 507 }],
    ]
    for (const [answer, want] of cases) {
      fetchStub.mockImplementation(async (url: string) => (url === '/api/runner/groups' ? answer() : json(snap())))
      expect(await r.submitGroup(body()), JSON.stringify(want)).toMatchObject(want)
      expect(session.getItem(OUTBOX)).toBeNull()
    }
  })

  it('says why the page sends its own work, lowering a plain first word and keeping a name', () => {
    expect(r.fallbackLine('Turned off with SWITCHGEN_RUNNER=off.')).toBe('This page sends the work itself: turned off with SWITCHGEN_RUNNER=off.')
    expect(r.fallbackLine('ComfyUI is down.')).toBe('This page sends the work itself: ComfyUI is down.')
  })
})

describe('a hand-over staged while an earlier one waits for its answer', () => {
  const G2 = '55555555-5555-4555-8555-555555555555'
  const G3 = '66666666-6666-4666-8666-666666666666'
  /** A group with a job id of its own. */
  const bodyOf = (id: string, device = 'dev', label = 'x'): RunnerMod.SubmitBody => {
    const b = body(id, device)
    return { ...b, group: { ...b.group, label }, jobs: b.jobs.map((j) => ({ ...j, id: `${id.slice(0, 8)}-0000-4000-8000-000000000000` })) }
  }
  const inTab = () => JSON.parse(session.getItem(OUTBOX) ?? '[]') as { at: number; body: RunnerMod.SubmitBody }[]

  it('is in the tab at once, in the order pressed, and keeps its place and time when staged again', async () => {
    const t = Date.now()
    const now = vi.spyOn(Date, 'now')
    now.mockReturnValue(t - 60_000)
    r.stage(bodyOf(G1))
    now.mockReturnValue(t - 30_000)
    r.stage(bodyOf(G2, ''))
    now.mockReturnValue(t)
    r.stage(bodyOf(G1, 'dev', 'again'))
    now.mockRestore()
    expect(inTab().map((e) => [e.body.group.id, e.at, e.body.group.label])).toEqual([
      [G1, t - 60_000, 'again'],
      [G2, t - 30_000, 'x'],
    ])
    expect(inTab()[1]!.body.group.device).toBe(r.deviceId())
    expect(fetchStub).not.toHaveBeenCalled()
    // A read of the state sends none of it: each goes by its own submitGroup, in its turn.
    await r.runnerStore.refresh()
    expect(groupsCalls()).toEqual([])
    expect(r.outboxPending().map((e) => e.groupId)).toEqual([G1, G2])
  })

  it('is sent, in the order pressed, by the page loaded again after the tab was thrown away', async () => {
    r.stage(bodyOf(G1))
    r.stage(bodyOf(G2))
    const posted: string[] = []
    fetchStub.mockImplementation(async (url: string, init?: RequestInit) => {
      if (url !== '/api/runner/groups') return json(snap())
      const id = JSON.parse(String(init!.body)).group.id as string
      posted.push(id)
      return json({ rev: 6, replayed: false, group: group({ id, jobIds: [] }), jobs: [] })
    })
    const m = await load()
    await vi.waitFor(() => expect(m.outboxPending()).toEqual([]), WAIT)
    expect(posted).toEqual([G1, G2])
  })

  it('stays where it was staged when its own hand-over goes', async () => {
    r.stage(bodyOf(G1))
    r.stage(bodyOf(G2))
    r.stage(bodyOf(G3))
    let atPost: string[] = []
    fetchStub.mockImplementation(async (url: string) => {
      if (url !== '/api/runner/groups') return json(snap())
      atPost = inTab().map((e) => e.body.group.id)
      return json({ error: 'bad uuid' }, 400)
    })
    expect(await r.submitGroup(bodyOf(G2))).toMatchObject({ ok: false, fallback: false, status: 400 })
    expect(atPost).toEqual([G1, G2, G3])
    expect(inTab().map((e) => e.body.group.id)).toEqual([G1, G3])
  })
})

describe('a hand-over left in the tab by a page that went', () => {
  const young = () => ({ at: Date.now() - 60_000, body: body('33333333-3333-4333-8333-333333333333') })
  const old = () => ({ at: Date.now() - 11 * 60_000, body: body('44444444-4444-4444-8444-444444444444') })

  it('is sent again under ten minutes, and dropped with its line after', async () => {
    session.setItem(OUTBOX, JSON.stringify([old(), young()]))
    const posted: string[] = []
    fetchStub.mockImplementation(async (url: string, init?: RequestInit) => {
      if (url !== '/api/runner/groups') return json(snap())
      posted.push(JSON.parse(String(init?.body)).group.id)
      return json({ rev: 6, replayed: true, group: group({ id: '33333333-3333-4333-8333-333333333333' }), jobs: [] })
    })
    const m = await load()
    await vi.waitFor(() => expect(m.outboxPending()).toHaveLength(0), WAIT)
    expect(posted).toEqual(['33333333-3333-4333-8333-333333333333'])
    expect(m.givenUpBatches()).toMatchObject([{ groupId: '44444444-4444-4444-8444-444444444444', line: m.OUTBOX_GIVEN_UP }])
    expect(m.OUTBOX_GIVEN_UP).toBe('The server never answered for this batch, so it was not sent.')
    expect(session.getItem(OUTBOX)).toBeNull()
    m.forgetGivenUp('44444444-4444-4444-8444-444444444444')
    expect(m.givenUpBatches()).toEqual([])
  })

  it('is dropped with the server\'s own words when refused, and with a line of its own when there is no queue', async () => {
    const refused = young()
    session.setItem(OUTBOX, JSON.stringify([refused]))
    fetchStub.mockImplementation(async (url: string) => (url === '/api/runner/groups' ? json({ error: 'group.label must be a string' }, 400) : json(snap())))
    let m = await load()
    await vi.waitFor(() => expect(m.givenUpBatches()).toMatchObject([{ line: 'group.label must be a string' }]), WAIT)

    session.setItem(OUTBOX, JSON.stringify([young()]))
    fetchStub.mockImplementation(async (url: string) => (url === '/api/runner/groups' ? json({ error: 'x', reason: 'off' }, 503) : json(snap())))
    m = await load()
    await vi.waitFor(() => expect(m.givenUpBatches()).toMatchObject([{ line: m.OUTBOX_NOT_TAKEN }]), WAIT)
  })

  it('stays while nothing answers, and goes again once the stream says where the server is', async () => {
    session.setItem(OUTBOX, JSON.stringify([young()]))
    let up = false
    const posted: string[] = []
    fetchStub.mockImplementation(async (url: string) => {
      if (url === '/api/runner/groups') {
        posted.push(url)
        if (!up) throw new TypeError('fetch failed')
        return json({ rev: 6, replayed: false, group: group({ id: '33333333-3333-4333-8333-333333333333' }), jobs: [] })
      }
      return json(snap())
    })
    const m = await load()
    await vi.waitFor(() => expect(posted).toHaveLength(1), WAIT)
    expect(m.outboxPending()).toHaveLength(1)
    up = true
    m.runnerStore.start()
    await vi.waitFor(() => expect(FakeES.all).toHaveLength(1), WAIT)
    FakeES.all[0]!.emit('state', snap())
    await vi.waitFor(() => expect(m.outboxPending()).toHaveLength(0), WAIT)
    expect(m.givenUpBatches()).toEqual([])
  })

  it('says of one its page pressed but never sent that it was never handed over', async () => {
    session.setItem(OUTBOX, JSON.stringify([{ ...old(), sent: false }]))
    const m = await load()
    await vi.waitFor(() => expect(m.givenUpBatches()).toMatchObject([{ groupId: '44444444-4444-4444-8444-444444444444', line: m.OUTBOX_TOO_OLD }]), WAIT)
    expect(m.OUTBOX_TOO_OLD).toBe('The page closed before it handed this batch over, and it is now too old to send.')
    expect(groupsCalls()).toEqual([])
  })

  it('says whether a send of it has left: not when staged, and from its first send on', async () => {
    vi.useFakeTimers()
    fetchStub.mockImplementation(async (url: string) => {
      if (url === '/api/runner/groups') throw new TypeError('fetch failed')
      return json(snap())
    })
    r.stage(body())
    expect(r.outboxPending()).toMatchObject([{ groupId: G1, sent: false }])
    expect(JSON.parse(session.getItem(OUTBOX)!)).toMatchObject([{ sent: false }])
    const p = r.submitGroup(body())
    await vi.advanceTimersByTimeAsync(10)
    expect(r.outboxPending()).toMatchObject([{ groupId: G1, sent: true }])
    expect(JSON.parse(session.getItem(OUTBOX)!)).toMatchObject([{ sent: true }])
    await vi.advanceTimersByTimeAsync(40_000)
    expect(await p).toMatchObject({ pending: true })
  })
})

describe('work the reader stopped before the server listed it', () => {
  const A = 'aaaaaaaa-1111-4111-8111-111111111111'
  const B = 'bbbbbbbb-1111-4111-8111-111111111111'
  const WITHDRAWN = 'switchgen.runner.withdrawn.v1'
  /** A group of two clips. */
  const pair = (): RunnerMod.SubmitBody => {
    const b = body()
    return { ...b, jobs: [A, B].map((id) => ({ ...b.jobs[0]!, id })) }
  }
  const takenAs = (ids: string[]) => json({ rev: 6, replayed: false, group: group({ id: G1, jobIds: ids }), jobs: ids.map((id) => job({ id, groupId: G1 })) })
  /** The job ids of each hand-over POSTed, in order. */
  const posted = () => groupsCalls().map(([, init]) => (JSON.parse(String((init as RequestInit).body)) as RunnerMod.SubmitBody).jobs.map((j) => j.id))
  const inTab = () => (JSON.parse(session.getItem(OUTBOX) ?? '[]') as { body: RunnerMod.SubmitBody }[]).map((e) => e.body.jobs.map((j) => j.id))

  it('comes out of a staged hand-over, in the tab too, and the desk\'s own submitGroup with the old body sends only what is left', async () => {
    r.stage(pair())
    expect(r.outboxPending()).toMatchObject([{ groupId: G1, jobIds: [A, B], sent: false }])
    r.withdraw(G1, [A])
    expect(r.outboxPending()).toMatchObject([{ groupId: G1, jobIds: [B], sent: false }])
    expect(inTab()).toEqual([[B]])
    expect(JSON.parse(session.getItem(WITHDRAWN)!)).toEqual([A])
    fetchStub.mockImplementation(async (url: string) => (url === '/api/runner/groups' ? takenAs([B]) : json(snap())))
    expect(await r.submitGroup(pair())).toMatchObject({ ok: true })
    expect(posted()).toEqual([[B]])
  })

  it('takes the hand-over out whole when all of it is withdrawn, and submitGroup then sends nothing', async () => {
    r.stage(pair())
    r.withdraw(G1, [A, B])
    expect(r.outboxPending()).toEqual([])
    expect(session.getItem(OUTBOX)).toBeNull()
    expect(await r.submitGroup(pair())).toEqual({ ok: false, fallback: false, pending: true })
    // A group withdrawn by its own id, before it was ever staged.
    const G2 = '55555555-5555-4555-8555-555555555555'
    r.withdraw(G2)
    expect(await r.submitGroup(body(G2))).toEqual({ ok: false, fallback: false, pending: true })
    expect(posted()).toEqual([])
    expect(r.stage(body(G2))).toBeNull()
    expect(r.outboxPending()).toEqual([])
  })

  it('is not sent by a hand-over still being asked about', async () => {
    vi.useFakeTimers()
    fetchStub.mockImplementation(async (url: string) => {
      if (url === '/api/runner/groups') throw new TypeError('fetch failed')
      return json(snap())
    })
    const p = r.submitGroup(pair())
    await vi.advanceTimersByTimeAsync(10)
    r.withdraw(G1, [A])
    await vi.advanceTimersByTimeAsync(1500)
    r.withdraw(G1)
    await vi.advanceTimersByTimeAsync(40_000)
    expect(await p).toMatchObject({ pending: true })
    // The first try had both; the one after, only B; after that, nothing.
    expect(posted()).toEqual([[A, B], [B]])
    expect(r.outboxPending()).toEqual([])
  })

  it('is kept in the tab: the page loaded after sends only what is left, and stages the old body without it', async () => {
    r.stage(pair())
    r.withdraw(G1, [A])
    fetchStub.mockImplementation(async (url: string) => (url === '/api/runner/groups' ? takenAs([B]) : json(snap())))
    const m = await load()
    await vi.waitFor(() => expect(m.outboxPending()).toEqual([]), WAIT)
    expect(posted()).toEqual([[B]])
    expect(m.stage(pair())!.jobs.map((j) => j.id)).toEqual([B])
  })

  it('takes the server\'s own list as the answer when an earlier send, whose answer was lost, got there with all of it', async () => {
    vi.useFakeTimers()
    let n = 0
    let listed = false
    fetchStub.mockImplementation(async (url: string) => {
      if (url === '/api/runner/groups') {
        if (n++ === 0) throw new TypeError('the answer was lost')
        listed = true
        return json({ error: 'A different group with this id is already here.', conflict: 'id' }, 409)
      }
      return json(listed ? snap({ rev: 7, groups: [group({ id: G1, jobIds: [A, B] })], jobs: [job({ id: A, groupId: G1 }), job({ id: B, groupId: G1 })] }) : snap())
    })
    const p = r.submitGroup(pair())
    await vi.advanceTimersByTimeAsync(10)
    r.withdraw(G1, [A])
    await vi.advanceTimersByTimeAsync(5000)
    // What the server has is the desk's to stop; the hand-over is settled.
    expect(await p).toMatchObject({ ok: true, replayed: true, group: { id: G1 }, jobs: [{ id: A }, { id: B }] })
    expect(r.outboxPending()).toEqual([])
  })
})

describe('a hand-over the reader stopped while the page that came back was sending it', () => {
  const G2 = '66666666-6666-4666-8666-666666666666'
  const J2 = '77777777-7777-4777-8777-777777777777'
  /** A batch left in the tab, already sent once, by the page that went. */
  const left = (id: string, jobId: string): RunnerMod.SubmitBody => {
    const b = body(id)
    return { ...b, group: { ...b.group, desk: 'images', kind: 'batch' }, jobs: [{ ...b.jobs[0]!, id: jobId }] }
  }
  /**
   * Load the page with two such batches in its tab, G1 then G2, and hold the
   * replay's POST of G1 until the test answers it. G2's is answered as G1's
   * is, once G1's has been dealt with: the replay sends one at a time, in
   * order, so G2 given up says G1's answer has been taken.
   */
  async function replaying(res: () => Response) {
    session.setItem(OUTBOX, JSON.stringify([G1, G2].map((id, i) => ({ at: Date.now(), body: left(id, [J1, J2][i]!), sent: true }))))
    let answer: () => void = () => {}
    fetchStub.mockImplementation(async (url: string, init?: RequestInit) => {
      if (url !== '/api/runner/groups') return json(snap())
      if ((JSON.parse(String(init?.body)) as RunnerMod.SubmitBody).group.id !== G1) return res()
      return new Promise<Response>((done) => (answer = () => done(res())))
    })
    const m = await load()
    m.runnerStore.start()
    await vi.waitFor(() => expect(groupsCalls()).toHaveLength(1), WAIT)
    return { m, answer: () => answer() }
  }

  for (const [said, res] of [
    ['the queue had stopped', () => json({ error: 'x', reason: 'The queue on the server is not running.' }, 503)],
    ['a refusal', () => json({ error: 'No.' }, 400)],
  ] as const) {
    it(`gives nothing up when the answer is ${said}, since the desk has already said it stopped`, async () => {
      const { m, answer } = await replaying(res)
      m.withdraw(G1)
      answer()
      await vi.waitFor(() => expect(m.givenUpBatches().map((g) => g.groupId)).toContain(G2), WAIT)
      expect(m.givenUpBatches().map((g) => g.groupId)).toEqual([G2])
    })
  }

  it('is still given up, with its line, when the reader did not stop it', async () => {
    const { m, answer } = await replaying(() => json({ error: 'x', reason: 'The queue on the server is not running.' }, 503))
    answer()
    await vi.waitFor(() => expect(m.givenUpBatches().map((g) => g.groupId)).toEqual([G1, G2]), WAIT)
    expect(m.givenUpBatches()[0]).toMatchObject({ groupId: G1, line: m.OUTBOX_NOT_TAKEN })
  })
})

describe('a hand-over that got no answer', () => {
  it('is sent again on its own while the stream stays up with nothing to say', async () => {
    vi.useFakeTimers()
    let up = false
    fetchStub.mockImplementation(async (url: string) => {
      if (url === '/api/runner/groups') {
        if (!up) throw new TypeError('fetch failed')
        return taken()
      }
      return json(snap())
    })
    r.runnerStore.start()
    await vi.advanceTimersByTimeAsync(10)
    FakeES.all[0]!.emit('state', snap())
    const p = r.submitGroup(body())
    await vi.advanceTimersByTimeAsync(40_000)
    expect(await p).toMatchObject({ pending: true })
    const before = groupsCalls().length
    up = true
    // No state, no event: the page hears nothing from the server meanwhile.
    await vi.advanceTimersByTimeAsync(31_000)
    expect(groupsCalls().length).toBe(before + 1)
    expect(r.outboxPending()).toEqual([])
    expect(r.runnerStore.snapshot().jobs.map((j) => j.id)).toEqual([J1])
  })

  it('is sent again a second later, then two, four, and so on up to half a minute, and given up only once too old', async () => {
    vi.useFakeTimers()
    const t0 = Date.now()
    const at: number[] = []
    fetchStub.mockImplementation(async (url: string) => {
      if (url === '/api/runner/groups') {
        at.push(Date.now() - t0)
        throw new TypeError('fetch failed')
      }
      return json(snap())
    })
    let pendingAt = -1
    const p = r.submitGroup(body()).then((x) => {
      pendingAt = Date.now() - t0
      return x
    })
    await vi.advanceTimersByTimeAsync(5 * 60_000)
    expect(await p).toMatchObject({ pending: true })
    const after = at.filter((x) => x > pendingAt)
    const gaps = after.map((x, i) => x - (i ? after[i - 1]! : pendingAt))
    expect(gaps.slice(0, 7)).toEqual([1000, 2000, 4000, 8000, 16_000, 30_000, 30_000])
    expect(r.givenUpBatches()).toEqual([])
    await vi.advanceTimersByTimeAsync(10 * 60_000)
    expect(r.outboxPending()).toEqual([])
    expect(r.givenUpBatches()).toMatchObject([{ groupId: G1, line: r.OUTBOX_GIVEN_UP }])
    // Nothing is left to send, so nothing more is asked.
    const n = fetchStub.mock.calls.length
    await vi.advanceTimersByTimeAsync(120_000)
    expect(fetchStub.mock.calls.length).toBe(n)
  })
})

describe('the store', () => {
  it('gives a fresh read, when asked, from a read that leaves after the call, not the one already out', async () => {
    let release: (v: Response) => void = () => {}
    let n = 0
    fetchStub.mockImplementation(() => {
      n++
      return n === 1 ? new Promise<Response>((res) => (release = res)) : Promise.resolve(json(snap({ rev: 9 })))
    })
    const out = r.runnerStore.refresh()
    const fresh = r.runnerStore.refresh({ fresh: true })
    // A plain read shares the one out.
    expect(r.runnerStore.refresh()).toBe(out)
    let done = false
    void fresh.then(() => (done = true))
    await new Promise((res) => setTimeout(res, 20))
    expect(done).toBe(false)
    expect(n).toBe(1)
    release(json(snap({ rev: 5 })))
    await out
    await fresh
    expect(n).toBe(2)
    expect(r.runnerStore.snapshot().rev).toBe(9)
  })

  it('reads, streams, applies revs in order, and never lets an older event put a job back', async () => {
    fetchStub.mockImplementation(async () => json(snap({ jobs: [job()], groups: [group()] })))
    r.runnerStore.start()
    await vi.waitFor(() => expect(FakeES.all).toHaveLength(1), WAIT)
    const es = FakeES.all[0]!
    expect(es.url).toBe('/api/runner/stream')
    es.emit('state', snap({ jobs: [job()], groups: [group()] }))
    expect(r.runnerStore.snapshot().connected).toBe(true)
    es.emit('job', { rev: 6, job: job({ status: 'queued', promptId: 'p1' }) })
    expect(r.runnerStore.snapshot()).toMatchObject({ rev: 6, jobs: [{ status: 'queued' }] })
    es.emit('job', { rev: 5, job: job({ status: 'waiting' }) })
    expect(r.runnerStore.snapshot().jobs[0]!.status).toBe('queued')
    // A job event at the current rev (the server's word that a write failed) is taken.
    es.emit('job', { rev: 6, job: job({ status: 'queued', promptId: 'p1', wait: { for: 'disk' } }) })
    expect(r.runnerStore.snapshot().jobs[0]!.wait).toEqual({ for: 'disk' })
    es.emit('lane', { rev: 7, lane: { held: { why: 'lost', scope: 'heavy', jobId: 'x', since: 1 } } })
    es.emit('gone', { rev: 8, jobs: ['j1'], groups: ['g1'] })
    expect(r.runnerStore.snapshot()).toMatchObject({ rev: 8, jobs: [], groups: [], lane: { held: { why: 'lost' } } })
  })

  it('reads the whole state again on a gap, at most twice while the server stays behind', async () => {
    fetchStub.mockImplementation(async () => json(snap({ jobs: [job()], groups: [group()] })))
    r.runnerStore.start()
    await vi.waitFor(() => expect(FakeES.all).toHaveLength(1), WAIT)
    const es = FakeES.all[0]!
    es.emit('state', snap({ jobs: [job()], groups: [group()] }))
    const before = fetchStub.mock.calls.length
    es.emit('job', { rev: 9, job: job({ status: 'running', promptId: 'p1' }) })
    expect(r.runnerStore.snapshot().jobs[0]!.status).toBe('waiting')
    await vi.waitFor(() => expect(fetchStub.mock.calls.length).toBe(before + 2), WAIT)
    await new Promise((res) => setTimeout(res, 50))
    expect(fetchStub.mock.calls.length).toBe(before + 2)
    fetchStub.mockImplementation(async () => json(snap({ rev: 9, jobs: [job({ status: 'running', promptId: 'p1' })], groups: [group()] })))
    es.emit('job', { rev: 11, job: job({ status: 'running', promptId: 'p1' }) })
    await vi.waitFor(() => expect(r.runnerStore.snapshot()).toMatchObject({ rev: 9, jobs: [{ status: 'running' }] }), WAIT)
  })

  it('takes a new start of the queue whole, and drops a read the old one answered after it', async () => {
    fetchStub.mockImplementation(async () => json(snap({ jobs: [job()], groups: [group()] })))
    r.runnerStore.start()
    await vi.waitFor(() => expect(FakeES.all).toHaveLength(1), WAIT)
    const es = FakeES.all[0]!
    es.emit('state', snap({ jobs: [job()], groups: [group()] }))
    let release: (v: Response) => void = () => {}
    fetchStub.mockImplementation(() => new Promise<Response>((res) => (release = res)))
    const slow = r.runnerStore.refresh()
    await vi.waitFor(() => expect(fetchStub).toHaveBeenCalled(), WAIT)
    es.emit('state', snap({ boot: 'b2', rev: 2, jobs: [] }))
    fetchStub.mockImplementation(async () => json(snap({ boot: 'b2', rev: 3, jobs: [] })))
    release(json(snap({ boot: 'b1', rev: 12, jobs: [job()] })))
    await slow
    expect(r.runnerStore.snapshot()).toMatchObject({ boot: 'b2', jobs: [] })
    await vi.waitFor(() => expect(r.runnerStore.snapshot().rev).toBe(3), WAIT)
  })

  it('keeps jobs the intake answered with, against a state from before it', async () => {
    fetchStub.mockImplementation(async (url: string) => (url === '/api/runner/groups' ? taken(7) : json(snap())))
    r.runnerStore.start()
    await vi.waitFor(() => expect(FakeES.all).toHaveLength(1), WAIT)
    const es = FakeES.all[0]!
    es.emit('state', snap())
    await r.submitGroup(body())
    es.emit('state', snap({ rev: 6 }))
    expect(r.runnerStore.snapshot().jobs.map((j) => j.id)).toEqual([J1])
  })

  it('settles as absent for a page or a 404 in place of the queue: nothing open, and no asking again', async () => {
    for (const answer of [() => html(200), () => json({ error: 'no' }, 404)]) {
      fetchStub.mockImplementation(async () => answer())
      const m = await load()
      FakeES.all = []
      m.runnerStore.start()
      await vi.waitFor(() => expect(m.runnerStore.snapshot().reason).toBe('This server has no queue of its own.'), WAIT)
      const n = fetchStub.mock.calls.length
      await new Promise((res) => setTimeout(res, 50))
      expect(FakeES.all).toHaveLength(0)
      expect(m.runnerStore.snapshot().available).toBe(false)
      expect(fetchStub.mock.calls.length).toBe(n)
    }
  })

  it('opens no stream while the queue is off with nothing in it', async () => {
    fetchStub.mockImplementation(async () => json(snap({ available: false, reason: 'Turned off with SWITCHGEN_RUNNER=off.' })))
    r.runnerStore.start()
    await vi.waitFor(() => expect(r.runnerStore.snapshot().boot).toBe('b1'), WAIT)
    await new Promise((res) => setTimeout(res, 20))
    expect(FakeES.all).toHaveLength(0)
  })

  it('gives the stream back while the tab is hidden, and reads again and reopens it when shown', async () => {
    const listeners = new Map<string, () => void>()
    const doc = { visibilityState: 'visible', addEventListener: (t: string, fn: () => void) => listeners.set(t, fn) }
    vi.stubGlobal('document', doc)
    fetchStub.mockImplementation(async () => json(snap({ jobs: [job()], groups: [group()] })))
    const m = await load()
    FakeES.all = []
    m.runnerStore.start()
    await vi.waitFor(() => expect(FakeES.all).toHaveLength(1), WAIT)
    doc.visibilityState = 'hidden'
    listeners.get('visibilitychange')!()
    expect(FakeES.all[0]!.closed).toBe(true)
    const reads = fetchStub.mock.calls.length
    doc.visibilityState = 'visible'
    listeners.get('visibilitychange')!()
    await vi.waitFor(() => expect(FakeES.all).toHaveLength(2), WAIT)
    expect(fetchStub.mock.calls.length).toBe(reads + 1)
  })
})

describe('words to the queue', () => {
  it('say yes only for a 200, and read the state when the stream is down', async () => {
    for (const [word, url] of [
      [() => r.stopJob('a b'), '/api/runner/jobs/a%20b/stop'],
      [() => r.stopGroup('g1'), '/api/runner/groups/g1/stop'],
      [() => r.laneWord('send', 1), '/api/runner/lane'],
      [() => r.dismiss(['j1']), '/api/runner/dismiss'],
    ] as const) {
      fetchStub.mockImplementation(async (u: string) => (u === url ? json({ error: 'no such job' }, 404) : json(snap())))
      expect(await word(), url).toBe(false)
      fetchStub.mockClear()
      fetchStub.mockImplementation(async (u: string) => (u === url ? json({ ok: true }) : json(snap())))
      expect(await word(), url).toBe(true)
      await vi.waitFor(() => expect(fetchStub.mock.calls.map(([u]) => u)).toEqual([url, '/api/runner']), WAIT)
    }
    fetchStub.mockImplementation(async (u: string, init?: RequestInit) => json({ body: init?.body, u }))
    await r.laneWord('stop', 42)
    expect(JSON.parse(String((fetchStub.mock.calls.find(([u]) => u === '/api/runner/lane')![1] as RequestInit).body))).toEqual({ action: 'stop', since: 42 })
  })

  it('name the hold the page showed, and read the hold as it now stands when the server says it has changed', async () => {
    const hold = (since: number) => ({ held: { why: 'lost', scope: 'heavy', jobId: 'x', since } })
    fetchStub.mockImplementation(async () => json(snap({ lane: hold(7) })))
    await r.runnerStore.refresh()
    // A read already on its way, which left before the hold changed.
    let release: (v: Response) => void = () => {}
    fetchStub.mockImplementation((u: string) => {
      if (u === '/api/runner/lane') return Promise.resolve(json({ error: 'The hold has changed since this page showed it.' }, 409))
      return new Promise<Response>((res) => (release = res))
    })
    const out = r.runnerStore.refresh()
    await vi.waitFor(() => expect(fetchStub).toHaveBeenCalledTimes(2), WAIT)
    expect(await r.laneWord('stop', 7)).toBe(false)
    const lane = fetchStub.mock.calls.find(([u]) => u === '/api/runner/lane')!
    expect(JSON.parse(String((lane[1] as RequestInit).body))).toEqual({ action: 'stop', since: 7 })
    const reads = () => fetchStub.mock.calls.filter(([u]) => u === '/api/runner').length
    const before = reads()
    fetchStub.mockImplementation(async () => json(snap({ rev: 7, lane: hold(9) })))
    release(json(snap({ lane: hold(7) })))
    await out
    // One more read, sent after the one that was out.
    await vi.waitFor(() => expect(reads()).toBe(before + 1), WAIT)
    await vi.waitFor(() => expect(r.runnerStore.snapshot().lane.held?.since).toBe(9), WAIT)
  })
})

describe('following a job', () => {
  it('tells queued, running and preview, and resolves with the filed job', async () => {
    fetchStub.mockImplementation(async () => json(snap({ jobs: [job()], groups: [group()] })))
    const seen: unknown[] = []
    const p = r.follow('j1', (e) => seen.push(e))
    await vi.waitFor(() => expect(FakeES.all).toHaveLength(1), WAIT)
    const es = FakeES.all[0]!
    es.emit('state', snap({ jobs: [job()], groups: [group()] }))
    es.emit('job', { rev: 6, job: job({ status: 'queued', promptId: 'p1' }) })
    es.emit('job', { rev: 7, job: job({ status: 'running', promptId: 'p1' }) })
    es.emit('progress', { id: 'j1', value: 3, max: 20, node: '5', classType: 'KSamplerAdvanced', pass: { index: 1, count: 2 }, at: 1, previewN: 0 })
    es.emit('preview', { id: 'j1', n: 1 })
    es.emit('job', { rev: 8, job: job({ status: 'done', promptId: 'p1', entryId: 'e1', entryNo: 4, durationMs: 1234, finishedAt: 99, primary: { filename: 'a.webm', subfolder: '', type: 'output', kind: 'video' } }) })
    expect(await p).toMatchObject({ entryId: 'e1', entryNo: 4, durationMs: 1234, finishedAt: 99, primary: { filename: 'a.webm' } })
    expect(seen).toEqual([
      { phase: 'queued', promptId: 'p1' },
      { phase: 'running', node: null, value: 0, max: 1, classType: null, pass: null },
      { phase: 'running', node: '5', value: 3, max: 20, classType: 'KSamplerAdvanced', pass: { index: 1, count: 2 } },
      { phase: 'preview', url: '/api/runner/jobs/j1/preview?n=1' },
    ])
  })

  it('rejects with the error a desk already knows how to say, under today\'s titles', async () => {
    // The same module graph as the runner, so ComfyError is one class.
    const { faultOf, faultTitle } = await import('../src/lib/faults')
    const { ComfyError, LostJob } = await import('../src/lib/comfy')
    const ended = (code: RunnerMod.RunnerErrorCode, over: Partial<RunnerMod.RunnerError> = {}) =>
      job({
        status: ({ lost: 'lost', unsent: 'unsent', stopped: 'stopped', skipped: 'skipped' } as Record<string, RunnerMod.RunnerStatus>)[code] ?? 'failed',
        promptId: 'p1',
        error: { code, message: null, node: null, nodeType: null, nodeErrors: null, mayExist: false, sent: true, after: null, ...over },
      })
    const cases: [RunnerMod.RunnerJob, string, 'comfy' | 'lost'][] = [
      [ended('refused', { message: 'bad', nodeErrors: { '3': { errors: [{ message: 'x' }] } } }), 'That job was rejected', 'comfy'],
      [ended('failed', { message: 'CUDA out of memory' }), 'The card ran out of memory', 'comfy'],
      [ended('stopped'), 'Correction', 'comfy'],
      [ended('skipped', { sent: false }), 'Correction', 'comfy'],
      [ended('lost'), 'We lost track of that job', 'lost'],
      [ended('unsent'), 'We lost track of that job', 'lost'],
      [ended('ended-unsent'), 'We lost track of that job', 'lost'],
      [ended('no-file'), 'That job did not finish', 'comfy'],
      [ended('no-frame'), 'That job did not finish', 'comfy'],
      [ended('internal'), 'That job did not finish', 'comfy'],
    ]
    for (const [j, title, kind] of cases) {
      const err = r.runnerFault(j)
      expect(err, j.error!.code).toBeInstanceOf(kind === 'lost' ? LostJob : ComfyError)
      expect(faultTitle(faultOf(err)), j.error!.code).toBe(title)
    }
    expect(r.runnerFault(ended('skipped', { sent: false })).message).toBe('Stopped before it was sent.')
    expect(r.runnerFault(ended('stopped')).message).toBe('Job stopped. Nothing was saved.')
    expect(r.runnerFault(ended('lost')).message).toBe(r.RUNNER_LOST)
    expect(r.runnerFault(ended('unsent')).message).toBe(r.RUNNER_UNSENT)
    expect(faultOf(r.runnerFault(ended('ended-unsent'))).mayExist).toBe(true)
    expect(r.runnerFault(ended('refused', { message: 'bad', node: '3', nodeType: 'KSampler' }))).toMatchObject({ node: '3', nodeType: 'KSampler', promptId: 'p1' })
    expect(r.runnerFault(ended('no-frame'), { 'no-frame': 'Shot 2 left no frame.' }).message).toBe('Shot 2 left no frame.')
    // The page's own lost sentence, word for word.
    expect(r.RUNNER_LOST).toBe('We lost track of this job. ComfyUI has no record of it any more, which usually means it restarted. Nothing was saved.')
  })

  it('rejects with the desk\'s own words for an ending only it can put well', async () => {
    const ended = job({ status: 'failed', error: { code: 'no-frame', message: null, node: null, nodeType: null, nodeErrors: null, mayExist: false, sent: false, after: null } })
    fetchStub.mockImplementation(async () => json(snap({ jobs: [ended] })))
    await r.runnerStore.refresh()
    await expect(r.follow('j1', () => {}, { words: { 'no-frame': 'Shot 2 opens on shot 1, which left no frame.' } })).rejects.toThrow('Shot 2 opens on shot 1, which left no frame.')
  })

  it('rejects a job the server has no record of only after a read that got through', async () => {
    let up = false
    fetchStub.mockImplementation(async () => {
      if (!up) throw new TypeError('fetch failed')
      return json(snap())
    })
    let settled: unknown = null
    const p = r.follow('nowhere', () => {}).catch((e: unknown) => (settled = e))
    await new Promise((res) => setTimeout(res, 30))
    // Nothing got through: nothing is known about the job.
    expect(settled).toBeNull()
    up = true
    await r.runnerStore.refresh()
    await vi.waitFor(() => expect(FakeES.all).toHaveLength(1), WAIT)
    FakeES.all[0]!.emit('state', snap())
    await vi.waitFor(() => expect(settled).not.toBeNull(), WAIT)
    await p
    const { LostJob } = await import('../src/lib/comfy')
    expect(settled).toBeInstanceOf(LostJob)
    expect((settled as Error).message).toBe('The server has no record of this job any more, so there is nothing to follow.')
  })

  it('keeps waiting while reads fail, once the state was known', async () => {
    fetchStub.mockImplementation(async () => json(snap()))
    await r.runnerStore.refresh()
    fetchStub.mockImplementation(async () => {
      throw new TypeError('fetch failed')
    })
    let settled: unknown = null
    void r.follow('nowhere', () => {}).catch((e: unknown) => (settled = e))
    await vi.waitFor(() => expect(fetchStub).toHaveBeenCalled(), WAIT)
    await new Promise((res) => setTimeout(res, 30))
    expect(settled).toBeNull()
  })

  it('asks once more before it gives a job up, when the state it holds lacks the job', async () => {
    fetchStub.mockImplementation(async () => json(snap()))
    await r.runnerStore.refresh()
    const reads = fetchStub.mock.calls.length
    await expect(r.follow('nowhere', () => {})).rejects.toThrow(/no record of this job/)
    expect(fetchStub.mock.calls.length).toBe(reads + 1)
  })

  it('waits for a job still being handed over from this tab', async () => {
    vi.useFakeTimers()
    fetchStub.mockImplementation(async (url: string) => {
      if (url === '/api/runner/groups') throw new TypeError('fetch failed')
      return json(snap())
    })
    await r.runnerStore.refresh()
    const handing = r.submitGroup(body())
    let settled = false
    void r.follow(J1, () => {}).then(
      () => (settled = true),
      () => (settled = true),
    )
    await vi.advanceTimersByTimeAsync(40_000)
    expect(await handing).toMatchObject({ pending: true })
    expect(settled).toBe(false)
  })

  it('rejects at once where the server has no queue, and when the page stops listening', async () => {
    fetchStub.mockImplementation(async () => html(200))
    r.runnerStore.start()
    await vi.waitFor(() => expect(r.runnerStore.snapshot().reason).toBe('This server has no queue of its own.'), WAIT)
    await expect(r.follow('j1', () => {})).rejects.toThrow(/no record of this job/)
    const ctl = new AbortController()
    ctl.abort(new Error('left the desk'))
    await expect(r.follow('j1', () => {}, { signal: ctl.signal })).rejects.toThrow('left the desk')
  })
})

describe('what a desk says', () => {
  it('uses the Video desk\'s own sentences for a clip, and names a picture or a shot as one', () => {
    expect(r.waitLine(job({ wait: null }))).toEqual({ stage: 'Waiting its turn', note: null })
    expect(r.waitLine(job({ wait: { for: 'turn' } }))).toEqual({ stage: 'Waiting its turn', note: 'Waits for the clip before it to finish, so that one’s memory can be released before this starts.' })
    expect(r.waitLine(job({ heavy: false, wait: { for: 'turn' } })).note).toBeNull()
    expect(r.waitLine(job({ wait: { for: 'queue', ahead: 2 } }))).toEqual({ stage: 'Waiting for the press', note: 'ComfyUI has 2 jobs to finish first. This clip waits for them, so the memory they hold can be released before it starts.' })
    expect(r.waitLine(job({ wait: { for: 'queue', ahead: 1 } })).note).toMatch(/^ComfyUI has 1 job to finish first/)
    expect(r.waitLine(job({ wait: { for: 'comfy' } }))).toEqual({ stage: 'Waiting for ComfyUI', note: 'ComfyUI is not answering; it may be restarting. This clip waits until it answers, then releases its memory and goes.' })
    expect(r.waitLine(job({ desk: 'reel', wait: { for: 'before' } })).stage).toBe('Waiting for the shot before it')
    expect(r.waitLine(job({ desk: 'images', heavy: false, wait: { for: 'before' } })).stage).toBe('Waiting for the picture before it')
    expect(r.waitLine(job({ wait: { for: 'heavy' } })).stage).toBe('Waiting for a heavy clip to go first')
    expect(r.waitLine(job({ wait: { for: 'disk' } })).stage).toBe('Waiting for room on the server')
    expect(r.waitLine(job({ status: 'releasing' })).stage).toBe('Releasing memory')
    expect(r.waitLine(job({ status: 'sending' })).stage).toBe('Sending it to the press')
    expect(r.waitLine(job({ status: 'queued' })).stage).toBe('Queued')
    expect(r.waitLine(job({ status: 'filing' })).stage).toBe('Filing')
    expect(r.waitLine(job({ status: 'running', stopRequested: true })).stage).toBe('Stopping')
  })

  it('says a clip is held, and why, only while the hold that covers it stands', async () => {
    fetchStub.mockImplementation(async () => json(snap({ lane: { held: { why: 'lost', scope: 'heavy', jobId: 'x', since: 1 } } })))
    await r.runnerStore.refresh()
    expect(r.waitLine(job({ wait: { for: 'held' } }))).toEqual({ stage: 'Held', note: 'Held until you say, because the heavy clip before it was lost.' })
    // A hold on heavy work does not hold a light job.
    expect(r.waitLine(job({ heavy: false, wait: { for: 'held' } }))).toEqual({ stage: 'Waiting its turn', note: null })
    fetchStub.mockImplementation(async () => json(snap({ rev: 6, lane: { held: { why: 'restart', scope: 'heavy', jobId: null, since: 2 } } })))
    await r.runnerStore.refresh()
    expect(r.waitLine(job({ wait: { for: 'held' } }))).toEqual({ stage: 'Held', note: r.HELD_AFTER_RESTART })
    expect(r.waitLine(job({ heavy: false, wait: { for: 'held' } })).stage).toBe('Waiting its turn')
  })

  it('says a job the word has just sent waits its turn, until its own wait follows', async () => {
    // The word cleared the hold; the job still says 'held' until the queue's next round.
    fetchStub.mockImplementation(async () => json(snap({ lane: { held: null } })))
    await r.runnerStore.refresh()
    expect(r.waitLine(job({ wait: { for: 'held' } }))).toEqual(r.waitLine(job({ wait: { for: 'turn' } })))
    expect(r.waitLine(job({ wait: { for: 'held' } })).stage).toBe('Waiting its turn')
    expect(r.reportedOf(job({ wait: { for: 'held' } }), null).stage).toBe('Waiting its turn')
  })

  it('gives the server\'s reason under Stopping while the queue is off, since the stop cannot land until it runs', async () => {
    const OFF = 'Turned off with SWITCHGEN_RUNNER=off.'
    fetchStub.mockImplementation(async () => json(snap({ available: false, reason: OFF })))
    await r.runnerStore.refresh()
    expect(r.waitLine(job({ status: 'running', promptId: 'p', stopRequested: true }))).toEqual({ stage: 'Stopping', note: OFF })
    expect(r.waitLine(job({ status: 'waiting', stopRequested: true }))).toEqual({ stage: 'Stopping', note: OFF })
    fetchStub.mockImplementation(async () => json(snap({ available: true, reason: null })))
    await r.runnerStore.refresh()
    expect(r.waitLine(job({ status: 'running', promptId: 'p', stopRequested: true }))).toEqual({ stage: 'Stopping', note: null })
  })

  it('says the machine restarted when that is why the work is held', async () => {
    fetchStub.mockImplementation(async () => json(snap({ lane: { held: { why: 'restart', scope: 'all', jobId: null, since: 1 } } })))
    await r.runnerStore.refresh()
    expect(r.waitLine(job({ wait: { for: 'held' } })).note).toBe(r.HELD_AFTER_RESTART)
    expect(r.HELD_AFTER_RESTART).toBe('The machine restarted while this work waited, so it is held until you say.')
  })

  it('gives the section bar a row with the prompt id from queued on, so no device counts the job as someone else\'s', () => {
    const want: Record<string, string> = { waiting: 'submitting', releasing: 'submitting', sending: 'submitting', queued: 'queued', running: 'running', filing: 'running', done: 'done', failed: 'error', lost: 'error', unsent: 'error', stopped: 'cancelled', skipped: 'cancelled' }
    const progress = { id: 'j1', value: 2, max: 9, node: null, classType: null, pass: { index: 2, count: 2 }, at: 0, previewN: 0 }
    for (const [status, bar] of Object.entries(want)) {
      const live = status === 'waiting' || status === 'releasing' || status === 'sending'
      const row = r.reportedOf(job({ status: status as RunnerMod.RunnerStatus, promptId: live ? null : 'p' }), progress)
      expect(row.status, status).toBe(bar)
      expect(row, status).toMatchObject({ key: 'j1', promptId: live ? null : 'p', value: 2, max: 9, pass: { index: 2, count: 2 }, startedAt: 1, label: 'Wan' })
    }
    expect(r.reportedOf(job({ status: 'lost', promptId: 'p' }), null).error).toBe(r.RUNNER_LOST)
  })

  it('gives the section bar Held for a job the standing hold covers', async () => {
    fetchStub.mockImplementation(async () => json(snap({ lane: { held: { why: 'lost', scope: 'heavy', jobId: 'x', since: 1 } } })))
    await r.runnerStore.refresh()
    expect(r.reportedOf(job({ wait: { for: 'held' } }), null).stage).toBe('Held')
  })
})

describe('whether a desk hands its work over', () => {
  const caps = (c: object) => fetchStub.mockImplementation(async (url: string) => (url === '/api/capabilities' ? json({ server: 'switchgen', ...c }) : json(snap())))

  it('needs the capability, the desk, and a live queue that has not said no since', async () => {
    fetchStub.mockImplementation(async (url: string) =>
      url === '/api/capabilities'
        ? json({ server: 'x', runner: true, runnerDesks: ['video', 'reel'], runnerReason: null })
        : json(snap({ available: false, reason: 'Turned off with SWITCHGEN_RUNNER=off.' })),
    )
    expect(await r.runnerAvailable('images')).toEqual({ ok: false, reason: 'The queue on the server does not take this desk’s work.' })
    r.runnerStore.start()
    await vi.waitFor(() => expect(r.runnerStore.snapshot().boot).toBe('b1'), WAIT)
    expect(await r.runnerAvailable('video')).toEqual({ ok: false, reason: 'Turned off with SWITCHGEN_RUNNER=off.' })
  })

  it('says yes where all three hold', async () => {
    caps({ runner: true, runnerDesks: ['video'], runnerReason: null })
    expect(await r.runnerAvailable('video')).toEqual({ ok: true, reason: null })
  })

  it('says no, with the server\'s own reason, for a server that says no or is too old to say', async () => {
    caps({ runner: false, runnerDesks: ['video'], runnerReason: 'Another SwitchGen server holds the archive, and the queue with it.' })
    expect(await r.runnerAvailable('video')).toEqual({ ok: false, reason: 'Another SwitchGen server holds the archive, and the queue with it.' })
    const { serverCapabilities } = await import('../src/lib/capabilities')
    expect((await serverCapabilities()).runnerDesks).toEqual([])
  })

  it('asks the server again once its queue, which said no, is running', async () => {
    const OFF = 'Turned off with SWITCHGEN_RUNNER=off.'
    let said: object = { server: 'x', runner: false, runnerDesks: [], runnerReason: OFF }
    let state = snap({ available: false, reason: OFF })
    fetchStub.mockImplementation(async (url: string) => (url === '/api/capabilities' ? json(said) : json(state)))
    const capsRead = () => fetchStub.mock.calls.filter(([u]) => u === '/api/capabilities').length
    expect(await r.runnerAvailable('video')).toEqual({ ok: false, reason: OFF })
    expect(await r.runnerAvailable('video')).toEqual({ ok: false, reason: OFF })
    expect(capsRead()).toBe(1)
    // The server started again with its queue on; the page was not loaded again.
    said = { server: 'x', runner: true, runnerDesks: ['video'], runnerReason: null }
    state = snap()
    expect(await r.runnerAvailable('video')).toEqual({ ok: true, reason: null })
    expect(capsRead()).toBe(2)
    expect(await r.runnerAvailable('video')).toEqual({ ok: true, reason: null })
    expect(capsRead()).toBe(2)
  })

  it('reads the state again when the store says the queue is off and no stream keeps it, so a queue that came back is found', async () => {
    let state = snap({ available: false, reason: 'Off.' })
    fetchStub.mockImplementation(async (url: string) =>
      url === '/api/capabilities' ? json({ server: 'x', runner: true, runnerDesks: ['images', 'reel'], runnerReason: null }) : json(state),
    )
    r.runnerStore.start()
    await vi.waitFor(() => expect(r.runnerStore.snapshot().boot).toBe('b1'), WAIT)
    expect(r.runnerStore.snapshot()).toMatchObject({ available: false, connected: false })
    expect(await r.runnerAvailable('images')).toEqual({ ok: false, reason: 'Off.' })
    // The queue came back, and nothing told the page.
    state = snap()
    expect(await r.runnerAvailable('images')).toEqual({ ok: true, reason: null })
    expect(r.runnerStore.snapshot().available).toBe(true)
  })

  it('reads the capability with care: no field is no queue, and only desks it knows', async () => {
    caps({})
    let m = await load()
    expect((await m.runnerAvailable('video')).ok).toBe(false)
    const { serverCapabilities } = await import('../src/lib/capabilities')
    expect(await serverCapabilities()).toMatchObject({ runner: false, runnerDesks: [], runnerReason: null })
    caps({ runner: true, runnerDesks: ['video', 'pictures', 7, 'reel'], runnerReason: 3 })
    m = await load()
    const c = await (await import('../src/lib/capabilities')).serverCapabilities()
    expect(c).toMatchObject({ runner: true, runnerDesks: ['video', 'reel'], runnerReason: null })
    caps({ runner: 'yes', runnerDesks: ['video'] })
    m = await load()
    expect(await (await import('../src/lib/capabilities')).serverCapabilities()).toMatchObject({ runner: false, runnerDesks: [] })
    void m
  })
})

describe('this browser\'s name', () => {
  it('is kept, and lasts as long as the page where nothing can be kept', async () => {
    const id = r.deviceId()
    expect(id).toMatch(/^[0-9a-f-]{36}$/)
    expect(local.getItem('switchgen.device.v1')).toBe(id)
    expect((await load()).deviceId()).toBe(id)
    vi.stubGlobal('localStorage', undefined)
    const m = await load()
    const lone = m.deviceId()
    expect(m.deviceId()).toBe(lone)
  })
})

describe('a queue that stands back while the page watches', () => {
  const HELD_ARCHIVE = 'Another SwitchGen server holds the archive, and the queue with it.'
  const running = () => ({ jobs: [job({ status: 'running', promptId: 'p' })], groups: [group()] })

  it('is taken from the stream at rev 0 in the same start, then taken back at its own rev when it runs again', async () => {
    fetchStub.mockImplementation(async () => json(snap({ rev: 42, ...running() })))
    r.runnerStore.start()
    await vi.waitFor(() => expect(FakeES.all).toHaveLength(1), WAIT)
    const es = FakeES.all[0]!
    es.emit('state', snap({ rev: 42, ...running() }))
    expect(r.runnerStore.snapshot()).toMatchObject({ available: true, rev: 42 })
    // The queue stands back: its list is closed, so it says rev 0.
    es.emit('state', snap({ rev: 0, available: false, reason: HELD_ARCHIVE, jobs: [job({ status: 'waiting' })], groups: [group()] }))
    expect(r.runnerStore.snapshot()).toMatchObject({ available: false, rev: 0, reason: HELD_ARCHIVE, jobs: [{ id: 'j1', status: 'waiting' }] })
    // Back on, at the rev it had saved, holding what waited.
    es.emit('state', snap({ rev: 43, available: true, jobs: [job({ status: 'waiting', wait: { for: 'held' } })], groups: [group()], lane: { held: { why: 'paused', scope: 'all', jobId: null, since: 1 } } }))
    expect(r.runnerStore.snapshot()).toMatchObject({ available: true, rev: 43, lane: { held: { why: 'paused' } } })
    expect(r.waitLine(r.runnerStore.snapshot().jobs[0]!).note).toBe(r.HELD_AFTER_PAUSE)
  })

  it('is taken from a read at rev 0 when the stream says nothing', async () => {
    fetchStub.mockImplementation(async () => json(snap({ rev: 42, ...running() })))
    await r.runnerStore.refresh()
    expect(r.runnerStore.snapshot().available).toBe(true)
    fetchStub.mockImplementation(async () => json(snap({ rev: 0, available: false, reason: 'Off.', jobs: [] })))
    await r.runnerStore.refresh()
    expect(r.runnerStore.snapshot()).toMatchObject({ available: false, reason: 'Off.', jobs: [] })
  })

  it('is not undone by a slow read that left before the stream said so', async () => {
    fetchStub.mockImplementation(async () => json(snap({ rev: 42, ...running() })))
    r.runnerStore.start()
    await vi.waitFor(() => expect(FakeES.all).toHaveLength(1), WAIT)
    const es = FakeES.all[0]!
    es.emit('state', snap({ rev: 42, ...running() }))
    let answer: ((v: Response) => void) | null = null
    fetchStub.mockImplementation(() => new Promise<Response>((res) => (answer = res)))
    const slow = r.runnerStore.refresh()
    await vi.waitFor(() => expect(answer).not.toBeNull(), WAIT)
    const off = snap({ rev: 42, available: false, reason: 'Off.', ...running() })
    es.emit('state', off)
    fetchStub.mockImplementation(async () => json(off))
    // The read that left first answers as the queue was.
    answer!(json(snap({ rev: 42, available: true, ...running() })))
    await slow
    expect(r.runnerStore.snapshot()).toMatchObject({ available: false, reason: 'Off.' })
    await r.runnerStore.refresh()
    expect(r.runnerStore.snapshot()).toMatchObject({ available: false, reason: 'Off.' })
  })

  it('leaves its saved work listed and waiting, said as waiting on a queue that is off, and followed', async () => {
    const OFF = 'Turned off with SWITCHGEN_RUNNER=off.'
    fetchStub.mockImplementation(async () => json(snap({ available: false, reason: OFF, jobs: [job()], groups: [group()] })))
    await r.runnerStore.refresh()
    expect(r.runnerStore.snapshot().jobs.map((j) => j.id)).toEqual(['j1'])
    expect(r.waitLine(job())).toEqual({ stage: 'Waiting for the server’s queue', note: OFF })
    expect(r.reportedOf(job()).stage).toBe('Waiting for the server’s queue')
    let settled = false
    void r.follow('j1', () => {}).then(
      () => (settled = true),
      () => (settled = true),
    )
    await r.runnerStore.refresh()
    await new Promise((res) => setTimeout(res, 50))
    expect(settled).toBe(false)
  })
})

describe('work the server holds', () => {
  it('is said in the words for why it is held', async () => {
    for (const [why, line] of [['lost', r.HELD_AFTER_LOSS], ['unsent', r.HELD_AFTER_UNSENT], ['restart', r.HELD_AFTER_RESTART], ['paused', r.HELD_AFTER_PAUSE]] as const) {
      fetchStub.mockImplementation(async () => json(snap({ lane: { held: { why, scope: 'all', jobId: null, since: 1 } } })))
      const m = await load()
      await m.runnerStore.refresh()
      expect(m.waitLine(job({ wait: { for: 'held' } })).note, why).toBe(line)
    }
    expect(r.HELD_AFTER_PAUSE).toBe('The queue on the server was off while this work waited, so it is held until you say.')
  })

  it('is every waiting job under a hold on all the work, and the heavy ones under a hold on heavy work', () => {
    expect(r.holdCovers(null, { heavy: true, createdAt: 1 })).toBe(false)
    expect(r.holdCovers({ why: 'lost', scope: 'heavy', jobId: null, since: 1 }, { heavy: false, createdAt: 1 })).toBe(false)
    expect(r.holdCovers({ why: 'lost', scope: 'heavy', jobId: null, since: 1 }, { heavy: true, createdAt: 1 })).toBe(true)
    expect(r.holdCovers({ why: 'restart', scope: 'all', jobId: null, since: 1 }, { heavy: false, createdAt: 1 })).toBe(true)
  })

  it('is, under a hold for a restart or a queue that was off, only the work made before the hold began, as the server rules', () => {
    for (const why of ['paused', 'restart'] as const) {
      const held = { why, scope: 'all' as const, jobId: null, since: 100 }
      expect(r.holdCovers(held, { heavy: false, createdAt: 100 }), why).toBe(true)
      expect(r.holdCovers(held, { heavy: true, createdAt: 99 }), why).toBe(true)
      expect(r.holdCovers(held, { heavy: false, createdAt: 101 }), why).toBe(false)
      expect(r.holdCovers(held, { heavy: true, createdAt: 101 }), why).toBe(false)
    }
    // A hold after a lost clip is about the heavy work, however new.
    expect(r.holdCovers({ why: 'lost', scope: 'heavy', jobId: 'x', since: 100 }, { heavy: true, createdAt: 500 })).toBe(true)
  })

  it('is, under a hold for a restart or a pause that a lost clip was added to, the heavy work made after it too, and still no light work', () => {
    for (const why of ['paused', 'restart'] as const) {
      const held = { why, scope: 'all' as const, jobId: 'x', since: 50, heavyAfter: 90 }
      expect(r.holdCovers(held, { heavy: false, createdAt: 100 }), why).toBe(false)
      expect(r.holdCovers(held, { heavy: true, createdAt: 100 }), why).toBe(true)
      expect(r.holdCovers(held, { heavy: false, createdAt: 40 }), why).toBe(true)
      expect(r.holdCovers({ ...held, heavyAfter: null }, { heavy: true, createdAt: 100 }), why).toBe(false)
      expect(r.holdCovers({ ...held, heavyAfter: undefined }, { heavy: true, createdAt: 100 }), why).toBe(false)
    }
  })

  it('says the heavy work made after such a hold began is held for the lost clip, and the work that waited through it for the pause', async () => {
    const lost = job({ id: 'x', status: 'lost', createdAt: 60 })
    const late = job({ id: 'p', wait: { for: 'held' }, createdAt: 100 })
    const old = job({ id: 'a', wait: { for: 'held' }, createdAt: 40 })
    const light = job({ id: 'l', desk: 'images', kind: 'image', heavy: false, wait: { for: 'turn' }, createdAt: 100 })
    fetchStub.mockImplementation(async () => json(snap({ lane: { held: { why: 'paused', scope: 'all', jobId: 'x', since: 50, heavyAfter: 90 } }, jobs: [lost, late, old, light] })))
    await r.runnerStore.refresh()
    expect(r.waitLine(late).note).toBe(r.HELD_AFTER_LOSS)
    expect(r.waitLine(old).note).toBe(r.HELD_AFTER_PAUSE)
    expect(r.waitLine(light).stage).not.toBe('Held')
    // A clip that may never have reached ComfyUI is said so.
    fetchStub.mockImplementation(async () =>
      json(snap({ rev: 6, lane: { held: { why: 'restart', scope: 'all', jobId: 'x', since: 50, heavyAfter: 90 } }, jobs: [{ ...lost, status: 'unsent' }, late] })),
    )
    await r.runnerStore.refresh()
    expect(r.waitLine(late).note).toBe(r.HELD_AFTER_UNSENT)
  })

  it('reads the state again when a word finds nothing held, since another device answered first', async () => {
    fetchStub.mockImplementation(async (u: string) => (u === '/api/runner/lane' ? json({ error: 'Nothing is held.' }, 409) : json(snap({ rev: 77 }))))
    expect(await r.laneWord('send', 1)).toBe(false)
    // This page's own copy is read again, not only some request made.
    await vi.waitFor(() => expect(r.runnerStore.snapshot().rev).toBe(77), WAIT)
  })
})

describe('a hand-over whose answer the page slept through', () => {
  // A group id of its own, so nothing another test's page replays is taken for it.
  const G4 = 'dddddddd-4444-4444-8444-444444444444'

  it('is let go, not given up, when the server lists it however old it is', async () => {
    session.setItem(OUTBOX, JSON.stringify([{ at: Date.now() - 60 * 60_000, body: body(G4) }]))
    const posted: string[] = []
    fetchStub.mockImplementation(async (url: string, init?: RequestInit) => {
      if (url === '/api/runner/groups') {
        const id = JSON.parse(String(init?.body)).group.id
        if (id === G4) posted.push(id)
        return json({ error: 'x' }, 400)
      }
      return json(snap({ jobs: [job({ id: J1, groupId: G4 })], groups: [group({ id: G4, jobIds: [J1] })] }))
    })
    const m = await load()
    await vi.waitFor(() => expect(m.outboxPending()).toHaveLength(0), WAIT)
    expect(m.givenUpBatches()).toEqual([])
    expect(posted).toEqual([])
    expect(session.getItem(OUTBOX)).toBeNull()
  })

  it('is not judged before the server\'s list has been read', async () => {
    session.setItem(OUTBOX, JSON.stringify([{ at: Date.now() - 60 * 60_000, body: body(G4) }]))
    fetchStub.mockImplementation(async () => {
      throw new TypeError('fetch failed')
    })
    const m = await load()
    await vi.waitFor(() => expect(fetchStub).toHaveBeenCalled(), WAIT)
    await new Promise((res) => setTimeout(res, 30))
    expect(m.givenUpBatches()).toEqual([])
    expect(m.outboxPending()).toHaveLength(1)
  })

  it('is let go the moment the stream names its group', async () => {
    vi.useFakeTimers()
    fetchStub.mockImplementation(async (url: string) => {
      if (url === '/api/runner/groups') throw new TypeError('fetch failed')
      return json(snap())
    })
    r.runnerStore.start()
    await vi.advanceTimersByTimeAsync(10)
    const es = FakeES.all[0]!
    es.emit('state', snap())
    const p = r.submitGroup(body())
    await vi.advanceTimersByTimeAsync(40_000)
    expect(await p).toMatchObject({ pending: true })
    expect(r.outboxPending()).toHaveLength(1)
    es.emit('group', { rev: 6, group: group({ id: G1, jobIds: [J1] }) })
    expect(r.outboxPending()).toHaveLength(0)
    expect(session.getItem(OUTBOX)).toBeNull()
  })
})

describe('the section bar\'s clock for work on the server', () => {
  it('times a clip from when it was made, and a picture or shot from its own send', () => {
    expect(r.reportedOf(job({ desk: 'video', createdAt: 5, sentAt: 9 })).startedAt).toBe(5)
    expect(r.reportedOf(job({ desk: 'images', createdAt: 5, sentAt: 9 })).startedAt).toBe(9)
    expect(r.reportedOf(job({ desk: 'reel', createdAt: 5, sentAt: 9 })).startedAt).toBe(9)
    // Not sent yet: from the moment the bar first shows it.
    const before = Date.now()
    const t = r.reportedOf(job({ desk: 'reel', createdAt: 5, sentAt: null })).startedAt!
    expect(t).toBeGreaterThanOrEqual(before)
    expect(t).toBeLessThanOrEqual(Date.now())
  })
})

describe('a job found missing by the read that brings it', () => {
  const reads = () => fetchStub.mock.calls.filter(([u]) => u === '/api/runner').length

  it('is asked about once more, and given up on that read, with no word from the stream', async () => {
    fetchStub.mockImplementation(async () => json(snap()))
    let settled: unknown = null
    const p = r.follow('nowhere', () => {}).catch((e: unknown) => (settled = e))
    await vi.waitFor(() => expect(settled).not.toBeNull(), WAIT)
    await p
    expect((settled as Error).message).toBe('The server has no record of this job any more, so there is nothing to follow.')
    // The read that found it missing, and one more.
    expect(reads()).toBeGreaterThanOrEqual(2)
  })

  it('is given up by a later read that gets through, once the one after the first failed', async () => {
    fetchStub.mockImplementation(async () => json(snap()))
    await r.runnerStore.refresh()
    let down = true
    fetchStub.mockImplementation(async () => {
      if (down) throw new TypeError('fetch failed')
      return json(snap())
    })
    const before = fetchStub.mock.calls.length
    let settled: unknown = null
    void r.follow('nowhere', () => {}).catch((e: unknown) => (settled = e))
    // It asks once more, and that read fails.
    await vi.waitFor(() => expect(fetchStub.mock.calls.length).toBeGreaterThan(before), WAIT)
    await new Promise((res) => setTimeout(res, 30))
    expect(settled).toBeNull()
    down = false
    await r.runnerStore.refresh()
    await vi.waitFor(() => expect(settled).not.toBeNull(), WAIT)
  })
})
