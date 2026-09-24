import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { OutputFile } from '../src/lib/comfy'
import type * as Desk from '../src/routes/Pictures'
import type * as R from '../src/lib/runner'

/**
 * The Pictures desk with the queue on the server. The queue's library is
 * stood in (its store, the hand-over, follow and stop), and so is ComfyUI's
 * run(), which a batch handed to the server must never call. Nothing is sent
 * anywhere.
 */
const m = vi.hoisted(() => ({
  run: vi.fn(),
  followPrompt: vi.fn(),
  cancelJob: vi.fn(),
  runnerAvailable: vi.fn(),
  withdraw: vi.fn(),
  submitGroup: vi.fn(),
  stopGroup: vi.fn(),
  follow: vi.fn(),
  givenUp: [] as { groupId: string; desk: string; label: string; at: number; line: string }[],
  subs: new Set<() => void>(),
  snap: null as unknown as R.RunnerSnapshot,
  /** The real store's own read, which waitLine (used as it is) takes the lane from. */
  realRefresh: null as null | (() => Promise<void>),
  /** The library itself, for a test that has the hand-over go through it. */
  real: null as unknown as typeof R,
}))
vi.mock('../src/lib/comfy', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../src/lib/comfy')>()),
  run: m.run,
  cancelJob: m.cancelJob,
  followPrompt: m.followPrompt,
}))
vi.mock('../src/lib/runner', async (importOriginal) => {
  const real = await importOriginal<typeof import('../src/lib/runner')>()
  m.realRefresh = real.runnerStore.refresh
  m.real = real
  return {
    ...real,
    runnerStore: {
      subscribe: (fn: () => void) => {
        m.subs.add(fn)
        return () => m.subs.delete(fn)
      },
      snapshot: () => m.snap,
      start: () => {},
      refresh: async () => {},
    },
    runnerAvailable: m.runnerAvailable,
    submitGroup: m.submitGroup,
    stopGroup: m.stopGroup,
    withdraw: m.withdraw,
    follow: m.follow,
    givenUpBatches: () => [...m.givenUp],
    forgetGivenUp: (id: string) => {
      const i = m.givenUp.findIndex((g) => g.groupId === id)
      if (i >= 0) m.givenUp.splice(i, 1)
    },
  }
})

const tab = new Map<string, string>()
const WAIT = { timeout: 3000, interval: 5 }

function blank(): R.RunnerSnapshot {
  return {
    v: 1, available: true, reason: null, boot: 'b1', rev: 1, comfy: { answering: true, since: 0 },
    lane: { held: null }, groups: [], jobs: [], progress: {}, connected: true,
  }
}
function setSnap(patch: Partial<R.RunnerSnapshot>) {
  m.snap = { ...m.snap, ...patch }
  for (const fn of [...m.subs]) fn()
}
/**
 * Give the real store the state the stand-in shows, by a read of its own:
 * waitLine, which the desk uses as it is, says a job is held only while that
 * store's lane holds it.
 */
async function feedStore(s: R.RunnerSnapshot) {
  const { connected: _c, ...state } = s
  vi.stubGlobal('fetch', async () => new Response(JSON.stringify(state), { status: 200, headers: { 'content-type': 'application/json' } }))
  await m.realRefresh!()
  for (const fn of [...m.subs]) fn()
}
/** The `i`th job of a batch of `total`, counted from nought here and from one where the server and the desk write it. */
function job(id: string, groupId: string, i: number, total: number, extra: Partial<R.RunnerJob> = {}): R.RunnerJob {
  return {
    id, groupId, desk: 'images', kind: 'image', seq: i, index: i + 1, total, label: `p${i + 1}`, prompt: 'x', device: 'other',
    heavy: false, status: 'waiting', wait: { for: 'turn' }, stopRequested: false, stopLanded: false, promptId: null, attempt: 0,
    createdAt: 1, sentAt: null, ranAt: null, finishedAt: null, endedAt: null, files: [], primary: null, frame: null, openedOn: null,
    entryId: null, entryNo: null, repeatOf: null, durationMs: 0, error: null, meta: { index: i + 1, total }, dismissed: false,
    ...extra,
  }
}

beforeEach(() => {
  for (const f of [m.run, m.followPrompt, m.cancelJob, m.runnerAvailable, m.withdraw, m.submitGroup, m.stopGroup, m.follow]) f.mockReset()
  m.givenUp.length = 0
  m.subs.clear()
  m.snap = blank()
  // As the real one answers: whether the queue runs as the server now says,
  // which the stand-in store shows, not yes whatever it shows.
  m.runnerAvailable.mockImplementation(async () => (m.snap.available ? { ok: true, reason: null } : { ok: false, reason: m.snap.reason ?? 'off' }))
  m.stopGroup.mockResolvedValue(true)
  tab.clear()
  vi.stubGlobal('sessionStorage', {
    getItem: (k: string) => tab.get(k) ?? null,
    setItem: (k: string, v: string) => void tab.set(k, String(v)),
    removeItem: (k: string) => void tab.delete(k),
  })
  vi.resetModules()
})
afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

const composition = {
  desk: 'images', mode: 't2i', familyId: 'sdxl-illustrious', model: 'm.safetensors',
  prompt: 'a lighthouse at dusk', negative: null, seed: 1, steps: 20, cfg: 5,
  sampler: 'euler', scheduler: 'normal', width: 1024, height: 1024, noLora: false,
}
const plan = (label: string, seed = 1) =>
  ({ graph: { '3': { class_type: 'KSampler', inputs: {} } }, label, composition: { ...composition, seed }, seed, familyLabel: 'Illustrious', modelLabel: 'M', variant: null, passes: { face: false, hand: false, hires: false }, loras: [] }) as unknown as Desk.RunPlan
const png = (filename: string): OutputFile => ({ filename, subfolder: '', type: 'output', kind: 'image' })

describe('a batch handed to the server', () => {
  it('goes as one batch of three, followed in order, with a no-file carried on and a failure ending it', async () => {
    const desk = await import('../src/routes/Pictures')
    const runner = await import('../src/lib/runner')
    let body: R.SubmitBody | null = null
    m.submitGroup.mockImplementation(async (b: R.SubmitBody) => {
      body = b
      return { ok: true, replayed: false, group: {} as R.RunnerGroup, jobs: [] }
    })
    const outcomes: ((on: (e: R.FollowEvent) => void) => Promise<unknown>)[] = [
      async (on) => {
        on({ phase: 'queued', promptId: 'p1' })
        on({ phase: 'running', node: '3', value: 5, max: 20 })
        return { files: [png('a.png')], primary: png('a.png'), frame: null, entryId: 'e1', entryNo: 7, repeatOf: null, openedOn: null, durationMs: 1234, finishedAt: 99 }
      },
      async () => {
        throw new (await import('../src/lib/comfy')).ComfyError(desk.pressSnapshot().job ? 'The job finished but wrote no file. Check ComfyUI’s own log for the reason.' : '')
      },
      async () => {
        throw new (await import('../src/lib/comfy')).ComfyError('bad node', { node: '3', nodeType: 'KSampler' })
      },
    ]
    let n = 0
    m.follow.mockImplementation((_id: string, on: (e: R.FollowEvent) => void) => outcomes[n++](on))
    const p = desk.startRuns([plan('one', 1), plan('two', 2), plan('three', 3)])
    expect(desk.pressSnapshot().job).toMatchObject({ runner: true, status: 'submitting', index: 1, total: 3 })
    await p
    expect(m.run).not.toHaveBeenCalled()
    expect(m.submitGroup).toHaveBeenCalledTimes(1)
    const b = body as unknown as R.SubmitBody
    expect(b.group).toMatchObject({ desk: 'images', kind: 'batch' })
    expect(b.jobs).toHaveLength(3)
    expect(b.jobs[1]).toMatchObject({ kind: 'image', primary: 'image', orFirst: true, noFile: 'fail', heavy: false, meta: { index: 2, total: 3 } })
    expect(b.jobs[1].record).toEqual(
      runner.recordTemplate(plan('two', 2).composition, { seed: 2, familyLabel: 'Illustrious', modelLabel: 'M', variant: null, passes: { face: false, hand: false, hires: false }, loras: [] }),
    )
    await vi.waitFor(() => expect(m.follow).toHaveBeenCalledTimes(3), WAIT)
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.status).toBe('error'), WAIT)
    const s = desk.pressSnapshot()
    expect(s.results[0]).toMatchObject({ id: 'e1', no: 7, at: 99, durationMs: 1234, promptId: 'p1' })
    expect(s.current?.id).toBe('e1')
    expect(s.lastMs).toBe(1234)
    expect(s.fault?.message).toBe('bad node')
    expect(s.job).toMatchObject({ index: 3, total: 3, runner: true })
    expect(tab.has('switchgen.picturesent.v1')).toBe(false)
    expect(tab.has('switchgen.pictures.runner.v1')).toBe(false)
  })

  it('says which pictures were never sent when one ends the batch, and none after a stop', async () => {
    const desk = await import('../src/routes/Pictures')
    const { ComfyError } = await import('../src/lib/comfy')
    m.submitGroup.mockImplementation(async (b: R.SubmitBody) => {
      const g = b.group.id
      setSnap({ jobs: b.jobs.map((j, i) => job(j.id, g, i, 3, { status: i === 0 ? 'failed' : 'skipped' })) })
      return { ok: true, replayed: false, group: {} as R.RunnerGroup, jobs: [] }
    })
    m.follow.mockRejectedValue(new ComfyError('broke'))
    await desk.startRuns([plan('one'), plan('two'), plan('three')])
    await vi.waitFor(() => expect(desk.pressSnapshot().unsent).toContain('Pictures 2 and 3 were never sent, because picture 1 failed'), WAIT)
  })

  it('is stopped on the server by stopPress, never by a cancel from here', async () => {
    const desk = await import('../src/routes/Pictures')
    let gid = ''
    m.submitGroup.mockImplementation(async (b: R.SubmitBody) => {
      gid = b.group.id
      return { ok: true, replayed: false, group: {} as R.RunnerGroup, jobs: [] }
    })
    m.follow.mockImplementation((_id: string, on: (e: R.FollowEvent) => void) => {
      on({ phase: 'queued', promptId: 'p1' })
      return new Promise(() => {})
    })
    await desk.startRuns([plan('one'), plan('two')])
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.status).toBe('queued'), WAIT)
    desk.stopPress()
    await vi.waitFor(() => expect(m.stopGroup).toHaveBeenCalledWith(gid), WAIT)
    expect(m.cancelJob).not.toHaveBeenCalled()
    expect(desk.pressSnapshot().job?.stage).toBe('Stopping')
  })

  it('shows the busy line on a 409, and sends nothing from here', async () => {
    const desk = await import('../src/routes/Pictures')
    m.submitGroup.mockResolvedValue({ ok: false, fallback: false, status: 409, error: 'busy', busy: 'images' })
    await desk.startRuns([plan('one')])
    expect(desk.pressSnapshot().fault).toMatchObject({ message: desk.BATCH_BUSY, title: 'Not sent', tone: 'correction' })
    expect(desk.pressSnapshot().job?.status).toBe('error')
    expect(m.run).not.toHaveBeenCalled()
  })

  it('sends from the page on a fallback, with the line saying why', async () => {
    const desk = await import('../src/routes/Pictures')
    m.submitGroup.mockResolvedValue({ ok: false, fallback: true, reason: 'The queue on the server is not running.' })
    m.run.mockImplementation((_g: unknown, on: (e: unknown) => void) => {
      on({ phase: 'queued', promptId: 'p1' })
      return new Promise(() => {})
    })
    await desk.startRuns([plan('one')])
    expect(m.run).toHaveBeenCalledTimes(1)
    expect(desk.pressSnapshot().fellBack).toBe('This page sends the work itself: the queue on the server is not running.')
    expect(desk.pressSnapshot().job?.runner).toBeUndefined()
  })

  it('never sends from the page when the server does not answer, and takes the batch up once the server has it', async () => {
    const desk = await import('../src/routes/Pictures')
    let b0: R.SubmitBody | null = null
    m.submitGroup.mockImplementation(async (b: R.SubmitBody) => {
      b0 = b
      return { ok: false, fallback: false, pending: true }
    })
    m.follow.mockImplementation(() => new Promise(() => {}))
    await desk.startRuns([plan('one')])
    expect(desk.pressSnapshot().job).toMatchObject({ status: 'submitting', stage: 'Handing it to the server' })
    expect(m.run).not.toHaveBeenCalled()
    const b = b0 as unknown as R.SubmitBody
    setSnap({
      groups: [{ id: b.group.id, desk: 'images', kind: 'batch', label: 'x', device: 'd', createdAt: 1, state: 'active', endedBy: null, endedAt: null, jobIds: [b.jobs[0].id], dismissed: false }],
      jobs: [job(b.jobs[0].id, b.group.id, 0, 1, { status: 'queued', promptId: 'p1' })],
    })
    await vi.waitFor(() => expect(m.follow).toHaveBeenCalledWith(b.jobs[0].id, expect.anything(), expect.anything()), WAIT)
    expect(desk.pressSnapshot().job).toMatchObject({ status: 'queued', promptId: 'p1' })
  })

  it('a batch stopped before the server answered is stopped when it shows up there', async () => {
    const desk = await import('../src/routes/Pictures')
    let b0: R.SubmitBody | null = null
    m.submitGroup.mockImplementation(async (b: R.SubmitBody) => {
      b0 = b
      return { ok: false, fallback: false, pending: true }
    })
    await desk.startRuns([plan('one')])
    m.stopGroup.mockResolvedValue(false)
    desk.stopPress()
    expect(desk.pressSnapshot().job?.status).toBe('cancelled')
    const b = b0 as unknown as R.SubmitBody
    m.stopGroup.mockClear()
    setSnap({
      groups: [{ id: b.group.id, desk: 'images', kind: 'batch', label: 'x', device: 'd', createdAt: 1, state: 'active', endedBy: null, endedAt: null, jobIds: [b.jobs[0].id], dismissed: false }],
      jobs: [job(b.jobs[0].id, b.group.id, 0, 1)],
    })
    await vi.waitFor(() => expect(m.stopGroup).toHaveBeenCalledWith(b.group.id), WAIT)
    expect(m.follow).not.toHaveBeenCalled()
  })

  it('stays in the page when the store says the queue is off and the server, asked again, says so too', async () => {
    m.snap = { ...blank(), available: false }
    const desk = await import('../src/routes/Pictures')
    m.run.mockImplementation(() => new Promise(() => {}))
    void desk.startRuns([plan('one')])
    await vi.waitFor(() => expect(m.run).toHaveBeenCalledTimes(1), WAIT)
    expect(m.runnerAvailable).toHaveBeenCalledTimes(1)
    expect(m.submitGroup).not.toHaveBeenCalled()
  })

  // A queue that was off when the page loaded, with nothing in it, keeps no
  // stream open, so nothing tells the page when it comes back. The desk asks
  // rather than going by the store's last word.
  it('hands the batch to a queue that came back with no word to the page', async () => {
    m.snap = { ...blank(), available: false, reason: 'off', connected: false }
    const desk = await import('../src/routes/Pictures')
    // The asking reads the queue again, and finds it running.
    m.runnerAvailable.mockImplementation(async () => {
      setSnap({ available: true, reason: null })
      return { ok: true, reason: null }
    })
    m.submitGroup.mockResolvedValue({ ok: true, replayed: false, group: {} as R.RunnerGroup, jobs: [] })
    m.follow.mockImplementation(() => new Promise(() => {}))
    m.run.mockImplementation(() => new Promise(() => {}))
    const shown = await desk.startRuns([plan('one'), plan('two')])
    expect(m.runnerAvailable).toHaveBeenCalledTimes(1)
    expect(m.submitGroup).toHaveBeenCalledTimes(1)
    expect(m.run).not.toHaveBeenCalled()
    expect(desk.pressSnapshot().job).toMatchObject({ id: shown, runner: true })
  })
})

describe('a batch taken up from the server', () => {
  it('is shown on load when another device is making one, and its record replaces nothing until pulled', async () => {
    const g = '11111111-1111-4111-8111-111111111111'
    const j1 = '22222222-2222-4222-8222-222222222222'
    m.snap = {
      ...blank(),
      groups: [{ id: g, desk: 'images', kind: 'batch', label: 'x', device: 'other', createdAt: 1, state: 'active', endedBy: null, endedAt: null, jobIds: [j1], dismissed: false }],
      jobs: [job(j1, g, 0, 1, { status: 'waiting', wait: { for: 'queue', ahead: 2 } })],
    }
    let finish: (v: unknown) => void = () => {}
    m.follow.mockImplementation(() => new Promise((r) => (finish = r)))
    const desk = await import('../src/routes/Pictures')
    const history = await import('../src/lib/history')
    expect(desk.pressSnapshot().job).toMatchObject({ id: j1, runner: true, status: 'submitting', stage: 'Waiting for the press' })
    expect(desk.serverWaitingLine(desk.pressSnapshot().job)).toContain('ComfyUI has 2 jobs to finish first')
    expect(desk.serverWaitingLine(desk.pressSnapshot().job)).toContain('kept on the SwitchGen server')
    finish({ files: [png('z.png')], primary: png('z.png'), frame: null, entryId: 'ez', entryNo: 3, repeatOf: null, openedOn: null, durationMs: 0, finishedAt: 5 })
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.status).toBe('done'), WAIT)
    expect(desk.pressSnapshot().current).toBeNull()
    expect(desk.pressSnapshot().lastMs).toBeNull()
    history.restore({ ...(composition as Record<string, unknown>), id: 'ez', no: 3, at: 5, file: png('z.png'), kind: 'image', promptId: 'p', durationMs: 0, familyLabel: 'I', modelLabel: 'M' } as never)
    await vi.waitFor(() => expect(desk.pressSnapshot().current?.id).toBe('ez'), WAIT)
    expect(desk.pressSnapshot().results.map((r) => r.id)).toEqual(['ez'])
  })

  it('says a given-up hand-over once', async () => {
    m.givenUp.push({ groupId: 'g', desk: 'images', label: 'M, 3 pictures', at: 1, line: 'The server never answered for this batch, so it was not sent.' })
    const desk = await import('../src/routes/Pictures')
    expect(desk.pressSnapshot().unsent).toBe('M, 3 pictures: The server never answered for this batch, so it was not sent.')
    await vi.waitFor(() => expect(m.givenUp).toHaveLength(0), WAIT)
  })
})

describe('the batch this tab handed over, after the page went', () => {
  it('is shown to its end on the next load: its picture, its fault and what was never sent', async () => {
    const g = '33333333-3333-4333-8333-333333333333'
    const ids = ['44444444-4444-4444-8444-444444444441', '44444444-4444-4444-8444-444444444442', '44444444-4444-4444-8444-444444444443']
    tab.set('switchgen.pictures.runner.v1', g)
    m.snap = {
      ...blank(),
      groups: [{ id: g, desk: 'images', kind: 'batch', label: 'x', device: 'me', createdAt: 1, state: 'ended', endedBy: { jobId: ids[1], why: 'failed' }, endedAt: 9, jobIds: ids, dismissed: false }],
      jobs: [
        job(ids[0], g, 0, 3, { status: 'done', entryId: 'e0' }),
        job(ids[1], g, 1, 3, { status: 'failed' }),
        job(ids[2], g, 2, 3, { status: 'skipped' }),
      ],
    }
    const { ComfyError } = await import('../src/lib/comfy')
    m.follow.mockImplementation(async (id: string) => {
      if (id === ids[0]) return { files: [png('a.png')], primary: png('a.png'), frame: null, entryId: 'e0', entryNo: 1, repeatOf: null, openedOn: null, durationMs: 0, finishedAt: 5 }
      throw new ComfyError('broke')
    })
    const desk = await import('../src/routes/Pictures')
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.status).toBe('error'), WAIT)
    expect(desk.pressSnapshot().fault?.message).toBe('broke')
    expect(desk.pressSnapshot().unsent).toContain('Picture 3 was never sent, because picture 2 failed')
    expect(m.follow).toHaveBeenCalledTimes(2)
    expect(tab.has('switchgen.pictures.runner.v1')).toBe(false)
  })

  it('is let go when the server has answered and has not got it', async () => {
    tab.set('switchgen.pictures.runner.v1', '55555555-5555-4555-8555-555555555555')
    await import('../src/routes/Pictures')
    expect(tab.has('switchgen.pictures.runner.v1')).toBe(false)
  })

  it('is kept while the server has not answered', async () => {
    tab.set('switchgen.pictures.runner.v1', '55555555-5555-4555-8555-555555555555')
    m.snap = { ...blank(), boot: '', connected: false, available: false }
    await import('../src/routes/Pictures')
    expect(tab.has('switchgen.pictures.runner.v1')).toBe(true)
  })
})

describe('Stop while the server is being asked', () => {
  it('frees the desk at once, and stops the batch the moment the server turns out to have it', async () => {
    const desk = await import('../src/routes/Pictures')
    let answer: (v: unknown) => void = () => {}
    let b0: R.SubmitBody | null = null
    m.submitGroup.mockImplementation((b: R.SubmitBody) => {
      b0 = b
      return new Promise((r) => (answer = r))
    })
    const shown = desk.startRuns([plan('one'), plan('two')])
    await vi.waitFor(() => expect(m.submitGroup).toHaveBeenCalled(), WAIT)
    m.stopGroup.mockResolvedValue(false)
    desk.stopPress()
    expect(desk.pressSnapshot().job?.status).toBe('cancelled')
    expect(desk.pressSnapshot().fault?.cancelled).toBe(true)
    const b = b0 as unknown as R.SubmitBody
    // Kept for the tab, so a reload still stops it when it shows up.
    expect(JSON.parse(tab.get('switchgen.pictures.dropped.v1') ?? '[]')).toContain(b.group.id)
    // The reader goes on to make something else, from the page.
    m.snap = { ...m.snap, available: false }
    m.run.mockImplementation(() => new Promise(() => {}))
    const next = await desk.startRuns([plan('three')])
    expect(m.run).toHaveBeenCalledTimes(1)
    expect(next).not.toBe(b.jobs[0].id)
    m.stopGroup.mockClear()
    m.stopGroup.mockResolvedValue(true)
    // The server had it after all.
    m.snap = { ...m.snap, available: true }
    setSnap({
      groups: [{ id: b.group.id, desk: 'images', kind: 'batch', label: 'x', device: 'd', createdAt: 1, state: 'active', endedBy: null, endedAt: null, jobIds: b.jobs.map((j) => j.id), dismissed: false }],
      jobs: b.jobs.map((j, i) => job(j.id, b.group.id, i, 2)),
    })
    answer({ ok: true, replayed: false, group: {}, jobs: [] })
    expect(await shown).toBe(b.jobs[0].id)
    await vi.waitFor(() => expect(m.stopGroup).toHaveBeenCalledWith(b.group.id), WAIT)
    expect(m.follow).not.toHaveBeenCalled()
    expect(desk.pressSnapshot().job?.runner).toBeUndefined()
  })
})

describe('Stop before the server answered, and the tab\'s outbox', () => {
  it('takes the batch out of the outbox, and asks the server once to stop it, since an earlier send may be there', async () => {
    const desk = await import('../src/routes/Pictures')
    let b0: R.SubmitBody | null = null
    m.submitGroup.mockImplementation((b: R.SubmitBody) => {
      b0 = b
      return new Promise(() => {})
    })
    void desk.startRuns([plan('one')])
    await vi.waitFor(() => expect(m.submitGroup).toHaveBeenCalled(), WAIT)
    desk.stopPress()
    expect(m.withdraw).toHaveBeenCalledWith((b0 as unknown as R.SubmitBody).group.id)
    expect(m.stopGroup).toHaveBeenCalledTimes(1)
  })

  it('takes it out too when Stop comes while the desk still asks whether the server takes the work, and hands nothing over', async () => {
    const desk = await import('../src/routes/Pictures')
    m.runnerAvailable.mockImplementation(() => new Promise(() => {}))
    void desk.startRuns([plan('one')])
    desk.stopPress()
    expect(m.withdraw).toHaveBeenCalledTimes(1)
    expect(m.submitGroup).not.toHaveBeenCalled()
  })

  it('is never sent again once stopped, by the hand-over still asking or by the outbox', async () => {
    vi.useFakeTimers()
    const OUTBOX = 'switchgen.runner.outbox.v1'
    const posts: string[] = []
    const state = { ...blank(), connected: undefined }
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      if (url === '/api/runner/groups') {
        posts.push(JSON.parse(String(init!.body)).group.id)
        throw new TypeError('fetch failed')
      }
      return new Response(JSON.stringify(state), { status: 200, headers: { 'content-type': 'application/json' } })
    })
    // The library's own hand-over and outbox, over a server that never answers.
    m.submitGroup.mockImplementation((b: R.SubmitBody) => m.real.submitGroup(b))
    m.withdraw.mockImplementation((groupId: string, jobIds?: string[]) => m.real.withdraw(groupId, jobIds))
    const desk = await import('../src/routes/Pictures')
    void desk.startRuns([plan('one')])
    await vi.advanceTimersByTimeAsync(10)
    expect(posts).toHaveLength(1)
    expect(tab.get(OUTBOX)).toContain(posts[0])
    desk.stopPress()
    expect(tab.has(OUTBOX)).toBe(false)
    await vi.advanceTimersByTimeAsync(15 * 60_000)
    expect(posts).toHaveLength(1)
    expect(m.run).not.toHaveBeenCalled()
  })
})

describe('a batch the server has while the press is busy', () => {
  const G = '66666666-6666-4666-8666-666666666666'
  const J = '77777777-7777-4777-8777-777777777777'
  const group = (): R.RunnerGroup => ({ id: G, desk: 'images', kind: 'batch', label: 'x', device: 'other', createdAt: 1, state: 'active', endedBy: null, endedAt: null, jobIds: [J], dismissed: false })
  /** The queue came back after it was off with the batch waiting, and holds it. */
  const paused = { held: { why: 'paused' as const, scope: 'all' as const, jobId: null, since: 5 } }

  it('is taken up once the page\'s own picture ends, with no word from the server after', async () => {
    const desk = await import('../src/routes/Pictures')
    m.follow.mockImplementation(() => new Promise(() => {}))
    setSnap({ available: false, reason: 'off', groups: [group()], jobs: [job(J, G, 0, 1)] })
    expect(desk.pressSnapshot().parked).toContain('one picture waits')
    let finish: (v: unknown) => void = () => {}
    m.run.mockImplementation(() => new Promise((res) => (finish = res)))
    await desk.startRuns([plan('mine')])
    expect(m.run).toHaveBeenCalledTimes(1)
    // The queue is back while the page's own picture runs, and holds the batch.
    setSnap({ available: true, reason: null, lane: paused })
    expect(desk.pressSnapshot().job?.runner).toBeUndefined()
    finish([png('a.png')])
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.id).toBe(J), WAIT)
    expect(m.follow).toHaveBeenCalledTimes(1)
  })

  it('leaves the fault of the page\'s own picture standing when it is taken up after it', async () => {
    const { ComfyError } = await import('../src/lib/comfy')
    const desk = await import('../src/routes/Pictures')
    m.follow.mockImplementation(() => new Promise(() => {}))
    setSnap({ available: false, reason: 'off', groups: [group()], jobs: [job(J, G, 0, 1)] })
    let fail: (e: unknown) => void = () => {}
    m.run.mockImplementation(() => new Promise((_res, rej) => (fail = rej)))
    await desk.startRuns([plan('mine')])
    await vi.waitFor(() => expect(m.run).toHaveBeenCalledTimes(1), WAIT)
    setSnap({ available: true, reason: null })
    fail(new ComfyError('out of memory'))
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.id).toBe(J), WAIT)
    expect(desk.pressSnapshot().fault?.message).toBe('out of memory')
  })

  it('clears an old fault of the page\'s own when another device\'s batch is taken up later, not as the press came free', async () => {
    const { ComfyError } = await import('../src/lib/comfy')
    const desk = await import('../src/routes/Pictures')
    m.follow.mockImplementation(() => new Promise(() => {}))
    setSnap({ available: false, reason: 'off' })
    m.run.mockRejectedValueOnce(new ComfyError('out of memory'))
    await desk.startRuns([plan('mine')])
    await vi.waitFor(() => expect(desk.pressSnapshot().fault?.message).toBe('out of memory'), WAIT)
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.status).toBe('error'), WAIT)
    // Past the look taken as the press came free.
    await new Promise((done) => setTimeout(done, 20))
    // The queue is back, and another device starts a batch.
    setSnap({ available: true, reason: null, groups: [group()], jobs: [job(J, G, 0, 1, { status: 'running', promptId: 'px' })] })
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.id).toBe(J), WAIT)
    expect(desk.pressSnapshot().fault).toBeNull()
  })

  it('is shown under a Make refused as busy because of it, with no word from the server after', async () => {
    const desk = await import('../src/routes/Pictures')
    m.follow.mockImplementation(() => new Promise(() => {}))
    m.submitGroup.mockImplementation(async () => {
      // The server lists the batch while the hand-over is out, and says nothing after.
      setSnap({ groups: [group()], jobs: [job(J, G, 0, 1)], lane: paused })
      return { ok: false, fallback: false, status: 409, error: 'busy', busy: 'images' }
    })
    await desk.startRuns([plan('one')])
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.id).toBe(J), WAIT)
    expect(desk.pressSnapshot().fault?.message).toBe(desk.BATCH_BUSY)
  })

  it('waits while pictures an earlier page sent from here are on the press, then is taken up after the last', async () => {
    const ends: ((r: unknown) => void)[] = []
    m.followPrompt.mockImplementation(() => new Promise((res) => ends.push(res)))
    m.follow.mockImplementation(() => new Promise(() => {}))
    const filing = { composition, seed: 1, familyLabel: 'I', modelLabel: 'M', variant: null, passes: { face: false, hand: false, hires: false }, loras: [] }
    tab.set('switchgen.picturesent.v1', JSON.stringify({
      writer: 'other',
      released: true,
      jobs: [
        { id: 'j1', promptId: 'p1', startedAt: 5, label: 'first', ...filing },
        { id: 'j2', promptId: 'p2', startedAt: 6, label: 'second', ...filing },
      ],
    }))
    m.snap = { ...blank(), groups: [group()], jobs: [job(J, G, 0, 1)], lane: paused }
    const desk = await import('../src/routes/Pictures')
    expect(desk.pressSnapshot().job?.id).toBe('j1')
    ends[0]!({ status: 'error', message: 'one broke', node: null, nodeType: null })
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.id).toBe('j2'), WAIT)
    await new Promise((res) => setTimeout(res, 20))
    expect(desk.pressSnapshot().job?.id).toBe('j2')
    expect(m.follow).not.toHaveBeenCalled()
    ends[1]!({ status: 'error', message: 'two broke', node: null, nodeType: null })
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.id).toBe(J), WAIT)
    expect(desk.pressSnapshot().fault?.message).toBe('two broke')
  })
})

describe('a 409 and the batch it names', () => {
  it('keeps the busy line when the other batch is taken up', async () => {
    const desk = await import('../src/routes/Pictures')
    m.submitGroup.mockResolvedValue({ ok: false, fallback: false, status: 409, error: 'busy', busy: 'images' })
    m.follow.mockImplementation(() => new Promise(() => {}))
    await desk.startRuns([plan('one')])
    const g = '66666666-6666-4666-8666-666666666666'
    const j = '77777777-7777-4777-8777-777777777777'
    setSnap({
      groups: [{ id: g, desk: 'images', kind: 'batch', label: 'x', device: 'other', createdAt: 1, state: 'active', endedBy: null, endedAt: null, jobIds: [j], dismissed: false }],
      jobs: [job(j, g, 0, 1, { status: 'running', promptId: 'px' })],
    })
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.id).toBe(j), WAIT)
    expect(desk.pressSnapshot().fault?.message).toBe(desk.BATCH_BUSY)
    expect(desk.pressSnapshot().job?.status).toBe('running')
  })
})

describe('what the batch hands over and shows', () => {
  it('keeps labels and prompts within what the server takes', async () => {
    const desk = await import('../src/routes/Pictures')
    let body: R.SubmitBody | null = null
    m.submitGroup.mockImplementation(async (b: R.SubmitBody) => {
      body = b
      return { ok: true, replayed: false, group: {} as R.RunnerGroup, jobs: [] }
    })
    m.follow.mockImplementation(() => new Promise(() => {}))
    const long = plan('L'.repeat(300))
    ;(long.composition as { prompt: string }).prompt = 'p'.repeat(5000)
    await desk.startRuns([long, plan('two')])
    const b = body as unknown as R.SubmitBody
    expect(b.group.label.length).toBeLessThanOrEqual(200)
    expect(b.jobs[0]!.label.length).toBeLessThanOrEqual(200)
    expect(b.jobs[0]!.prompt.length).toBeLessThanOrEqual(4000)
    // The record keeps the prompt whole.
    expect(b.jobs[0]!.record.prompt).toBe('p'.repeat(5000))
  })

  it('puts the record already on file on the plate for an answer from the cache, and claims no time', async () => {
    const desk = await import('../src/routes/Pictures')
    const history = await import('../src/lib/history')
    history.restore({ ...(composition as Record<string, unknown>), id: 'e-old', no: 4, at: 5, file: png('same.png'), kind: 'image', promptId: 'p0', durationMs: 900, familyLabel: 'I', modelLabel: 'M' } as never)
    m.submitGroup.mockResolvedValue({ ok: true, replayed: false, group: {} as R.RunnerGroup, jobs: [] })
    m.follow.mockResolvedValue({ files: [{ ...png('same.png'), cached: true }], primary: png('same.png'), frame: null, entryId: 'e-old', entryNo: 4, repeatOf: 'e-old', openedOn: null, durationMs: 777, finishedAt: 50 })
    await desk.startRuns([plan('one')])
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.status).toBe('done'), WAIT)
    expect(desk.pressSnapshot().current?.id).toBe('e-old')
    expect(desk.pressSnapshot().results.map((r) => r.id)).toEqual(['e-old'])
    expect(desk.pressSnapshot().lastMs).toBeNull()
  })

  it('says Stopping when another page stopped the batch', async () => {
    const desk = await import('../src/routes/Pictures')
    let b0: R.SubmitBody | null = null
    m.submitGroup.mockImplementation(async (b: R.SubmitBody) => {
      b0 = b
      return { ok: true, replayed: false, group: {} as R.RunnerGroup, jobs: [] }
    })
    m.follow.mockImplementation(() => new Promise(() => {}))
    await desk.startRuns([plan('one')])
    const b = b0 as unknown as R.SubmitBody
    setSnap({
      groups: [{ id: b.group.id, desk: 'images', kind: 'batch', label: 'x', device: 'd', createdAt: 1, state: 'active', endedBy: null, endedAt: null, jobIds: [b.jobs[0]!.id], dismissed: false }],
      jobs: [job(b.jobs[0]!.id, b.group.id, 0, 1, { status: 'queued', promptId: 'p1' })],
    })
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.stage).toBe('Queued'), WAIT)
    setSnap({ jobs: [job(b.jobs[0]!.id, b.group.id, 0, 1, { status: 'queued', promptId: 'p1', stopRequested: true })] })
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.stage).toBe('Stopping'), WAIT)
    expect(m.stopGroup).not.toHaveBeenCalled()
  })
})

describe('a batch the server holds', () => {
  const G = '88888888-8888-4888-8888-888888888888'
  const ids = ['99999999-9999-4999-8999-999999999991', '99999999-9999-4999-8999-999999999992', '99999999-9999-4999-8999-999999999993']
  const batch = (jobs: R.RunnerJob[], lane: R.RunnerSnapshot['lane'] = { held: null }): R.RunnerSnapshot => ({
    ...blank(),
    lane,
    groups: [{ id: G, desk: 'images', kind: 'batch', label: 'x', device: 'other', createdAt: 1, state: 'active', endedBy: null, endedAt: null, jobIds: ids, dismissed: false }],
    jobs,
  })

  it('says which of it is held and where the word is given, and not that it goes in turn', async () => {
    const { WAITS_ON_SERVER } = await import('../src/lib/wakeLock')
    m.snap = batch(
      [job(ids[0]!, G, 0, 3, { wait: { for: 'held' } }), job(ids[1]!, G, 1, 3, { wait: { for: 'before' } }), job(ids[2]!, G, 2, 3, { wait: { for: 'before' } })],
      { held: { why: 'restart', scope: 'all', jobId: null, since: 1 } },
    )
    m.follow.mockImplementation(() => new Promise(() => {}))
    const desk = await import('../src/routes/Pictures')
    await feedStore(m.snap)
    await vi.waitFor(() => expect(desk.pressSnapshot().job).toMatchObject({ id: ids[0], runner: true, status: 'submitting', stage: 'Held' }), WAIT)
    expect(desk.pressSnapshot().job?.onHold).toEqual({ press: true, rest: 2 })
    let line = desk.serverWaitingLine(desk.pressSnapshot().job)!
    // The note is the store's own, for the hold it holds.
    expect(line).toContain('The machine restarted while this work waited, so it is held until you say.')
    expect(line).toContain('2 more pictures wait on the SwitchGen server and are held with it.')
    expect(line).toContain('The notice about held work sends them')
    expect(line).not.toContain('sent in turn')

    // The word is given: the hold goes, and the batch goes in turn.
    setSnap(batch([job(ids[0]!, G, 0, 3, { wait: { for: 'turn' } }), job(ids[1]!, G, 1, 3, { wait: { for: 'before' } }), job(ids[2]!, G, 2, 3, { wait: { for: 'before' } })]))
    await feedStore(m.snap)
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.onHold).toBeNull(), WAIT)
    line = desk.serverWaitingLine(desk.pressSnapshot().job)!
    expect(line).toContain('2 more pictures wait on the SwitchGen server and are sent one at a time.')
    expect(line).toContain(WAITS_ON_SERVER)
    expect(line).not.toContain('held')
  })

  it('says the picture on the press waits its turn once the word has let the hold go, before its own wait follows', async () => {
    const held = { held: { why: 'restart' as const, scope: 'all' as const, jobId: null, since: 1 } }
    const jobs = () => [job(ids[0]!, G, 0, 3, { wait: { for: 'held' } }), job(ids[1]!, G, 1, 3, { wait: { for: 'before' } }), job(ids[2]!, G, 2, 3, { wait: { for: 'before' } })]
    m.snap = batch(jobs(), held)
    m.follow.mockImplementation(() => new Promise(() => {}))
    const desk = await import('../src/routes/Pictures')
    await feedStore(m.snap)
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.stage).toBe('Held'), WAIT)
    // The word cleared the hold at once; the picture's wait says 'held' until the queue's next round.
    setSnap(batch(jobs()))
    await feedStore(m.snap)
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.stage).toBe('Waiting its turn'), WAIT)
    expect(desk.pressSnapshot().job?.onHold).toBeNull()
    const line = desk.serverWaitingLine(desk.pressSnapshot().job)!
    expect(line).not.toContain('Held until you say')
    expect(line).not.toContain('held until you say')
  })

  it('counts only its heavy pictures under a hold on heavy work', async () => {
    m.snap = batch(
      [job(ids[0]!, G, 0, 3, { wait: { for: 'turn' } }), job(ids[1]!, G, 1, 3, { wait: { for: 'before' } }), job(ids[2]!, G, 2, 3, { heavy: true, wait: { for: 'before' } })],
      { held: { why: 'lost', scope: 'heavy', jobId: 'x', since: 1 } },
    )
    m.follow.mockImplementation(() => new Promise(() => {}))
    const desk = await import('../src/routes/Pictures')
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.id).toBe(ids[0]), WAIT)
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.onHold).toEqual({ press: false, rest: 1 }), WAIT)
    const line = desk.serverWaitingLine(desk.pressSnapshot().job)!
    expect(line).toContain('2 more pictures wait on the SwitchGen server, and one of them is held until you say.')
    expect(line).toContain('The notice about held work sends it')
  })

  it('says the same of the press and the rest, whichever of them is held', async () => {
    const { WAITS_ON_SERVER } = await import('../src/lib/wakeLock')
    const desk = await import('../src/routes/Pictures')
    const running = desk.serverWaitingLine({ status: 'running', stage: 'Drawing · step 3 of 20', index: 1, total: 3, runner: true, note: null, onHold: { press: false, rest: 2 } })!
    expect(running).toBe('2 more pictures wait on the SwitchGen server, held until you say. The notice about held work sends them, or calls them off, with any other work held.')
    expect(running).not.toContain(WAITS_ON_SERVER)
    const last = desk.serverWaitingLine({ status: 'submitting', stage: 'Held', index: 3, total: 3, runner: true, note: 'Held until you say.', onHold: { press: true, rest: 0 } })!
    expect(last).toBe('Held until you say. The notice about held work sends it, or calls it off, with any other work held.')
  })
})

describe('a stop the server did not take', () => {
  const G = 'aaaaaaaa-1111-4aaa-8aaa-aaaaaaaaaaaa'
  const ids = ['bbbbbbbb-1111-4bbb-8bbb-bbbbbbbbbbb1', 'bbbbbbbb-1111-4bbb-8bbb-bbbbbbbbbbb2', 'bbbbbbbb-1111-4bbb-8bbb-bbbbbbbbbbb3']
  const drawing = (status: 'running' | 'queued'): R.RunnerSnapshot => ({
    ...blank(),
    groups: [{ id: G, desk: 'images', kind: 'batch', label: 'x', device: 'other', createdAt: 1, state: 'active', endedBy: null, endedAt: null, jobIds: ids, dismissed: false }],
    jobs: [job(ids[0]!, G, 0, 3, { status, wait: null, promptId: 'p1' }), job(ids[1]!, G, 1, 3, { wait: { for: 'before' } }), job(ids[2]!, G, 2, 3, { wait: { for: 'before' } })],
    progress: { [ids[0]!]: { id: ids[0]!, classType: 'KSampler', value: 3, max: 20, node: '3', pass: null, at: 1, previewN: 0 } },
  })

  for (const status of ['running', 'queued'] as const) {
    it(`gives the picture back its real stage at once, with the server's list saying ${status}`, async () => {
      m.snap = drawing(status)
      let events = 0
      m.follow.mockImplementation((id: string, on: (e: R.FollowEvent) => void) => {
        if (id === ids[0]) {
          on({ phase: 'queued', promptId: 'p1' })
          on({ phase: 'running', node: '3', value: 3, max: 20, classType: 'KSampler', pass: null })
          events = 2
        }
        // A long step: nothing more is reported.
        return new Promise(() => {})
      })
      m.stopGroup.mockResolvedValue(false)
      const desk = await import('../src/routes/Pictures')
      await vi.waitFor(() => expect(desk.pressSnapshot().job).toMatchObject({ id: ids[0], status: 'running', stage: 'Drawing · step 3 of 20' }), WAIT)
      desk.stopPress()
      await vi.waitFor(() => expect(m.stopGroup).toHaveBeenCalledWith(G), WAIT)
      await vi.waitFor(() => expect(desk.pressSnapshot().job?.stage).toBe('Drawing · step 3 of 20'), WAIT)
      expect(events).toBe(2)
      expect(desk.pressSnapshot().job?.status).toBe('running')
    })
  }
})

const OFF_G = '12121212-1212-4212-8212-121212121212'
const OFF_IDS = ['34343434-3434-4434-8434-343434343431', '34343434-3434-4434-8434-343434343432']
const OFF = 'Turned off with SWITCHGEN_RUNNER=off.'
const offGroup = (state: 'active' | 'ended' = 'active', label = 'a lighthouse at dusk, 2 pictures'): R.RunnerGroup =>
  ({ id: OFF_G, desk: 'images', kind: 'batch', label, device: 'other', createdAt: 1, state, endedBy: null, endedAt: null, jobIds: OFF_IDS, dismissed: false })

describe('a batch the server keeps while its queue is off', () => {
  it('is not taken up while the queue is off: the press stays free for the page\'s own pictures, with a line saying why, and no Stop', async () => {
    m.snap = { ...blank(), available: false, reason: OFF, groups: [offGroup()], jobs: [job(OFF_IDS[0], OFF_G, 0, 2, { wait: null }), job(OFF_IDS[1], OFF_G, 1, 2, { wait: null })] }
    m.follow.mockImplementation(() => new Promise(() => {}))
    const desk = await import('../src/routes/Pictures')
    expect(m.follow).not.toHaveBeenCalled()
    expect(desk.pressSnapshot().job).toBeNull()
    const line = desk.pressSnapshot().parked
    expect(line).toContain('“a lighthouse at dusk, 2 pictures”: 2 pictures wait on the SwitchGen server, whose queue is off.')
    expect(line).toContain(OFF)
    expect(line).toContain('cannot stop them until the queue is back')
    m.run.mockImplementation(() => new Promise(() => {}))
    const shown = await desk.startRuns([plan('mine')])
    expect(shown).not.toBeNull()
    expect(m.run).toHaveBeenCalledTimes(1)
    expect(m.submitGroup).not.toHaveBeenCalled()
    desk.stopPress()
    expect(m.stopGroup).not.toHaveBeenCalled()
  })

  it('is let go from the press when the queue goes off, and the press is the page\'s again', async () => {
    m.snap = { ...blank(), groups: [offGroup()], jobs: [job(OFF_IDS[0], OFF_G, 0, 2, { status: 'queued', promptId: 'p1', wait: null }), job(OFF_IDS[1], OFF_G, 1, 2, { wait: { for: 'before' } })] }
    const signals: (AbortSignal | undefined)[] = []
    m.follow.mockImplementation((_id: string, _on: unknown, opts: { signal?: AbortSignal }) => {
      signals.push(opts?.signal)
      return new Promise(() => {})
    })
    const desk = await import('../src/routes/Pictures')
    await vi.waitFor(() => expect(desk.pressSnapshot().job).toMatchObject({ id: OFF_IDS[0], runner: true }), WAIT)
    expect(desk.pressSnapshot().parked).toBeNull()
    setSnap({ available: false, reason: 'Another SwitchGen server holds the archive, and the queue with it.' })
    expect(desk.pressSnapshot().job).toBeNull()
    expect(signals[0]?.aborted).toBe(true)
    expect(desk.pressSnapshot().parked).toContain('Another SwitchGen server holds the archive')
    // The press is free for the page's own work.
    m.run.mockImplementation(() => new Promise(() => {}))
    expect(await desk.startRuns([plan('mine')])).not.toBeNull()
    expect(m.run).toHaveBeenCalledTimes(1)
    // Stop goes to the page's own picture, never to the server.
    desk.stopPress()
    expect(m.stopGroup).not.toHaveBeenCalled()
    expect(m.follow).toHaveBeenCalledTimes(1)
  })

  it('is taken up again once the queue is back, while the press is free', async () => {
    m.snap = { ...blank(), groups: [offGroup()], jobs: [job(OFF_IDS[0], OFF_G, 0, 2, { status: 'queued', promptId: 'p1', wait: null }), job(OFF_IDS[1], OFF_G, 1, 2, { wait: { for: 'before' } })] }
    m.follow.mockImplementation(() => new Promise(() => {}))
    const desk = await import('../src/routes/Pictures')
    await vi.waitFor(() => expect(m.follow).toHaveBeenCalledTimes(1), WAIT)
    setSnap({ available: false, reason: OFF })
    expect(desk.pressSnapshot().job).toBeNull()
    setSnap({ available: true, reason: null })
    await vi.waitFor(() => expect(m.follow).toHaveBeenCalledTimes(2), WAIT)
    expect(desk.pressSnapshot().job).toMatchObject({ id: OFF_IDS[0], runner: true })
    expect(desk.pressSnapshot().parked).toBeNull()
  })

  it('keeps a stop the reader asked for, and asks it again once the queue is back, never while it is off', async () => {
    m.snap = { ...blank(), groups: [offGroup()], jobs: [job(OFF_IDS[0], OFF_G, 0, 2, { status: 'queued', promptId: 'p1', wait: null }), job(OFF_IDS[1], OFF_G, 1, 2, { wait: { for: 'before' } })] }
    m.follow.mockImplementation(() => new Promise(() => {}))
    let answer: (v: boolean) => void = () => {}
    m.stopGroup.mockImplementation(() => new Promise((r) => (answer = r)))
    const desk = await import('../src/routes/Pictures')
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.id).toBe(OFF_IDS[0]), WAIT)
    desk.stopPress()
    await vi.waitFor(() => expect(m.stopGroup).toHaveBeenCalledTimes(1), WAIT)
    setSnap({ available: false, reason: OFF })
    answer(false)
    expect(desk.pressSnapshot().job).toBeNull()
    expect(desk.pressSnapshot().parked).toContain('You asked to stop them')
    expect(JSON.parse(tab.get('switchgen.pictures.dropped.v1') ?? '[]')).toContain(OFF_G)
    m.stopGroup.mockReset()
    m.stopGroup.mockResolvedValue(true)
    setSnap({ rev: 2 })
    await new Promise((r) => setTimeout(r, 20))
    expect(m.stopGroup).not.toHaveBeenCalled()
    setSnap({ available: true, reason: null })
    await vi.waitFor(() => expect(m.stopGroup).toHaveBeenCalledWith(OFF_G), WAIT)
    expect(m.follow).toHaveBeenCalledTimes(1)
  })

  it('shows a hand-over with no answer yet, which the server lists while its queue is off, in the line, and does not follow it', async () => {
    const desk = await import('../src/routes/Pictures')
    let b0: R.SubmitBody | null = null
    m.submitGroup.mockImplementation(async (b: R.SubmitBody) => {
      b0 = b
      return { ok: false, fallback: false, pending: true }
    })
    m.follow.mockImplementation(() => new Promise(() => {}))
    await desk.startRuns([plan('one')])
    expect(desk.pressSnapshot().job?.stage).toBe('Handing it to the server')
    const b = b0 as unknown as R.SubmitBody
    setSnap({
      available: false, reason: OFF,
      groups: [{ id: b.group.id, desk: 'images', kind: 'batch', label: 'one', device: 'd', createdAt: 1, state: 'active', endedBy: null, endedAt: null, jobIds: [b.jobs[0].id], dismissed: false }],
      jobs: [job(b.jobs[0].id, b.group.id, 0, 1, { wait: null })],
    })
    expect(m.follow).not.toHaveBeenCalled()
    expect(desk.pressSnapshot().job).toBeNull()
    expect(desk.pressSnapshot().parked).toContain('“one”: one picture waits')
    // Kept for the tab, and taken up with its plans once the queue is back.
    expect(tab.get('switchgen.pictures.runner.v1')).toBe(b.group.id)
    setSnap({ available: true, reason: null })
    await vi.waitFor(() => expect(m.follow).toHaveBeenCalledWith(b.jobs[0].id, expect.anything(), expect.anything()), WAIT)
  })

  it('does not follow a hand-over answered just as the queue went off', async () => {
    const desk = await import('../src/routes/Pictures')
    m.submitGroup.mockImplementation(async () => {
      setSnap({ available: false, reason: OFF })
      return { ok: true, replayed: false, group: {} as R.RunnerGroup, jobs: [] }
    })
    await desk.startRuns([plan('one')])
    expect(m.follow).not.toHaveBeenCalled()
    expect(desk.pressSnapshot().job).toBeNull()
  })

  it('keeps the tab\'s note of its batch on a list from a queue that is off', async () => {
    tab.set('switchgen.pictures.runner.v1', '55555555-5555-4555-8555-555555555555')
    m.snap = { ...blank(), available: false, reason: OFF }
    await import('../src/routes/Pictures')
    expect(tab.has('switchgen.pictures.runner.v1')).toBe(true)
  })
})

describe('the line under a batch waiting on the server', () => {
  it('does not say it carries on by itself while the queue is off, since it is held for the reader\'s word once the queue is back', async () => {
    const { WAITS_ON_SERVER } = await import('../src/lib/wakeLock')
    const desk = await import('../src/routes/Pictures')
    const j = { status: 'submitting' as const, stage: 'Waiting for the server’s queue', index: 1, total: 3, runner: true, note: OFF, onHold: null }
    expect(desk.serverWaitingLine(j, true)).toContain(WAITS_ON_SERVER)
    const off = desk.serverWaitingLine(j, false)!
    expect(off).not.toContain(WAITS_ON_SERVER)
    expect(off).toContain(OFF)
    m.snap = { ...blank(), available: false }
    expect(desk.serverWaitingLine(j)).not.toContain(WAITS_ON_SERVER)
    expect(desk.serverWaitingLine({ ...j, note: null, index: 3 }, false)).toBeNull()
  })
})
