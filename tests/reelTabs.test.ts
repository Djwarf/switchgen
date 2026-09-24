import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type * as Engine from '../src/components/reel/engine'
import type { OutputFile, ProgressEvent } from '../src/lib/comfy'
import { shotPlan } from '../src/lib/continuation'
import { newComposition } from '../src/lib/session'
import { FAMILIES, defaultsFor, type Params } from '../src/lib/workflows'

// A shot left on the press by another tab or an earlier page, saved as the
// reel's pending entry, and what this page does with it on load.
const m = vi.hoisted(() => ({ run: vi.fn(), cancelJob: vi.fn(), getJob: vi.fn(), fetchPastRun: vi.fn() }))
vi.mock('../src/lib/comfy', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../src/lib/comfy')>()),
  run: m.run,
  cancelJob: m.cancelJob,
  getJob: m.getJob,
  fetchPastRun: m.fetchPastRun,
}))

const KEY = 'switchgen.reelrun.v1'
const made = { signature: 's', seed: 1, openedOn: null, frames: 17, fps: 24, width: 832, height: 480 }
const pendingOf = (extra: Record<string, unknown>) => ({
  shotId: 'shot2',
  promptId: 'p-left',
  label: 'Shot 2',
  order: ['shot1', 'shot2'],
  startedAt: Date.now() - 1000,
  made,
  frames: 17,
  composition: { kind: 'video' },
  familyLabel: 'x',
  modelLabel: 'y',
  ...extra,
})

let engine: typeof Engine
/** Save a run as another page left it, then load the engine as this page does. */
async function load(saved: unknown) {
  vi.resetModules()
  const session = await import('../src/lib/session')
  session.store.set(KEY, JSON.stringify(saved))
  engine = await import('../src/components/reel/engine')
  return session
}
beforeEach(() => {
  for (const f of Object.values(m)) f.mockReset()
})
afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

/** What the saved run holds now. */
const savedRun = (session: typeof import('../src/lib/session')) => JSON.parse(session.store.get(KEY) ?? 'null')

// A two-shot T2V reel, planned the way the desk plans it.
const T2V = FAMILIES.find((f) => f.id === 'wan22-14b-t2v')!
function params(): Params {
  const d = defaultsFor(T2V, '')
  return {
    model: '',
    positive: '',
    negative: d.negative ?? '',
    seed: 500,
    steps: d.steps,
    cfg: d.cfg,
    width: d.width,
    height: d.height,
    sampler: d.sampler,
    scheduler: d.scheduler,
    length: 17,
    fps: d.fps || 24,
  }
}
const planOf = (prompts: string[]) =>
  shotPlan({ base: T2V, params: params(), shots: prompts.map((prompt) => ({ prompt, seed: undefined })), freshSeeds: false }).jobs
const ctx: Engine.RunContext = {
  familyLabel: 'x',
  modelLabel: 'y',
  compositionFor: (job) => newComposition('video', { mode: 't2v', familyId: T2V.id, model: '', prompt: job.params.positive }),
}
const clipFiles = (n: number): OutputFile[] => [
  { filename: `${n}.webm`, subfolder: 'reel', type: 'output', kind: 'video' },
  { filename: `${n}.png`, subfolder: 'reel', type: 'output', kind: 'image' },
]
/** A run that reports it was queued as `promptId`, then waits until the test lets it finish. */
function hanging(promptId: string) {
  let finish: (files: OutputFile[]) => void = () => {}
  m.run.mockImplementationOnce(
    (_wf: unknown, on: (e: ProgressEvent) => void) =>
      new Promise<OutputFile[]>((resolve) => {
        on({ phase: 'queued', promptId })
        finish = resolve
      }),
  )
  return { finish: (files: OutputFile[]) => finish(files) }
}
/** A window the engine can listen on, and a way to fire its events. */
function stubWindow() {
  const win = new EventTarget()
  vi.stubGlobal('window', win)
  return {
    fire: (type: string, extra: Record<string, unknown> = {}) => win.dispatchEvent(Object.assign(new Event(type), extra)),
  }
}
const pressOf = (extra: Record<string, unknown>) => ({ owner: 'other', beat: Date.now(), released: false, shotId: 'shot1', ...extra })

describe('one tab follows the shot on the press', () => {
  it('leaves a shot another live tab follows alone, and notes where that tab is', async () => {
    const session = await load({ shots: [], pending: pendingOf({ owner: 'other', beat: Date.now(), released: false }) })
    expect(engine.reelRun.busy()).toBe(false)
    expect(engine.reelRun.snapshot().status).toBe('idle')
    expect(engine.reelRun.snapshot().elsewhere).toEqual({ shotId: 'shot2', left: false })
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(m.getJob).not.toHaveBeenCalled()
    // Clearing is refused while that tab renders, and its entry stays.
    engine.reelRun.clear()
    expect(savedRun(session).pending.owner).toBe('other')
  })

  it('keeps another tab\'s entry through a save of its own', async () => {
    vi.useFakeTimers()
    let job: { id: string; status: string } | null = { id: 'p-left', status: 'in_progress' }
    m.getJob.mockImplementation(async () => job)
    const session = await load({ shots: [], pending: pendingOf({ owner: 'old', beat: 0, released: true }) })
    expect(engine.reelRun.busy()).toBe(true)
    // Another tab saves a shot of its own while this one follows the old one.
    const theirs = pendingOf({ owner: 'other', beat: Date.now(), released: false, shotId: 'shot9', promptId: 'p-other' })
    session.store.set(KEY, JSON.stringify({ shots: [], pending: theirs, press: null }))
    // The followed job is stopped in ComfyUI, and this tab writes its ending.
    job = { id: 'p-left', status: 'cancelled' }
    await vi.advanceTimersByTimeAsync(4000)
    expect(engine.reelRun.busy()).toBe(false)
    expect(savedRun(session).pending).toMatchObject({ owner: 'other', promptId: 'p-other' })
  })

  it('picks up a shot its page released on the way out', async () => {
    m.getJob.mockResolvedValue({ id: 'p-left', status: 'in_progress' })
    await load({ shots: [], pending: pendingOf({ owner: 'other', beat: Date.now(), released: true }) })
    expect(engine.reelRun.busy()).toBe(true)
    expect(engine.reelRun.snapshot().states.shot2?.status).toBe('running')
  })

  it('picks up an entry saved before entries were stamped', async () => {
    m.getJob.mockResolvedValue({ id: 'p-left', status: 'in_progress' })
    await load({ shots: [], pending: pendingOf({}) })
    expect(engine.reelRun.busy()).toBe(true)
  })
})

describe('stopping a picked-up shot', () => {
  it('reads Stopped when the stop takes it out of the queue', async () => {
    let gone = false
    m.getJob.mockImplementation(async () => (gone ? null : { id: 'p-left', status: 'pending' }))
    m.cancelJob.mockImplementation(async () => {
      gone = true
      return true
    })
    await load({ shots: [], pending: pendingOf({ owner: 'old', beat: 0, released: true }) })
    await vi.waitFor(() => expect(m.getJob).toHaveBeenCalled(), { timeout: 5000 })
    engine.reelRun.stop()
    await vi.waitFor(() => expect(engine.reelRun.busy()).toBe(false), { timeout: 10_000 })
    const run = engine.reelRun.snapshot()
    expect(run.status).toBe('stopped')
    expect(run.states.shot2?.status).toBe('stopped')
    expect(run.note).toBe('Shot 2 was stopped.')
  }, 15_000)

  it('files a finished picked-up shot from its own history record', async () => {
    m.getJob.mockResolvedValue({ id: 'p-left', status: 'completed' })
    m.fetchPastRun.mockResolvedValue({
      promptId: 'p-left',
      graph: {},
      status: 'success',
      startedAt: 1,
      finishedAt: 2,
      clientId: null,
      error: null,
      files: [{ filename: 'c.webm', subfolder: 'reel', type: 'output', kind: 'video' }],
    })
    await load({ shots: [], pending: pendingOf({ owner: 'old', beat: 0, released: true }) })
    await vi.waitFor(() => expect(engine.reelRun.busy()).toBe(false), { timeout: 5000 })
    expect(m.fetchPastRun).toHaveBeenCalledWith('p-left')
    expect(engine.reelRun.snapshot().states.shot2?.status).toBe('done')
  })
})

describe('a pass another tab is walking', () => {
  it('holds this tab\'s own press while that tab has a pass on its way', async () => {
    await load({ shots: [], pending: null, press: pressOf({}) })
    expect(engine.reelRun.snapshot().elsewhere).toEqual({ shotId: 'shot1', left: false })
    const jobs = planOf(['a', 'b'])
    engine.reelRun.renderAll(['shot1', 'shot2'], jobs, ctx)
    engine.reelRun.renderOne(1, ['shot1', 'shot2'], jobs, ctx)
    await new Promise((resolve) => setTimeout(resolve, 20))
    expect(m.run).not.toHaveBeenCalled()
    expect(engine.reelRun.busy()).toBe(false)
  })

  it('takes no notice of a pass whose page let it go, or that has gone quiet', async () => {
    await load({ shots: [], pending: null, press: pressOf({ released: true }) })
    expect(engine.reelRun.snapshot().elsewhere).toBeNull()
    await load({ shots: [], pending: null, press: pressOf({ beat: 0 }) })
    expect(engine.reelRun.snapshot().elsewhere).toBeNull()
  })

  it('follows that tab\'s saves: the press free again, then taken again', async () => {
    const win = stubWindow()
    await load({ shots: [], pending: pendingOf({ owner: 'other', beat: Date.now(), released: false }) })
    expect(engine.reelRun.snapshot().elsewhere).toEqual({ shotId: 'shot2', left: false })
    expect(engine.reelRun.snapshot().status).toBe('idle')
    win.fire('storage', { key: KEY, newValue: JSON.stringify({ shots: [], pending: null, press: null }) })
    expect(engine.reelRun.snapshot().elsewhere).toBeNull()
    win.fire('storage', { key: KEY, newValue: JSON.stringify({ shots: [], pending: null, press: pressOf({}) }) })
    expect(engine.reelRun.snapshot().elsewhere).toEqual({ shotId: 'shot1', left: false })
  })

  it('saves this tab\'s own pass with the shot it is on, and lets it go when the pass ends', async () => {
    const session = await load({ shots: [], pending: null })
    const run = hanging('p1')
    engine.reelRun.renderAll(['s1'], planOf(['a']), ctx)
    await vi.waitFor(() => expect(savedRun(session)?.pending?.promptId).toBe('p1'), { timeout: 5000 })
    expect(savedRun(session).press).toMatchObject({ shotId: 's1', released: false })
    run.finish(clipFiles(1))
    await vi.waitFor(() => expect(engine.reelRun.busy()).toBe(false), { timeout: 5000 })
    expect(savedRun(session).press).toBeNull()
    expect(savedRun(session).pending).toBeNull()
  })
})

describe('a page the browser kept, coming back', () => {
  it('ends its pass without sending or filing anything when another page took the reel over', async () => {
    const win = stubWindow()
    const session = await load({ shots: [], pending: null })
    const history = await import('../src/lib/history')
    const run = hanging('p1')
    engine.reelRun.renderAll(['s1', 's2'], planOf(['a', 'b']), ctx)
    await vi.waitFor(() => expect(savedRun(session)?.pending?.promptId).toBe('p1'), { timeout: 5000 })

    win.fire('pagehide')
    expect(savedRun(session).pending.released).toBe(true)
    expect(savedRun(session).press.released).toBe(true)

    // A page that loaded meanwhile took the shot and the pass.
    const theirs = {
      shots: [],
      pending: { ...savedRun(session).pending, owner: 'other', beat: Date.now(), released: false },
      press: { owner: 'other', beat: Date.now(), released: false, shotId: 's1' },
    }
    session.store.set(KEY, JSON.stringify(theirs))
    win.fire('pageshow', { persisted: true })

    await vi.waitFor(() => expect(engine.reelRun.busy()).toBe(false), { timeout: 5000 })
    const snap = engine.reelRun.snapshot()
    expect(snap.status).toBe('stopped')
    expect(snap.note).toMatch(/another tab took the reel over/)
    expect(snap.elsewhere).toEqual({ shotId: 's1', left: false })
    expect(savedRun(session)).toEqual(theirs)

    // The job lands; the page that took it over files it, not this one.
    run.finish(clipFiles(1))
    await new Promise((resolve) => setTimeout(resolve, 20))
    expect(history.all()).toEqual([])
    expect(m.run).toHaveBeenCalledTimes(1)
  })

  it('takes its pass back when nobody took it over, and files the shot as usual', async () => {
    const win = stubWindow()
    const session = await load({ shots: [], pending: null })
    const history = await import('../src/lib/history')
    const run = hanging('p1')
    engine.reelRun.renderAll(['s1'], planOf(['a']), ctx)
    await vi.waitFor(() => expect(savedRun(session)?.pending?.promptId).toBe('p1'), { timeout: 5000 })

    win.fire('pagehide')
    expect(savedRun(session).pending.released).toBe(true)
    win.fire('pageshow', { persisted: true })
    expect(savedRun(session).pending.released).toBe(false)
    expect(savedRun(session).press.released).toBe(false)

    run.finish(clipFiles(1))
    await vi.waitFor(() => expect(engine.reelRun.busy()).toBe(false), { timeout: 5000 })
    expect(engine.reelRun.snapshot().states.s1?.status).toBe('done')
    expect(history.all()).toHaveLength(1)
  })

  it('lets a picked-up shot go when another page took it over, without reading its record', async () => {
    vi.useFakeTimers()
    const win = stubWindow()
    m.getJob.mockResolvedValue({ id: 'p-left', status: 'in_progress' })
    m.fetchPastRun.mockResolvedValue({
      promptId: 'p-left', graph: {}, status: 'success', startedAt: 1, finishedAt: 2, clientId: null, error: null,
      files: [{ filename: 'c.webm', subfolder: 'reel', type: 'output', kind: 'video' }],
    })
    const session = await load({ shots: [], pending: pendingOf({ owner: 'old', beat: 0, released: true }) })
    expect(engine.reelRun.busy()).toBe(true)

    win.fire('pagehide')
    const theirs = {
      shots: [],
      pending: pendingOf({ owner: 'other', beat: Date.now(), released: false }),
      press: { owner: 'other', beat: Date.now(), released: false, shotId: 'shot2' },
    }
    session.store.set(KEY, JSON.stringify(theirs))
    win.fire('pageshow', { persisted: true })
    // The job finishes; only the page that took it over may file it.
    m.getJob.mockResolvedValue({ id: 'p-left', status: 'completed' })
    await vi.advanceTimersByTimeAsync(8000)

    expect(engine.reelRun.busy()).toBe(false)
    expect(engine.reelRun.snapshot().status).toBe('stopped')
    expect(engine.reelRun.snapshot().elsewhere).toEqual({ shotId: 'shot2', left: false })
    expect(m.fetchPastRun).not.toHaveBeenCalled()
  })
})
