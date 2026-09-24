import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type * as Engine from '../src/components/reel/engine'
import type { OutputFile, ProgressEvent } from '../src/lib/comfy'
import type * as Session from '../src/lib/session'

/**
 * The reel's shot on its way to ComfyUI, and a tab the phone threw away. A
 * shot is saved under the id it is sent with before it goes, so a page lost
 * while the send is out leaves the next one a number to ask ComfyUI about. A
 * tab's own page id is kept in the tab, so a page reloaded after a discard
 * knows the shot and pass its last page held for its own. The queue and the
 * wake lock are stand-ins; the engine, the plan and the tab's storage are real.
 */
const m = vi.hoisted(() => ({ run: vi.fn(), cancelJob: vi.fn(), getJob: vi.fn(), fetchPastRun: vi.fn(), hold: vi.fn(), letGo: vi.fn() }))
vi.mock('../src/lib/comfy', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../src/lib/comfy')>()),
  run: m.run,
  cancelJob: m.cancelJob,
  getJob: m.getJob,
  fetchPastRun: m.fetchPastRun,
}))
vi.mock('../src/lib/wakeLock', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../src/lib/wakeLock')>()),
  holdAwake: (why: string) => {
    m.hold(why)
    return m.letGo
  },
}))

const KEY = 'switchgen.reelrun.v1'
const TAB_KEY = 'switchgen.reeltab.v1'
const V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/
/** vi.waitFor with room for a slow runner: its own default gives up after one second. */
const waitFor = <T>(fn: () => T) => vi.waitFor(fn, { timeout: 8000, interval: 10 })

let engine: typeof Engine
/**
 * The tab: its session storage, holding the id its last page ran under when
 * there was one, and a document that says whether the browser threw that page
 * away.
 */
function stubTab(previous: string | null, discarded: boolean): Map<string, string> {
  const tab = new Map<string, string>()
  if (previous) tab.set(TAB_KEY, previous)
  vi.stubGlobal('sessionStorage', {
    getItem: (k: string) => tab.get(k) ?? null,
    setItem: (k: string, v: string) => void tab.set(k, String(v)),
    removeItem: (k: string) => void tab.delete(k),
  })
  vi.stubGlobal('document', Object.assign(new EventTarget(), { visibilityState: 'visible', wasDiscarded: discarded }))
  return tab
}
/** Save a run as another page left it, then load the engine as this page does. */
async function load(saved: unknown): Promise<typeof Session> {
  vi.resetModules()
  const session = await import('../src/lib/session')
  session.store.set(KEY, JSON.stringify(saved))
  engine = await import('../src/components/reel/engine')
  return session
}
const savedRun = (session: typeof Session) => JSON.parse(session.store.get(KEY) ?? 'null')
const made = { signature: 's', seed: 1, openedOn: null, frames: 17, fps: 24, width: 832, height: 480 }
const pendingOf = (extra: Record<string, unknown>) => ({
  shotId: 's2',
  promptId: 'p-left',
  label: 'Shot 2',
  order: ['s1', 's2', 's3'],
  startedAt: Date.now() - 1000,
  made,
  frames: 17,
  composition: { kind: 'video' },
  familyLabel: 'x',
  modelLabel: 'y',
  ...extra,
})
async function planOf(prompts: string[]) {
  const { shotPlan } = await import('../src/lib/continuation')
  const { FAMILIES, defaultsFor } = await import('../src/lib/workflows')
  const T2V = FAMILIES.find((f) => f.id === 'wan22-14b-t2v')!
  const d = defaultsFor(T2V, '')
  const params = { model: '', positive: '', negative: d.negative ?? '', seed: 500, steps: d.steps, cfg: d.cfg, width: d.width, height: d.height, sampler: d.sampler, scheduler: d.scheduler, length: 17, fps: d.fps || 24 }
  return shotPlan({ base: T2V, params, shots: prompts.map((prompt) => ({ prompt, seed: undefined })), freshSeeds: false }).jobs
}
async function ctxOf(): Promise<Engine.RunContext> {
  const { newComposition } = await import('../src/lib/session')
  return {
    familyLabel: 'x',
    modelLabel: 'y',
    compositionFor: (job) => newComposition('video', { mode: 't2v', familyId: 'wan22-14b-t2v', model: '', prompt: job.params.positive }),
    memory: () => ({ level: 'ok', reason: null, release: false }),
  }
}
const clipFiles = (n: number): OutputFile[] => [
  { filename: `${n}.webm`, subfolder: 'reel', type: 'output', kind: 'video' },
  { filename: `${n}.png`, subfolder: 'reel', type: 'output', kind: 'image' },
]
type Send = (wf: unknown, on: (e: ProgressEvent) => void, opts?: { promptId?: string }) => Promise<OutputFile[]>

beforeEach(() => {
  for (const f of Object.values(m)) f.mockReset()
  m.cancelJob.mockResolvedValue(true)
})
afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe('a shot on its way to ComfyUI', () => {
  it('is saved under the id it goes with, as still being sent, before it goes, and as ComfyUI\'s once it answers', async () => {
    stubTab(null, false)
    const session = await load({ shots: [], pending: null })
    let atSend: Record<string, unknown> | null = null
    let sendAs: string | undefined
    let finish: (files: OutputFile[]) => void = () => {}
    m.run.mockImplementationOnce(((_wf, on, opts) =>
      new Promise<OutputFile[]>((resolve) => {
        atSend = savedRun(session).pending
        sendAs = opts?.promptId
        on({ phase: 'queued', promptId: sendAs! })
        finish = resolve
      })) as Send)
    engine.reelRun.renderAll(['s1'], await planOf(['a']), await ctxOf())
    await waitFor(() => expect(m.run).toHaveBeenCalled())
    expect(sendAs).toMatch(V4)
    expect(atSend).toMatchObject({ shotId: 's1', promptId: sendAs, sending: true })
    expect(savedRun(session).pending).toMatchObject({ promptId: sendAs, sending: false })
    finish(clipFiles(1))
    await waitFor(() => expect(engine.reelRun.busy()).toBe(false))
  })

  it('is shown as not sent when the page before went while sending it and ComfyUI never had it, and is not sent again', async () => {
    // ComfyUI is asked again four seconds on, on a clock moved by hand.
    vi.useFakeTimers()
    stubTab(null, false)
    m.getJob.mockResolvedValue(null)
    await load({ shots: [], pending: pendingOf({ owner: 'old', beat: 0, released: true, sending: true }) })
    expect(engine.reelRun.snapshot().states.s2!.stage).toMatch(/Asking ComfyUI whether it arrived/)
    await vi.advanceTimersByTimeAsync(4000)
    await waitFor(() => expect(engine.reelRun.busy()).toBe(false))
    // Asked twice: ComfyUI checks a graph before it queues it.
    expect(m.getJob).toHaveBeenCalledTimes(2)
    const run = engine.reelRun.snapshot()
    expect(run.states.s2!.stage).toBe('Not sent')
    expect(run.note).toMatch(/was being sent when the page before this one went, and ComfyUI has no job under its number/)
    expect(run.note).toMatch(/Shot 3 after it has no clip/)
    expect(m.run).not.toHaveBeenCalled()
  })

  it('is followed as a shot on the press once ComfyUI shows it has it', async () => {
    // On a faked clock, so the follow's next ask, four seconds on, never
    // reaches the stand-ins of the tests after this one.
    vi.useFakeTimers()
    stubTab(null, false)
    m.getJob.mockResolvedValue({ id: 'p-left', status: 'in_progress' })
    const session = await load({ shots: [], pending: pendingOf({ owner: 'old', beat: 0, released: true, sending: true }) })
    await waitFor(() => expect(savedRun(session).pending.sending).toBe(false))
    expect(engine.reelRun.snapshot().states.s2!.stage).toBe('Drawing')
    engine.reelRun.stop()
  })

  it('carries no number until ComfyUI shows it, and a Stop pressed meanwhile is sent once it does', async () => {
    vi.useFakeTimers()
    stubTab(null, false)
    let there = false
    m.getJob.mockImplementation(async () => (there ? { id: 'p-left', status: 'pending' } : null))
    await load({ shots: [], pending: pendingOf({ owner: 'old', beat: 0, released: true, sending: true }) })
    // The section bar reads a number as the queue's word that the job is in it.
    expect(engine.reelRun.snapshot().states.s2!.promptId).toBeNull()
    await vi.advanceTimersByTimeAsync(0)
    expect(m.getJob).toHaveBeenCalledTimes(1)
    engine.reelRun.stop()
    expect(m.cancelJob).not.toHaveBeenCalled()
    there = true
    await vi.advanceTimersByTimeAsync(4000)
    expect(m.cancelJob).toHaveBeenCalledWith('p-left')
    expect(engine.reelRun.snapshot().states.s2!.promptId).toBe('p-left')
  })

  it('is asked about once, as before, when it was saved without the mark', async () => {
    stubTab(null, false)
    m.getJob.mockResolvedValue(null)
    await load({ shots: [], pending: pendingOf({ owner: 'old', beat: 0, released: true }) })
    await waitFor(() => expect(engine.reelRun.busy()).toBe(false))
    expect(m.getJob).toHaveBeenCalledTimes(1)
    expect(engine.reelRun.snapshot().states.s2!.stage).toBe('Failed')
  })
})

describe('a tab the phone threw away', () => {
  it('takes the shot its own last page held at once, with no pagehide and a fresh beat', async () => {
    vi.useFakeTimers()
    stubTab('prev', true)
    m.getJob.mockResolvedValue({ id: 'p-left', status: 'in_progress' })
    const session = await load({
      shots: [],
      pending: pendingOf({ owner: 'prev', beat: Date.now(), released: false }),
      press: { owner: 'prev', beat: Date.now(), released: false, shotId: 's2' },
    })
    expect(engine.reelRun.busy()).toBe(true)
    expect(engine.reelRun.snapshot().elsewhere).toBeNull()
    expect(savedRun(session).pending.owner).not.toBe('prev')
    engine.reelRun.stop()
  })

  it('is told apart from a copied tab, which was not thrown away and still waits', async () => {
    stubTab('prev', false)
    await load({ shots: [], pending: pendingOf({ owner: 'prev', beat: Date.now(), released: false }) })
    expect(engine.reelRun.busy()).toBe(false)
    expect(engine.reelRun.snapshot().elsewhere).toEqual({ shotId: 's2', left: false })
  })

  it('still leaves another tab\'s shot alone', async () => {
    stubTab('prev', true)
    await load({ shots: [], pending: pendingOf({ owner: 'other', beat: Date.now(), released: false }) })
    expect(engine.reelRun.busy()).toBe(false)
    expect(engine.reelRun.snapshot().elsewhere).toEqual({ shotId: 's2', left: false })
  })

  it('frees a pass it held with no shot on the press, and says which shots were never sent', async () => {
    const tab = stubTab('prev', true)
    const session = await load({
      shots: [],
      pending: null,
      press: { owner: 'prev', beat: Date.now(), released: false, shotId: 's2', order: ['s1', 's2', 's3'], queue: ['s1', 's2', 's3'], startedAt: Date.now() - 5000 },
    })
    const run = engine.reelRun.snapshot()
    expect(run.elsewhere).toBeNull()
    expect(run.status).toBe('stopped')
    expect(run.note).toBe(
      'The page before this one went before it sent shots 2 and 3, so they have no clip and nothing is rendering them. Render the reel sends them from here.',
    )
    expect(savedRun(session).press).toBeNull()
    expect(tab.get(TAB_KEY)).not.toBe('prev')
  })
})

describe('a pass the page before let go between shots', () => {
  it('is said once, after a plain reload, naming the shot it never sent', async () => {
    stubTab('prev', false)
    const shot = { shotId: 's1', clip: clipFiles(1)[0], frame: clipFiles(1)[1], files: clipFiles(1), entryId: null, durationMs: 5, finishedAt: 1, made }
    const session = await load({
      shots: [shot],
      pending: null,
      press: { owner: 'prev', beat: Date.now(), released: true, shotId: 's2', order: ['s1', 's2'], queue: ['s1', 's2'], startedAt: 0 },
    })
    expect(engine.reelRun.snapshot().note).toBe(
      'The page before this one went before it sent shot 2, so it has no clip and nothing is rendering it. Render what is missing sends it from here.',
    )
    expect(savedRun(session).press).toBeNull()
    // The page after this one says nothing more of it.
    vi.resetModules()
    engine = await import('../src/components/reel/engine')
    expect(engine.reelRun.snapshot().note).toBeNull()
  })

  it('is not this page\'s to speak of when another tab let it go', async () => {
    stubTab('prev', false)
    await load({
      shots: [],
      pending: null,
      press: { owner: 'other', beat: Date.now(), released: true, shotId: 's2', order: ['s1', 's2'], queue: ['s1', 's2'], startedAt: 0 },
    })
    expect(engine.reelRun.snapshot().status).toBe('idle')
    expect(engine.reelRun.snapshot().note).toBeNull()
  })
})

describe('the screen through a pass', () => {
  it('is let sleep once ComfyUI has the last shot, not when that shot lands', async () => {
    stubTab(null, false)
    await load({ shots: [], pending: null })
    const finish: ((files: OutputFile[]) => void)[] = []
    for (const id of ['p1', 'p2']) {
      m.run.mockImplementationOnce(((_wf, on) =>
        new Promise<OutputFile[]>((resolve) => {
          on({ phase: 'queued', promptId: id })
          finish.push(resolve)
        })) as Send)
    }
    engine.reelRun.renderAll(['s1', 's2'], await planOf(['a', 'b']), await ctxOf())
    await waitFor(() => expect(m.run).toHaveBeenCalledTimes(1))
    expect(m.hold).toHaveBeenCalledTimes(1)
    // Shot 2 still waits in the page.
    expect(m.letGo).not.toHaveBeenCalled()
    finish[0]!(clipFiles(1))
    await waitFor(() => expect(m.run).toHaveBeenCalledTimes(2))
    expect(m.letGo).toHaveBeenCalledTimes(1)
    expect(engine.reelRun.busy()).toBe(true)
    finish[1]!(clipFiles(2))
    await waitFor(() => expect(engine.reelRun.busy()).toBe(false))
    expect(m.letGo).toHaveBeenCalledTimes(1)
  })
})
