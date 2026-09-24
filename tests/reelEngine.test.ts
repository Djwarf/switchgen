import { beforeEach, describe, expect, it, vi } from 'vitest'
import type * as Engine from '../src/components/reel/engine'
import type { ApiWorkflow, OutputFile, ProgressEvent } from '../src/lib/comfy'
import { NODE_IDS, shotPlan, type ShotJob } from '../src/lib/continuation'
import { newComposition } from '../src/lib/session'
import { FAMILIES, defaultsFor, type Params } from '../src/lib/workflows'

// Nothing here reaches ComfyUI. The queue, the stop and the memory release are
// stand-ins the tests drive by hand; the engine, the plan and the archive are
// the real modules.
const m = vi.hoisted(() => ({
  run: vi.fn(),
  cancelJob: vi.fn(),
  getJob: vi.fn(),
  pastRuns: vi.fn(),
  release: vi.fn(),
  idle: vi.fn(),
  others: vi.fn(),
}))

vi.mock('../src/lib/comfy', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../src/lib/comfy')>()),
  run: m.run,
  cancelJob: m.cancelJob,
  getJob: m.getJob,
  pastRuns: m.pastRuns,
}))

// The queue is idle whenever the engine asks. The real wait reads /comfy/queue,
// which has no answer here, and it waits through a ComfyUI that does not
// answer for as long as that lasts.
vi.mock('../src/lib/clipMemory', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../src/lib/clipMemory')>()),
  releaseComfyMemory: m.release,
  waitForIdleComfy: m.idle,
  releaseIfOthersAhead: m.others,
}))

type Run = (wf: ApiWorkflow, on: (e: ProgressEvent) => void) => Promise<OutputFile[]>

const FIVE_B = FAMILIES.find((f) => f.id === 'wan22-5b')!

function params(): Params {
  const model = FIVE_B.models[0]!
  const d = defaultsFor(FIVE_B, model)
  return {
    model,
    positive: '',
    negative: d.negative ?? '',
    seed: 100,
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

/** A Random reel, as the desk plans one: the seed is drawn per press, so it is not an edit. */
const jobsFor = (prompts: string[]): ShotJob[] =>
  shotPlan({ base: FIVE_B, params: params(), shots: prompts.map((prompt) => ({ prompt })), freshSeeds: true }).jobs

const ctx: Engine.RunContext = {
  familyLabel: FIVE_B.label,
  modelLabel: FIVE_B.models[0]!,
  compositionFor: (job) =>
    newComposition('video', { mode: 't2v', familyId: FIVE_B.id, model: job.params.model, prompt: job.params.positive }),
}

/** What a finished shot writes: the clip, and the handoff frame named the way the tap names it. */
let made = 0
function filesOf(wf: ApiWorkflow): OutputFile[] {
  made++
  const frame = String(wf[NODE_IDS.frameSave]?.inputs.filename_prefix ?? 'frame')
  const clipPrefix = frame.replace(/\.frame$/, '')
  const [folder, stem] = [clipPrefix.slice(0, clipPrefix.lastIndexOf('/')), clipPrefix.slice(clipPrefix.lastIndexOf('/') + 1)]
  return [
    { filename: `${stem}_${String(made).padStart(5, '0')}_.webm`, subfolder: folder, type: 'output', kind: 'video' },
    { filename: `${stem}.frame_${String(made).padStart(5, '0')}_.png`, subfolder: folder, type: 'output', kind: 'image' },
  ]
}

const finishes: Run = async (wf, on) => {
  on({ phase: 'queued', promptId: `p${made + 1}` })
  return filesOf(wf)
}

let engine: typeof Engine

beforeEach(async () => {
  for (const f of Object.values(m)) f.mockReset()
  m.release.mockResolvedValue(true)
  m.idle.mockResolvedValue(true)
  m.others.mockResolvedValue(undefined)
  m.cancelJob.mockResolvedValue(true)
  made = 0
  vi.resetModules()
  engine = await import('../src/components/reel/engine')
})

const settled = () =>
  vi.waitFor(() => {
    if (engine.reelRun.busy()) throw new Error('still walking')
  }, { timeout: 5000 })

async function renderedReel(prompts: string[]) {
  const order = prompts.map((_, i) => `shot${i + 1}`)
  const jobs = jobsFor(prompts)
  m.run.mockImplementation(finishes)
  engine.reelRun.renderAll(order, jobs, ctx)
  await settled()
  return { order, jobs, states: engine.reelRun.snapshot().states }
}

describe('the reel engine', () => {
  it('renders every shot in order, each continued shot opening on the frame before it', async () => {
    const { order, jobs, states } = await renderedReel(['a car on a coast road', 'past a lighthouse', 'into a tunnel'])
    expect(engine.reelRun.snapshot().status).toBe('done')
    expect(m.run).toHaveBeenCalledTimes(3)
    for (const id of order) expect(states[id]?.status).toBe('done')
    // Shot 2's graph was built on shot 1's handoff frame.
    const shot2 = m.run.mock.calls[1]![0] as ApiWorkflow
    const opened = Object.values(shot2).filter((n) => n.class_type === 'LoadImage').map((n) => n.inputs.image)
    expect(opened).toContain(`${states.shot1!.frame!.subfolder}/${states.shot1!.frame!.filename} [output]`)
    expect(order.map((_, i) => engine.currencyOf(i, order, jobs, states))).toEqual(['current', 'current', 'current'])
  })

  it('keeps every earlier clip when a forced second pass is stopped on its first shot', async () => {
    const { order, jobs, states: first } = await renderedReel(['a car on a coast road', 'past a lighthouse', 'into a tunnel'])
    const { ComfyError } = await import('../src/lib/comfy')

    let stopRun: (err: unknown) => void = () => undefined
    m.run.mockImplementation(((_wf, on) =>
      new Promise((_resolve, reject) => {
        stopRun = reject
        on({ phase: 'queued', promptId: 'p-again' })
      })) as Run)
    m.cancelJob.mockImplementation(async (id: string) => {
      stopRun(new ComfyError('Interrupted', { cancelled: true, promptId: id }))
      return true
    })

    engine.reelRun.renderAll(order, jobs, ctx, { force: true })
    // The pass covers the whole reel, and every shot waits for it.
    expect(engine.reelRun.snapshot().queue).toEqual(order)
    await vi.waitFor(() => expect(engine.reelRun.snapshot().states.shot1?.promptId).toBe('p-again'), { timeout: 5000 })
    engine.reelRun.stop()
    await settled()

    const run = engine.reelRun.snapshot()
    expect(m.cancelJob).toHaveBeenCalledWith('p-again')
    expect(m.run).toHaveBeenCalledTimes(4)
    expect(run.status).toBe('stopped')
    expect(run.queue).toEqual(order)
    for (const id of order) {
      expect(run.states[id]?.status).toBe('done')
      expect(run.states[id]?.clip).toEqual(first[id]!.clip)
      expect(run.states[id]?.frame).toEqual(first[id]!.frame)
    }
  })

  it('never sends a clip the memory verdict refuses, and frees memory before one it cautions', async () => {
    const order = ['shot1', 'shot2']
    const jobs = jobsFor(['a car on a coast road', 'past a lighthouse'])
    m.run.mockImplementation(finishes)

    engine.reelRun.renderAll(order, jobs, {
      ...ctx,
      memory: (job) =>
        job.index === 0
          ? { level: 'caution', reason: 'Larger than the size measured to fit.', release: true }
          : { level: 'refuse', reason: 'Too large for memory.', release: true },
    })
    await settled()

    const run = engine.reelRun.snapshot()
    expect(m.release).toHaveBeenCalledTimes(1)
    expect(m.release.mock.invocationCallOrder[0]!).toBeLessThan(m.run.mock.invocationCallOrder[0]!)
    expect(m.run).toHaveBeenCalledTimes(1)
    expect(run.states.shot1?.status).toBe('done')
    expect(run.states.shot2?.status).toBe('error')
    expect(run.states.shot2?.error).toBe('Too large for memory.')
    expect(run.status).toBe('error')
  })
})

describe('what a whole-reel pass would render', () => {
  it('an edited line re-renders that shot and every shot that opens on it', async () => {
    const { order, states } = await renderedReel(['a car on a coast road', 'past a lighthouse', 'into a tunnel'])
    const edited = jobsFor(['a car on a coast road', 'past a windmill', 'into a tunnel'])
    expect(order.map((_, i) => engine.currencyOf(i, order, edited, states))).toEqual(['current', 'changed', 'current'])
    expect(engine.shotsToRender(order, edited, states)).toEqual([1, 2])
  })

  it('a cut leaves the shot after it stale, opening on a frame that is no longer before it', async () => {
    const { states } = await renderedReel(['a car on a coast road', 'past a lighthouse', 'into a tunnel'])
    const order = ['shot1', 'shot3']
    const cut = jobsFor(['a car on a coast road', 'into a tunnel'])
    expect(order.map((_, i) => engine.currencyOf(i, order, cut, states))).toEqual(['current', 'stale'])
    expect(engine.shotsToRender(order, cut, states)).toEqual([1])
  })

  it('a move changes how the moved shots start, and strands the one after them', async () => {
    const { states } = await renderedReel(['a car on a coast road', 'past a lighthouse', 'into a tunnel'])
    const order = ['shot2', 'shot1', 'shot3']
    const moved = jobsFor(['past a lighthouse', 'a car on a coast road', 'into a tunnel'])
    expect(order.map((_, i) => engine.currencyOf(i, order, moved, states))).toEqual(['changed', 'changed', 'stale'])
    expect(engine.shotsToRender(order, moved, states)).toEqual([0, 1, 2])
  })

  it('renders nothing for a reel that is current, and everything when forced', async () => {
    const { order, jobs, states } = await renderedReel(['a car on a coast road', 'past a lighthouse'])
    expect(engine.shotsToRender(order, jobs, states)).toEqual([])
    expect(engine.shotsToRender(order, jobs, states, true)).toEqual([0, 1])
  })
})
