import { readFileSync } from 'node:fs'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type * as Engine from '../src/components/reel/engine'
import type { ApiWorkflow, OutputFile, ProgressEvent } from '../src/lib/comfy'
import { NODE_IDS, shotPlan, type ShotJob } from '../src/lib/continuation'
import { newComposition } from '../src/lib/session'
import { FAMILIES, defaultsFor, type Params } from '../src/lib/workflows'

// The reel's walk through its shots, with the queue, the stop, the wait for an
// idle ComfyUI and the memory release stood in for. Nothing reaches ComfyUI.
const m = vi.hoisted(() => ({
  run: vi.fn(),
  cancelJob: vi.fn(),
  getJob: vi.fn(),
  fetchPastRun: vi.fn(),
  release: vi.fn(),
  wait: vi.fn(),
}))

vi.mock('../src/lib/comfy', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../src/lib/comfy')>()),
  run: m.run,
  cancelJob: m.cancelJob,
  getJob: m.getJob,
  fetchPastRun: m.fetchPastRun,
}))
vi.mock('../src/lib/clipMemory', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../src/lib/clipMemory')>()),
  releaseComfyMemory: m.release,
  waitForIdleComfy: m.wait,
}))

type Run = (wf: ApiWorkflow, on: (e: ProgressEvent) => void) => Promise<OutputFile[]>
const fam = (id: string) => FAMILIES.find((f) => f.id === id)!

function params(def = fam('wan22-5b')): Params {
  const model = def.dualModel ? '' : def.models[0]!
  const d = defaultsFor(def, model)
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

const jobsFor = (prompts: string[], fresh = true, id = 'wan22-5b'): ShotJob[] =>
  shotPlan({ base: fam(id), params: params(fam(id)), shots: prompts.map((prompt) => ({ prompt })), freshSeeds: fresh }).jobs

const ctx: Engine.RunContext = {
  familyLabel: 'x',
  modelLabel: 'y',
  compositionFor: (job) =>
    newComposition('video', { mode: 't2v', familyId: 'wan22-5b', model: job.params.model, prompt: job.params.positive }),
}
const releasing = { ...ctx, memory: () => ({ level: 'ok' as const, reason: null, release: true }) }

let made = 0
/** The clip and last frame a shot's graph asks for, named the way ComfyUI names them. */
function filesOf(wf: ApiWorkflow): OutputFile[] {
  made++
  const frame = String(wf[NODE_IDS.frameSave]?.inputs.filename_prefix ?? 'frame')
  const clipPrefix = frame.replace(/\.frame$/, '')
  const folder = clipPrefix.slice(0, clipPrefix.lastIndexOf('/'))
  const stem = clipPrefix.slice(clipPrefix.lastIndexOf('/') + 1)
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
  m.cancelJob.mockResolvedValue(true)
  m.wait.mockResolvedValue(true)
  made = 0
  vi.resetModules()
  engine = await import('../src/components/reel/engine')
})

const settled = () =>
  vi.waitFor(() => {
    if (engine.reelRun.busy()) throw new Error('still walking')
  })

describe('a shot that needs memory released first', () => {
  // ComfyUI applies a release to the next prompt it takes, so the reel waits
  // for the queue to empty, releases, and submits, with nothing in between.
  it('waits for ComfyUI to be idle, says so, then releases, then submits', async () => {
    let letGo: (v: boolean) => void = () => {}
    m.wait.mockImplementation(
      (_s: AbortSignal, onWait: (n: number) => void) =>
        new Promise<boolean>((resolve) => {
          onWait(2)
          letGo = resolve
        }),
    )
    m.run.mockImplementation(finishes)
    engine.reelRun.renderAll(['s1'], jobsFor(['a']), releasing)
    await vi.waitFor(() => expect(engine.reelRun.snapshot().states.s1?.stage).toBe('Waiting for ComfyUI to finish 2 other jobs'))
    expect(m.release).not.toHaveBeenCalled()
    expect(m.run).not.toHaveBeenCalled()
    letGo(true)
    await settled()
    expect(m.wait.mock.invocationCallOrder[0]!).toBeLessThan(m.release.mock.invocationCallOrder[0]!)
    expect(m.release.mock.invocationCallOrder[0]!).toBeLessThan(m.run.mock.invocationCallOrder[0]!)
    expect(engine.reelRun.snapshot().status).toBe('done')
  })

  it('sends nothing when stopped during the wait', async () => {
    m.wait.mockImplementation(
      (s: AbortSignal) => new Promise<boolean>((resolve) => s.addEventListener('abort', () => resolve(false))),
    )
    m.run.mockImplementation(finishes)
    engine.reelRun.renderAll(['s1', 's2'], jobsFor(['a', 'b']), releasing)
    await vi.waitFor(() => expect(m.wait).toHaveBeenCalled())
    engine.reelRun.stop()
    await settled()
    expect(m.release).not.toHaveBeenCalled()
    expect(m.run).not.toHaveBeenCalled()
    const r = engine.reelRun.snapshot()
    expect(r.status).toBe('stopped')
    expect(r.note).toBe('Stopped during shot 1.')
  })

  it('does not wait for a shot whose verdict needs no release', async () => {
    m.run.mockImplementation(finishes)
    engine.reelRun.renderAll(['s1'], jobsFor(['a']), { ...ctx, memory: () => ({ level: 'ok', reason: null, release: false }) })
    await settled()
    expect(m.wait).not.toHaveBeenCalled()
    expect(m.release).not.toHaveBeenCalled()
  })
})

describe('what the reel says when a pass ends early', () => {
  it('says only that the one shot failed when it was rendered alone', async () => {
    m.run.mockImplementation(finishes)
    const order = ['s1', 's2', 's3']
    const jobs = jobsFor(['a', 'b', 'c'])
    engine.reelRun.renderAll(order, jobs, ctx)
    await settled()
    m.run.mockImplementation(async () => {
      throw new Error('boom')
    })
    engine.reelRun.renderOne(1, order, jobs, ctx)
    await settled()
    expect(engine.reelRun.snapshot().note).toBe('Shot 2 failed.')
  })

  it('names the shot that was going to open on the failed one\'s last frame', async () => {
    let n = 0
    m.run.mockImplementation(async (wf: ApiWorkflow, on: (e: ProgressEvent) => void) => {
      n++
      if (n === 2) throw new Error('boom')
      return finishes(wf, on)
    })
    engine.reelRun.renderAll(['s1', 's2', 's3'], jobsFor(['a', 'b', 'c']), ctx)
    await settled()
    expect(engine.reelRun.snapshot().note).toBe(
      'Shot 2 failed, and shot 3 was going to open on its last frame, so the rest of this pass was not sent.',
    )
  })

  it('does not talk of last frames on a style that cannot chain', async () => {
    let n = 0
    m.run.mockImplementation(async (wf: ApiWorkflow, on: (e: ProgressEvent) => void) => {
      n++
      if (n === 1) throw new Error('boom')
      return finishes(wf, on)
    })
    engine.reelRun.renderAll(['s1', 's2'], jobsFor(['a', 'b'], true, 'wan22-14b-t2v'), ctx)
    await settled()
    expect(engine.reelRun.snapshot().note).toBe('Shot 1 failed, so the rest of this pass was not sent.')
  })

  it('keeps a shot\'s earlier clip when its re-render is stopped, and says so', async () => {
    m.run.mockImplementation(finishes)
    const order = ['s1', 's2', 's3']
    const jobs = jobsFor(['a', 'b', 'c'])
    engine.reelRun.renderAll(order, jobs, ctx)
    await settled()
    const { ComfyError } = await import('../src/lib/comfy')
    let stopRun: (e: unknown) => void = () => {}
    m.run.mockImplementation(((_wf, on) =>
      new Promise((_resolve, reject) => {
        stopRun = reject
        on({ phase: 'queued', promptId: 'again' })
      })) as Run)
    m.cancelJob.mockImplementation(async (id: string) => {
      stopRun(new ComfyError('Interrupted', { cancelled: true, promptId: id }))
      return true
    })
    engine.reelRun.renderOne(2, order, jobs, ctx)
    await vi.waitFor(() => expect(engine.reelRun.snapshot().states.s3?.promptId).toBe('again'))
    engine.reelRun.stop()
    await settled()
    expect(engine.reelRun.snapshot().note).toBe('Stopped during shot 3. Its earlier clip is kept.')
  })
})

describe('the cutting room', () => {
  // A key on <Assembly> tied to the strip remounted it whenever a shot was
  // added, and a cut in flight lost its button state and its answer. There is
  // no DOM in this suite to press the button in, so the page is read instead.
  it('is not remounted when the strip changes', () => {
    const page = readFileSync(new URL('../src/routes/Reel.tsx', import.meta.url), 'utf8')
    const tags = page.match(/<Assembly\b[^>]*>/g) ?? []
    expect(tags.length).toBeGreaterThan(0)
    for (const tag of tags) expect(tag).not.toMatch(/\bkey=/)
  })
})
