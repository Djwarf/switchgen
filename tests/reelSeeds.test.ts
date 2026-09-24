import { beforeEach, describe, expect, it, vi } from 'vitest'
import type * as Engine from '../src/components/reel/engine'
import { recipeFor } from '../src/components/reel/recipe'
import type { ApiWorkflow, OutputFile, ProgressEvent } from '../src/lib/comfy'
import { shotPlan } from '../src/lib/continuation'
import { newComposition } from '../src/lib/session'
import { FAMILIES, defaultsFor, type Params } from '../src/lib/workflows'

// A reel's seeds across edits to the strip, with the queue stood in for.
const m = vi.hoisted(() => ({ run: vi.fn(), cancelJob: vi.fn(), getJob: vi.fn(), fetchPastRun: vi.fn() }))
vi.mock('../src/lib/comfy', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../src/lib/comfy')>()),
  run: m.run,
  cancelJob: m.cancelJob,
  getJob: m.getJob,
  fetchPastRun: m.fetchPastRun,
}))

const T2V = FAMILIES.find((f) => f.id === 'wan22-14b-t2v')!
function params(seed: number): Params {
  const d = defaultsFor(T2V, '')
  return {
    model: '',
    positive: '',
    negative: d.negative ?? '',
    seed,
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
const ctx: Engine.RunContext = {
  familyLabel: 'x',
  modelLabel: 'y',
  compositionFor: (job) => newComposition('video', { mode: 't2v', familyId: T2V.id, model: '', prompt: job.params.positive }),
}

let made = 0
const finishes = async (wf: ApiWorkflow, on: (e: ProgressEvent) => void): Promise<OutputFile[]> => {
  on({ phase: 'queued', promptId: `p${made + 1}` })
  made++
  const prefix = String(Object.values(wf).find((n) => n.class_type.startsWith('Save'))?.inputs.filename_prefix ?? 'x')
  return [
    { filename: `${made}.webm`, subfolder: prefix, type: 'output', kind: 'video' },
    { filename: `${made}.png`, subfolder: prefix, type: 'output', kind: 'image' },
  ]
}

let engine: typeof Engine
let store: typeof import('../src/components/reel/store')
beforeEach(async () => {
  for (const f of Object.values(m)) f.mockReset()
  made = 0
  vi.resetModules()
  engine = await import('../src/components/reel/engine')
  store = await import('../src/components/reel/store')
})
const settled = () =>
  vi.waitFor(() => {
    if (engine.reelRun.busy()) throw new Error('still walking')
  }, { timeout: 5000 })

type Shot = ReturnType<typeof import('../src/components/reel/store').newShot>
/** Plan the draft the way the desk does. */
const planOf = (shots: Shot[], fixed: boolean, seed: number) =>
  shotPlan({
    base: T2V,
    params: params(seed),
    shots: shots.map((s) => ({ prompt: s.prompt, seed: store.shotSeed(s, fixed) })),
    freshSeeds: !fixed,
  }).jobs

describe('a fixed reel through a cut', () => {
  it('keeps the rendered takes current when a shot before them is cut', async () => {
    m.run.mockImplementation(finishes)
    const shots = ['a', 'b', 'c', 'd'].map((p) => store.newShot(p))
    engine.reelRun.renderAll(shots.map((s) => s.id), planOf(shots, true, 500), ctx)
    await settled()
    const states = engine.reelRun.snapshot().states

    // Without keeping the seeds, every shot after the cut moves up a rung of
    // the ladder and reads as changed.
    const bare = shots.slice(1)
    const order = bare.map((s) => s.id)
    expect(order.map((_, i) => engine.currencyOf(i, order, planOf(bare, true, 500), states))).toEqual([
      'changed',
      'changed',
      'changed',
    ])

    const keep = store.seedsToKeep(shots, states)
    const kept = shots.map((s) => (keep.has(s.id) ? { ...s, keptSeed: keep.get(s.id)! } : s)).slice(1)
    const jobs = planOf(kept, true, 500)
    expect(order.map((_, i) => engine.currencyOf(i, order, jobs, states))).toEqual(['current', 'current', 'current'])
    expect(engine.shotsToRender(order, jobs, states)).toEqual([])

    // A shot added in the gap takes the ladder at its place, and only it is due.
    const added = [kept[0]!, store.newShot('new'), ...kept.slice(1)]
    const addedJobs = planOf(added, true, 500)
    expect(addedJobs[1]!.params.seed).toBe(501)
    expect(engine.shotsToRender(added.map((s) => s.id), addedJobs, states)).toEqual([1])
  })
})

describe('a new seed on a fixed reel', () => {
  // Typing a reel seed, or the dice, reads every take made with the old one as
  // changed. The next cut kept each take's old seed all the same, so the strip
  // read current again and the new seed was never rendered.
  it('keeps no take made with the old seed, so the cut leaves every shot due', async () => {
    m.run.mockImplementation(finishes)
    const shots = ['a', 'b', 'c', 'd'].map((p) => store.newShot(p))
    engine.reelRun.renderAll(shots.map((s) => s.id), planOf(shots, true, 500), ctx)
    await settled()
    const states = engine.reelRun.snapshot().states

    const keep = store.seedsToKeep(shots, states, 900)
    expect(keep.size).toBe(0)
    const rest = shots.slice(1).map((s) => (keep.has(s.id) ? { ...s, keptSeed: keep.get(s.id)! } : s))
    expect(engine.shotsToRender(rest.map((s) => s.id), planOf(rest, true, 900), states)).toEqual([0, 1, 2])

    // At the seed the takes were made with, all of them are kept, as before.
    expect(store.seedsToKeep(shots, states, 500).size).toBe(4)
  })

  it('keeps no take for a shot whose own seed was cleared, and keeps the ladder shot after it', async () => {
    m.run.mockImplementation(finishes)
    const shots = [{ ...store.newShot('a'), seed: 7 }, store.newShot('b')]
    engine.reelRun.renderAll(shots.map((s) => s.id), planOf(shots, true, 500), ctx)
    await settled()
    const states = engine.reelRun.snapshot().states

    const cleared = [{ ...shots[0]!, seed: null, keptSeed: null }, shots[1]!]
    const keep = store.seedsToKeep(cleared, states, 500)
    expect(keep.has(shots[0]!.id)).toBe(false)
    expect(keep.get(shots[1]!.id)).toBe(501)
  })

  it('keeps every take of a Random pass when it is fixed, and none against another seed', async () => {
    m.run.mockImplementation(finishes)
    const shots = ['a', 'b'].map((p) => store.newShot(p))
    // Random draws the reel's seed at the press; this pass drew 300.
    engine.reelRun.renderAll(shots.map((s) => s.id), planOf(shots, false, 300), ctx)
    await settled()
    const states = engine.reelRun.snapshot().states

    expect(shots.map((s) => store.seedsToKeep(shots, states).get(s.id))).toEqual([300, 301])
    expect(store.seedsToKeep(shots, states, 999).size).toBe(0)
  })
})

describe('Random lets kept seeds go', () => {
  it('ignores a kept seed on Random, so the press draws fresh ones, and never a typed one', () => {
    const s = { ...store.newShot('a'), keptSeed: 777 }
    expect(store.shotSeed(s, true)).toBe(777)
    expect(store.shotSeed(s, false)).toBeUndefined()
    expect(planOf([s], false, 500)[0]!.seedFixed).toBe(false)
    expect(store.shotSeed({ ...s, seed: 5 }, false)).toBe(5)
    expect(store.shotSeed({ ...s, seed: 5 }, true)).toBe(5)
  })

  it('keeps and releases in one write each, and a copy does not take the kept seed', () => {
    const id = store.reel.add()
    store.reel.keepSeeds(new Map([[id, 42]]))
    expect(store.reel.get().shots.find((x) => x.id === id)!.keptSeed).toBe(42)
    const copy = store.reel.duplicate(id)!
    expect(store.reel.get().shots.find((x) => x.id === copy)!.keptSeed).toBeNull()
    store.reel.releaseSeeds()
    expect(store.reel.get().shots.every((x) => x.keptSeed === null)).toBe(true)
  })
})

describe('switching the reel between the 14B pairs', () => {
  const spec = { min: 1, max: 9999, step: 1 }
  const choice = (id: string) => {
    const def = FAMILIES.find((f) => f.id === id)!
    return { def, model: '', label: def.label, chainable: true, why: null, width: spec, height: spec, frames: spec }
  }

  it('gives each pair its own length', () => {
    expect(recipeFor(choice('wan22-14b-t2v')).length).toBe(81)
    expect(recipeFor(choice('wan22-14b-i2v')).length).toBe(49)
    expect(recipeFor(choice('wan22-14b-i2v')).familyId).toBe('wan22-14b-i2v')
  })
})
