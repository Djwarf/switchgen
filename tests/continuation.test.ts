import { describe, expect, it } from 'vitest'
import {
  CHAIN_PREFIX,
  NODE_IDS,
  PLACEHOLDER_IMAGE,
  clipFrames,
  deriveBookend,
  deriveChainTap,
  deriveContinuation,
  instantiateShot,
  jobSignature,
  shotPlan,
  snapLength,
  type ShotJob,
  type ShotSpec,
} from '../src/lib/continuation'
import { FAMILIES, defaultsFor, type Params } from '../src/lib/workflows'

type Graph = Record<string, { class_type: string; inputs: Record<string, unknown> }>

const family = (id: string) => FAMILIES.find((f) => f.id === id)!

function paramsFor(id: string): Params {
  const def = family(id)
  const model = def.dualModel ? '' : def.models[0]!
  const d = defaultsFor(def, model)
  return {
    model,
    positive: '',
    negative: d.negative ?? '',
    seed: 7,
    steps: d.steps,
    cfg: d.cfg,
    width: d.width,
    height: d.height,
    sampler: d.sampler,
    scheduler: d.scheduler,
    length: d.length || undefined,
    fps: d.fps || undefined,
  }
}

const plan = (id: string, shots: ShotSpec[]) => shotPlan({ base: family(id), params: paramsFor(id), shots })
const HANDOFF = `${CHAIN_PREFIX}/shot_00001_.png [output]`

/** The one clip writer: a Save node that is not the handoff tap's. */
const clipWriter = (wf: Graph) =>
  Object.entries(wf).find(([id, n]) => n.class_type.startsWith('Save') && id !== NODE_IDS.frameSave)

describe('which families can carry a reel', () => {
  // The validator used to turn a null derivation into a SKIP, so a family that
  // lost its continuation read the same as one that never had it. These pin
  // down which families must chain.
  it('taps and continues the 5B and the 14B image-to-video pair', () => {
    for (const id of ['wan22-5b', 'wan22-14b-i2v']) {
      expect(deriveChainTap(family(id)), id).not.toBeNull()
      expect(deriveContinuation(family(id)), id).not.toBeNull()
    }
  })

  it('pins both ends only where the conditioning node can take an end frame', () => {
    expect(deriveBookend(family('wan22-14b-i2v'))).not.toBeNull()
    expect(deriveBookend(family('wan22-5b'))).toBeNull()
  })

  it('does not continue the 14B text-to-video pair, which has no slot for a frame', () => {
    expect(deriveChainTap(family('wan22-14b-t2v'))).not.toBeNull()
    expect(deriveContinuation(family('wan22-14b-t2v'))).toBeNull()
  })
})

describe('a two-shot reel on the 5B', () => {
  const jobs = plan('wan22-5b', [{ prompt: 'a car on a coast road' }, { prompt: 'the car passes a lighthouse' }]).jobs

  it('opens shot 2 on the frame shot 1 ends on', () => {
    expect(jobs).toHaveLength(2)
    expect(jobs[0]!.start.from).toBe('none')
    expect(jobs[1]!.start.from).toBe('previous')
    expect(jobs[0]!.outputPrefix).not.toBe(jobs[1]!.outputPrefix)
  })

  it('writes the handoff into the loader, and refuses to build without one', () => {
    const wf = instantiateShot(jobs[1]!, HANDOFF) as Graph
    const loaders = Object.values(wf).filter((n) => n.class_type === 'LoadImage')
    expect(loaders.map((n) => n.inputs.image)).toContain(HANDOFF)
    expect(loaders.map((n) => n.inputs.image)).not.toContain(PLACEHOLDER_IMAGE)
    expect(() => instantiateShot(jobs[1]!, null)).toThrow(/no handoff frame/)
  })

  it('keeps the repeated first frame out of a continued clip, and only out of that', () => {
    const wf = instantiateShot(jobs[1]!, HANDOFF) as Graph
    const drop = wf[NODE_IDS.dropFirst]
    const tap = wf[NODE_IDS.lastFrame]
    expect(drop?.class_type).toBe('ImageFromBatch')
    expect(drop?.inputs.batch_index).toBe(1)
    // The clip writer reads the trimmed batch; the tap still reads the whole
    // decode, so the next handoff is the real final frame.
    const writer = clipWriter(wf)
    expect(writer).toBeDefined()
    expect(Object.values(writer![1].inputs)).toContainEqual([NODE_IDS.dropFirst, 0])
    const decode = (drop!.inputs.image as [string, number])[0]
    expect(wf[decode]?.class_type).toMatch(/^VAEDecode/)
    expect(tap?.inputs.image).toEqual([decode, 0])
    expect(tap?.inputs.batch_index).toBe(-1)

    // The opening shot has nothing to repeat, so it has no trim.
    const first = instantiateShot(jobs[0]!, null) as Graph
    expect(first[NODE_IDS.dropFirst]).toBeUndefined()
  })

  it('counts the frames each clip holds', () => {
    const length = jobs[0]!.params.length!
    expect(clipFrames(jobs[0]!)).toBe(length)
    expect(clipFrames(jobs[1]!)).toBe(length - 1)
  })
})

describe('jobSignature', () => {
  const [job] = plan('wan22-5b', [{ prompt: 'a car on a coast road' }]).jobs
  const variant = (patch: Partial<ShotJob['params']>, rest: Partial<ShotJob> = {}): ShotJob => ({
    ...job!,
    ...rest,
    params: { ...job!.params, ...patch },
  })

  it('ignores what does not change the clip: the seed, the image and the file name', () => {
    const same = jobSignature(job!)
    expect(jobSignature(variant({ seed: 99 }))).toBe(same)
    expect(jobSignature(variant({ image: 'other.png' }))).toBe(same)
    expect(jobSignature(variant({}, { outputPrefix: 'elsewhere/001' }))).toBe(same)
  })

  it('changes with the words, the length and the start', () => {
    const same = jobSignature(job!)
    expect(jobSignature(variant({ positive: 'a boat' }))).not.toBe(same)
    expect(jobSignature(variant({ length: (job!.params.length ?? 0) + 4 }))).not.toBe(same)
    expect(jobSignature(variant({}, { start: { from: 'given', image: 'x.png' } }))).not.toBe(same)
  })
})

describe('a family that only starts from a picture', () => {
  it('refuses shot 1 with no opening frame, and says what fixes it', () => {
    const jobs = plan('wan22-14b-i2v', [{ prompt: 'a cat wakes' }, { prompt: 'the cat stretches' }]).jobs
    expect(jobs[0]!.blocked).toMatch(/^Shot 1 needs an opening frame\./)
    expect(jobs[0]!.blocked).toContain('cannot start from words alone')
    expect(jobs[0]!.blocked).toContain('Pin a start frame on shot 1')
    // Shot 2 opens on shot 1's last frame, so it has one.
    expect(jobs[1]!.start.from).toBe('previous')
    expect(jobs[1]!.blocked).toBeNull()
  })

  it('takes a pinned opening frame', () => {
    const jobs = plan('wan22-14b-i2v', [{ prompt: 'a cat wakes', startImage: 'cat.png' }]).jobs
    expect(jobs[0]!.blocked).toBeNull()
    expect(jobs[0]!.start).toEqual({ from: 'given', image: 'cat.png' })
  })
})

describe('the reel bench on a change of style', () => {
  // The Reel desk's recipeFor() takes the new family's length as
  // snapLength(defaults.length || 81). It is private to src/routes/Reel.tsx,
  // so this checks the figures it reads: moving between the two 14B pairs
  // must not keep the text-to-video pair's 81 frames, which the registry
  // records being killed for memory on the image-to-video pair.
  const lengthFor = (id: string) => {
    const def = family(id)
    return snapLength(defaultsFor(def, def.dualModel ? '' : def.models[0]!).length || 81)
  }

  it('gives the image-to-video pair its own 49 frames', () => {
    expect(lengthFor('wan22-14b-t2v')).toBe(81)
    expect(lengthFor('wan22-14b-i2v')).toBe(49)
  })
})
