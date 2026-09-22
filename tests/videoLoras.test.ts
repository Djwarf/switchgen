import { afterEach, describe, expect, it, vi } from 'vitest'
import { loadStack, saveStack } from '../src/lib/loras'
import { canTakeVideoLoras, withVideoLoras } from '../src/lib/refine'
import {
  chainVideoStack,
  collapsePairs,
  expandVideoStack,
  pairedHalf,
  partnerOf,
  rackFromRecord,
  restoreRack,
  videoLorasToRun,
} from '../src/lib/videoLoras'
import { FAMILIES } from '../src/lib/workflows'

type Graph = Record<string, { class_type: string; inputs: Record<string, unknown> }>

const family = (id: string) => FAMILIES.find((f) => f.id === id)!
const loaderCount = (g: Graph) => Object.values(g).filter((n) => n.class_type === 'LoraLoaderModelOnly').length

/**
 * The node ids a sampler's model passes through, from the sampler up to the
 * first node with no model input of its own, which should be the weights
 * loader. Counting loaders says nothing about wiring: a loader added and left
 * dangling counts the same, and ComfyUI prunes it, so the add-on does nothing.
 */
function upstream(graph: Graph, samplerId: string): string[] {
  const path: string[] = []
  let cur = graph[samplerId]?.inputs.model
  while (Array.isArray(cur) && typeof cur[0] === 'string' && path.length < 32) {
    const id = cur[0]
    path.push(id)
    cur = graph[id]?.inputs.model
  }
  return path
}

const samplers = (g: Graph) => Object.entries(g).filter(([, n]) => /^KSampler/.test(n.class_type)).map(([id]) => id)
const addOnsOn = (g: Graph, path: string[]) =>
  path.filter((id) => g[id]?.class_type === 'LoraLoaderModelOnly').map((id) => String(g[id]!.inputs.lora_name))
const endsAtWeights = (g: Graph, path: string[]) =>
  /^(UNETLoader|UnetLoaderGGUF|CheckpointLoaderSimple)$/.test(g[path.at(-1) ?? '']?.class_type ?? '')

describe('HIGH and LOW pairs', () => {
  it('reads the half off the filename and names its partner', () => {
    expect(pairedHalf('Wan22_I2V_NSFW_General_HIGH.safetensors')).toEqual({ stem: 'Wan22_I2V_NSFW_General.safetensors', half: 'high' })
    expect(partnerOf('Wan22_I2V_NSFW_General_HIGH.safetensors')).toBe('Wan22_I2V_NSFW_General_LOW.safetensors')
    expect(pairedHalf('plain.safetensors')).toBeNull()
  })

  it('expands a pair to one spec per half only when the partner is installed', () => {
    const dual = family('wan22-14b-i2v')
    const both = new Set(['a_HIGH.safetensors', 'a_LOW.safetensors'])
    expect(expandVideoStack([{ name: 'a_HIGH.safetensors', strength: 0.7 }], dual, both)).toEqual([
      { name: 'a_HIGH.safetensors', strength: 0.7, half: 'high' },
      { name: 'a_LOW.safetensors', strength: 0.7, half: 'low' },
    ])
    expect(expandVideoStack([{ name: 'a_HIGH.safetensors', strength: 0.7 }], dual, new Set(['a_HIGH.safetensors']))).toEqual([
      { name: 'a_HIGH.safetensors', strength: 0.7, half: 'both' },
    ])
  })

  it('folds a pair back to one row for the rack', () => {
    const rows = collapsePairs([{ name: 'a_HIGH.safetensors' }, { name: 'a_LOW.safetensors' }, { name: 'b.safetensors' }])
    expect(rows.map((r) => r.name)).toEqual(['a_HIGH.safetensors', 'b.safetensors'])
  })
})

describe('chaining into video graphs', () => {
  it('adds one loader on a one-model family, on the sampler\'s path', () => {
    const def = family('wan22-5b')
    expect(canTakeVideoLoras(def)).toBe(true)
    const out = withVideoLoras(def, [{ name: 'x.safetensors', strength: 0.8 }])!
    expect(loaderCount(out.graph)).toBe(loaderCount(def.graph) + 1)
    const ids = samplers(out.graph)
    expect(ids.length).toBeGreaterThan(0)
    for (const id of ids) {
      const path = upstream(out.graph, id)
      expect(addOnsOn(out.graph, path)).toContain('x.safetensors')
      expect(endsAtWeights(out.graph, path)).toBe(true)
    }
  })

  it('adds one loader per half on a two-model family, in front of the family\'s own', () => {
    const def = family('wan22-14b-i2v')
    const before = loaderCount(def.graph)
    const out = withVideoLoras(def, [{ name: 'x.safetensors', strength: 0.8 }])!
    expect(loaderCount(out.graph)).toBe(before + 2)
    expect(out.derived.loraNodes).toHaveLength(2)
    // Every sampler reaches its weights through one of the new loaders, and
    // the reader's add-on sits before the family's own, which stays on.
    const ids = samplers(out.graph)
    expect(ids).toHaveLength(2)
    for (const id of ids) {
      const path = upstream(out.graph, id)
      expect(path.some((p) => out.derived.loraNodes.includes(p))).toBe(true)
      expect(endsAtWeights(out.graph, path)).toBe(true)
      const names = addOnsOn(out.graph, path)
      expect(names).toContain('x.safetensors')
      expect(names.length).toBeGreaterThan(1)
      expect(names.at(-1)).toBe('x.safetensors')
    }
  })

  it('sends a half-marked spec to its half only', () => {
    const def = family('wan22-14b-t2v')
    const out = chainVideoStack(def, [{ name: 'p_HIGH.safetensors', strength: 0.5 }], new Set(['p_HIGH.safetensors', 'p_LOW.safetensors']))!
    const names = Object.values(out.graph).filter((n) => n.class_type === 'LoraLoaderModelOnly').map((n) => n.inputs.lora_name)
    expect(names.sort()).toEqual(['p_HIGH.safetensors', 'p_LOW.safetensors'])
    // The high-noise pass is the one that starts at step 0.
    const ids = samplers(out.graph)
    const high = ids.find((id) => out.graph[id]!.inputs.start_at_step === 0)
    const low = ids.find((id) => Number(out.graph[id]!.inputs.start_at_step) > 0)
    expect(high && low).toBeTruthy()
    const onHigh = addOnsOn(out.graph, upstream(out.graph, high!))
    const onLow = addOnsOn(out.graph, upstream(out.graph, low!))
    expect(onHigh).toContain('p_HIGH.safetensors')
    expect(onHigh).not.toContain('p_LOW.safetensors')
    expect(onLow).toContain('p_LOW.safetensors')
    expect(onLow).not.toContain('p_HIGH.safetensors')
  })
})

// The Wan 2.2 I2V pair's graph loads this pair itself, one file per half.
const BUILT_IN_HIGH = 'Wan22_I2V_NSFW_General_HIGH.safetensors'
const BUILT_IN_LOW = 'Wan22_I2V_NSFW_General_LOW.safetensors'
const HIGH = 'Foo_HIGH.safetensors'
const LOW = 'Foo_LOW.safetensors'
const pair = new Set([HIGH, LOW])
const loaders = (g: Graph) =>
  Object.entries(g)
    .filter(([, n]) => n.class_type === 'LoraLoaderModelOnly')
    .map(([id, n]) => [id, n.inputs.lora_name])

describe('each half at its own strength', () => {
  const t2v = family('wan22-14b-t2v')

  it('keeps two rows of a pair at their own strengths', () => {
    const both = new Set(['a_HIGH.safetensors', 'a_LOW.safetensors'])
    const specs = [
      { name: 'a_HIGH.safetensors', strength: 0.7 },
      { name: 'a_LOW.safetensors', strength: 0.4 },
    ]
    expect(expandVideoStack(specs, family('wan22-14b-i2v'), both)).toEqual([
      { name: 'a_HIGH.safetensors', strength: 0.7, half: 'high' },
      { name: 'a_LOW.safetensors', strength: 0.4, half: 'low' },
    ])
  })

  it('never pulls in a partner the rack left out, and sends the row to both halves', () => {
    const both = new Set(['a_HIGH.safetensors', 'a_LOW.safetensors'])
    const out = expandVideoStack([{ name: 'a_HIGH.safetensors', strength: 0.7 }], t2v, both, new Set(['a_LOW.safetensors']))
    expect(out).toEqual([{ name: 'a_HIGH.safetensors', strength: 0.7, half: 'both' }])
  })

  it('keeps a half at 0 as its own half\'s row, so that half gets nothing', () => {
    const specs = [
      { name: HIGH, strength: 0.8 },
      { name: LOW, strength: 0 },
    ]
    expect(expandVideoStack(specs, t2v, pair)).toEqual([
      { name: HIGH, strength: 0.8, half: 'high' },
      { name: LOW, strength: 0, half: 'low' },
    ])
    // Only the high half is patched: no LOW file anywhere, and no HIGH file
    // on the low half.
    const out = chainVideoStack(t2v, specs, pair)!
    expect(loaders(out.graph)).toEqual([['__lora_high_1', HIGH]])
    const ids = samplers(out.graph)
    const low = ids.find((id) => Number(out.graph[id]!.inputs.start_at_step) > 0)!
    expect(addOnsOn(out.graph, upstream(out.graph, low))).toEqual([])
  })

  it('hands back no derived graph when every row is at 0', () => {
    expect(chainVideoStack(t2v, [{ name: HIGH, strength: 0 }, { name: LOW, strength: 0 }], pair)).toBeNull()
  })
})

describe('the family\'s own add-ons', () => {
  const i2v = family('wan22-14b-i2v')
  const installed = new Set([BUILT_IN_HIGH, BUILT_IN_LOW])

  it('are taken out of what runs, so neither half is patched twice with them', () => {
    expect(videoLorasToRun(i2v, [{ name: BUILT_IN_HIGH, strength: 1 }], installed)).toEqual([])
    expect(videoLorasToRun(i2v, [{ name: BUILT_IN_LOW, strength: 0.5 }], installed)).toEqual([])
    expect(chainVideoStack(i2v, [{ name: BUILT_IN_HIGH, strength: 1 }, { name: BUILT_IN_LOW, strength: 1 }], installed)).toBeNull()
  })

  it('leave the reader\'s own add-ons alone', () => {
    const run = videoLorasToRun(i2v, [{ name: BUILT_IN_HIGH, strength: 1 }, { name: 'x.safetensors', strength: 0.6 }], installed)
    expect(run).toEqual([{ name: 'x.safetensors', strength: 0.6, half: 'both' }])
  })

  it('are never put back on the rack from a record', () => {
    const rows = rackFromRecord(
      [
        { name: BUILT_IN_HIGH, strength: 0.85, half: 'high' },
        { name: BUILT_IN_LOW, strength: 0.85, half: 'low' },
        { name: 'x.safetensors', strength: 0.6, half: 'both' },
      ],
      i2v,
      installed,
    )
    expect(rows).toEqual([{ file: 'x.safetensors', strength: 0.6, enabled: true }])
  })
})

describe('a rack rebuilt from a record', () => {
  const t2v = family('wan22-14b-t2v')
  const row = (file: string, strength: number, enabled = true) => ({ file, strength, enabled })

  it('keeps a row per half when the halves ran at two strengths, and one when at one', () => {
    expect(rackFromRecord([{ name: HIGH, strength: 0.8, half: 'high' }, { name: LOW, strength: 0.4, half: 'low' }], t2v, pair)).toEqual([
      row(HIGH, 0.8),
      row(LOW, 0.4),
    ])
    expect(rackFromRecord([{ name: HIGH, strength: 0.8, half: 'high' }, { name: LOW, strength: 0.8, half: 'low' }], t2v, pair)).toEqual([
      row(HIGH, 0.8),
    ])
  })

  it('puts a half that ran alone back with its partner at 0', () => {
    expect(rackFromRecord([{ name: HIGH, strength: 0.8, half: 'high' }], t2v, pair)).toEqual([row(HIGH, 0.8), row(LOW, 0)])
    expect(rackFromRecord([{ name: LOW, strength: 0.5, half: 'low' }], t2v, pair)).toEqual([row(HIGH, 0), row(LOW, 0.5)])
  })

  it('switches the partner off for a half that ran on both halves, even when nobody knows what is installed', () => {
    expect(rackFromRecord([{ name: HIGH, strength: 1, half: 'both' }], t2v, pair)).toEqual([row(HIGH, 1), row(LOW, 1, false)])
    expect(rackFromRecord([{ name: HIGH, strength: 1, half: 'both' }], t2v, null)).toEqual([row(HIGH, 1), row(LOW, 1, false)])
  })

  it('folds a pair on a record filed before halves were recorded to the strength that ran', () => {
    // Such a pair ran both halves at the strength of whichever row came first.
    expect(rackFromRecord([{ name: HIGH, strength: 1 }, { name: LOW, strength: 0.6 }], t2v, pair)).toEqual([row(HIGH, 1)])
    expect(rackFromRecord([{ name: LOW, strength: 0.6 }, { name: HIGH, strength: 1 }], t2v, pair)).toEqual([row(HIGH, 0.6)])
  })
})

describe('putting a clip\'s add-ons back on its rack', () => {
  const memory = new Map<string, string>()
  const shim = {
    getItem: (k: string) => memory.get(k) ?? null,
    setItem: (k: string, v: string) => void memory.set(k, String(v)),
    removeItem: (k: string) => void memory.delete(k),
  }
  afterEach(() => {
    memory.clear()
    vi.unstubAllGlobals()
  })

  it('empties a saved rack for a clip that used none, says so, and undoes it', () => {
    vi.stubGlobal('localStorage', shim)
    const saved = [{ file: 'x.safetensors', strength: 0.6, enabled: true }]
    saveStack('wan22-5b', saved)
    const r = restoreRack({ familyId: 'wan22-5b', loras: [] }, null)
    expect(loadStack('wan22-5b')).toEqual([])
    expect(r.note).toBe('The add-ons on the rack were taken off, because this clip used none.')
    r.undo()
    expect(loadStack('wan22-5b')).toEqual(saved)
  })

  it('fills the rack with what the clip used, and says nothing when there was no rack to replace', () => {
    vi.stubGlobal('localStorage', shim)
    const r = restoreRack({ familyId: 'wan22-14b-t2v', loras: [{ name: HIGH, strength: 0.8, half: 'high' }] }, pair)
    expect(loadStack('wan22-14b-t2v')).toEqual([
      { file: HIGH, strength: 0.8, enabled: true },
      { file: LOW, strength: 0, enabled: true },
    ])
    expect(r.note).toBeNull()
  })
})
