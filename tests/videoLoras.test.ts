import { describe, expect, it } from 'vitest'
import { canTakeVideoLoras, withVideoLoras } from '../src/lib/refine'
import { chainVideoStack, collapsePairs, expandVideoStack, pairedHalf, partnerOf } from '../src/lib/videoLoras'
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
