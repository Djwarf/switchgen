import { describe, expect, it } from 'vitest'
import { canTakeVideoLoras, withVideoLoras } from '../src/lib/refine'
import { chainVideoStack, collapsePairs, expandVideoStack, pairedHalf, partnerOf } from '../src/lib/videoLoras'
import { FAMILIES } from '../src/lib/workflows'

const family = (id: string) => FAMILIES.find((f) => f.id === id)!
const loaderCount = (g: Record<string, { class_type: string }>) =>
  Object.values(g).filter((n) => n.class_type === 'LoraLoaderModelOnly').length

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
  it('adds one loader on a one-model family', () => {
    const def = family('wan22-5b')
    expect(canTakeVideoLoras(def)).toBe(true)
    const out = withVideoLoras(def, [{ name: 'x.safetensors', strength: 0.8 }])!
    expect(loaderCount(out.graph)).toBe(loaderCount(def.graph) + 1)
  })

  it('adds one loader per half on a two-model family, in front of the family\'s own', () => {
    const def = family('wan22-14b-i2v')
    const before = loaderCount(def.graph)
    const out = withVideoLoras(def, [{ name: 'x.safetensors', strength: 0.8 }])!
    expect(loaderCount(out.graph)).toBe(before + 2)
    expect(out.derived.loraNodes).toHaveLength(2)
    // Every sampler still resolves to a loader through the chain.
    for (const n of Object.values(out.graph)) {
      if (n.class_type !== 'KSamplerAdvanced') continue
      let cur = n.inputs.model as [string, number]
      let hops = 0
      while (hops++ < 10) {
        const up = out.graph[cur[0]]
        if (/^(UNETLoader|UnetLoaderGGUF)$/.test(up.class_type)) break
        cur = up.inputs.model as [string, number]
      }
      expect(hops).toBeLessThan(10)
    }
  })

  it('sends a half-marked spec to its half only', () => {
    const def = family('wan22-14b-t2v')
    const out = chainVideoStack(def, [{ name: 'p_HIGH.safetensors', strength: 0.5 }], new Set(['p_HIGH.safetensors', 'p_LOW.safetensors']))!
    const names = Object.values(out.graph).filter((n) => n.class_type === 'LoraLoaderModelOnly').map((n) => n.inputs.lora_name)
    expect(names.sort()).toEqual(['p_HIGH.safetensors', 'p_LOW.safetensors'])
  })
})
