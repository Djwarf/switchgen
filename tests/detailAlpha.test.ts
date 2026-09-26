import { describe, expect, it } from 'vitest'
import { decodesWithAlpha, deriveAutoDetail, deriveHiresFix } from '../src/lib/refine'
import { FAMILIES } from '../src/lib/workflows'

/**
 * Qwen-Image 2.1 decodes a picture with a transparency channel, and the face
 * detailer (Impact Pack) fails on one with "'Tensor' object has no attribute
 * 'copy'". The face and hand passes therefore hand it the colour only, through
 * ComfyUI's own Split Image with Alpha; every other family's graph is unchanged.
 */
const family = (id: string) => {
  const def = FAMILIES.find((f) => f.id === id)
  if (!def) throw new Error(`no family ${id}`)
  return def
}
type Graph = Record<string, { class_type: string; inputs: Record<string, unknown> }>
const nodeOf = (g: Graph, type: string) => Object.entries(g).find(([, n]) => n.class_type === type)

describe('the face and hand passes on a picture with a transparency channel', () => {
  for (const target of ['face', 'hand'] as const) {
    it(`give the ${target} pass Qwen-Image 2.1's colour only`, () => {
      const d = deriveAutoDetail(family('qwen-image-21'), target)
      expect(d).not.toBeNull()
      const g = d!.graph as Graph
      const split = nodeOf(g, 'SplitImageWithAlpha')
      const fd = nodeOf(g, 'FaceDetailer')
      expect(split).toBeDefined()
      expect(fd![1].inputs.image).toEqual([split![0], 0])
      // The split reads the decoded picture itself.
      const decode = nodeOf(g, 'VAEDecode')
      expect(split![1].inputs.image).toEqual([decode![0], 0])
    })
  }

  it('does the same for a Qwen-Image 2.1 picture made larger first', () => {
    const bigger = deriveHiresFix(family('qwen-image-21'))
    expect(bigger).not.toBeNull()
    expect(decodesWithAlpha(bigger!)).toBe(true)
    const g = deriveAutoDetail(bigger!, 'face')!.graph as Graph
    const split = nodeOf(g, 'SplitImageWithAlpha')
    expect(split).toBeDefined()
    expect(nodeOf(g, 'FaceDetailer')![1].inputs.image).toEqual([split![0], 0])
  })

  it('leaves every other family as it was', () => {
    for (const def of FAMILIES.filter((f) => f.mode === 'image' && f.id !== 'qwen-image-21')) {
      const d = deriveAutoDetail(def, 'face')
      if (!d) continue
      expect(nodeOf(d.graph as Graph, 'SplitImageWithAlpha'), def.id).toBeUndefined()
    }
  })
})
