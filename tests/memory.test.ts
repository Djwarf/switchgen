import { describe, expect, it } from 'vitest'
import { clipMemory } from '../src/lib/clipMemory'
import { feasibility, type Hardware, type ModelFile } from '../src/lib/hardware'
import { withVideoLoras } from '../src/lib/refine'
import { FAMILIES } from '../src/lib/workflows'

const GiB = 1024 ** 3
const family = (id: string) => FAMILIES.find((f) => f.id === id)!

function machine(ramGiB: number): Hardware {
  return {
    cpu: { cores: 8, model: 'test' },
    ram: { total: ramGiB * GiB, free: ramGiB * GiB },
    gpu: { name: 'test', vramTotal: 16 * GiB, vramUsed: 0, vramFree: 16 * GiB },
    disk: null,
    platform: 'test',
  }
}

/** Roughly the 32 GB machine the two points were measured on, as the OS reports it. */
const MEASURED_MACHINE = machine(31)
const at = (frames: number, width = 832, height = 480) => ({ width, height, frames })

describe('clipMemory: the 14B pairs against the two measured points', () => {
  const pair = family('wan22-14b-t2v')

  it('lets the size that survived through, and still frees memory first', () => {
    expect(clipMemory(pair, at(49), MEASURED_MACHINE)).toEqual({ level: 'ok', reason: null, release: true })
  })

  it('cautions between the two points, up to and including the edge', () => {
    for (const frames of [53, 81]) {
      const v = clipMemory(pair, at(frames), MEASURED_MACHINE)
      expect(v.level).toBe('caution')
      expect(v.release).toBe(true)
      expect(v.reason).toContain('49 frames at 832 × 480')
    }
  })

  it('refuses above the measured edge, with or without a memory reading', () => {
    for (const hw of [MEASURED_MACHINE, null]) {
      const v = clipMemory(pair, at(85), hw)
      expect(v.level).toBe('refuse')
      expect(v.reason).toMatch(/^Too large for memory\./)
      expect(v.reason).toContain('28.1 GB')
    }
    // Size is frames times pixels, so a larger frame crosses the edge sooner.
    expect(clipMemory(pair, at(81, 1280, 720), MEASURED_MACHINE).level).toBe('refuse')
  })

  it('refuses nothing on a machine with clearly more memory, which nobody measured', () => {
    const roomy = machine(64)
    const v = clipMemory(pair, at(121), roomy)
    expect(v.level).toBe('caution')
    expect(v.reason).toContain('which was not measured')
    expect(clipMemory(pair, at(81, 1280, 720), roomy).level).not.toBe('refuse')
  })

  it('leaves every one-model family alone', () => {
    for (const id of ['wan22-5b', 'hunyuan-video', 'ltxv-0_9_6-gguf']) {
      expect(clipMemory(family(id), at(241, 1280, 720), MEASURED_MACHINE)).toEqual({ level: 'ok', reason: null, release: false })
    }
  })
})

describe('feasibility: what a family costs in memory', () => {
  /** Every weight file a graph names, each given `each` bytes. */
  function sizesFor(graph: Record<string, { inputs: Record<string, unknown> }>, each: number): Map<string, ModelFile> {
    const sizes = new Map<string, ModelFile>()
    for (const n of Object.values(graph)) {
      for (const [k, v] of Object.entries(n.inputs)) {
        if (/^(ckpt|unet|clip|vae|lora)_name\d*$/.test(k) && typeof v === 'string') sizes.set(v, { name: v, rel: v, folder: 'x', size: each, mtime: 0 })
      }
    }
    return sizes
  }

  it('counts both of a DualCLIPLoader family\'s text encoders', () => {
    const def = family('hunyuan-video')
    const dual = Object.values(def.graph).find((n) => n.class_type === 'DualCLIPLoader')
    expect(dual).toBeDefined()
    const encoders = [dual!.inputs.clip_name1, dual!.inputs.clip_name2] as string[]
    expect(encoders.every((e) => typeof e === 'string' && e.length > 0)).toBe(true)

    const sizes = sizesFor(def.graph, GiB)
    const v = feasibility(def, sizes, machine(64))
    const counted = v.footprint.files.map((f) => f.name)
    for (const e of encoders) expect(counted).toContain(e)
    expect(v.footprint.weightBytes).toBe(sizes.size * GiB)
    expect(v.footprint.unknown).toEqual([])
  })

  it('prices the graph that will be sent, add-ons included', () => {
    const def = family('wan22-5b')
    const sizes = sizesFor(def.graph, GiB)
    sizes.set('extra.safetensors', { name: 'extra.safetensors', rel: 'Lora/extra.safetensors', folder: 'Lora', size: GiB / 4, mtime: 0 })
    const plain = feasibility(def, sizes, machine(64))
    const chained = withVideoLoras(def, [{ name: 'extra.safetensors', strength: 0.8 }])!
    const withAddOn = feasibility(def, sizes, machine(64), chained.graph)
    expect(withAddOn.footprint.weightBytes - plain.footprint.weightBytes).toBe(GiB / 4)
  })

  it('blocks a family the machine cannot hold at all', () => {
    const def = family('hunyuan-video')
    const v = feasibility(def, sizesFor(def.graph, 8 * GiB), machine(16))
    expect(v.level).toBe('blocked')
    expect(v.selectable).toBe(false)
  })
})
