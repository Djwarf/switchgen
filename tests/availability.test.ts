import { describe, expect, it } from 'vitest'
import { availabilityOf, inventoryFrom, missingFilesFor, missingWhy, passBlocks } from '../src/lib/availability'
import { feasibility, modelGraph, type Hardware, type ModelFile } from '../src/lib/hardware'
import { intentReport } from '../src/lib/intent'
import { DETECTORS, UPSCALE_MODEL } from '../src/lib/refine'
import { FAMILIES } from '../src/lib/workflows'

const family = (id: string) => FAMILIES.find((f) => f.id === id)!

/** A fake /object_info that lists a file only under the GGUF encoder loader. */
function objectInfo(files: Partial<Record<string, string[]>>) {
  const node = (field: string, list: string[] = []) => ({ input: { required: { [field]: [list] } } })
  return {
    CheckpointLoaderSimple: node('ckpt_name', files.ckpt),
    UNETLoader: node('unet_name', files.unet),
    UnetLoaderGGUF: node('unet_name', files.gguf),
    CLIPLoader: node('clip_name', files.clip),
    CLIPLoaderGGUF: node('clip_name', files.clipGguf),
    VAELoader: node('vae_name', files.vae),
    LoraLoaderModelOnly: node('lora_name', files.lora),
    KSampler: { input: { required: { sampler_name: [['euler']], scheduler: [['simple']] } } },
  }
}

describe('availability', () => {
  it('reads GGUF text encoders, which CLIPLoader alone never lists', () => {
    const def = family('wan22-14b-t2v')
    const inv = inventoryFrom(objectInfo({
      gguf: ['Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf', 'Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf'],
      clipGguf: ['umt5-xxl-encoder-Q4_K_M.gguf'],
      vae: ['wan_2.1_vae.safetensors'],
    }))
    expect(missingFilesFor(def, inv)).toEqual([])
    expect(availabilityOf(def, inv, null, new Map())).toEqual({ ok: true, verdict: null })
  })

  it('names the missing file instead of offering the family', () => {
    const def = family('wan22-14b-t2v')
    const inv = inventoryFrom(objectInfo({
      gguf: ['Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf', 'Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf'],
      vae: ['wan_2.1_vae.safetensors'],
    }))
    expect(missingFilesFor(def, inv)).toEqual(['umt5-xxl-encoder-Q4_K_M.gguf'])
    const a = availabilityOf(def, inv, null, new Map())
    expect(a.ok).toBe(false)
    if (!a.ok) expect(a.why).toContain('umt5-xxl-encoder-Q4_K_M.gguf')
  })

  it('refuses a family the machine cannot hold, with the verdict as the reason', () => {
    const def = family('wan22-5b')
    const inv = inventoryFrom(objectInfo({
      unet: ['wan2.2_ti2v_5B_fp16.safetensors'],
      clip: ['umt5_xxl_fp8_e4m3fn_scaled.safetensors'],
      vae: ['wan2.2_vae.safetensors'],
    }))
    const sizes = new Map([['wan2.2_ti2v_5B_fp16.safetensors', { name: 'wan2.2_ti2v_5B_fp16.safetensors', rel: 'x', folder: 'x', size: 60 * 1024 ** 3, mtime: 0 }]])
    const tiny = { cpu: { cores: 1, model: 'x' }, ram: { total: 8 * 1024 ** 3, free: 4 * 1024 ** 3 }, gpu: null, disk: null, platform: 'test' }
    const a = availabilityOf(def, inv, tiny as never, sizes as never)
    expect(a.ok).toBe(false)
    if (!a.ok) expect(a.why).toMatch(/RAM/)
  })
})

describe('the quality passes against what ComfyUI has', () => {
  const combo = (field: string, options: string[]) => ({ input: { required: { [field]: ['COMBO', { options }] } } })
  const bare = { input: { required: {} } }
  const everything = (upscalers: string[] = [UPSCALE_MODEL]) => ({
    UpscaleModelLoader: combo('model_name', upscalers),
    ImpactGaussianBlurMask: bare,
    UltralyticsDetectorProvider: combo('model_name', [DETECTORS.face, DETECTORS.hand]),
    FaceDetailer: bare,
  })

  it('reads a list ComfyUI sends in its newer COMBO shape', () => {
    const inv = inventoryFrom(everything())
    expect([...inv.upscalers]).toContain(UPSCALE_MODEL)
    expect(passBlocks(inv).refine).toBeNull()
  })

  it('names the file a pass needs when ComfyUI does not list it', () => {
    expect(passBlocks(inventoryFrom(everything([]))).refine).toContain(UPSCALE_MODEL)
  })

  it('names the node pack when the detector loader is missing', () => {
    const { UltralyticsDetectorProvider: _gone, ...rest } = everything()
    const blocks = passBlocks(inventoryFrom(rest))
    expect(blocks.face).toContain('the ComfyUI Impact Subpack')
    expect(blocks.hand).toContain('the ComfyUI Impact Subpack')
  })

  it('blocks nothing when everything is there, or when ComfyUI said nothing at all', () => {
    expect(passBlocks(inventoryFrom(everything()))).toEqual({ refine: null, face: null, hand: null })
    expect(passBlocks(inventoryFrom({}))).toEqual({ refine: null, face: null, hand: null })
  })
})

describe('a family whose files only the GGUF node pack can list', () => {
  const GGUF_LOADERS = ['UnetLoaderGGUF', 'CLIPLoaderGGUF', 'DualCLIPLoaderGGUF']
  /** An /object_info with every file the family needs except its .gguf ones, and no GGUF loaders. */
  function withoutPack(id: string) {
    const needed = missingFilesFor(family(id), inventoryFrom({})).filter((f) => !/\.gguf$/i.test(f))
    const info = objectInfo({ ckpt: needed, unet: needed, clip: needed, vae: needed, lora: needed }) as Record<string, unknown>
    for (const n of GGUF_LOADERS) delete info[n]
    return inventoryFrom(info)
  }

  it('puts it down to the missing node pack, not to files the reader already has', () => {
    for (const id of ['qwen-image-edit', 'wan22-14b-t2v']) {
      const a = availabilityOf(family(id), withoutPack(id), null, new Map())
      expect(a.ok, id).toBe(false)
      if (a.ok) continue
      expect(a.why).toContain('ComfyUI-GGUF node pack')
      expect(a.why).not.toMatch(/[\w-]\.gguf/)
    }
  })

  const onDisk = (names: string[]) =>
    new Map(names.map((name): [string, ModelFile] => [name, { name, rel: name, folder: 'x', size: 1, mtime: 0 }]))
  const t2v = family('wan22-14b-t2v')
  const ggufs = missingFilesFor(t2v, inventoryFrom({})).filter((f) => /\.gguf$/i.test(f))

  it('names a .gguf file the disk listing does not have as well as the pack', () => {
    // Told of the pack alone, a reader would install it and then find the
    // encoder missing too.
    const a = availabilityOf(t2v, withoutPack('wan22-14b-t2v'), null, onDisk(['wan_2.1_vae.safetensors']))
    expect(a.ok).toBe(false)
    if (a.ok) return
    expect(a.why).toContain('ComfyUI-GGUF node pack')
    expect(a.why).toContain('umt5-xxl-encoder-Q4_K_M.gguf')
  })

  it('names only the pack when every .gguf file is on disk', () => {
    expect(ggufs.length).toBeGreaterThan(0)
    const a = availabilityOf(t2v, withoutPack('wan22-14b-t2v'), null, onDisk(['wan_2.1_vae.safetensors', ...ggufs]))
    expect(a.ok).toBe(false)
    if (a.ok) return
    expect(a.why).toContain('ComfyUI-GGUF node pack')
    expect(a.why).not.toMatch(/[\w-]\.gguf/)
  })
})

describe('the sentence for missing files', () => {
  it('lists them plainly, with no "and" between every one', () => {
    const inv = inventoryFrom({ KSampler: { input: { required: {} } } })
    expect(missingWhy(['a.safetensors', 'b.safetensors', 'c.safetensors'], inv, new Map())).toBe(
      'needs a.safetensors, b.safetensors, c.safetensors',
    )
  })
})

describe('pricing one file of a family that lists several', () => {
  const GiB = 1024 ** 3
  const il = family('sdxl-illustrious')
  const hw: Hardware = {
    cpu: { cores: 8, model: 'test' },
    ram: { total: 64 * GiB, free: 64 * GiB },
    gpu: { name: 'test', vramTotal: 16 * GiB, vramUsed: 0, vramFree: 16 * GiB },
    disk: null,
    platform: 'test',
  }
  const file = (name: string, size: number): ModelFile => ({ name, rel: name, folder: 'checkpoints', size, mtime: 0 })
  const sizes = new Map([
    ['waiMatureIllustrious_v30.safetensors', file('waiMatureIllustrious_v30.safetensors', 7 * GiB)],
    ['NoobAI-XL-v1.1.safetensors', file('NoobAI-XL-v1.1.safetensors', 12 * GiB)],
  ])

  it('writes the chosen file into the graph', () => {
    const graph = modelGraph(il, 'NoobAI-XL-v1.1.safetensors')
    expect(Object.values(graph).some((n) => n.inputs.ckpt_name === 'NoobAI-XL-v1.1.safetensors')).toBe(true)
    expect(Object.values(il.graph).some((n) => n.inputs.ckpt_name === 'NoobAI-XL-v1.1.safetensors')).toBe(false)
  })

  it('prices each file at its own size', () => {
    const bytes = (m: string) => feasibility(il, sizes, hw, modelGraph(il, m)).footprint.weightBytes
    expect(bytes('NoobAI-XL-v1.1.safetensors') - bytes('waiMatureIllustrious_v30.safetensors')).toBe(5 * GiB)
  })

  it('ranks two files of one family on their own footprints', () => {
    const report = intentReport(
      { intent: 'anime', explicit: false },
      { families: [il], installed: [...sizes.keys()], sizes, hardware: hw },
    )
    const footprint = (m: string) => report.ranked.find((r) => r.model === m)?.verdict?.footprint.weightBytes
    expect(footprint('NoobAI-XL-v1.1.safetensors')).toBeDefined()
    expect(footprint('NoobAI-XL-v1.1.safetensors')).not.toBe(footprint('waiMatureIllustrious_v30.safetensors'))
  })

  it('prices the file it is asked about when deciding a family is available', () => {
    // All four of the family's checkpoints are installed here, so each of the
    // two files asked about is available and priced on its own size.
    const inv = inventoryFrom(objectInfo({ ckpt: il.models }))
    const bytes = (m: string) => {
      const a = availabilityOf(il, inv, hw, sizes, m)
      return a.ok ? a.verdict?.footprint.weightBytes : undefined
    }
    expect(bytes('NoobAI-XL-v1.1.safetensors')! - bytes('waiMatureIllustrious_v30.safetensors')!).toBe(5 * GiB)
  })
})

describe('a family that lists several weight files, with one of them installed', () => {
  // A family that lists several checkpoints loads one per run, so the one
  // asked about is all it needs; one placed on its own used to read as
  // needing the other three, and was left out of every list on the desk.
  const GiB = 1024 ** 3
  const il = family('sdxl-illustrious')
  const wai = 'waiMatureIllustrious_v30.safetensors'
  const hw: Hardware = {
    cpu: { cores: 8, model: 'test' },
    ram: { total: 64 * GiB, free: 64 * GiB },
    gpu: { name: 'test', vramTotal: 16 * GiB, vramUsed: 0, vramFree: 16 * GiB },
    disk: null,
    platform: 'test',
  }
  const sizes = new Map([[wai, { name: wai, rel: wai, folder: 'checkpoints', size: 7 * GiB, mtime: 0 } as ModelFile]])

  it('is available for the file that is there', () => {
    const inv = inventoryFrom(objectInfo({ ckpt: [wai] }))
    expect(missingFilesFor(il, inv, wai)).toEqual([])
    expect(availabilityOf(il, inv, hw, sizes, wai).ok).toBe(true)
  })

  it('still needs every file when no file is named', () => {
    const inv = inventoryFrom(objectInfo({ ckpt: [wai] }))
    expect(missingFilesFor(il, inv)).toEqual(il.models.filter((m) => m !== wai))
  })

  it('still needs the files the graph names beside the weights', () => {
    const z = family('z-image')
    const inv = inventoryFrom(objectInfo({ unet: ['Z-Image-Turbo-fp8mix.safetensors'], vae: ['ae.safetensors'] }))
    expect(missingFilesFor(z, inv, 'Z-Image-Turbo-fp8mix.safetensors')).toEqual(['qwen_3_4b.safetensors', 'Flux/ae.safetensors'])
  })

  it('needs both halves of a pair that loads both, whichever half is named', () => {
    const pair = family('wan22-14b-t2v')
    const [high] = pair.models
    const inv = inventoryFrom(objectInfo({
      gguf: [high!],
      clipGguf: ['umt5-xxl-encoder-Q4_K_M.gguf'],
      vae: ['wan_2.1_vae.safetensors'],
    }))
    expect(missingFilesFor(pair, inv, high)).not.toEqual([])
    expect(availabilityOf(pair, inv, null, new Map(), high).ok).toBe(false)
  })
})
