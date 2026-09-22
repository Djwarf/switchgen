import { describe, expect, it } from 'vitest'
import { availabilityOf, inventoryFrom, missingFilesFor } from '../src/lib/availability'
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
