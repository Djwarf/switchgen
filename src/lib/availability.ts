/**
 * Can this family run here? One answer, used by every desk and the validator.
 *
 * Three desks each carried a copy of this check and the copies had drifted:
 * the Reel desk skipped the memory-fit verdict, so it could offer a family that
 * would be killed for RAM, and all three read the text encoder list off
 * CLIPLoader alone. GGUF encoders are listed by CLIPLoaderGGUF, not CLIPLoader,
 * so the two Wan 2.2 14B families read as "needs umt5-xxl-encoder-Q4_K_M.gguf"
 * on every desk with the file sitting installed. The validator already knew
 * that; this module makes the desks know it too.
 */
import { optionsFor } from './comfy'
import { feasibility, type Hardware, type ModelFile, type Verdict } from './hardware'
import { modelsOf, sidecarsOf, type FamilyDef } from './workflows'

/** Everything ComfyUI says it can load, read once off /object_info. */
export type Inventory = {
  /** Checkpoints and diffusion models, whatever loader lists them. */
  weights: Set<string>
  /** Text encoders, from the plain, GGUF and dual loaders alike. */
  clips: Set<string>
  vaes: Set<string>
  loras: Set<string>
  samplers: string[]
  schedulers: string[]
}

const many = (info: Record<string, unknown>, pairs: readonly [string, string][]) =>
  new Set(pairs.flatMap(([node, field]) => optionsFor(info, node, field)))

export function inventoryFrom(info: Record<string, unknown>): Inventory {
  return {
    weights: many(info, [
      ['CheckpointLoaderSimple', 'ckpt_name'],
      ['UNETLoader', 'unet_name'],
      ['UnetLoaderGGUF', 'unet_name'],
    ]),
    clips: many(info, [
      ['CLIPLoader', 'clip_name'],
      ['CLIPLoaderGGUF', 'clip_name'],
      ['DualCLIPLoader', 'clip_name1'],
      ['DualCLIPLoader', 'clip_name2'],
      ['DualCLIPLoaderGGUF', 'clip_name1'],
      ['DualCLIPLoaderGGUF', 'clip_name2'],
    ]),
    vaes: many(info, [['VAELoader', 'vae_name']]),
    loras: many(info, [
      ['LoraLoaderModelOnly', 'lora_name'],
      ['LoraLoader', 'lora_name'],
    ]),
    samplers: optionsFor(info, 'KSampler', 'sampler_name'),
    schedulers: optionsFor(info, 'KSampler', 'scheduler'),
  }
}

/**
 * Every file the family's graph names that ComfyUI does not list. Empty means
 * the graph can be queued without an opaque backend error naming a file.
 */
export function missingFilesFor(def: FamilyDef, inv: Inventory): string[] {
  const { clip, vae } = sidecarsOf(def)
  const missing = [
    ...clip.filter((c) => !inv.clips.has(c)),
    ...(vae && !inv.vaes.has(vae) ? [vae] : []),
    ...modelsOf(def).filter((m) => !inv.weights.has(m)),
    ...Object.values(def.graph)
      .map((n) => n.inputs['lora_name'])
      .filter((l): l is string => typeof l === 'string' && !inv.loras.has(l)),
  ]
  return [...new Set(missing)]
}

export type Availability =
  | { ok: true; verdict: Verdict | null }
  | { ok: false; why: string }

/**
 * Files first, then memory. A family with a file missing is not offered and
 * the file is named; one that is installed but cannot be held in RAM is not
 * offered and the verdict's own sentence says by how much. `hardware` null
 * means the machine has not been measured, which yields a null verdict rather
 * than a refusal.
 */
export function availabilityOf(
  def: FamilyDef,
  inv: Inventory,
  hardware: Hardware | null,
  sizes: Map<string, ModelFile>,
): Availability {
  const missing = missingFilesFor(def, inv)
  if (missing.length) return { ok: false, why: `needs ${missing.join(', ')}` }
  const verdict = hardware ? feasibility(def, sizes, hardware) : null
  if (verdict && !verdict.selectable) return { ok: false, why: verdict.reason }
  return { ok: true, verdict }
}
