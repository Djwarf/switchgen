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
import { feasibility, modelGraph, type Hardware, type ModelFile, type Verdict } from './hardware'
import { DETECTORS, UPSCALE_MODEL } from './refine'
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
  /** Upscale models, which only the region pass loads. */
  upscalers: Set<string>
  /** Face, hand and person detectors, which only the detail passes load. */
  detectors: Set<string>
  /** Every node class ComfyUI offers. Empty when /object_info said nothing. */
  nodes: Set<string>
}

/**
 * The files one node input can take, in either of the two shapes ComfyUI
 * uses for a list. Most loaders still send the bare list as the first entry;
 * UpscaleModelLoader already sends `['COMBO', { options }]`, and read the old
 * way it listed nothing, so the region pass would have looked unrunnable on
 * a machine that has the upscaler.
 */
function listed(info: Record<string, unknown>, node: string, field: string): string[] {
  const input = (info?.[node] as { input?: Record<string, Record<string, unknown[]>> } | undefined)?.input
  const spec = input?.required?.[field] ?? input?.optional?.[field]
  if (!Array.isArray(spec)) return []
  const [head, extra] = spec
  if (Array.isArray(head)) return head.filter((v): v is string => typeof v === 'string')
  const options = (extra as { options?: unknown } | undefined)?.options
  if (head === 'COMBO' && Array.isArray(options)) return options.filter((v): v is string => typeof v === 'string')
  return []
}

const many = (info: Record<string, unknown>, pairs: readonly [string, string][]) =>
  new Set(pairs.flatMap(([node, field]) => listed(info, node, field)))

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
    samplers: listed(info, 'KSampler', 'sampler_name'),
    schedulers: listed(info, 'KSampler', 'scheduler'),
    upscalers: many(info, [['UpscaleModelLoader', 'model_name']]),
    detectors: many(info, [['UltralyticsDetectorProvider', 'model_name']]),
    nodes: new Set(Object.keys(info ?? {})),
  }
}

/**
 * The loaders a node pack brings, for a file ComfyUI cannot list because the
 * node that would read it is not installed. ComfyUI lists a .gguf only through
 * these, so without the pack the file looks missing while it sits on disk, and
 * "needs Wan2.2-...gguf" sent the reader looking for a file they already have.
 */
const PACK_LOADERS: readonly { pack: string; loaders: readonly string[]; files: RegExp }[] = [
  {
    pack: 'the ComfyUI-GGUF node pack',
    loaders: ['UnetLoaderGGUF', 'CLIPLoaderGGUF', 'DualCLIPLoaderGGUF'],
    files: /\.gguf$/i,
  },
]

/**
 * The node pack a file needs before ComfyUI can list it, or null when the file
 * is simply not there. Null as well when /object_info said nothing about its
 * nodes, since an absent node is then not evidence of anything.
 */
export function packNeededFor(file: string, inv: Inventory): string | null {
  if (!inv.nodes.size) return null
  for (const p of PACK_LOADERS) {
    if (p.files.test(file) && !p.loaders.some((n) => inv.nodes.has(n))) return p.pack
  }
  return null
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

/**
 * Why these files cannot be loaded, starting "needs". A file whose node pack is
 * missing is put down to the pack, once, and only the rest are named as files.
 */
export function missingWhy(missing: readonly string[], inv: Inventory): string {
  const packs = new Set<string>()
  const files: string[] = []
  for (const file of missing) {
    const pack = packNeededFor(file, inv)
    if (pack) packs.add(pack)
    else files.push(file)
  }
  const parts = [...[...packs].map((p) => `${p} to read its .gguf files`), ...files]
  return `needs ${parts.join(', and ')}`
}

export type Availability =
  | { ok: true; verdict: Verdict | null }
  | { ok: false; why: string }

/**
 * Files first, then memory. A family with a file missing is not offered and
 * the file is named, or the node pack that would read it; one that is
 * installed but cannot be held in RAM is not offered and the verdict's own
 * sentence says by how much. `hardware` null means the machine has not been
 * measured, which yields a null verdict rather than a refusal.
 *
 * `model` is the weight file this row stands for, when the family lists more
 * than one. The memory verdict is priced on that file, not on the family's
 * default: two quants of one model differ by gigabytes.
 */
export function availabilityOf(
  def: FamilyDef,
  inv: Inventory,
  hardware: Hardware | null,
  sizes: Map<string, ModelFile>,
  model?: string,
): Availability {
  const missing = missingFilesFor(def, inv)
  if (missing.length) return { ok: false, why: missingWhy(missing, inv) }
  const graph = model ? modelGraph(def, model) : def.graph
  const verdict = hardware ? feasibility(def, sizes, hardware, graph) : null
  if (verdict && !verdict.selectable) return { ok: false, why: verdict.reason }
  return { ok: true, verdict }
}

// ---------------------------------------------------------------------------
// The quality passes
// ---------------------------------------------------------------------------

export type PassKind = 'refine' | 'face' | 'hand'

/** A sentence per pass that cannot be queued here, naming what is missing. Null when it can. */
export type PassBlocks = Record<PassKind, string | null>

export const NO_PASS_BLOCKS: PassBlocks = { refine: null, face: null, hand: null }

/** Which node pack each pass node comes from, for naming what to install when it is absent. */
const PASS_NODES: Record<PassKind, readonly (readonly [node: string, pack: string])[]> = {
  refine: [
    ['UpscaleModelLoader', 'the UpscaleModelLoader node'],
    ['ImpactGaussianBlurMask', 'the ComfyUI Impact Pack'],
  ],
  face: [
    ['UltralyticsDetectorProvider', 'the ComfyUI Impact Subpack'],
    ['FaceDetailer', 'the ComfyUI Impact Pack'],
  ],
  hand: [
    ['UltralyticsDetectorProvider', 'the ComfyUI Impact Subpack'],
    ['FaceDetailer', 'the ComfyUI Impact Pack'],
  ],
}

const PASS_NAME: Record<PassKind, string> = {
  refine: 'The region pass',
  face: 'The face pass',
  hand: 'The hand pass',
}

/**
 * What stops each quality pass from running here, read off /object_info.
 *
 * refine.ts writes the upscaler and the two detectors into the graphs it
 * builds, and a machine without them had every pass offered anyway: ComfyUI
 * then refused the job over a file the page never mentioned. A node pack that
 * is missing is named before a file, because without it the file cannot be
 * listed at all. Nothing is blocked when /object_info said nothing, since an
 * empty answer is not evidence that a file is gone.
 */
export function passBlocks(inv: Inventory): PassBlocks {
  if (!inv.nodes.size) return NO_PASS_BLOCKS
  const block = (kind: PassKind, file: string, have: Set<string>, folder: string): string | null => {
    const pack = PASS_NODES[kind].find(([node]) => !inv.nodes.has(node))?.[1]
    if (pack) return `${PASS_NAME[kind]} needs ${pack}, which this ComfyUI does not have.`
    if (!have.has(file)) return `${PASS_NAME[kind]} needs ${file} in ComfyUI’s ${folder} folder, and ComfyUI does not list it.`
    return null
  }
  return {
    refine: block('refine', UPSCALE_MODEL, inv.upscalers, 'upscale_models'),
    face: block('face', DETECTORS.face, inv.detectors, 'ultralytics'),
    hand: block('hand', DETECTORS.hand, inv.detectors, 'ultralytics'),
  }
}
