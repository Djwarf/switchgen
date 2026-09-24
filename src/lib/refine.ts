/**
 * Quality derivations: region refine, automatic detailing, hires fix, LoRA stacks.
 *
 * WHY THIS FILE EXISTS, IN ARITHMETIC.
 *
 * A diffusion model spends its capacity evenly over the latent grid. The VAE
 * downsamples 8x in each direction, so a 1024x1024 image is a 128x128 latent:
 * 16384 cells for the whole frame. Now take a region that occupies 5% of that
 * frame, which is roughly a 229x229 pixel patch. In latent space that patch is
 * 29x29 cells, and a smaller feature inside it, a nipple or a set of labia or
 * a knuckle, is 6x6 cells or less. Correct anatomy cannot be reconstructed
 * from 36 numbers. Neither can a face at distance, and neither can fingers.
 *
 * This is not a failure of the checkpoint. Swapping to a better base model
 * moves the same 36 cells around. The only fix is to give the region more
 * cells, which means MORE REAL PIXELS AT THE REGION:
 *
 *   crop the region -> upscale the crop to full working resolution
 *   -> re render the crop alone at low denoise -> composite it back
 *
 * A 229x229 patch blown up to 1024x1024 is 128x128 latent cells. The same
 * feature that had 36 cells now has roughly 700. That is the entire trick,
 * and it is why every one of these derivations is about resolution rather
 * than about prompt wording.
 *
 * WHAT CAN BE AUTOMATIC AND WHAT CANNOT.
 *
 * The installed Ultralytics detectors are exactly three:
 *   bbox/face_yolov8m.pt, bbox/hand_yolov8s.pt, segm/person_yolov8m-seg.pt
 * There is NO detector for breasts, nipples, vulvas or penises. None is
 * installed and none is being pretended into existence here. Faces and hands
 * get an automatic pass (deriveAutoDetail). Every other region needs the user
 * to draw a mask, which is what deriveRefine consumes. Do not "improve" this
 * file by inventing a detector model name; the enum above is what the live
 * schema offers and anything else fails at queue time.
 *
 * COST. Each refine pass renders a full working resolution frame, so it costs
 * about one whole generation. Hires fix costs about 1.6x a single pass. Say so
 * in the UI rather than letting a click quietly take a minute of GPU.
 *
 * Every derivation returns null when the family cannot support it. Callers
 * must hide the option on null, never silently fall back to something else.
 *
 * Node classes and enum values below were read from the live /object_info on
 * this machine. Anything added here must be checked the same way.
 *
 * Dependency direction: this module imports workflows.ts. Do not import this
 * module from workflows.ts or registry.ts.
 */
import type { ApiWorkflow, FileRef } from './comfy'
import type { HistoryEntry } from './history'
import type { Binding, FamilyDef } from './registry'
import { instantiate, type Params } from './workflows'

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

type Graph = FamilyDef['graph']
type Ref = [nodeId: string, slot: number]

/** Knobs a derivation adds on top of the base family bindings. */
export type ExtraKey =
  // region refine
  | 'maskImage'
  | 'cropX'
  | 'cropY'
  | 'cropWidth'
  | 'cropHeight'
  | 'targetWidth'
  | 'targetHeight'
  | 'maskGrow'
  | 'maskFeatherKernel'
  | 'maskFeatherSigma'
  | 'refinePrompt'
  | 'refineDenoise'
  // automatic detailing
  | 'detailPrompt'
  | 'detailDenoise'
  | 'detailSteps'
  | 'detailThreshold'
  | 'detailDilation'
  | 'detailCropFactor'
  | 'detailGuideSize'
  | 'detailMaxSize'
  | 'detailFeather'
  // hires fix
  | 'hiresScale'
  | 'hiresDenoise'
  | 'hiresSteps'

export type DerivationKind = 'refine' | 'autodetail' | 'hires' | 'loras'

export type Derivation = {
  /** Which derivations have been applied, in the order they were applied. */
  kinds: DerivationKind[]
  extra: Partial<Record<ExtraKey, Binding[]>>
  /** Node ids of the LoRA chain, in application order. Empty when none. */
  loraNodes: string[]
  /** Rough cost as a multiple of one plain generation of this family. */
  cost: number
  /** One line of honest UI copy about what this pass does and what it costs. */
  note: string
}

/** A family graph with one or more quality derivations applied. */
export type DerivedDef = FamilyDef & { derived: Derivation }

const isDerived = (def: FamilyDef | DerivedDef): def is DerivedDef =>
  'derived' in def && !!(def as DerivedDef).derived

function priorDerivation(def: FamilyDef | DerivedDef): Derivation {
  if (isDerived(def)) {
    return {
      kinds: [...def.derived.kinds],
      extra: { ...def.derived.extra },
      loraNodes: [...def.derived.loraNodes],
      cost: def.derived.cost,
      note: def.derived.note,
    }
  }
  return { kinds: [], extra: {}, loraNodes: [], cost: 1, note: '' }
}

// ---------------------------------------------------------------------------
// Graph plumbing
// ---------------------------------------------------------------------------

const isRef = (v: unknown): v is Ref =>
  Array.isArray(v) && v.length === 2 && typeof v[0] === 'string' && typeof v[1] === 'number'

const cloneGraph = (g: Graph): Graph => JSON.parse(JSON.stringify(g)) as Graph

function uniqueId(graph: Graph, base: string): string {
  if (!(base in graph)) return base
  for (let i = 2; ; i += 1) {
    const id = `${base}_${i}`
    if (!(id in graph)) return id
  }
}

/**
 * Repoint every consumer of `from` at `to`. Matching is on BOTH node id and
 * output slot: CheckpointLoaderSimple emits MODEL, CLIP and VAE from one node,
 * so rewiring by node id alone would hand the VAE socket a MODEL.
 */
function rewire(graph: Graph, from: Ref, to: Ref, skip: ReadonlySet<string>) {
  for (const [id, node] of Object.entries(graph)) {
    if (skip.has(id)) continue
    for (const [k, v] of Object.entries(node.inputs)) {
      if (isRef(v) && v[0] === from[0] && v[1] === from[1]) node.inputs[k] = [to[0], to[1]] as Ref
    }
  }
}

const EMPTY_LATENT = /^Empty.*Latent/
const IMAGE_SINK = /^(SaveImage|PreviewImage|SaveImageWebsocket|SaveAnimatedPNG)$/

/** The Empty*Latent* node a text-to-image graph samples from. */
function findEmptyLatent(graph: Graph): string | null {
  for (const [id, n] of Object.entries(graph)) {
    if (EMPTY_LATENT.test(n.class_type) && 'width' in n.inputs && 'height' in n.inputs) return id
  }
  return null
}

/** The plain KSampler. Custom sampler graphs (SamplerCustomAdvanced) have none. */
function findKSampler(graph: Graph): string | null {
  for (const [id, n] of Object.entries(graph)) if (n.class_type === 'KSampler') return id
  return null
}

function findVaeDecode(graph: Graph): string | null {
  for (const [id, n] of Object.entries(graph)) if (n.class_type === 'VAEDecode') return id
  return null
}

function findVaeRef(graph: Graph): Ref | null {
  const dec = findVaeDecode(graph)
  if (dec) {
    const v = graph[dec].inputs.vae
    if (isRef(v)) return v
  }
  for (const [id, n] of Object.entries(graph)) if (n.class_type === 'VAELoader') return [id, 0]
  for (const [id, n] of Object.entries(graph)) {
    if (n.class_type === 'CheckpointLoaderSimple') return [id, 2]
  }
  return null
}

/** The node that finally shows the picture, plus the image ref feeding it. */
function findImageSink(graph: Graph): { id: string; source: Ref } | null {
  for (const [id, n] of Object.entries(graph)) {
    if (!IMAGE_SINK.test(n.class_type)) continue
    const v = n.inputs.images
    if (isRef(v)) return { id, source: v }
  }
  return null
}

/**
 * The CLIP the family encodes prompts with. Taken from the positive prompt
 * node so clip-skip is respected: on SDXL the text encoder reads
 * CLIPSetLastLayer, not the raw checkpoint CLIP, and a detailer that re
 * encoded on the raw CLIP would render in a subtly different style.
 */
function findClipRef(def: FamilyDef, graph: Graph): Ref | null {
  const pos = def.bindings.positive?.[0]?.[0]
  if (pos && graph[pos]) {
    const v = graph[pos].inputs.clip
    if (isRef(v)) return v
  }
  for (const [id, n] of Object.entries(graph)) if (n.class_type === 'CLIPSetLastLayer') return [id, 0]
  for (const [id, n] of Object.entries(graph)) if (n.class_type === 'CLIPLoader') return [id, 0]
  for (const [id, n] of Object.entries(graph)) {
    if (n.class_type === 'CheckpointLoaderSimple') return [id, 1]
  }
  return null
}

const MODEL_LOADER = /^(CheckpointLoaderSimple|UNETLoader|UnetLoaderGGUF)$/

function findModelLoaders(graph: Graph): string[] {
  return Object.entries(graph)
    .filter(([, n]) => MODEL_LOADER.test(n.class_type))
    .map(([id]) => id)
}

/** Sampler nodes exposing a writable denoise float. */
function findDenoiseTargets(graph: Graph): Binding[] {
  const out: Binding[] = []
  for (const [id, n] of Object.entries(graph)) {
    if ('denoise' in n.inputs && !isRef(n.inputs.denoise)) out.push([id, 'denoise'])
  }
  return out
}

function mergeBindings(
  into: Partial<Record<ExtraKey, Binding[]>>,
  key: ExtraKey,
  binds: Binding[],
): void {
  into[key] = [...(into[key] ?? []), ...binds]
}

// ---------------------------------------------------------------------------
// Latent arithmetic, exported so the UI can show the user the real numbers
// instead of a vague promise that refining "improves detail".
// ---------------------------------------------------------------------------

/** VAE spatial downsample factor. Every model installed here uses 8. */
const VAE_STRIDE = 8

/** Latent cells a frame of this pixel size gets. */
function latentCells(width: number, height: number): number {
  return Math.floor(width / VAE_STRIDE) * Math.floor(height / VAE_STRIDE)
}

export type DetailGain = {
  /** Latent cells the region has inside the full frame today. */
  before: number
  /** Latent cells the same region gets once cropped and rendered at target. */
  after: number
  /** after / before, rounded to one decimal. */
  gain: number
  /** Share of the frame the region occupies, as a percentage. */
  sharePct: number
}

/**
 * How much detail a region actually gains from a refine pass. `region` is the
 * mask bounds in source pixels; `target` is the resolution the crop will be
 * re rendered at. Drive the UI copy from this, so the user can see when a
 * region is already big enough that refining buys nothing.
 */
export function detailGain(
  region: { width: number; height: number },
  image: { width: number; height: number },
  target: { width: number; height: number },
): DetailGain {
  const before = Math.max(1, latentCells(region.width, region.height))
  const after = Math.max(1, latentCells(target.width, target.height))
  const frame = Math.max(1, image.width * image.height)
  return {
    before,
    after,
    gain: Math.round((after / before) * 10) / 10,
    sharePct: Math.round(((region.width * region.height) / frame) * 1000) / 10,
  }
}

// ---------------------------------------------------------------------------
// Region refine: the anatomy fix
// ---------------------------------------------------------------------------

/** The useful denoise band for a region re render. */
export const REFINE_DENOISE = { min: 0.35, max: 0.55, default: 0.45 } as const

/**
 * Upscale model, verified present in UpscaleModelLoader.model_name on the
 * machine this was written on. Another machine may not have it, so the desks
 * check it against /object_info (availability.ts, passBlocks) before offering
 * a region pass.
 */
export const UPSCALE_MODEL = '4x-UltraSharp.pth'

/**
 * Detector files, verified present in UltralyticsDetectorProvider.model_name on
 * the machine this was written on. Checked the same way before a face or hand
 * pass is offered.
 */
export const DETECTORS = {
  face: 'bbox/face_yolov8m.pt',
  hand: 'bbox/hand_yolov8s.pt',
  person: 'segm/person_yolov8m-seg.pt',
} as const

export type Rect = { x: number; y: number; width: number; height: number }

export type RefinePlan = {
  /** Crop rectangle in source pixels, clamped inside the image. */
  crop: Rect
  /** Resolution the crop is re rendered at. */
  target: { width: number; height: number }
  gain: DetailGain
}

const snap = (n: number, step: number) => Math.max(step, Math.round(n / step) * step)

/**
 * Turn a drawn mask's bounding box into a crop rectangle and a render size.
 *
 * Clamping matters: ComfyUI's ImageCrop and CropMask slice the tensor, so a
 * rectangle that runs off the edge silently yields a SMALLER image than asked
 * for, and the composite would then stretch the refined patch. Everything is
 * clamped inside the frame here, once, rather than hoped about later.
 *
 * `padding` is context, not decoration. The sampler needs to see the skin,
 * shadow and body line around a region to render it consistently with the rest
 * of the frame; a crop tight to the brush stroke produces a correct patch that
 * does not belong to the body it sits on.
 */
export function planRefine(
  bounds: Rect,
  image: { width: number; height: number },
  opts?: { padding?: number; targetLongEdge?: number },
): RefinePlan {
  const padding = Math.max(0, Math.round(opts?.padding ?? 64))
  const longEdge = Math.max(512, Math.round(opts?.targetLongEdge ?? 1024))

  const x0 = Math.max(0, Math.floor(bounds.x) - padding)
  const y0 = Math.max(0, Math.floor(bounds.y) - padding)
  const x1 = Math.min(image.width, Math.ceil(bounds.x + bounds.width) + padding)
  const y1 = Math.min(image.height, Math.ceil(bounds.y + bounds.height) + padding)

  const crop: Rect = {
    x: x0,
    y: y0,
    width: Math.max(8, x1 - x0),
    height: Math.max(8, y1 - y0),
  }

  // Render size keeps the crop's aspect ratio and puts `longEdge` on the long
  // side, snapped to 16 so both dimensions divide cleanly by the VAE stride.
  // Snapping can shift the aspect by under one percent, which the scale back
  // to the crop rectangle absorbs invisibly.
  const ratio = crop.width / crop.height
  const target =
    ratio >= 1
      ? { width: snap(longEdge, 16), height: snap(longEdge / ratio, 16) }
      : { width: snap(longEdge * ratio, 16), height: snap(longEdge, 16) }

  return { crop, target, gain: detailGain(crop, image, target) }
}

/**
 * STEPS AND DENOISE IN COMFYUI. Read this before "fixing" the step counts.
 *
 * A1111 and Forge multiply: ask for 30 steps at denoise 0.45 and you get 13
 * real sampling iterations, so those UIs need the step count scaled up by
 * 1/denoise to keep quality at partial denoise.
 *
 * ComfyUI does the opposite and already compensates. comfy/samplers.py,
 * KSampler.set_steps: `new_steps = int(steps/denoise)`, then it keeps
 * `sigmas[-(steps + 1):]`. The schedule is stretched, and exactly `steps`
 * iterations run. Impact Pack's detailers go through the same common_ksampler,
 * so FaceDetailer counts steps the same way.
 *
 * So a refine pass asks for the family's own step count and gets the family's
 * own quality. Dividing by denoise here would silently more than double the
 * GPU time of every refine for no gain at all.
 */

export type RefineParams = {
  /** Source image filename, already uploaded to ComfyUI's input folder. */
  image: string
  /**
   * Mask filename. Upload a PNG the size of the source with WHITE strokes on
   * a BLACK background, fully opaque. The graph reads the red channel, because
   * LoadImageMask on the alpha channel returns 1 minus alpha and an opaque RGB
   * PNG therefore reads back as an empty mask.
   */
  mask: string
  crop: Rect
  target: { width: number; height: number }
  /** 0.35 to 0.55 is the useful band. Below it nothing changes, above it the region stops matching the body. */
  denoise: number
  /** Pixels to widen the mask by before rendering, so the sampler owns the boundary rather than meeting it. */
  grow: number
  /** Pixels of blur on the mask edge. This is what stops the composite showing a seam. */
  feather: number
  /** Prompt for this region. Defaults to the parent prompt, but is meant to be edited: a region prompt should name the anatomy, not the scene. */
  prompt: string
  negative?: string
  seed?: number
  steps?: number
  cfg?: number
}

export const REFINE_DEFAULTS = { grow: 12, feather: 16, padding: 64, targetLongEdge: 1024 } as const

/**
 * THE ANATOMY FIX.
 *
 * Re renders one user drawn region at full working resolution and composites
 * it back. Shape of the derived graph:
 *
 *   LoadImage(source) ------------------------------+
 *   LoadImageMask(mask,red) -> GrowMask -> blur -+   |
 *        |                                      |   |
 *        +-> CropMask(rect) -------------------+ |  |
 *   ImageCrop(source, rect)                    | |  |
 *        -> ImageUpscaleWithModel(4x-UltraSharp)  |  |
 *        -> ImageScale(target)                    |  |
 *        -> VAEEncode -> SetLatentNoiseMask(cropped mask)
 *        -> KSampler(denoise 0.45)  [the family's own sampler, model and CLIP]
 *        -> VAEDecode -> ImageScale(back to rect size)
 *        -> ImageCompositeMasked(dest=source, x, y, mask=cropped mask) -> SaveImage
 *
 * Three decisions worth not undoing:
 *
 * 1. The crop goes through the 4x upscale model rather than a plain lanczos
 *    resize. Lanczos on a 200px patch invents no structure, so the VAE encodes
 *    a blurred patch and the low denoise pass has nothing to sharpen. The
 *    ESRGAN pass gives the encoder real edges to work with. ComfyUI tiles this
 *    internally, so it does not blow up VRAM, it just costs seconds.
 *
 * 2. SetLatentNoiseMask is applied. Without it the whole crop is denoised and
 *    the surrounding skin drifts in tone, which the composite then cuts a hole
 *    in. With it, the surround is restored from the source latent at every
 *    step, so the model still SEES the body around the region and conditions on
 *    it, while only the masked cells actually move. That is the difference
 *    between a correct nipple and a correct nipple on a slightly different
 *    breast.
 *
 * 3. The composite mask is cropped to the same rectangle. ImageCompositeMasked
 *    interpolates its mask to the SOURCE image's size, so handing it a full
 *    frame mask would stretch the whole mask over the patch.
 *
 * Returns null for edit and video families, for custom sampler graphs with no
 * denoise input (flux2-klein), and for any family missing a VAE or an image
 * sink.
 */
export function deriveRefine(def: FamilyDef | DerivedDef): DerivedDef | null {
  if (def.mode !== 'image') return null
  // A hires graph has a second sampler chained after the first. Feeding it a
  // cropped region would sample the crop twice and decode it at 1.5x the size
  // the composite expects. Refine and hires are alternatives, not a stack.
  if (isDerived(def) && def.derived.kinds.includes('hires')) return null

  const graph = cloneGraph(def.graph)
  const latentId = findEmptyLatent(graph)
  const vae = findVaeRef(graph)
  const decodeId = findVaeDecode(graph)
  const sink = findImageSink(graph)
  const denoise = findDenoiseTargets(graph)
  if (!latentId || !vae || !decodeId || !sink || !denoise.length) return null

  const SRC = uniqueId(graph, '__rf_src')
  const MASK = uniqueId(graph, '__rf_mask')
  const GROW = uniqueId(graph, '__rf_grow')
  const BLUR = uniqueId(graph, '__rf_blur')
  const MCROP = uniqueId(graph, '__rf_maskcrop')
  const CROP = uniqueId(graph, '__rf_crop')
  const UPMODEL = uniqueId(graph, '__rf_upmodel')
  const UPSCALE = uniqueId(graph, '__rf_upscale')
  const FIT = uniqueId(graph, '__rf_fit')
  const ENC = uniqueId(graph, '__rf_encode')
  const NOISE = uniqueId(graph, '__rf_noisemask')
  const BACK = uniqueId(graph, '__rf_back')
  const COMP = uniqueId(graph, '__rf_composite')

  graph[SRC] = { class_type: 'LoadImage', inputs: { image: 'example.png' } }
  graph[MASK] = { class_type: 'LoadImageMask', inputs: { image: 'example.png', channel: 'red' } }
  graph[GROW] = {
    class_type: 'GrowMask',
    inputs: { mask: [MASK, 0], expand: REFINE_DEFAULTS.grow, tapered_corners: true },
  }
  // ImpactGaussianBlurMask blurs the mask BLOB. The native FeatherMask fades
  // the four borders of the mask rectangle instead, which does nothing useful
  // for a brush stroke in the middle of a frame. Do not swap them.
  graph[BLUR] = {
    class_type: 'ImpactGaussianBlurMask',
    inputs: { mask: [GROW, 0], kernel_size: REFINE_DEFAULTS.feather, sigma: REFINE_DEFAULTS.feather / 2 },
  }
  graph[MCROP] = {
    class_type: 'CropMask',
    inputs: { mask: [BLUR, 0], x: 0, y: 0, width: 512, height: 512 },
  }
  // ImageCrop is marked deprecated in favour of ImageCropV2, and is still the
  // right node here for two reasons: V2 takes its rectangle as one BOUNDING_BOX
  // object rather than four writable ints, and V2 emits a PreviewImage, which
  // would land the raw crop in the job's output files next to the finished
  // picture. Keep ImageCrop until V2 stops doing both.
  graph[CROP] = {
    class_type: 'ImageCrop',
    inputs: { image: [SRC, 0], width: 512, height: 512, x: 0, y: 0 },
  }
  graph[UPMODEL] = { class_type: 'UpscaleModelLoader', inputs: { model_name: UPSCALE_MODEL } }
  graph[UPSCALE] = {
    class_type: 'ImageUpscaleWithModel',
    inputs: { upscale_model: [UPMODEL, 0], image: [CROP, 0] },
  }
  graph[FIT] = {
    class_type: 'ImageScale',
    inputs: { image: [UPSCALE, 0], upscale_method: 'lanczos', width: 1024, height: 1024, crop: 'disabled' },
  }
  graph[ENC] = { class_type: 'VAEEncode', inputs: { pixels: [FIT, 0], vae } }
  graph[NOISE] = { class_type: 'SetLatentNoiseMask', inputs: { samples: [ENC, 0], mask: [MCROP, 0] } }

  // The family's sampler now denoises the cropped region instead of an empty frame.
  rewire(graph, [latentId, 0], [NOISE, 0], new Set([NOISE]))
  delete graph[latentId]

  // Decoded patch goes back to the crop rectangle's own size, then lands in the
  // untouched original at the rectangle's coordinates.
  graph[BACK] = {
    class_type: 'ImageScale',
    inputs: { image: [decodeId, 0], upscale_method: 'lanczos', width: 512, height: 512, crop: 'disabled' },
  }
  graph[COMP] = {
    class_type: 'ImageCompositeMasked',
    inputs: {
      destination: [SRC, 0],
      source: [BACK, 0],
      x: 0,
      y: 0,
      resize_source: false,
      mask: [MCROP, 0],
    },
  }
  graph[sink.id].inputs.images = [COMP, 0] as Ref

  // Width and height came from the latent node that no longer exists. In a
  // refine the crop rectangle decides the size, so those controls do not apply.
  const bindings: FamilyDef['bindings'] = { ...def.bindings }
  delete bindings.width
  delete bindings.height
  delete bindings.megapixels
  bindings.image = [[SRC, 'image']]
  bindings.denoise = denoise

  const derived = priorDerivation(def)
  derived.kinds.push('refine')
  // A refine REPLACES the generation rather than adding to it: there is no
  // empty latent left, the crop is the whole job. So cost stays at one.
  derived.cost = Math.max(1, derived.cost)
  derived.note =
    'Re renders the masked region alone at full resolution, then composites it back. Costs about one full generation.'

  mergeBindings(derived.extra, 'maskImage', [[MASK, 'image']])
  mergeBindings(derived.extra, 'maskGrow', [[GROW, 'expand']])
  mergeBindings(derived.extra, 'maskFeatherKernel', [[BLUR, 'kernel_size']])
  mergeBindings(derived.extra, 'maskFeatherSigma', [[BLUR, 'sigma']])
  mergeBindings(derived.extra, 'cropX', [
    [CROP, 'x'],
    [MCROP, 'x'],
    [COMP, 'x'],
  ])
  mergeBindings(derived.extra, 'cropY', [
    [CROP, 'y'],
    [MCROP, 'y'],
    [COMP, 'y'],
  ])
  mergeBindings(derived.extra, 'cropWidth', [
    [CROP, 'width'],
    [MCROP, 'width'],
    [BACK, 'width'],
  ])
  mergeBindings(derived.extra, 'cropHeight', [
    [CROP, 'height'],
    [MCROP, 'height'],
    [BACK, 'height'],
  ])
  mergeBindings(derived.extra, 'targetWidth', [[FIT, 'width']])
  mergeBindings(derived.extra, 'targetHeight', [[FIT, 'height']])
  mergeBindings(derived.extra, 'refineDenoise', denoise)
  if (def.bindings.positive) mergeBindings(derived.extra, 'refinePrompt', def.bindings.positive)

  return {
    ...def,
    id: `${def.id}__refine`,
    label: `${def.label}: region refine`,
    graph,
    bindings,
    derived,
  }
}

/**
 * Build the queue-ready graph for a region refine.
 *
 * Writes the family parameters first, then the refine knobs, so the region
 * prompt wins over the parent prompt on the shared text node.
 */
export function instantiateRefine(def: DerivedDef, base: Params, r: RefineParams): ApiWorkflow {
  // Not scaled by 1/denoise. See the note above set_steps: ComfyUI already does.
  const steps = r.steps ?? base.steps
  const wf = instantiate(def, {
    ...base,
    image: r.image,
    seed: r.seed ?? base.seed,
    steps,
    cfg: r.cfg ?? base.cfg,
    negative: r.negative ?? base.negative,
    denoise: r.denoise,
  })
  const feather = Math.max(0, Math.min(100, Math.round(r.feather)))
  writeExtras(wf, def, {
    maskImage: r.mask,
    cropX: Math.max(0, Math.round(r.crop.x)),
    cropY: Math.max(0, Math.round(r.crop.y)),
    cropWidth: Math.max(1, Math.round(r.crop.width)),
    cropHeight: Math.max(1, Math.round(r.crop.height)),
    targetWidth: Math.max(64, Math.round(r.target.width)),
    targetHeight: Math.max(64, Math.round(r.target.height)),
    maskGrow: Math.round(r.grow),
    maskFeatherKernel: feather,
    maskFeatherSigma: Math.max(0.1, feather / 2),
    refineDenoise: r.denoise,
    refinePrompt: r.prompt,
  })
  return wf
}

// ---------------------------------------------------------------------------
// Automatic detailing, for the two regions a detector actually covers
// ---------------------------------------------------------------------------

export type DetailTarget = 'face' | 'hand'

/**
 * How the face and hand passes are tuned. The graph below is built from this,
 * and so is every sentence that describes the pass (detailSentence), so the
 * words cannot drift from what runs again.
 *
 * Hands are usually wrong rather than merely soft, so they need more freedom
 * to be rebuilt. Faces are usually right but low on pixels, so a gentler pass
 * keeps the likeness. cropFactor pulls in surrounding context: a hand needs
 * the wrist and forearm in frame to be posed correctly, a face needs less.
 */
export const DETAIL_TUNING: Record<
  DetailTarget,
  { denoise: number; guide: number; cropFactor: number; dilation: number }
> = {
  face: { denoise: 0.4, guide: 768, cropFactor: 2.5, dilation: 10 },
  hand: { denoise: 0.5, guide: 768, cropFactor: 3.0, dilation: 12 },
}

/** FaceDetailer's max_size: the longest side the padded crop is ever enlarged to. */
export const DETAIL_MAX_SIZE = 1024

/**
 * What the pass does to one detection `box` pixels across, worked out the way
 * the Impact Pack's enhance_detail does it (modules/impact/core.py). The crop
 * is the box with `cropFactor` times its size around it. It is scaled so the
 * box reaches `guide` pixels, unless that would make the crop longer than
 * max_size, in which case the crop is scaled to max_size instead; and with
 * force_inpaint on, it is never scaled down. With these settings the cap
 * always wins for a box smaller than the guide, so the box comes back at about
 * max_size / cropFactor: 410 pixels for a face, 340 for a hand. Arithmetic,
 * not a measurement, and for a square box away from the picture's edges,
 * where the crop is not cut short.
 */
export function detailRedraw(target: DetailTarget, box: number): { crop: number; redrawn: number; cells: number } {
  const t = DETAIL_TUNING[target]
  const crop = box * t.cropFactor
  const scale = Math.max(1, Math.min(t.guide / box, DETAIL_MAX_SIZE / crop))
  const redrawn = box * scale
  // A latent cell is eight pixels a side.
  return { crop, redrawn, cells: Math.round((redrawn / 8) ** 2) }
}

/** The crop factors the tuning uses, as a reader would say them. */
const TIMES_WORDS: Record<number, string> = { 2: 'two', 2.5: 'two and a half', 3: 'three', 4: 'four' }

const grouped = (n: number) => Math.round(n).toLocaleString('en-GB')
/** Rounded the way a reader says a figure: to ten, then to a hundred past a thousand. */
const about = (n: number) => grouped(n >= 1000 ? Math.round(n / 100) * 100 : Math.round(n / 10) * 10)

/**
 * The pass in one sentence, from the tuning above, with an 80 pixel box as the
 * example. It used to say a face "re renders at up to 1024px" and "at 1024 it
 * has nine thousand" cells, which is neither what 1024 pixels holds (16,384)
 * nor what the graph does to a small face: the 1024 cap applies to the padded
 * crop, not to the face.
 */
export function detailSentence(target: DetailTarget): string {
  const t = DETAIL_TUNING[target]
  const box = 80
  const r = detailRedraw(target, box)
  const factor = TIMES_WORDS[t.cropFactor] ?? String(t.cropFactor)
  const what = target === 'face' ? 'face' : 'hand'
  const around = target === 'face' ? '' : ', so the wrist and forearm are in frame'
  return `Finds every ${what} with a detector and cuts out an area ${factor} times its size around it${around}. The cut is enlarged to at most ${DETAIL_MAX_SIZE} pixels on its longer side and drawn again, so a ${what} ${box} pixels across comes back about ${about(r.redrawn)} across: some ${about(r.cells)} latent cells where it had ${grouped((box / 8) ** 2)}.`
}

/**
 * Automatic detail pass over every detected face or hand.
 *
 * Same physics as deriveRefine, run without a drawn mask because YOLO can find
 * these two region types on its own. FaceDetailer crops each detection with
 * context around it, enlarges the crop, re renders it at low denoise and
 * pastes it back with a feathered edge. What that comes to for one box is
 * detailRedraw's arithmetic above.
 *
 * The node is called FaceDetailer but it details whatever the bbox detector
 * hands it, which is why the hand detector goes through the same node.
 *
 * There is no detector for breasts or genitalia. Those regions go through
 * deriveRefine with a drawn mask. Do not add a target here without a model
 * file that exists in UltralyticsDetectorProvider.model_name.
 *
 * Returns null unless the family samples through a plain KSampler and exposes
 * a MODEL, CLIP, VAE and both conditionings, which excludes the custom sampler
 * graphs and the edit family, whose conditioning carries reference latents that
 * a detached crop pass would misread.
 */
export function deriveAutoDetail(
  def: FamilyDef | DerivedDef,
  target: DetailTarget,
): DerivedDef | null {
  if (def.mode !== 'image') return null

  const graph = cloneGraph(def.graph)
  const ksId = findKSampler(graph)
  const sink = findImageSink(graph)
  const vae = findVaeRef(graph)
  const clip = findClipRef(def, graph)
  if (!ksId || !sink || !vae || !clip) return null

  const ks = graph[ksId].inputs
  const model = ks.model
  const positive = ks.positive
  const negative = ks.negative
  if (!isRef(model) || !isRef(positive) || !isRef(negative)) return null

  const DET = uniqueId(graph, `__ad_${target}_detector`)
  const FD = uniqueId(graph, `__ad_${target}`)

  graph[DET] = {
    class_type: 'UltralyticsDetectorProvider',
    inputs: { model_name: DETECTORS[target] },
  }

  // See DETAIL_TUNING for why a hand and a face are tuned apart.
  const tuned = DETAIL_TUNING[target]

  graph[FD] = {
    class_type: 'FaceDetailer',
    inputs: {
      image: [sink.source[0], sink.source[1]] as Ref,
      model,
      clip,
      vae,
      positive,
      negative,
      bbox_detector: [DET, 0],
      guide_size: tuned.guide,
      // bbox: measure the detection box itself against guide_size. crop_region
      // would measure the padded crop, which under-upscales the actual face.
      guide_size_for: true,
      max_size: DETAIL_MAX_SIZE,
      seed: 0,
      steps: 20,
      cfg: 7,
      sampler_name: 'euler',
      scheduler: 'normal',
      denoise: tuned.denoise,
      feather: 5,
      noise_mask: true,
      force_inpaint: true,
      bbox_threshold: 0.5,
      bbox_dilation: tuned.dilation,
      bbox_crop_factor: tuned.cropFactor,
      sam_detection_hint: 'center-1',
      sam_dilation: 0,
      sam_threshold: 0.93,
      sam_bbox_expansion: 0,
      sam_mask_hint_threshold: 0.7,
      sam_mask_hint_use_negative: 'False',
      drop_size: 10,
      // Empty wildcard means the detailer reuses the conditioning above. Put a
      // region prompt here and it re encodes on `clip` instead.
      wildcard: '',
      cycle: 1,
    },
  }
  graph[sink.id].inputs.images = [FD, 0] as Ref

  // The detailer runs the same sampler settings as the main pass, so the
  // family's seed, steps, cfg, sampler and scheduler drive both.
  const bindings: FamilyDef['bindings'] = { ...def.bindings }
  const also = (key: 'seed' | 'steps' | 'cfg' | 'sampler' | 'scheduler', input: string) => {
    const current = bindings[key]
    if (current) bindings[key] = [...current, [FD, input]]
  }
  also('seed', 'seed')
  also('steps', 'steps')
  also('cfg', 'cfg')
  also('sampler', 'sampler_name')
  also('scheduler', 'scheduler')

  const derived = priorDerivation(def)
  derived.kinds.push('autodetail')
  // Each detection is a full sampler run at up to 1024px, with the family's own
  // step count, because ComfyUI runs every step it is given regardless of
  // denoise. Two hands is two more generations. Say so rather than surprising
  // the user with a three minute job from one checkbox.
  derived.cost += 1
  derived.note =
    target === 'hand'
      ? 'Finds hands and re renders each one at up to 1024px. Adds roughly one generation per hand found.'
      : 'Finds faces and re renders each one at up to 1024px. Adds roughly one generation per face found.'

  mergeBindings(derived.extra, 'detailPrompt', [[FD, 'wildcard']])
  mergeBindings(derived.extra, 'detailDenoise', [[FD, 'denoise']])
  mergeBindings(derived.extra, 'detailSteps', [[FD, 'steps']])
  mergeBindings(derived.extra, 'detailThreshold', [[FD, 'bbox_threshold']])
  mergeBindings(derived.extra, 'detailDilation', [[FD, 'bbox_dilation']])
  mergeBindings(derived.extra, 'detailCropFactor', [[FD, 'bbox_crop_factor']])
  mergeBindings(derived.extra, 'detailGuideSize', [[FD, 'guide_size']])
  mergeBindings(derived.extra, 'detailMaxSize', [[FD, 'max_size']])
  mergeBindings(derived.extra, 'detailFeather', [[FD, 'feather']])

  return {
    ...def,
    id: `${def.id}__detail_${target}`,
    label: `${def.label}: ${target} detail`,
    graph,
    bindings,
    derived,
  }
}

// ---------------------------------------------------------------------------
// Hires fix
// ---------------------------------------------------------------------------

const HIRES_DEFAULTS = { scale: 1.5, denoise: 0.45 } as const

/**
 * Two pass generation: sample at the family's native size, upscale the LATENT,
 * sample again at low denoise.
 *
 * Why this helps anatomy rather than just looking sharper: pass one composes
 * the picture at a size the model was trained on, so the body is proportioned
 * correctly. Pass two runs on a latent 1.5x larger in each direction, which is
 * 2.25x the cells, so every region including breasts, hands and faces has more
 * than twice the capacity to resolve. At 0.45 denoise the composition from
 * pass one survives and only the detail is rewritten.
 *
 * The upscale is on the latent, not on pixels: a pixel upscale followed by a
 * re encode throws away the structure the first pass just solved.
 * nearest-exact is deliberate. Smooth latent interpolation hands pass two a
 * blurred starting point that a 0.45 denoise cannot recover from.
 *
 * Returns null for families using a custom sampler graph, such as flux2-klein,
 * where the step schedule comes from a scheduler node with no denoise input
 * and a second pass would have to be guessed at.
 */
export function deriveHiresFix(def: FamilyDef | DerivedDef): DerivedDef | null {
  if (def.mode !== 'image') return null

  const graph = cloneGraph(def.graph)
  const ksId = findKSampler(graph)
  const latentId = findEmptyLatent(graph)
  const decodeId = findVaeDecode(graph)
  if (!ksId || !latentId || !decodeId) return null
  // Only insert a second pass where the decode reads the sampler directly.
  // Anything else means a graph shape this derivation has not been verified
  // against, and guessing is how you get a silently wrong picture.
  const decoded = graph[decodeId].inputs.samples
  if (!isRef(decoded) || decoded[0] !== ksId) return null

  const UP = uniqueId(graph, '__hr_upscale')
  const KS2 = uniqueId(graph, '__hr_sampler')

  graph[UP] = {
    class_type: 'LatentUpscaleBy',
    inputs: { samples: [ksId, 0], upscale_method: 'nearest-exact', scale_by: HIRES_DEFAULTS.scale },
  }
  graph[KS2] = {
    class_type: 'KSampler',
    inputs: { ...graph[ksId].inputs, latent_image: [UP, 0], denoise: HIRES_DEFAULTS.denoise },
  }
  graph[decodeId].inputs.samples = [KS2, 0] as Ref

  // Both passes share the family's sampler settings. Pass two's denoise is a
  // separate knob and is deliberately NOT added to bindings.denoise, which
  // would otherwise be written with the first pass's value of 1.
  const bindings: FamilyDef['bindings'] = { ...def.bindings }
  const also = (key: 'seed' | 'steps' | 'cfg' | 'sampler' | 'scheduler', input: string) => {
    const current = bindings[key]
    if (current) bindings[key] = [...current, [KS2, input]]
  }
  also('seed', 'seed')
  also('cfg', 'cfg')
  also('sampler', 'sampler_name')
  also('scheduler', 'scheduler')
  // steps is deliberately NOT shared with pass one. Pass two runs every step
  // it is given (see the set_steps note above) on 2.25x the pixels, so reusing
  // the family's full count would more than triple the job. Callers should
  // pass hiresSteps from hiresStepsFor(); the graph default is the same number.
  graph[KS2].inputs.steps = hiresStepsFor(def.defaults.steps)

  const derived = priorDerivation(def)
  derived.kinds.push('hires')
  // Pass two at 1.5x linear is 2.25x the pixels, at 60% of the step count.
  derived.cost += 1.35
  derived.note =
    'Renders at native size, upscales the latent, then refines the larger frame. Roughly 2.3x the time of a single pass, and noticeably more VRAM.'

  mergeBindings(derived.extra, 'hiresScale', [[UP, 'scale_by']])
  mergeBindings(derived.extra, 'hiresDenoise', [[KS2, 'denoise']])
  mergeBindings(derived.extra, 'hiresSteps', [[KS2, 'steps']])

  return {
    ...def,
    id: `${def.id}__hires`,
    label: `${def.label}: hires fix`,
    graph,
    bindings,
    derived,
  }
}

/**
 * Step count for the second pass.
 *
 * Pass two only has to resolve detail on a composition pass one already solved,
 * and every step it is given is a real step on a much larger latent. Sixty
 * percent of the family's count is the point where more steps stop changing
 * the picture and only cost time.
 */
export function hiresStepsFor(baseSteps: number): number {
  return Math.max(8, Math.round(baseSteps * 0.6))
}

/** Output size a hires pass will produce, for the UI and for feasibility checks. */
export function hiresSize(
  size: { width: number; height: number },
  scale = HIRES_DEFAULTS.scale,
): { width: number; height: number } {
  return {
    width: Math.round((size.width * scale) / 8) * 8,
    height: Math.round((size.height * scale) / 8) * 8,
  }
}

// ---------------------------------------------------------------------------
// LoRA stacks
// ---------------------------------------------------------------------------

export type LoraSpec = {
  /** Filename exactly as LoraLoader.lora_name lists it, subfolder included. */
  name: string
  /** Strength on the diffusion model. 0.6 to 0.9 is the usual band for an anatomy LoRA. */
  strength: number
  /**
   * Strength on the text encoder. Defaults to the model strength. Ignored on
   * families whose text encoder is loaded separately from the diffusion model.
   */
  clipStrength?: number
}

/**
 * Whether a LoRA chain can be inserted: one diffusion model loader, and not a
 * dual model family. Checked without building a graph, so the UI can ask
 * cheaply.
 */
export function canTakeLoras(def: FamilyDef | DerivedDef): boolean {
  return !def.dualModel && findModelLoaders(def.graph).length === 1
}

/**
 * Insert a chain of LoRA loaders between the model loader and its first consumer.
 *
 * This is the other half of the anatomy answer, and it works at a different
 * level from the resolution passes above. A refine pass gives a region enough
 * latent cells to be drawn correctly; a LoRA changes what the model believes
 * correct looks like. The booru trained SDXL families already carry explicit
 * anatomy in their weights, so a LoRA there is for style or for a specific
 * feature. The photoreal bases have far less of it, and that is exactly where a
 * LoRA moves the result most.
 *
 * Two wirings, chosen by what the family loads:
 *
 *   CheckpointLoaderSimple: the checkpoint's own CLIP is bundled and matched to
 *     the weights, so LoraLoader patches MODEL and CLIP together. Skipping the
 *     CLIP patch loses every trigger token the LoRA trained.
 *
 *   UNETLoader or UnetLoaderGGUF: the text encoder is a separate, often
 *     quantized file shared between families. LoraLoaderModelOnly patches only
 *     the diffusion model, which is what these LoRAs are trained against.
 *
 * Order in the array is the order of application, first entry closest to the
 * loader. The model binding still points at the loader node, which has not
 * moved, so it keeps resolving.
 *
 * Returns null for dual model families: the Wan high and low noise halves take
 * different LoRAs at different strengths, which one flat list cannot express.
 */
export function withLoras(def: FamilyDef | DerivedDef, loras: LoraSpec[]): DerivedDef | null {
  if (!canTakeLoras(def)) return null

  const graph = cloneGraph(def.graph)
  const loaderId = findModelLoaders(graph)[0]

  const base = priorDerivation(def)
  const usable = loras.filter(l => l.name && l.strength !== 0)
  if (!usable.length) {
    return { ...def, graph, bindings: { ...def.bindings }, derived: base }
  }

  const checkpoint = graph[loaderId].class_type === 'CheckpointLoaderSimple'
  const clipRef: Ref | null = checkpoint ? [loaderId, 1] : null

  const added: string[] = []
  let model: Ref = [loaderId, 0]
  let clip: Ref | null = clipRef

  for (const [i, lora] of usable.entries()) {
    const id = uniqueId(graph, `__lora_${i + 1}`)
    if (checkpoint && clip) {
      graph[id] = {
        class_type: 'LoraLoader',
        inputs: {
          model,
          clip,
          lora_name: lora.name,
          strength_model: lora.strength,
          strength_clip: lora.clipStrength ?? lora.strength,
        },
      }
      clip = [id, 1]
    } else {
      graph[id] = {
        class_type: 'LoraLoaderModelOnly',
        inputs: { model, lora_name: lora.name, strength_model: lora.strength },
      }
    }
    model = [id, 0]
    added.push(id)
  }

  const skip = new Set(added)
  rewire(graph, [loaderId, 0], model, skip)
  if (checkpoint && clip) rewire(graph, [loaderId, 1], clip, skip)

  const derived = base
  if (!derived.kinds.includes('loras')) derived.kinds.push('loras')
  derived.loraNodes = added
  derived.note = `${added.length} add-on${added.length === 1 ? '' : 's'} applied to the model${checkpoint ? ' and text encoder' : ''}.`

  return {
    ...def,
    id: `${def.id}__lora${added.length}`,
    label: def.label,
    graph,
    // The model binding still names the loader node, which has not moved.
    bindings: { ...def.bindings },
    derived,
  }
}

// ---------------------------------------------------------------------------
// Add-ons on a clip
// ---------------------------------------------------------------------------

export type VideoLoraSpec = LoraSpec & {
  /** Which half of a two-model family takes it. Absent or 'both' means both. */
  half?: 'high' | 'low' | 'both'
}

export type VideoLoraSlots =
  | { kind: 'single'; loader: string }
  | { kind: 'dual'; high: string; low: string }

/** Follow `.model` links upstream from a sampler to the loader that feeds it. */
function upstreamLoader(graph: Graph, from: unknown): string | null {
  if (!isRef(from)) return null
  let id = from[0]
  for (let hops = 0; hops < 8; hops += 1) {
    const n = graph[id]
    if (!n) return null
    if (MODEL_LOADER.test(n.class_type)) return id
    const m = n.inputs.model
    if (!isRef(m)) return null
    id = m[0]
  }
  return null
}

/**
 * Where a chain can go in a video family's graph.
 *
 * One loader: the same place withLoras() uses. Two loaders, the Wan 2.2 14B
 * shape: the half whose sampler adds the noise (add_noise enable, start step
 * 0) is the high-noise half; the other is the low. Read off the samplers
 * rather than the filenames, with the filenames as the fallback.
 */
function videoLoraSlots(def: FamilyDef | DerivedDef): VideoLoraSlots | null {
  const loaders = findModelLoaders(def.graph)
  if (!def.dualModel && loaders.length === 1) return { kind: 'single', loader: loaders[0] }
  if (!def.dualModel || loaders.length !== 2) return null
  let high: string | null = null
  let low: string | null = null
  for (const n of Object.values(def.graph)) {
    if (n.class_type !== 'KSamplerAdvanced') continue
    const loader = upstreamLoader(def.graph, n.inputs.model)
    if (!loader) continue
    const first = n.inputs.add_noise === 'enable' || Number(n.inputs.start_at_step) === 0
    if (first) high = loader
    else low = loader
  }
  const byName = (re: RegExp) =>
    loaders.find(id => re.test(String(def.graph[id].inputs.unet_name ?? def.graph[id].inputs.ckpt_name ?? ''))) ?? null
  high = high ?? byName(/high/i)
  low = low ?? byName(/low/i)
  if (!high || !low || high === low) return null
  return { kind: 'dual', high, low }
}

/** Cheap check for the UI: can this family take a chain at all. */
export function canTakeVideoLoras(def: FamilyDef | DerivedDef): boolean {
  return videoLoraSlots(def) !== null
}

/**
 * Insert add-on chains into a video family's graph.
 *
 * A one-model family is withLoras() exactly. A two-model family gets one
 * chain per half, `__lora_high_N` and `__lora_low_N`, each wired between its
 * loader and whatever consumed the loader before, so a family that already
 * carries an add-on of its own (the Wan 2.2 I2V pair) keeps it: the reader's
 * add-ons go first and the family's own after. A spec marked for one half
 * goes to that half only; anything else goes to both.
 */
export function withVideoLoras(def: FamilyDef | DerivedDef, specs: VideoLoraSpec[]): DerivedDef | null {
  const slots = videoLoraSlots(def)
  if (!slots) return null
  if (slots.kind === 'single') return withLoras(def, specs)

  const graph = cloneGraph(def.graph)
  const base = priorDerivation(def)
  const usable = specs.filter(l => l.name && l.strength !== 0)
  if (!usable.length) return { ...def, graph, bindings: { ...def.bindings }, derived: base }

  const added: string[] = []
  const chain = (loaderId: string, half: 'high' | 'low', list: VideoLoraSpec[]) => {
    let model: Ref = [loaderId, 0]
    const mine: string[] = []
    for (const [i, lora] of list.entries()) {
      const id = uniqueId(graph, `__lora_${half}_${i + 1}`)
      graph[id] = {
        class_type: 'LoraLoaderModelOnly',
        inputs: { model, lora_name: lora.name, strength_model: lora.strength },
      }
      model = [id, 0]
      mine.push(id)
    }
    if (mine.length) rewire(graph, [loaderId, 0], model, new Set(mine))
    added.push(...mine)
  }
  chain(slots.high, 'high', usable.filter(l => l.half !== 'low'))
  chain(slots.low, 'low', usable.filter(l => l.half !== 'high'))

  const derived = base
  if (!derived.kinds.includes('loras')) derived.kinds.push('loras')
  derived.loraNodes = added
  derived.note = `${added.length} add-on loader${added.length === 1 ? '' : 's'} across the high and low halves.`

  return {
    ...def,
    id: `${def.id}__lora${added.length}`,
    label: def.label,
    graph,
    bindings: { ...def.bindings },
    derived,
  }
}

// ---------------------------------------------------------------------------
// Writing the extra knobs
// ---------------------------------------------------------------------------

/**
 * Write derivation knobs into a built graph. Values that are undefined are
 * left at the graph's own defaults, and keys this derivation does not expose
 * are ignored, so one caller can pass a full knob set to any derived family.
 */
export function writeExtras(
  wf: ApiWorkflow,
  def: DerivedDef,
  values: Partial<Record<ExtraKey, string | number>>,
): void {
  for (const [key, value] of Object.entries(values) as [ExtraKey, string | number | undefined][]) {
    if (value === undefined) continue
    for (const [nodeId, input] of def.derived.extra[key] ?? []) {
      const node = wf[nodeId]
      if (node) node.inputs[input] = value
    }
  }
}

/** Build a graph for any derived family that is not a region refine. */
export function instantiateDerived(
  def: DerivedDef,
  params: Params,
  extras?: Partial<Record<ExtraKey, string | number>>,
): ApiWorkflow {
  const wf = instantiate(def, params)
  if (extras) writeExtras(wf, def, extras)
  return wf
}

// ---------------------------------------------------------------------------
// What a family can actually do, for building the UI without guessing
// ---------------------------------------------------------------------------

export type Capabilities = {
  refine: boolean
  faceDetail: boolean
  handDetail: boolean
  hires: boolean
  loras: boolean
}

/**
 * Which quality passes a family supports. The UI hides what is false rather
 * than offering it and failing at queue time, or worse, quietly doing
 * something else.
 */
export function capabilitiesOf(def: FamilyDef | DerivedDef): Capabilities {
  return {
    refine: deriveRefine(def) !== null,
    faceDetail: deriveAutoDetail(def, 'face') !== null,
    handDetail: deriveAutoDetail(def, 'hand') !== null,
    hires: deriveHiresFix(def) !== null,
    loras: canTakeLoras(def),
  }
}

// ---------------------------------------------------------------------------
// Which finished pictures can be made again from their record
// ---------------------------------------------------------------------------

/**
 * Whether a picture's graph can be rebuilt from its archive record.
 *
 * The face, hand, larger render and "make another" rows all re-render a
 * picture from its record. A region pass is filed against the ORIGINAL picture
 * with the region prompt, and nothing on the record says which region or what
 * mask: rebuilt, it redraws the whole original frame at the region's strength
 * and throws the refine away. It is filed with the variant `refine` for that
 * reason. A variant this build does not know came from a newer one sharing the
 * archive, and what it ran cannot be read off this record here either.
 *
 * Region passes filed before `refine` existed say image to image with no pixel
 * budget, which plain image to image always files. A record recovered from
 * ComfyUI that does not say how it was sized looks the same, and cannot be
 * rebuilt faithfully either.
 *
 * Here rather than on one desk, so every surface that offers to run a record
 * again (the desk's own rows, the archive's "Load these settings") asks the
 * same question.
 */
export function rebuildable(entry: Pick<HistoryEntry, 'variant' | 'mode' | 'megapixels'>): boolean {
  const v = entry.variant
  if (v !== null && v !== 'img2img' && v !== 'nolora' && v !== 'i2v') return false
  return !(entry.mode === 'i2i' && entry.megapixels == null)
}

/**
 * Where a region pass was drawn: the record of that picture, the output file
 * it names, the copy in ComfyUI's input folder, or nothing left.
 */
export type RegionOrigin =
  | { kind: 'record'; entry: HistoryEntry }
  | { kind: 'output'; ref: FileRef; fromEntryId?: string }
  | { kind: 'input'; name: string; fromEntryId?: string }
  | { kind: 'gone' }

/**
 * The picture a region pass was drawn on, found again.
 *
 * Null for anything that is not a region pass. The record it came from is
 * preferred, when it is still in the archive and its file is still on disk;
 * then the output file the source names; then the copy the pass itself loaded
 * from ComfyUI's input folder, which is all an uploaded picture ever had.
 * Else it is gone, and the caller says so rather than opening an empty bench.
 */
export function regionOrigin(
  entry: Pick<HistoryEntry, 'variant' | 'source'>,
  records: readonly HistoryEntry[],
): RegionOrigin | null {
  if (entry.variant !== 'refine') return null
  const src = entry.source
  const fromEntryId = src?.fromEntryId || undefined
  const from = fromEntryId ? records.find((r) => r.id === fromEntryId) : undefined
  if (from && !from.missing) return { kind: 'record', entry: from }
  if (src?.ref) return { kind: 'output', ref: src.ref, fromEntryId }
  if (src?.name) return { kind: 'input', name: src.name, fromEntryId }
  return { kind: 'gone' }
}

/**
 * The picture a region pass was drawn on, as the record the Pictures desk
 * opens its region bench on, carrying the pass's own words: the same three
 * places the desk's "Make another like this" looks. Null when that picture is
 * no longer anywhere.
 *
 * Only a record crosses from the Archive to the Pictures desk, so a picture known only
 * by its file is handed over as a record of that file, claiming no maker. It
 * goes as the picture to paint on, never as a pass: a region pass drawn on
 * another region pass is painted on that one, not traced back past it.
 */
export function regionPicture(entry: HistoryEntry, origin: RegionOrigin): HistoryEntry | null {
  if (origin.kind === 'gone') return null
  if (origin.kind === 'record') return { ...origin.entry, prompt: entry.prompt, variant: null }
  const file =
    origin.kind === 'output'
      ? origin.ref
      : {
          filename: origin.name.slice(origin.name.lastIndexOf('/') + 1),
          subfolder: origin.name.includes('/') ? origin.name.slice(0, origin.name.lastIndexOf('/')) : '',
          type: 'input',
        }
  return {
    ...entry,
    id: origin.fromEntryId ?? '',
    file,
    files: undefined,
    variant: null,
    source: undefined,
    width: null,
    height: null,
    model: '',
    modelLabel: '',
    familyId: '',
    familyLabel: '',
    missing: undefined,
  }
}

/**
 * The regions that have no detector, for UI copy. These require a drawn mask
 * and a refine pass. Naming them plainly is the point: a user fixing anatomy
 * needs to know which tool applies to which region.
 */
export const MASK_ONLY_REGIONS = [
  'breasts and nipples',
  'vulva',
  'penis',
  'feet',
  'any other region',
] as const
