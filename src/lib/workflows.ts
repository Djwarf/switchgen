/**
 * Workflow instantiation.
 *
 * Graphs are DATA (src/lib/registry.ts), produced and adversarially verified by
 * the research swarm against ComfyUI's live schema. This module only clones a
 * graph and writes user parameters into the bound node inputs. It never builds
 * a graph procedurally, so adding an architecture means adding registry data,
 * not code.
 *
 * Three parameters have no binding to write through: the ModelSampling shift,
 * the CLIPSetLastLayer clip skip, and the dual-model step split. The registry
 * is generated and must not be hand-edited, and its `bindings` map carries no
 * key for any of them. They are written by finding the node that owns them
 * instead. {@link instantiate} does that itself, so every desk gets the same
 * behaviour and no desk has to remember to ask for it; the writers are also
 * exported for the few places that already hold a built graph.
 */
import type { ApiNode, ApiWorkflow } from './comfy'
import { FAMILY_DEFS, type FamilyDef, type Binding } from './registry'

export type { FamilyDef }
export type Mode = FamilyDef['mode']

export const FAMILIES: FamilyDef[] = FAMILY_DEFS
export const BY_ID: Record<string, FamilyDef> = Object.fromEntries(FAMILY_DEFS.map(f => [f.id, f]))

/**
 * Any node map, built or straight out of the registry. `FamilyDef['graph']` and
 * `ApiWorkflow` are the same shape; the readers and writers below take either.
 */
type NodeMap = Record<string, ApiNode>

/** Effective defaults for one model: family defaults with per-model overrides applied. */
export function defaultsFor(def: FamilyDef, model: string): FamilyDef['defaults'] {
  const over = (def.perModel?.[model] ?? {}) as Partial<FamilyDef['defaults']>
  return { ...def.defaults, ...over }
}

export type Params = {
  model: string
  positive: string
  negative: string
  seed: number
  steps: number
  cfg: number
  width: number
  height: number
  sampler: string
  scheduler: string
  length?: number
  fps?: number
  /** Uploaded filename, for edit, image-to-image and image-to-video families. */
  image?: string
  /** Image-to-image strength: 1 = ignore the source, 0.4 = stay close to it. */
  denoise?: number
  /** Pixel budget for the scaled source image, in megapixels. */
  megapixels?: number
  /**
   * ModelSampling* shift. Written by {@link applyShift}; ignored by a family
   * whose graph carries no ModelSampling node.
   */
  shift?: number
  /**
   * CLIPSetLastLayer `stop_at_clip_layer`, e.g. -2 on Illustrious. Negative.
   * Written by {@link applyClipSkip}; ignored when the graph has no such node.
   */
  clipSkip?: number
  /**
   * Dual-model pass boundary: the step at which the high-noise model hands over
   * to the low-noise one. Defaults to half the steps. Only dual-model families
   * have anywhere to put it.
   */
  split?: number
}

function write(wf: ApiWorkflow, binds: Binding[] | undefined, value: unknown) {
  if (!binds || value === undefined || value === null) return
  for (const [nodeId, input] of binds) {
    const node = wf[nodeId]
    if (node) node.inputs[input] = value
  }
}

// ---------------------------------------------------------------------------
// The three unbound parameters
//
// Each writer is a no-op when the value is absent or the graph has no node that
// takes it, and each returns whether it wrote anything, so a caller that wants
// to say "this family has no shift" can ask without building a second graph.
// ---------------------------------------------------------------------------

const SHIFT_NODE = /^ModelSampling/
const CLIP_SKIP_NODE = 'CLIPSetLastLayer'
const ADVANCED_SAMPLER = 'KSamplerAdvanced'

const clamp = (n: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, n))

/**
 * Write the sampling shift onto every ModelSampling* node in a graph.
 *
 * Z-Image, Chroma and the Wan families all expose the shift, each through a
 * differently named node (ModelSamplingAuraFlow, ModelSamplingSD3), so the
 * match is on the prefix and on the input actually being a number.
 */
export function applyShift(wf: NodeMap, shift: number | null | undefined): boolean {
  if (typeof shift !== 'number' || !Number.isFinite(shift)) return false
  let wrote = false
  for (const node of Object.values(wf)) {
    if (SHIFT_NODE.test(node.class_type) && typeof node.inputs.shift === 'number') {
      node.inputs.shift = shift
      wrote = true
    }
  }
  return wrote
}

/** The shift a family's own graph ships with, or null when it has no shift. */
export function graphShift(wf: NodeMap): number | null {
  for (const node of Object.values(wf)) {
    if (SHIFT_NODE.test(node.class_type) && typeof node.inputs.shift === 'number') {
      return node.inputs.shift
    }
  }
  return null
}

/**
 * Write the clip skip onto every CLIPSetLastLayer node in a graph.
 *
 * ComfyUI counts layers backwards, so the value is negative: -1 is the last
 * layer, -2 the Illustrious and Pony house setting. A positive number is taken
 * as the same layer said the other way round, because "clip skip 2" is how the
 * checkpoints' own cards write it. Zero is not a layer and is ignored.
 */
export function applyClipSkip(wf: NodeMap, clipSkip: number | null | undefined): boolean {
  if (typeof clipSkip !== 'number' || !Number.isFinite(clipSkip)) return false
  const n = Math.round(clipSkip)
  if (n === 0) return false
  const layer = clamp(n < 0 ? n : -n, -24, -1)
  let wrote = false
  for (const node of Object.values(wf)) {
    if (node.class_type === CLIP_SKIP_NODE && 'stop_at_clip_layer' in node.inputs) {
      node.inputs.stop_at_clip_layer = layer
      wrote = true
    }
  }
  return wrote
}

/** The clip skip a family's own graph ships with, or null when it sets none. */
export function graphClipSkip(wf: NodeMap): number | null {
  for (const node of Object.values(wf)) {
    if (node.class_type === CLIP_SKIP_NODE && typeof node.inputs.stop_at_clip_layer === 'number') {
      return node.inputs.stop_at_clip_layer
    }
  }
  return null
}

/**
 * Find the handover between two KSamplerAdvanced nodes sharing one latent
 * chain: the first stops at a step, the second starts at that same step.
 *
 * This is the whole mechanism of the Wan 2.2 14B families. The two numbers must
 * agree, or the boundary steps are either sampled twice or skipped, so they are
 * found and written as a pair rather than bound separately.
 */
function splitPairs(wf: NodeMap): { first: ApiNode; second: ApiNode }[] {
  const advanced = Object.values(wf).filter(n => n.class_type === ADVANCED_SAMPLER)
  const pairs: { first: ApiNode; second: ApiNode }[] = []
  for (const first of advanced) {
    const end = first.inputs.end_at_step
    if (typeof end !== 'number' || end <= 0) continue
    for (const second of advanced) {
      if (second === first) continue
      const start = second.inputs.start_at_step
      const tail = second.inputs.end_at_step
      if (start !== end) continue
      if (typeof tail !== 'number' || tail <= end) continue
      pairs.push({ first, second })
    }
  }
  return pairs
}

/** Move the dual-model pass boundary. */
export function applySplit(wf: NodeMap, split: number | null | undefined): boolean {
  if (typeof split !== 'number' || !Number.isFinite(split)) return false
  const boundary = Math.max(1, Math.floor(split))
  const pairs = splitPairs(wf)
  for (const { first, second } of pairs) {
    first.inputs.end_at_step = boundary
    second.inputs.start_at_step = boundary
  }
  return pairs.length > 0
}

/** The pass boundary a dual-model graph ships with, or null when it has none. */
export function graphSplit(wf: NodeMap): number | null {
  const [pair] = splitPairs(wf)
  return pair ? (pair.first.inputs.end_at_step as number) : null
}

export function instantiate(def: FamilyDef, p: Params): ApiWorkflow {
  const wf: ApiWorkflow = JSON.parse(JSON.stringify(def.graph))
  const b = def.bindings

  // A dual-model family (Wan 2.2 14B high/low-noise pairs) carries a fixed,
  // matched pair of files in its graph. Overwriting both loaders with one
  // picked file would load the same noise model twice and ruin the step split.
  if (!def.dualModel) write(wf, b.model, p.model)

  write(wf, b.positive, p.positive)
  write(wf, b.negative, p.negative)
  write(wf, b.seed, Math.floor(p.seed))
  write(wf, b.steps, Math.floor(p.steps))
  write(wf, b.cfg, p.cfg)
  write(wf, b.sampler, p.sampler)
  write(wf, b.scheduler, p.scheduler)
  write(wf, b.width, Math.floor(p.width))
  write(wf, b.height, Math.floor(p.height))
  if (p.length) write(wf, b.length, Math.floor(p.length))
  if (p.fps) write(wf, b.fps, p.fps)
  if (p.image) write(wf, b.image, p.image)
  if (p.denoise !== undefined) write(wf, b.denoise, p.denoise)
  if (p.megapixels !== undefined) write(wf, b.megapixels, p.megapixels)

  // The three the registry has no binding for. Written here rather than at each
  // desk, so a value the composer collects can never be silently dropped on the
  // way to the queue and then filed in the archive as what produced the result.
  applyShift(wf, p.shift)
  applyClipSkip(wf, p.clipSkip)
  // A dual-model family splits its steps between the high-noise and low-noise
  // halves. Half the run is both the documented default and the boundary every
  // registry preset hardcodes, so an unstated split reproduces the graph and a
  // changed step count no longer leaves the boundary behind.
  if (def.dualModel) applySplit(wf, p.split ?? Math.round(Math.floor(p.steps) / 2))

  return wf
}

/** Sidecar files a family needs beyond the model itself. */
export function sidecarsOf(def: FamilyDef): { clip: string[]; vae: string } {
  const clip = new Set<string>()
  let vae = ''
  for (const node of Object.values(def.graph)) {
    const cn = node.inputs['clip_name']
    if (typeof cn === 'string') clip.add(cn)
    const vn = node.inputs['vae_name']
    if (typeof vn === 'string') vae = vn
  }
  return { clip: [...clip], vae }
}

/** Every model file referenced by any loader in a family's graph. */
export function modelsOf(def: FamilyDef): string[] {
  const out = new Set<string>(def.models)
  for (const node of Object.values(def.graph)) {
    for (const k of ['ckpt_name', 'unet_name'] as const) {
      const v = node.inputs[k]
      if (typeof v === 'string') out.add(v)
    }
  }
  return [...out]
}

// ---------------------------------------------------------------------------
// Image-to-image, derived rather than hand-authored.
//
// Every text-to-image graph ends up sampling from an Empty*Latent* node. Swap
// that for a VAEEncode of an uploaded image and the same graph becomes img2img.
// Deriving it means any family we add later gets img2img for free, instead of
// needing a second hand-verified graph.
// ---------------------------------------------------------------------------

const EMPTY_LATENT = /^Empty.*Latent/

function findLatentNode(graph: FamilyDef['graph']): string | null {
  for (const [id, n] of Object.entries(graph)) {
    if (EMPTY_LATENT.test(n.class_type) && 'width' in n.inputs && 'height' in n.inputs) return id
  }
  return null
}

function findVaeRef(graph: FamilyDef['graph']): unknown {
  for (const n of Object.values(graph)) {
    if (n.class_type === 'VAEDecode' && Array.isArray(n.inputs.vae)) return n.inputs.vae
  }
  for (const [id, n] of Object.entries(graph)) {
    if (n.class_type === 'VAELoader') return [id, 0]
  }
  return null
}

/** True when the family's sampler exposes a denoise control we can drive. */
function findDenoiseTargets(graph: FamilyDef['graph']): Binding[] {
  const out: Binding[] = []
  for (const [id, n] of Object.entries(graph)) {
    if ('denoise' in n.inputs && !Array.isArray(n.inputs.denoise)) out.push([id, 'denoise'])
  }
  return out
}

/**
 * Derive an image-to-image variant of a text-to-image family.
 * Returns null when the family cannot support it, for example a custom-sampler graph
 * whose step count comes from a scheduler node with no denoise input. Callers
 * must treat null as "not offered" rather than falling back to text-to-image,
 * which would silently ignore the user's source image.
 */
export function deriveImg2Img(def: FamilyDef): FamilyDef | null {
  if (def.mode !== 'image') return null
  const graph: FamilyDef['graph'] = JSON.parse(JSON.stringify(def.graph))
  const latentId = findLatentNode(graph)
  const vae = findVaeRef(graph)
  const denoise = findDenoiseTargets(graph)
  if (!latentId || !vae || !denoise.length) return null

  const LOAD = '__i2i_load'
  const FIT = '__i2i_fit'
  const ENCODE = '__i2i_encode'
  graph[LOAD] = { class_type: 'LoadImage', inputs: { image: 'example.png' } }
  // Scale the source to a sane pixel budget BEFORE the VAE sees it. Without
  // this a 4000x3000 photo is encoded at full resolution and OOMs a 16GB card.
  // resolution_steps 16 snaps both dimensions to multiples of 16 while keeping
  // the aspect ratio - so it neither crops nor stretches.
  graph[FIT] = {
    class_type: 'ImageScaleToTotalPixels',
    inputs: { image: [LOAD, 0], upscale_method: 'lanczos', megapixels: 1.0, resolution_steps: 16 },
  }
  graph[ENCODE] = { class_type: 'VAEEncode', inputs: { pixels: [FIT, 0], vae } }

  // Re-point every consumer of the empty latent at the encoded source image.
  for (const node of Object.values(graph)) {
    for (const [k, v] of Object.entries(node.inputs)) {
      if (Array.isArray(v) && v.length === 2 && v[0] === latentId) node.inputs[k] = [ENCODE, 0]
    }
  }
  delete graph[latentId]

  // Width/height came from the node we just removed; in img2img the source
  // image dictates the output size, so those controls no longer apply.
  const bindings = { ...def.bindings }
  delete bindings.width
  delete bindings.height
  bindings.image = [[LOAD, 'image']]
  bindings.megapixels = [[FIT, 'megapixels']]
  bindings.denoise = denoise

  return {
    ...def,
    id: `${def.id}__img2img`,
    label: `${def.label}, image to image`,
    graph,
    bindings,
    notes: def.notes,
  }
}

/** Families that support image-to-image, keyed by the source family id. */
export const IMG2IMG: Record<string, FamilyDef> = Object.fromEntries(
  FAMILY_DEFS.map(d => [d.id, deriveImg2Img(d)]).filter(([, v]) => v) as [string, FamilyDef][],
)
