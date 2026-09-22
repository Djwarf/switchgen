/**
 * Continued video: shots that start where the last one stopped.
 *
 * A single Wan generation is five seconds. A scene is not. This module turns a
 * list of shots into a list of jobs where shot N+1 begins on the final frame of
 * shot N, so the clips can be laid end to end and read as one continuous take.
 *
 * WHAT IS DERIVED, NOT AUTHORED
 * Every graph here starts as a clone of a registry family that was already
 * validated against the live /object_info, exactly as deriveImg2Img does in
 * workflows.ts. Nothing invents a graph from nothing. A family that cannot
 * carry the rewiring returns null, so the caller hides the control instead of
 * queueing eight minutes of GPU time that produces the wrong thing.
 *
 * THE HANDOFF, AND WHY IT IS A PNG
 * Each chainable graph gets a tap: ImageFromBatch(batch_index -1, length 1) on
 * the decode output, then SaveImage. Verified in
 * ComfyUI/comfy_extras/nodes_images.py: a negative batch_index has the batch
 * size added to it before clamping, so -1 really is the last frame on this
 * install. The tap sits BEFORE the WEBM encoder, so the handoff frame never
 * carries vp9 crf 32 loss. Pulling the same frame out of the finished clip
 * would hand the next shot a compressed frame and compound two different
 * artefacts instead of one.
 *
 * The next shot reads that PNG straight from the output directory using
 * ComfyUI's annotated path form, "subfolder/name.png [output]". LoadImage
 * declares VALIDATE_INPUTS against folder_paths.exists_annotated_filepath
 * (nodes.py:1808), which is why a filename outside the input directory passes.
 * The suffix is parsed as name[:-9] (folder_paths.py:258), so the format is
 * exact: one space, then the bracket. No re upload, no download round trip.
 *
 * DRIFT IS REAL. IT IS NOT SOLVED HERE.
 * Every hop re encodes the model's own output: decode to pixels, then VAE
 * encode that image again as the next shot's condition. Colour creeps, contrast
 * flattens, fine texture softens. By the fourth or fifth hop a face is visibly
 * a different face. Three honest mitigations, all partial:
 *
 *   1. Fewer, longer shots. A 121 frame shot is one hop. Two 61 frame shots
 *      covering the same time are two hops with a seam in the middle. Length
 *      costs VRAM once; hops cost quality forever.
 *   2. A VACE reference_image re anchors appearance on every shot, because the
 *      reference is a fixed file rather than the previous output. It fights
 *      identity drift. It does not fix colour drift in the chained frame.
 *   3. Re anchor on a schedule. shotPlan's reanchorEvery restarts the chain
 *      from a clean keyframe every few shots, which resets accumulation and
 *      buys a cut point in exchange for a visible jump.
 *
 * ShotJob.hops reports how many generations deep a shot sits. Show it. A user
 * who can see the number can decide for themselves.
 *
 * WHAT THE HARDWARE ALLOWS
 * Each shot is a separate generation of several minutes, and a reel of ten is
 * ten of them. Nothing here batches: the plan is a queue, meant to be run one
 * job at a time so the peak stays where it was measured.
 */
import type { ApiWorkflow, FileRef, OutputFile } from './comfy'
import type { FamilyDef } from './registry'
import { instantiate, type Params } from './workflows'

type Graph = FamilyDef['graph']
type GraphNode = Graph[string]
/** Any node map: a registry graph or an instantiated workflow. Same shape. */
export type GraphLike = Record<string, GraphNode>
type Link = [nodeId: string, slot: number]

/** Node ids this module inserts. Prefixed so they cannot collide with registry ids. */
export const NODE_IDS = {
  start: '__cont_start',
  end: '__cont_end',
  reference: '__cont_ref',
  lastFrame: '__cont_last',
  frameSave: '__cont_frame',
  trim: '__cont_trim',
} as const

/** Output path the handoff frames are written under. */
export const CHAIN_PREFIX = 'switchgen/chain'

/** Default reel folder. Shots are numbered so the files sort into cutting order. */
export const REEL_PREFIX = 'switchgen/reel'

/**
 * Placeholder for an inserted LoadImage. It is a real file in ComfyUI's input
 * directory, which matters because scripts/validate-workflows checks combo
 * values against the live option list.
 */
export const PLACEHOLDER_IMAGE = 'example.png'

/**
 * Nodes that accept a start frame, checked against the live schema.
 *   Wan22ImageToVideoLatent  optional start_image, bakes it into the latent
 *   WanImageToVideo          optional start_image, conditioning concat route
 *   WanFirstLastFrameToVideo optional start_image and end_image
 */
const START_IMAGE_HOSTS = ['Wan22ImageToVideoLatent', 'WanImageToVideo', 'WanFirstLastFrameToVideo']

const DECODE_NODES = ['VAEDecode', 'VAEDecodeTiled']

const EMPTY_VIDEO_LATENT = /^Empty.*(Video|Latent)/

// ---------------------------------------------------------------------------
// Graph primitives
// ---------------------------------------------------------------------------

const isLink = (v: unknown): v is Link =>
  Array.isArray(v) && v.length === 2 && typeof v[0] === 'string' && typeof v[1] === 'number'

function clone<T>(v: T): T {
  return JSON.parse(JSON.stringify(v)) as T
}

function idsOfClass(graph: GraphLike, ...classes: string[]): string[] {
  return Object.entries(graph)
    .filter(([, n]) => classes.includes(n.class_type))
    .map(([id]) => id)
}

/** The decode that feeds the clip writer. Multiple decodes resolve by who saves. */
export function findDecodeNode(graph: GraphLike): string | null {
  const ids = idsOfClass(graph, ...DECODE_NODES)
  if (ids.length <= 1) return ids[0] ?? null
  const feedsSaver = ids.find(id =>
    Object.values(graph).some(
      n => n.class_type.startsWith('Save') && Object.values(n.inputs).some(v => isLink(v) && v[0] === id),
    ),
  )
  return feedsSaver ?? ids[0] ?? null
}

/** Whatever a family's VAEDecode is using, or the loader itself. */
function findVaeRef(graph: GraphLike): Link | null {
  for (const n of Object.values(graph)) {
    if (DECODE_NODES.includes(n.class_type) && isLink(n.inputs.vae)) return n.inputs.vae
  }
  const loader = idsOfClass(graph, 'VAELoader')[0]
  return loader ? [loader, 0] : null
}

/**
 * Insert ImageFromBatch on a decode output and return the new node id.
 *
 * Mutates `graph`. batch_index -1 with length 1 is the last decoded frame:
 * ImageFromBatch adds the batch size to a negative index before clamping, so
 * the slice is the final frame rather than an empty tensor.
 */
export function lastFrameNode(
  graph: GraphLike,
  decodeNodeId: string,
  opts: { id?: string; batchIndex?: number; length?: number } = {},
): string | null {
  if (!graph[decodeNodeId]) return null
  const id = opts.id ?? NODE_IDS.lastFrame
  graph[id] = {
    class_type: 'ImageFromBatch',
    inputs: { image: [decodeNodeId, 0], batch_index: opts.batchIndex ?? -1, length: opts.length ?? 1 },
  }
  return id
}

/**
 * Last frame plus a SaveImage, so the shot publishes the frame the next shot
 * consumes. Mutates `graph`. Idempotent: calling it twice adds one tap.
 */
export function addChainTap(
  graph: GraphLike,
  opts: { prefix?: string; decodeNodeId?: string } = {},
): { pick: string; save: string } | null {
  if (graph[NODE_IDS.frameSave]) return { pick: NODE_IDS.lastFrame, save: NODE_IDS.frameSave }
  const decode = opts.decodeNodeId ?? findDecodeNode(graph)
  if (!decode) return null
  const pick = lastFrameNode(graph, decode)
  if (!pick) return null
  graph[NODE_IDS.frameSave] = {
    class_type: 'SaveImage',
    inputs: { images: [pick, 0], filename_prefix: opts.prefix ?? CHAIN_PREFIX },
  }
  return { pick, save: NODE_IDS.frameSave }
}

/**
 * Point a node's image input at a LoadImage, reusing one if the family already
 * wired it. Returns the loader id. Mutates `graph`.
 */
function ensureImageLoader(graph: GraphLike, hostId: string, input: string, preferredId: string): string {
  const host = graph[hostId]
  const existing = host && host.inputs[input]
  if (isLink(existing) && graph[existing[0]]?.class_type === 'LoadImage') return existing[0]
  graph[preferredId] = { class_type: 'LoadImage', inputs: { image: PLACEHOLDER_IMAGE } }
  if (host) host.inputs[input] = [preferredId, 0]
  return preferredId
}

/** Replace one node in place, remapping the output slots its consumers read. */
function replaceNode(graph: GraphLike, id: string, node: GraphNode, slotMap?: Record<number, number>) {
  graph[id] = node
  if (!slotMap) return
  for (const [otherId, n] of Object.entries(graph)) {
    if (otherId === id) continue
    for (const [k, v] of Object.entries(n.inputs)) {
      if (isLink(v) && v[0] === id && slotMap[v[1]] !== undefined) n.inputs[k] = [id, slotMap[v[1]] as number]
    }
  }
}

// ---------------------------------------------------------------------------
// Image setters
//
// FamilyDef.bindings carries a fixed set of keys generated into registry.ts,
// with one slot for an image. A bookend shot needs two files and a VACE shot
// needs a reference, so those travel as graph writes rather than bindings. The
// setters find their target by following the link into the node that consumes
// it, which keeps working when registry ids change under a regeneration.
// ---------------------------------------------------------------------------

function loaderFeeding(wf: GraphLike, input: string): string | null {
  for (const n of Object.values(wf)) {
    const v = n.inputs[input]
    if (isLink(v) && wf[v[0]]?.class_type === 'LoadImage') return v[0]
  }
  return null
}

function setImageInput(wf: GraphLike, input: string, filename: string): boolean {
  const id = loaderFeeding(wf, input)
  if (!id) return false
  const node = wf[id]
  if (!node) return false
  node.inputs.image = filename
  return true
}

/** The frame a shot opens on. Returns false when the graph has no start slot. */
export function setStartImage(wf: ApiWorkflow, filename: string): boolean {
  return setImageInput(wf as GraphLike, 'start_image', filename)
}

/** The frame a shot is required to land on. Bookend graphs only. */
export function setEndImage(wf: ApiWorkflow, filename: string): boolean {
  return setImageInput(wf as GraphLike, 'end_image', filename)
}

/** The appearance anchor. VACE graphs only. */
export function setReferenceImage(wf: ApiWorkflow, filename: string): boolean {
  return setImageInput(wf as GraphLike, 'reference_image', filename)
}

/**
 * Rename a workflow's outputs so a reel's clips sort into cutting order.
 * Video writers take `prefix`, the handoff frame takes `prefix.frame`.
 */
export function setOutputPrefix(wf: ApiWorkflow, prefix: string): void {
  for (const [id, node] of Object.entries(wf as GraphLike)) {
    if (!('filename_prefix' in node.inputs)) continue
    node.inputs.filename_prefix = id === NODE_IDS.frameSave ? `${prefix}.frame` : prefix
  }
}

/**
 * ComfyUI's annotated path, which LoadImage accepts for files outside the
 * input directory. Exact format: one space before the bracket.
 */
export function annotatedRef(f: FileRef): string {
  const path = f.subfolder ? `${f.subfolder}/${f.filename}` : f.filename
  const type = f.type || 'output'
  return type === 'input' ? path : `${path} [${type}]`
}

/**
 * The handoff frame among a finished run's outputs.
 *
 * A chainable video graph writes exactly one still: the tap. Preference goes to
 * a filename that looks like the tap, then to the last image, so a family that
 * saves a poster frame of its own cannot quietly become the chain source.
 */
export function chainFrameOf(files: readonly OutputFile[]): OutputFile | null {
  const images = files.filter(f => f.kind === 'image')
  if (!images.length) return null
  const tagged = images.filter(f => /(^chain|\.frame)/.test(f.filename))
  const pick = tagged.length ? tagged : images
  return pick[pick.length - 1] ?? null
}

/** Wan latent maths wants 4k+1 frames. Snap a request to the nearest legal length. */
export function snapLength(frames: number): number {
  const k = Math.max(1, Math.round((frames - 1) / 4))
  return k * 4 + 1
}

// ---------------------------------------------------------------------------
// Derivations
// ---------------------------------------------------------------------------

/** What a family can be asked to do beyond a plain shot. */
export type ShotVariant = 'continuation' | 'bookend' | 'vace'

/**
 * A shot that opens on a supplied frame.
 *
 * wan22-5b conditions through the latent: Wan22ImageToVideoLatent takes an
 * IMAGE LINK on start_image, not a filename, so a LoadImage node is inserted
 * and bound. The 14B I2V family already carries its own LoadImage, and that one
 * is reused rather than duplicated.
 *
 * Returns null for text to video families such as wan22-14b-t2v. Their weights
 * have no image conditioning channels, and swapping in a latent node that
 * expects a different VAE would build a graph nobody has run. Hide the control.
 */
export function deriveContinuation(def: FamilyDef): FamilyDef | null {
  if (def.mode !== 'video') return null
  const graph = clone(def.graph)
  const hostId = idsOfClass(graph, ...START_IMAGE_HOSTS)[0]
  if (!hostId) return null
  if (!addChainTap(graph)) return null

  const loadId = ensureImageLoader(graph, hostId, 'start_image', NODE_IDS.start)
  const bindings = { ...def.bindings, image: [[loadId, 'image'] as [string, string]] }

  return {
    ...def,
    id: `${def.id}__continue`,
    label: `${def.label}: continued shot`,
    graph,
    bindings,
  }
}

/**
 * A shot pinned at both ends, through WanFirstLastFrameToVideo.
 *
 * This is the difference between storyboarding and hoping. Give it the frame
 * the shot starts on and the frame it has to reach, and the model fills the
 * span between them.
 *
 * The swap is safe in place: WanImageToVideo and WanFirstLastFrameToVideo both
 * return positive, negative, latent on slots 0, 1, 2, so every downstream link
 * stays valid. Only the clip vision input changes name.
 *
 * Returns null for wan22-5b. The 5B takes its image through the latent rather
 * than through conditioning, so there is no end frame path on that model, and
 * no validated graph that would give it one.
 */
export function deriveBookend(def: FamilyDef): FamilyDef | null {
  if (def.mode !== 'video') return null
  const graph = clone(def.graph)
  const hostId = idsOfClass(graph, 'WanFirstLastFrameToVideo')[0] ?? idsOfClass(graph, 'WanImageToVideo')[0]
  if (!hostId) return null
  const host = graph[hostId]
  if (!host) return null
  if (!addChainTap(graph)) return null

  if (host.class_type === 'WanImageToVideo') {
    const inputs: Record<string, unknown> = { ...host.inputs }
    if ('clip_vision_output' in inputs) {
      inputs.clip_vision_start_image = inputs.clip_vision_output
      delete inputs.clip_vision_output
    }
    replaceNode(graph, hostId, { class_type: 'WanFirstLastFrameToVideo', inputs })
  }

  const startId = ensureImageLoader(graph, hostId, 'start_image', NODE_IDS.start)
  ensureImageLoader(graph, hostId, 'end_image', NODE_IDS.end)
  const bindings = { ...def.bindings, image: [[startId, 'image'] as [string, string]] }

  return {
    ...def,
    id: `${def.id}__bookend`,
    label: `${def.label}: first and last frame`,
    graph,
    bindings,
  }
}

/** True when a family's weights are VACE, which is what WanVaceToVideo needs. */
export function isVaceFamily(def: FamilyDef): boolean {
  if (idsOfClass(def.graph, 'WanVaceToVideo').length) return true
  if (def.models.some(m => /vace/i.test(m))) return true
  return Object.values(def.graph).some(n =>
    ['unet_name', 'ckpt_name'].some(k => typeof n.inputs[k] === 'string' && /vace/i.test(n.inputs[k] as string)),
  )
}

/**
 * A VACE shot with a reference_image, the strongest anchor against drift.
 *
 * VACE prepends the encoded reference as an extra latent frame, which is why
 * the node returns trim_latent and why a TrimVideoLatent is inserted between
 * the sampler and the decode. Skip that trim and the clip opens on a ghost of
 * the reference.
 *
 * GATED, DELIBERATELY. Two VACE checkpoints are installed
 * (Wan2.1_14B_VACE-Q4_K_M.gguf and Wan2.1-VACE-1.3B-Q8_0.gguf) but no VACE
 * family is registered yet, so this returns null for everything today. The
 * rewiring is schema correct and waiting. What it will not do is repoint a Wan
 * 2.2 family's loader, VAE and text encoder at VACE weights in one unverified
 * jump, because that is authoring a family, not deriving one.
 *
 * Wan 2.2's TI2V latent node is deliberately not a host here. VACE ships as Wan
 * 2.1 weights against the 2.1 VAE, so a graph built around Wan22ImageToVideoLatent
 * is the wrong body to graft it onto.
 */
export function deriveVaceShot(def: FamilyDef): FamilyDef | null {
  if (def.mode !== 'video' || !isVaceFamily(def)) return null
  const graph = clone(def.graph)

  let hostId: string | null = idsOfClass(graph, 'WanVaceToVideo')[0] ?? null
  if (!hostId) {
    const cond = idsOfClass(graph, 'WanImageToVideo', 'WanFirstLastFrameToVideo')[0]
    hostId = cond ? buildVaceFromConditioning(graph, cond) : buildVaceFromEmptyLatent(graph)
  }
  if (!hostId || !insertVaceTrim(graph, hostId)) return null
  if (!addChainTap(graph)) return null
  pruneOrphanLoaders(graph)

  const refId = ensureImageLoader(graph, hostId, 'reference_image', NODE_IDS.reference)
  const bindings = { ...def.bindings, image: [[refId, 'image'] as [string, string]] }
  // Size and length now live on the VACE node, whatever the family used before.
  for (const key of ['width', 'height', 'length'] as const) {
    bindings[key] = [[hostId, key] as [string, string]]
  }

  return {
    ...def,
    id: `${def.id}__vace`,
    label: `${def.label}: anchored shot`,
    graph,
    bindings,
  }
}

/** Swap a conditioning video node for VACE. Slots 0, 1, 2 line up, so links hold. */
function buildVaceFromConditioning(graph: GraphLike, hostId: string): string | null {
  const host = graph[hostId]
  if (!host) return null
  const inputs: Record<string, unknown> = {
    positive: host.inputs.positive,
    negative: host.inputs.negative,
    vae: host.inputs.vae,
    width: host.inputs.width,
    height: host.inputs.height,
    length: host.inputs.length,
    batch_size: host.inputs.batch_size ?? 1,
    strength: 1.0,
  }
  replaceNode(graph, hostId, { class_type: 'WanVaceToVideo', inputs })
  return hostId
}

/**
 * Build VACE where a text to video family has only an empty latent. The node
 * takes over that id, so width, height and length bindings survive, and the
 * samplers are repointed at the conditioning VACE returns.
 */
function buildVaceFromEmptyLatent(graph: GraphLike): string | null {
  const latentId = Object.entries(graph).find(
    ([, n]) => EMPTY_VIDEO_LATENT.test(n.class_type) && 'width' in n.inputs && 'length' in n.inputs,
  )?.[0]
  const vae = findVaeRef(graph)
  if (!latentId || !vae) return null

  const latent = graph[latentId]
  if (!latent) return null

  const consumer = Object.values(graph).find(n => isLink(n.inputs.positive) && isLink(n.inputs.negative))
  if (!consumer) return null

  const inputs: Record<string, unknown> = {
    positive: consumer.inputs.positive,
    negative: consumer.inputs.negative,
    vae,
    width: latent.inputs.width,
    height: latent.inputs.height,
    length: latent.inputs.length,
    batch_size: latent.inputs.batch_size ?? 1,
    strength: 1.0,
  }
  // The empty latent sat on slot 0. VACE returns its latent on slot 2.
  replaceNode(graph, latentId, { class_type: 'WanVaceToVideo', inputs }, { 0: 2 })

  for (const [id, n] of Object.entries(graph)) {
    if (id === latentId) continue
    if (isLink(n.inputs.positive)) n.inputs.positive = [latentId, 0]
    if (isLink(n.inputs.negative)) n.inputs.negative = [latentId, 1]
  }
  return latentId
}

/**
 * Drop LoadImage nodes nothing reads any more. WanVaceToVideo has no start_image,
 * so a family that had one leaves its loader stranded.
 */
function pruneOrphanLoaders(graph: GraphLike) {
  for (const id of idsOfClass(graph, 'LoadImage')) {
    const used = Object.entries(graph).some(
      ([otherId, n]) => otherId !== id && Object.values(n.inputs).some(v => isLink(v) && v[0] === id),
    )
    if (!used) delete graph[id]
  }
}

/** Trim the reference frame off the sampled latent before it is decoded. */
function insertVaceTrim(graph: GraphLike, vaceId: string): boolean {
  if (graph[NODE_IDS.trim]) return true
  const decodeId = findDecodeNode(graph)
  const decode = decodeId ? graph[decodeId] : null
  if (!decode || !isLink(decode.inputs.samples)) return false
  graph[NODE_IDS.trim] = {
    class_type: 'TrimVideoLatent',
    inputs: { samples: decode.inputs.samples, trim_amount: [vaceId, 3] },
  }
  decode.inputs.samples = [NODE_IDS.trim, 0]
  return true
}

/**
 * A plain shot that still publishes its last frame, for the opening shot of a
 * reel. The first shot has nothing to continue from, but everything after it
 * continues from this.
 */
export function deriveChainTap(def: FamilyDef): FamilyDef | null {
  if (def.mode !== 'video') return null
  const graph = clone(def.graph)
  if (!addChainTap(graph)) return null
  return { ...def, id: `${def.id}__tap`, label: `${def.label}: opening shot`, graph }
}

/** Every continued shot variant a family supports. Null entries are not offered. */
export function shotVariants(def: FamilyDef): Record<ShotVariant, FamilyDef | null> {
  return {
    continuation: deriveContinuation(def),
    bookend: deriveBookend(def),
    vace: deriveVaceShot(def),
  }
}

/**
 * One short line for a variant a family cannot do, so the interface can say why
 * a control is missing instead of leaving a hole. Null means it is available.
 */
export function explainUnavailable(def: FamilyDef, variant: ShotVariant): string | null {
  if (def.mode !== 'video') return 'Video families only.'
  switch (variant) {
    case 'continuation':
      return deriveContinuation(def)
        ? null
        : `${def.label} generates from text alone. It has no slot for an opening frame.`
    case 'bookend':
      if (deriveBookend(def)) return null
      return deriveContinuation(def)
        ? `${def.label} conditions through the latent. That carries an opening frame but not a closing one.`
        : `${def.label} generates from text alone. There is no frame to pin at either end.`
    case 'vace':
      return deriveVaceShot(def) ? null : 'VACE weights are installed but no VACE family is registered yet.'
  }
}

/** Families keyed by source id, in the shape workflows.ts uses for IMG2IMG. */
function variantMap(pick: (def: FamilyDef) => FamilyDef | null, defs: readonly FamilyDef[]) {
  const out: Record<string, FamilyDef> = {}
  for (const def of defs) {
    const derived = pick(def)
    if (derived) out[def.id] = derived
  }
  return out
}

/** Build the lookup tables. Call once with FAMILIES, or with a filtered list. */
export function continuationTables(defs: readonly FamilyDef[]): {
  continuation: Record<string, FamilyDef>
  bookend: Record<string, FamilyDef>
  vace: Record<string, FamilyDef>
  tap: Record<string, FamilyDef>
} {
  return {
    continuation: variantMap(deriveContinuation, defs),
    bookend: variantMap(deriveBookend, defs),
    vace: variantMap(deriveVaceShot, defs),
    tap: variantMap(deriveChainTap, defs),
  }
}

// ---------------------------------------------------------------------------
// Planning a reel
// ---------------------------------------------------------------------------

/** One shot as the user describes it. Only the prompt is required. */
export type ShotSpec = {
  prompt: string
  negative?: string
  seed?: number
  /** Frames. Snapped to 4k+1. Longer is cheaper than more shots. */
  length?: number
  label?: string
  /** Open on this file instead of on the previous shot. Resets the drift count. */
  startImage?: string
  /** Land on this file. Needs a family that supports the bookend variant. */
  endImage?: string
  /** Appearance anchor. Needs a VACE family. */
  referenceImage?: string
}

export type ShotStart =
  | { from: 'none' }
  | { from: 'given'; image: string }
  | { from: 'anchor'; image: string }
  | { from: 'previous' }

export type ShotJob = {
  index: number
  key: string
  label: string
  def: FamilyDef
  params: Params
  start: ShotStart
  endImage?: string
  referenceImage?: string
  /** Output path for this shot's clip. Numbered, so the reel sorts into order. */
  outputPrefix: string
  /**
   * Generations since a clean keyframe. 0 means this shot opens on a real file.
   * 4 means four rounds of encode and decode sit between this shot and anything
   * the user actually chose.
   */
  hops: number
  notes: string[]
}

export type ShotPlan = {
  jobs: ShotJob[]
  /** Frames across the whole reel. */
  frames: number
  /** Running time at the plan's fps. */
  seconds: number
  warnings: string[]
}

export type ShotPlanInput = {
  /** The family for a shot that starts from nothing. */
  base: FamilyDef
  /** Shared settings. Per shot fields override prompt, seed, negative and length. */
  params: Params
  shots: readonly ShotSpec[]
  /** Defaults to deriveContinuation(base). Pass null to force unchained shots. */
  continuation?: FamilyDef | null
  /** Defaults to deriveBookend(base), used only by shots that name an end frame. */
  bookend?: FamilyDef | null
  /** Defaults to deriveVaceShot(base), used only by shots that name a reference. */
  vace?: FamilyDef | null
  /** A clean keyframe the reel can fall back to. */
  anchorImage?: string
  /** Restart from the anchor every N shots. 0 or absent means never. */
  reanchorEvery?: number
  /** Output folder for the reel. */
  prefix?: string
}

/**
 * Turn shots into an ordered queue of jobs.
 *
 * Shot 1 starts from whatever it was given: a file, an anchor, or nothing at
 * all. Every shot after it starts from the shot before, unless it names its own
 * frame or the reanchor schedule pulls it back to the anchor.
 *
 * The plan is pure. It queues nothing, uploads nothing, and does not need the
 * previous shot's filename to exist yet: that arrives at instantiateShot.
 */
export function shotPlan(input: ShotPlanInput): ShotPlan {
  const { base, params, shots } = input
  const prefix = input.prefix ?? REEL_PREFIX
  const cont = input.continuation === undefined ? deriveContinuation(base) : input.continuation
  const bookend = input.bookend === undefined ? deriveBookend(base) : input.bookend
  const vace = input.vace === undefined ? deriveVaceShot(base) : input.vace
  const tap = deriveChainTap(base) ?? base

  const warnings: string[] = []
  if (!cont && shots.length > 1) {
    warnings.push(
      `${base.label} cannot open a shot on a frame, so these clips will not continue from each other.`,
    )
  }

  const reanchor = Math.max(0, Math.floor(input.reanchorEvery ?? 0))
  const jobs: ShotJob[] = []
  let hops = 0
  let frames = 0

  shots.forEach((spec, index) => {
    const notes: string[] = []
    const anchorDue = reanchor > 0 && index > 0 && index % reanchor === 0

    let start: ShotStart = { from: 'none' }
    if (spec.startImage) start = { from: 'given', image: spec.startImage }
    else if (input.anchorImage && (index === 0 || anchorDue)) start = { from: 'anchor', image: input.anchorImage }
    else if (index > 0 && cont) start = { from: 'previous' }

    // A family with no opening frame slot cannot use any of that.
    if (!cont && start.from !== 'none') {
      notes.push(`${base.label} has no slot for an opening frame, so this shot starts fresh.`)
      start = { from: 'none' }
    }

    if (index > 0 && start.from === 'none') {
      notes.push('Opens cold. Nothing carries over from the shot before it.')
      hops = 0
    } else if (start.from === 'previous') {
      hops += 1
    } else {
      hops = 0
    }

    // Variant selection, in order of what the shot actually asked for.
    let def = start.from === 'none' ? tap : (cont ?? tap)
    let endImage = spec.endImage
    let referenceImage = spec.referenceImage

    if (endImage) {
      if (bookend) def = bookend
      else {
        endImage = undefined
        notes.push(`${base.label} cannot pin a closing frame, so the end image was dropped.`)
      }
    }
    if (referenceImage) {
      if (vace) def = vace
      else {
        referenceImage = undefined
        notes.push('No VACE family is registered, so the reference image was dropped.')
      }
    }

    const wanted = spec.length ?? params.length ?? def.defaults.length
    const length = snapLength(wanted > 0 ? wanted : 81)
    frames += length

    if (hops === 3) notes.push('Third generation off the last clean frame. Colour has started to move.')
    if (hops >= 5) notes.push('Five hops deep. Expect a visibly different look from where the reel started.')

    const startImage =
      start.from === 'given' || start.from === 'anchor' ? start.image : undefined

    jobs.push({
      index,
      key: `shot-${String(index + 1).padStart(3, '0')}`,
      label: spec.label ?? `Shot ${index + 1}`,
      def,
      params: {
        ...params,
        positive: spec.prompt,
        negative: spec.negative ?? params.negative,
        seed: spec.seed ?? params.seed + index,
        length,
        image: startImage,
      },
      start,
      endImage,
      referenceImage,
      outputPrefix: `${prefix}/${String(index + 1).padStart(3, '0')}`,
      hops,
      notes,
    })
  })

  const fps = params.fps || base.defaults.fps || 24
  const deepest = jobs.reduce((m, j) => Math.max(m, j.hops), 0)
  if (deepest >= 4 && !input.anchorImage) {
    warnings.push(
      `Nothing resets this chain: the last shots sit ${deepest} generations from a clean frame. Set an anchor image, or plan fewer and longer shots.`,
    )
  }
  if (base.dualModel) {
    warnings.push('Each shot loads both halves of the noise pair. Run the queue one job at a time.')
  }

  return { jobs, frames, seconds: frames / fps, warnings }
}

/**
 * Build the workflow for one job.
 *
 * `previous` is the handoff frame from the shot before, as returned by
 * chainFrameOf. A job whose start is 'previous' cannot be built without it:
 * this throws rather than quietly producing an unrelated shot that costs
 * minutes of GPU time to discover.
 */
export function instantiateShot(job: ShotJob, previous?: FileRef | string | null): ApiWorkflow {
  const params: Params = { ...job.params }

  if (job.start.from === 'previous') {
    const ref = typeof previous === 'string' ? previous : previous ? annotatedRef(previous) : null
    if (!ref) {
      throw new Error(`${job.label} continues from the shot before it, but no handoff frame was supplied.`)
    }
    params.image = ref
  }

  const wf = instantiate(job.def, params)
  if (job.endImage) setEndImage(wf, job.endImage)
  if (job.referenceImage) setReferenceImage(wf, job.referenceImage)
  setOutputPrefix(wf, job.outputPrefix)
  return wf
}

/**
 * Sanity check before a reel is queued.
 *
 * Chained frames are resized with a centre crop by every Wan image node, so a
 * shot that changes aspect ratio mid reel silently loses the edges of the frame
 * it was handed. Sizes stay constant across a plan by construction; this guards
 * a caller that edits params between jobs.
 */
export function checkReel(jobs: readonly ShotJob[]): string[] {
  const issues: string[] = []
  const first = jobs[0]
  if (!first) return issues
  const w = first.params.width
  const h = first.params.height
  for (const job of jobs) {
    if (job.params.width !== w || job.params.height !== h) {
      issues.push(
        `${job.label} is ${job.params.width}x${job.params.height} while the reel is ${w}x${h}. The handoff frame will be centre cropped.`,
      )
    }
    if (job.start.from === 'previous' && job.index === 0) {
      issues.push('The opening shot cannot continue from anything.')
    }
  }
  return issues
}
