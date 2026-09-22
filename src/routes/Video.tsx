/**
 * The Video desk.
 *
 * A room of its own, not a mode of the picture desk: its own draft, its own
 * jobs, its own vocabulary. Text-to-video and image-to-video both live here,
 * chosen with a tab rather than a dropdown, because they are two intents and
 * not two settings.
 *
 * Three facts shape everything below.
 *
 * 1. A clip takes minutes. The desk therefore never blocks: the job engine is
 *    a module singleton, so leaving the desk — or the route being unmounted by
 *    the shell — does not drop a running job, and the elapsed clock, the live
 *    step count and the latent preview keep running while you write the next
 *    prompt.
 * 2. `Wan22ImageToVideoLatent.start_image` is an IMAGE *link*, not a filename.
 *    Verified against the live /object_info: `optional.start_image = ["IMAGE", {}]`.
 *    Image-to-video therefore inserts a LoadImage node and wires it in; writing
 *    a string into that input passes validation and then fails at the server.
 * 3. Frames are an implementation detail. Simple mode speaks in seconds and
 *    shows the frame count as a quiet caption; expert mode speaks in frames,
 *    snapped to the step the live schema declares (length step 4, min 1 — so
 *    49, 81, 121, 161 — and width/height step 32).
 *
 * Cancellation is `cancelJob(promptId)` and nothing else, so stopping a clip
 * can never stop somebody else's picture.
 */

import { CataloguePanel } from '../components/advanced/CataloguePanel'
import { faultBody, faultOf, faultTitle, faultWhere, type Fault } from '../lib/faults'
import { availabilityOf, inventoryFrom } from '../lib/availability'
import { measureImage as measure } from '../lib/images'
import { clamp } from '../lib/num'
import { ServerDown } from '../components/ServerDown'
import { Notice, RING } from '../components/type'
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
  type DragEvent as ReactDragEvent,
  type KeyboardEvent as ReactKeyboardEvent,
  type ReactNode,
} from 'react'

import {
  cancelJob,
  connect,
  fileUrl,
  getJob,
  listJobs,
  objectInfo,
  run,
  systemStats,
  uploadImage,
  watchConnection,
  type ApiWorkflow,
  type ConnectionState,
  type FileRef,
  type OutputFile,
  type ProgressEvent as ComfyProgress,
} from '../lib/comfy'

import {
  FAMILIES,
  defaultsFor,
  instantiate,
  type FamilyDef,
} from '../lib/workflows'

import {
  gb,
  modelFiles,
  probeHardware,
  type Hardware,
  type ModelFile,
  type Verdict,
} from '../lib/hardware'

import { history, type HistoryEntry } from '../lib/history'

import {
  applyDefaults,
  clearTouched,
  deskStore,
  randomSeed,
  recordOf,
  reuseIntoDesk,
  settings,
  toParams,
  type AppliedReuse,
  type Composition,
  type FamilyDefaults,
  type SourceRef,
} from '../lib/session'

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export type PlayerSlot = {
  /** Browser-loadable URL for the clip. */
  src: string
  file: FileRef
  /** Frame rate from the record, never probed. */
  fps: number
  /** Frame count from the record. */
  frames: number
  /** The archive record, when the clip has one. */
  entry: HistoryEntry | null
}

export type VideoProps = {
  /**
   * The shell injects the full player here. Without it the desk shows its own
   * plain clip frame, which is enough to watch and to save, and nothing more.
   */
  renderPlayer?: (slot: PlayerSlot) => ReactNode
  /** Navigate elsewhere in the shell, e.g. `#/archive?q=is:video`. */
  onNavigate?: (hash: string) => void
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------


const isTyping = (e: KeyboardEvent): boolean => {
  const t = e.target as HTMLElement | null
  return !!t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))
}

/** `4 min 10 s`, `42 s`, `0.9 s`. Never a bare decimal minute. */
function duration(ms: number): string {
  const s = Math.max(0, ms) / 1000
  if (s < 10) return `${s.toFixed(1)} s`
  if (s < 60) return `${Math.round(s)} s`
  const m = Math.floor(s / 60)
  const rest = Math.round(s - m * 60)
  if (rest === 0) return `${m} min`
  return `${m} min ${rest} s`
}

/** `02:14`, for the elapsed clock that ticks beside a running job. */
function stopwatch(ms: number): string {
  const total = Math.floor(Math.max(0, ms) / 1000)
  const m = Math.floor(total / 60)
  const s = total % 60
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

/** `next`, `second`, `third` — queue positions read as English, not as maths. */
function ordinal(n: number): string {
  const words = ['', 'next', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth']
  return words[n] ?? `${n}th`
}

const seconds = (frames: number, fps: number) => (fps > 0 ? frames / fps : 0)

/** `5.0 seconds`. One decimal is the honest resolution of a frame count. */
const clipLength = (frames: number, fps: number) => `${seconds(frames, fps).toFixed(1)} seconds`

const times = (w: number, h: number) => `${w} × ${h}`

const dateline = (at: number) =>
  new Date(at).toLocaleString('en-GB', {
    day: 'numeric',
    month: 'long',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })

const prettyModel = (file: string) =>
  file
    .replace(/\.(safetensors|gguf|ckpt|sft)$/i, '')
    .replace(/[_.]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()

/**
 * ComfyUI's /view sets no Content-Disposition, so a plain link opens the clip
 * instead of saving it. Fetch it and hand the browser a blob it must download.
 */
async function saveAs(url: string, name: string): Promise<void> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`The file could not be read (HTTP ${res.status}).`)
  const blob = await res.blob()
  const href = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = href
  a.download = name
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(href), 10_000)
}

// ---------------------------------------------------------------------------
// Stage names — so a 60-second model load stops looking like a crash
// ---------------------------------------------------------------------------

const STAGES: Record<string, string> = {
  UnetLoaderGGUF: 'Loading the model',
  UNETLoader: 'Loading the model',
  CheckpointLoaderSimple: 'Loading the model',
  CLIPLoader: 'Reading the prompt',
  CLIPTextEncode: 'Reading the prompt',
  LoraLoaderModelOnly: 'Loading add-ons',
  LoadImage: 'Preparing your picture',
  VAEEncode: 'Preparing your picture',
  ImageScaleToTotalPixels: 'Preparing your picture',
  Wan22ImageToVideoLatent: 'Preparing the frames',
  WanImageToVideo: 'Preparing the frames',
  EmptyHunyuanLatentVideo: 'Preparing the frames',
  ModelSamplingSD3: 'Setting the schedule',
  ModelSamplingAuraFlow: 'Setting the schedule',
  KSampler: 'Drawing',
  KSamplerAdvanced: 'Drawing',
  SamplerCustomAdvanced: 'Drawing',
  VAEDecode: 'Developing the frames',
  VAEDecodeTiled: 'Developing the frames',
  SaveWEBM: 'Encoding the clip',
  SaveVideo: 'Encoding the clip',
  SaveImage: 'Writing the file',
}

function stageOf(graph: ApiWorkflow, nodeId: string | null): string {
  if (!nodeId) return 'Working'
  const cls = graph[nodeId]?.class_type
  return (cls && STAGES[cls]) || 'Working'
}

// ---------------------------------------------------------------------------
// Graph shaping
// ---------------------------------------------------------------------------

const I2V_LOAD = '__i2v_load'

/** Latent nodes that accept a first frame. Both take an IMAGE link. */
const START_FRAME_NODES = ['Wan22ImageToVideoLatent', 'WanImageToVideo']

/**
 * Give a text-to-video family a start frame.
 *
 * `start_image` is an optional IMAGE input — a `[nodeId, slot]` link — so the
 * only way to supply one is to insert a LoadImage node and wire it in. A family
 * that already binds `image` (the 14B I2V pair) is returned unchanged.
 *
 * Returns null when the family's latent node cannot take a first frame, which
 * callers must treat as "not offered" rather than silently running text-to-video
 * and throwing the reader's picture away.
 */
export function deriveImageToVideo(def: FamilyDef): FamilyDef | null {
  if (def.mode !== 'video') return null
  if (def.bindings.image) return def

  const graph: FamilyDef['graph'] = JSON.parse(JSON.stringify(def.graph))
  const target = Object.entries(graph).find(([, n]) => START_FRAME_NODES.includes(n.class_type))
  if (!target) return null

  graph[I2V_LOAD] = { class_type: 'LoadImage', inputs: { image: 'example.png' } }
  graph[target[0]].inputs.start_image = [I2V_LOAD, 0]

  return {
    ...def,
    id: `${def.id}__i2v`,
    label: `${def.label}, from a picture`,
    graph,
    bindings: { ...def.bindings, image: [[I2V_LOAD, 'image']] },
  }
}

/**
 * ModelSampling* shift, written by class_type rather than by a binding, because
 * the registry carries no binding for it and must not be hand-edited.
 */
function applyShift(wf: ApiWorkflow, shift: number | null | undefined): void {
  if (shift === null || shift === undefined || !Number.isFinite(shift)) return
  for (const node of Object.values(wf)) {
    if (node.class_type.startsWith('ModelSampling') && typeof node.inputs.shift === 'number') {
      node.inputs.shift = shift
    }
  }
}

/** The shift the family's own graph ships with, when it has one. */
function graphShift(def: FamilyDef): number | null {
  for (const node of Object.values(def.graph)) {
    if (node.class_type.startsWith('ModelSampling') && typeof node.inputs.shift === 'number') {
      return node.inputs.shift
    }
  }
  return null
}

function latentClassOf(def: FamilyDef): string | null {
  for (const node of Object.values(def.graph)) {
    if (START_FRAME_NODES.includes(node.class_type)) return node.class_type
    if (/^Empty.*Latent/.test(node.class_type)) return node.class_type
  }
  return null
}

// ---------------------------------------------------------------------------
// What this machine can actually run
// ---------------------------------------------------------------------------

/** One numeric input as the live schema declares it. */
type NumSpec = { min: number; max: number; step: number }

const SPEC_FALLBACK: Record<'size' | 'frames', NumSpec> = {
  size: { min: 32, max: 16384, step: 16 },
  frames: { min: 1, max: 16384, step: 4 },
}

export type VideoFamily = {
  def: FamilyDef
  /** The weight file this entry selects. A dual-model pair carries its own. */
  model: string
  label: string
  verdict: Verdict | null
  /** True when a start frame can be wired into this family's latent node. */
  canStartFromPicture: boolean
  width: NumSpec
  height: NumSpec
  frames: NumSpec
}

type Catalogue = {
  families: VideoFamily[]
  /** Families with a recipe but a missing file, so the absence is explicable. */
  blocked: { label: string; why: string }[]
  samplers: string[]
  schedulers: string[]
  hardware: Hardware | null
  vramFree: number | null
}

function readSpec(info: Record<string, unknown>, cls: string | null, field: string, fallback: NumSpec): NumSpec {
  if (!cls) return fallback
  const node = info[cls] as { input?: { required?: Record<string, unknown> } } | undefined
  const raw = node?.input?.required?.[field]
  if (!Array.isArray(raw) || raw.length < 2) return fallback
  const opts = raw[1] as { min?: number; max?: number; step?: number }
  return {
    min: typeof opts?.min === 'number' ? opts.min : fallback.min,
    max: typeof opts?.max === 'number' ? opts.max : fallback.max,
    step: typeof opts?.step === 'number' && opts.step > 0 ? opts.step : fallback.step,
  }
}

async function loadCatalogue(): Promise<Catalogue> {
  const [info, hardware, sizes, stats] = await Promise.all([
    objectInfo(),
    probeHardware().catch(() => null),
    modelFiles().catch(() => new Map<string, ModelFile>()),
    systemStats().catch(() => null),
  ])

  const inv = inventoryFrom(info)
  const weights = inv.weights

  const families: VideoFamily[] = []
  const blocked: Catalogue['blocked'] = []

  for (const def of FAMILIES) {
    if (def.mode !== 'video') continue

    const avail = availabilityOf(def, inv, hardware, sizes)
    if (!avail.ok) {
      blocked.push({ label: def.label, why: avail.why })
      continue
    }
    const verdict = avail.verdict

    const model = def.dualModel ? '' : (def.models.find((m) => weights.has(m)) ?? def.models[0] ?? '')

    const latent = latentClassOf(def)
    families.push({
      def,
      model,
      label: def.label,
      verdict,
      canStartFromPicture: deriveImageToVideo(def) !== null,
      width: readSpec(info, latent, 'width', SPEC_FALLBACK.size),
      height: readSpec(info, latent, 'height', SPEC_FALLBACK.size),
      frames: readSpec(info, latent, 'length', SPEC_FALLBACK.frames),
    })
  }

  const devices = (stats as { devices?: { vram_free?: number }[] } | null)?.devices
  const vramFree = Array.isArray(devices) && typeof devices[0]?.vram_free === 'number'
    ? devices[0].vram_free
    : (hardware?.gpu?.vramFree ?? null)

  return {
    families,
    blocked,
    samplers: inv.samplers,
    schedulers: inv.schedulers,
    hardware,
    vramFree,
  }
}

let cataloguePromise: Promise<Catalogue> | null = null

/**
 * Read once per page load, not once per visit to the desk. A failed read is
 * forgotten, so the retry is a real retry and not the same rejected promise.
 */
/** Forget the cached catalogue, so the next read sees files that just landed. */
export function resetCatalogue(): void {
  cataloguePromise = null
}

function catalogue(): Promise<Catalogue> {
  if (!cataloguePromise) {
    cataloguePromise = loadCatalogue().catch((err: unknown) => {
      cataloguePromise = null
      throw err
    })
  }
  return cataloguePromise
}

/** Snap a value onto the grid the live schema declares. */
function snap(value: number, spec: NumSpec): number {
  const steps = Math.round((value - spec.min) / spec.step)
  return clamp(spec.min + steps * spec.step, spec.min, spec.max)
}

function defaultsOf(family: VideoFamily): FamilyDefaults {
  const d = defaultsFor(family.def, family.model)
  return {
    familyId: family.def.id,
    model: family.model,
    steps: d.steps,
    cfg: d.cfg,
    width: d.width,
    height: d.height,
    sampler: d.sampler,
    scheduler: d.scheduler,
    length: d.length || undefined,
    fps: d.fps || undefined,
    negative: d.negative,
  }
}

function modelLabelOf(family: VideoFamily): string {
  const over = family.def.perModel?.[family.model] as { label?: unknown } | undefined
  if (typeof over?.label === 'string') return over.label
  return family.model ? prettyModel(family.model) : family.def.label
}

/** The registry's own guidance, rendered in the margin where it applies. */
function marginaliaOf(family: VideoFamily): string[] {
  const out: string[] = []
  const over = family.def.perModel?.[family.model] as Record<string, unknown> | undefined
  const presets = over?.presets as Record<string, { note?: unknown }> | undefined
  if (presets) {
    for (const preset of Object.values(presets)) {
      if (typeof preset?.note === 'string') out.push(preset.note)
    }
  }
  if (typeof over?.note === 'string') out.push(over.note)
  return out
}

// ---------------------------------------------------------------------------
// Shapes — every option comes from the family's own data
// ---------------------------------------------------------------------------

export type Shape = { width: number; height: number; label: string; note?: string }

function collectPairs(value: unknown, into: Shape[], note?: string): void {
  if (!value || typeof value !== 'object') return
  const rec = value as Record<string, unknown>
  if (typeof rec.width === 'number' && typeof rec.height === 'number') {
    into.push({
      width: rec.width,
      height: rec.height,
      label: '',
      note: typeof rec.note === 'string' ? rec.note : note,
    })
    return
  }
  for (const child of Object.values(rec)) collectPairs(child, into, note)
}

function orientation(w: number, h: number): string {
  if (w > h * 1.05) return 'Wide'
  if (h > w * 1.05) return 'Tall'
  return 'Square'
}

/**
 * Shapes offered for a family: its own recipe, every width/height pair its
 * per-model notes document, and the transpose of each, snapped to the live
 * step. Wan families also offer 832 × 480, which their own registry note names
 * as the trained resolution and which is markedly quicker than 720p.
 */
function shapesFor(family: VideoFamily): Shape[] {
  const d = defaultsFor(family.def, family.model)
  const raw: Shape[] = [{ width: d.width, height: d.height, label: '' }]
  collectPairs(family.def.perModel, raw)
  if (/^wan/.test(family.def.id)) raw.push({ width: 832, height: 480, label: '' })

  const out: Shape[] = []
  const seen = new Set<string>()
  for (const candidate of raw) {
    for (const pair of [candidate, { ...candidate, width: candidate.height, height: candidate.width }]) {
      const width = snap(pair.width, family.width)
      const height = snap(pair.height, family.height)
      const key = `${width}x${height}`
      if (seen.has(key)) continue
      seen.add(key)
      out.push({ width, height, label: `${orientation(width, height)} ${times(width, height)}`, note: pair.note })
    }
  }
  // Largest last: the picker reads as a ladder of cost.
  return out.sort((a, b) => a.width * a.height - b.width * b.height).slice(0, 6)
}

/** Length chips, in seconds, all landing on a frame count the node accepts. */
function lengthsFor(family: VideoFamily, fps: number): number[] {
  const d = defaultsFor(family.def, family.model)
  const wanted = [2, 3, 5].map((s) => snap(Math.round(s * fps), family.frames))
  if (d.length) wanted.push(snap(d.length, family.frames))
  return [...new Set(wanted)].filter((f) => f >= family.frames.min).sort((a, b) => a - b)
}

// ---------------------------------------------------------------------------
// Honest estimates
// ---------------------------------------------------------------------------

type Estimate = { ms: number; runs: number }

/**
 * The median of this machine's own finished runs of the same shape. Three runs
 * is the floor; below that we print nothing rather than invent a number.
 */
function measuredEstimate(
  records: readonly HistoryEntry[],
  c: Pick<Composition, 'familyId' | 'length' | 'width' | 'height' | 'mode'>,
): Estimate | null {
  const like = records
    .filter(
      (e) =>
        e.kind === 'video' &&
        e.familyId === c.familyId &&
        e.mode === c.mode &&
        e.length === c.length &&
        e.width === c.width &&
        e.height === c.height &&
        e.durationMs > 0,
    )
    .slice(0, 8)
  if (like.length < 3) return null
  const sorted = like.map((e) => e.durationMs).sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  const ms = sorted.length % 2 ? sorted[mid] : Math.round((sorted[mid - 1] + sorted[mid]) / 2)
  return { ms, runs: like.length }
}

/** `5-7` in the registry means minutes on this exact card. */
function registryMinutes(family: VideoFamily): string | null {
  const per = family.def.perModel ?? {}
  for (const value of Object.values(per)) {
    const rec = value as Record<string, unknown>
    if (rec && typeof rec._default === 'boolean' && rec._default && typeof rec.est_minutes_5060ti === 'string') {
      return rec.est_minutes_5060ti
    }
  }
  return null
}

// ---------------------------------------------------------------------------
// The job engine — a module singleton, so a clip survives leaving the desk
// ---------------------------------------------------------------------------

export type VideoJobStatus = 'submitting' | 'queued' | 'running' | 'done' | 'error' | 'cancelled'

export type VideoJob = {
  id: string
  promptId: string | null
  status: VideoJobStatus
  familyLabel: string
  modelLabel: string
  prompt: string
  /** Sampler steps. 0/1 until the sampler reports. */
  value: number
  max: number
  stage: string
  previewUrl: string | null
  startedAt: number
  samplingAt: number | null
  finishedAt: number | null
  error: string | null
  /** The classified failure, with ComfyUI's node and per-input detail when it gave them. */
  fault: Fault | null
  files: OutputFile[]
  entryId: string | null
  composition: Composition
  graph: ApiWorkflow
  frames: number
  fps: number
  /** Position in ComfyUI's own queue; 0 means it is the one running. */
  queuePos: number | null
  cancelRequested: boolean
  sighted: boolean
  misses: number
}

let jobs: VideoJob[] = []
const jobListeners = new Set<() => void>()
let pollTimer: ReturnType<typeof setInterval> | null = null

const unfinished = (j: VideoJob) => j.status === 'submitting' || j.status === 'queued' || j.status === 'running'

function announce(): void {
  for (const fn of [...jobListeners]) {
    try {
      fn()
    } catch {
      /* one broken subscriber must not stop the others */
    }
  }
  managePoll()
}

function patchJob(id: string, patch: Partial<VideoJob>): void {
  let changed = false
  jobs = jobs.map((j) => {
    if (j.id !== id) return j
    changed = true
    return { ...j, ...patch }
  })
  if (changed) announce()
}

function jobById(id: string): VideoJob | undefined {
  return jobs.find((j) => j.id === id)
}

export const videoJobs = {
  subscribe(fn: () => void): () => void {
    jobListeners.add(fn)
    return () => {
      jobListeners.delete(fn)
    }
  },
  snapshot: (): VideoJob[] => jobs,
  running: (): VideoJob | null => jobs.find((j) => j.status === 'running') ?? null,
  pending: (): VideoJob[] => jobs.filter(unfinished),
}

function dismissJob(id: string): void {
  jobs = jobs.filter((j) => j.id !== id)
  announce()
}

async function stopJob(id: string): Promise<void> {
  const job = jobById(id)
  if (!job || !unfinished(job)) return
  if (!job.promptId) {
    // Still in flight to the queue. Mark it, and the queued handler stops it
    // the moment ComfyUI hands us an id.
    patchJob(id, { cancelRequested: true, stage: 'Stopping' })
    return
  }
  patchJob(id, { cancelRequested: true, stage: 'Stopping' })
  try {
    await cancelJob(job.promptId)
    // `false` means it had already finished; the terminal event settles it.
  } catch (err) {
    patchJob(id, { error: (err as Error).message })
  }
}

type StartOptions = {
  composition: Composition
  graph: ApiWorkflow
  familyLabel: string
  modelLabel: string
}

function startJob(opts: StartOptions): string {
  const id = globalThis.crypto?.randomUUID?.() ?? `job_${Date.now()}_${Math.random().toString(36).slice(2)}`
  const job: VideoJob = {
    id,
    promptId: null,
    status: 'submitting',
    familyLabel: opts.familyLabel,
    modelLabel: opts.modelLabel,
    prompt: opts.composition.prompt,
    value: 0,
    max: 0,
    stage: 'Sending it to the press',
    previewUrl: null,
    startedAt: Date.now(),
    samplingAt: null,
    finishedAt: null,
    error: null,
    fault: null,
    files: [],
    entryId: null,
    composition: opts.composition,
    graph: opts.graph,
    frames: opts.composition.length ?? 0,
    fps: opts.composition.fps ?? 0,
    queuePos: null,
    cancelRequested: false,
    sighted: false,
    misses: 0,
  }
  jobs = [job, ...jobs]
  announce()

  const onEvent = (e: ComfyProgress) => {
    const current = jobById(id)
    if (!current) return
    if (e.phase === 'queued') {
      patchJob(id, { promptId: e.promptId, status: 'queued', stage: 'Queued' })
      if (current.cancelRequested) void cancelJob(e.promptId).catch(() => undefined)
    } else if (e.phase === 'running') {
      const sampling = e.max > 1
      patchJob(id, {
        status: 'running',
        value: e.value,
        max: e.max,
        stage: stageOf(current.graph, e.node),
        samplingAt: sampling && current.samplingAt === null ? Date.now() : current.samplingAt,
        queuePos: 0,
      })
    } else if (e.phase === 'preview') {
      patchJob(id, { previewUrl: e.url })
    }
  }

  void run(opts.graph, onEvent)
    .then((files) => {
      const current = jobById(id)
      if (!current) return
      const finishedAt = Date.now()
      const file = files.find((f) => f.kind === 'video') ?? files[0] ?? null
      let entryId: string | null = null
      if (file) {
        try {
          const entry = history.add(
            recordOf(current.composition, {
              file,
              files: files.length > 1 ? files : undefined,
              kind: file.kind,
              promptId: current.promptId ?? '',
              durationMs: finishedAt - current.startedAt,
              seed: current.composition.seed,
              familyLabel: current.familyLabel,
              modelLabel: current.modelLabel,
              at: finishedAt,
            }),
          )
          entryId = entry.id
        } catch {
          // A full archive must never cost the reader the clip itself.
        }
      }
      patchJob(id, {
        status: 'done',
        files,
        entryId,
        finishedAt,
        stage: 'Done',
        previewUrl: null,
        queuePos: null,
      })
    })
    .catch((err: unknown) => {
      // The same classification the Pictures desk uses, so a clip that failed
      // on a bad frame names the node instead of saying something went wrong.
      const f = faultOf(err)
      patchJob(id, {
        status: f.cancelled ? 'cancelled' : 'error',
        error: f.message || 'Something went wrong.',
        fault: f,
        finishedAt: Date.now(),
        previewUrl: null,
        stage: f.cancelled ? 'Stopped' : 'Failed',
        queuePos: null,
      })
    })

  return id
}

/**
 * Reconcile against ComfyUI's own queue every five seconds, so a job that the
 * server has forgotten is reported rather than spinning forever, and so the
 * desk can say honestly how many clips are ahead of this one.
 */
function managePoll(): void {
  const live = jobs.some(unfinished)
  if (live && !pollTimer) pollTimer = setInterval(() => void reconcile(), 5000)
  if (!live && pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

async function reconcile(): Promise<void> {
  const watching = jobs.filter(unfinished)
  if (!watching.length) return

  let ids: string[]
  try {
    const page = await listJobs({ status: ['pending', 'in_progress'], limit: 100 })
    ids = page.jobs.map((j) => j.id)
  } catch {
    return // the connection notice covers an unreachable server
  }

  for (const job of watching) {
    if (!job.promptId) continue
    const at = ids.indexOf(job.promptId)
    if (at >= 0) {
      patchJob(job.id, { queuePos: at, sighted: true, misses: 0 })
      continue
    }
    if (!job.sighted && Date.now() - job.startedAt < 20_000) continue

    const misses = job.misses + 1
    if (misses < 2) {
      patchJob(job.id, { misses })
      continue
    }
    const server = await getJob(job.promptId).catch(() => null)
    if (server && (server.status === 'completed' || server.status === 'failed' || server.status === 'cancelled')) {
      // The terminal socket event is the authority; give it one more cycle.
      patchJob(job.id, { misses: 0, sighted: true })
      continue
    }
    if (jobById(job.id)?.status === 'done') continue
    patchJob(job.id, {
      status: 'error',
      error:
        'We lost track of this job. ComfyUI no longer lists it. Check the archive. It may have finished anyway.',
      finishedAt: Date.now(),
      stage: 'Lost',
      queuePos: null,
    })
  }
}

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------

/** A clock that ticks only while something is running. */
function useNow(active: boolean): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!active) return
    setNow(Date.now())
    const t = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(t)
  }, [active])
  return now
}

/** Hold to confirm. A mis-click must not destroy four minutes of GPU time. */
function useHold(onConfirm: () => void, ms = 600) {
  const [progress, setProgress] = useState(0)
  const frame = useRef<number | null>(null)
  const start = useRef(0)

  const stop = useCallback(() => {
    if (frame.current !== null) cancelAnimationFrame(frame.current)
    frame.current = null
    setProgress(0)
  }, [])

  const begin = useCallback(() => {
    if (frame.current !== null) return
    start.current = performance.now()
    const tick = () => {
      const p = clamp((performance.now() - start.current) / ms, 0, 1)
      setProgress(p)
      if (p >= 1) {
        stop()
        onConfirm()
        return
      }
      frame.current = requestAnimationFrame(tick)
    }
    frame.current = requestAnimationFrame(tick)
  }, [ms, onConfirm, stop])

  useEffect(() => stop, [stop])

  return {
    progress,
    handlers: {
      onPointerDown: begin,
      onPointerUp: stop,
      onPointerLeave: stop,
      onPointerCancel: stop,
      onKeyDown: (e: ReactKeyboardEvent) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          begin()
        }
      },
      onKeyUp: stop,
      onBlur: stop,
    },
  }
}

// ---------------------------------------------------------------------------
// Presentational pieces
// ---------------------------------------------------------------------------

function Head({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="mb-2 flex items-baseline justify-between border-b border-grey-300 pb-1">
      <h3 className="text-[0.625rem] font-semibold uppercase tracking-[0.18em] text-grey-700">{title}</h3>
      {children}
    </div>
  )
}

/**
 * Stopping a clip takes a deliberate 600 ms hold with a burgundy wipe. A
 * mis-click must never destroy four minutes of GPU time, and the wipe shows
 * the reader exactly how much of the hold is left.
 */
function HoldToStop({ jobId, label = 'Hold to stop' }: { jobId: string; label?: string }) {
  const stop = useCallback(() => {
    void stopJob(jobId)
  }, [jobId])
  const hold = useHold(stop, 600)
  return (
    <button
      type="button"
      {...hold.handlers}
      aria-label="Hold to stop this clip"
      className={`sg-hold relative block w-full overflow-hidden border border-burgundy-900 px-4 py-2 text-[0.625rem] font-semibold uppercase tracking-[0.16em] text-burgundy-900 ${RING}`}
    >
      <span
        aria-hidden
        className="absolute inset-y-0 left-0 bg-burgundy-900"
        style={{ width: `${Math.round(hold.progress * 100)}%` }}
      />
      <span className="relative" style={{ color: hold.progress > 0.5 ? 'var(--color-newsprint)' : undefined }}>
        {label}
      </span>
    </button>
  )
}

/**
 * A number the reader can actually type into.
 *
 * Snapping on every keystroke makes a field with a step of 32 impossible to
 * use — you type `8` and it becomes `32` before you can reach the `3`. So the
 * text is held as typed and committed on blur or Enter, where it is snapped
 * and clamped and shown back.
 */
function NumberField({
  value,
  min,
  max,
  step,
  commit,
  label,
  italic,
}: {
  value: number
  min: number
  max: number
  step: number
  commit: (n: number) => void
  label: string
  italic?: boolean
}) {
  const [text, setText] = useState(() => String(value))
  useEffect(() => setText(String(value)), [value])

  const settle = () => {
    const n = Number(text)
    if (!Number.isFinite(n)) {
      setText(String(value))
      return
    }
    commit(n)
    setText(String(value))
  }

  return (
    <input
      type="number"
      inputMode="decimal"
      aria-label={label}
      className={`field tabular-nums ${italic ? 'italic text-grey-500' : ''}`}
      value={text}
      min={min}
      max={max}
      step={step}
      onChange={(e) => setText(e.target.value)}
      onBlur={settle}
      onKeyDown={(e) => {
        if (e.key === 'Enter') {
          e.preventDefault()
          settle()
        }
      }}
    />
  )
}

function Marginalia({ text }: { text: string }) {
  return (
    <p className="mb-3 border-l-[3px] border-burgundy-900 pl-3 text-small italic leading-snug text-grey-700">
      {text}
    </p>
  )
}

function ExpertField({
  label,
  hint,
  id,
  children,
}: {
  label: string
  hint?: string
  id?: string
  children: ReactNode
}) {
  return (
    <div className="mb-3" id={id}>
      <div className="flex items-baseline justify-between gap-2">
        <label className="text-[0.625rem] font-semibold uppercase tracking-[0.18em] text-grey-700">{label}</label>
        {hint ? <span className="text-caption italic text-grey-500 tabular-nums">{hint}</span> : null}
      </div>
      <div className="mt-1">{children}</div>
    </div>
  )
}

/** Boxes sharing hairlines. Active is black-on-newsprint, never burgundy. */
function Chips<T extends string | number>({
  options,
  value,
  onChange,
  ariaLabel,
}: {
  options: { value: T; label: string; caption?: string; disabled?: boolean; title?: string }[]
  value: T
  onChange: (v: T) => void
  ariaLabel: string
}) {
  return (
    <div className="flex flex-wrap border border-grey-300" role="group" aria-label={ariaLabel}>
      {options.map((o, i) => {
        const active = o.value === value
        return (
          <button
            key={String(o.value)}
            type="button"
            title={o.title}
            disabled={o.disabled}
            aria-pressed={active}
            onClick={() => onChange(o.value)}
            className={[
              'flex-1 px-2 py-1.5 text-left text-small leading-tight transition-colors',
              RING,
              i > 0 ? 'border-l border-grey-300' : '',
              active ? 'bg-ink text-newsprint' : 'bg-transparent text-ink hover:bg-newsprint-aged',
              o.disabled ? 'cursor-not-allowed text-grey-400' : 'cursor-pointer',
            ].join(' ')}
          >
            <span className="block whitespace-nowrap">{o.label}</span>
            {o.caption ? (
              <span className={`block text-caption tabular-nums ${active ? 'text-grey-300' : 'text-grey-500'}`}>
                {o.caption}
              </span>
            ) : null}
          </button>
        )
      })}
    </div>
  )
}

// ---------------------------------------------------------------------------
// The desk
// ---------------------------------------------------------------------------

const store = deskStore('video')

/** Why the press cannot run yet, in the reader's words. Null means it can. */
function reasonFor(family: VideoFamily | null, c: Composition): string | null {
  if (!family) return 'No video model is installed.'
  if (!c.prompt.trim()) return 'Describe the shot first.'
  if (c.mode === 'i2v' && !family.canStartFromPicture) return `${family.label} works from words only.`
  if (c.mode === 'i2v' && !c.source?.name) return 'Add a start frame, or work from words.'
  return null
}

const EXAMPLES: { prompt: string; note: string }[] = [
  {
    prompt: 'A tram crosses a wet junction at dusk, headlights smearing in the rain',
    note: 'wide · five seconds',
  },
  {
    prompt: 'Steam lifts off a harbour at first light, gulls crossing the frame',
    note: 'wide · three seconds',
  },
  {
    prompt: 'A curtain moves in a draught beside an open window, afternoon light',
    note: 'tall · five seconds',
  },
]

export default function Video({ renderPlayer, onNavigate }: VideoProps = {}) {
  const composition = useSyncExternalStore(store.subscribe, store.get)
  const prefs = useSyncExternalStore(settings.subscribe, settings.get)
  const records = useSyncExternalStore(history.subscribe, history.all)
  const allJobs = useSyncExternalStore(videoJobs.subscribe, videoJobs.snapshot)

  const [cat, setCat] = useState<Catalogue | null>(null)
  const [catError, setCatError] = useState<string | null>(null)
  const [connection, setConnection] = useState<ConnectionState>('connecting')
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [notice, setNotice] = useState<{ kind: 'info' | 'correction' | 'error'; title: string; body: string } | null>(null)
  const [showWorkflow, setShowWorkflow] = useState(false)
  const [viewing, setViewing] = useState<HistoryEntry | null>(null)
  const [justFinished, setJustFinished] = useState<{ id: string; ms: number } | null>(null)
  const [reuseNotice, setReuseNotice] = useState<{ applied: AppliedReuse; no: number } | null>(null)

  const fileInput = useRef<HTMLInputElement | null>(null)
  const promptRef = useRef<HTMLTextAreaElement | null>(null)
  const pendingFocus = useRef<string | null>(null)

  // --- what the machine has ------------------------------------------------
  // A failed read retries itself every five seconds, so the desk comes back on
  // its own when ComfyUI starts. No reload, no button hunt.
  const [attempt, setAttempt] = useState(0)
  useEffect(() => {
    let alive = true
    let retry: ReturnType<typeof setTimeout> | null = null
    connect()
    catalogue().then(
      (c) => {
        if (!alive) return
        setCat(c)
        setCatError(null)
      },
      (err: Error) => {
        if (!alive) return
        setCatError(err.message)
        retry = setTimeout(() => setAttempt((a) => a + 1), 5000)
      },
    )
    return () => {
      alive = false
      if (retry) clearTimeout(retry)
    }
  }, [attempt])

  useEffect(() => watchConnection(setConnection), [])

  const family = useMemo<VideoFamily | null>(() => {
    if (!cat || !cat.families.length) return null
    return cat.families.find((f) => f.def.id === composition.familyId) ?? cat.families[0]
  }, [cat, composition.familyId])

  // Load the family's verified recipe the first time, and whenever the chosen
  // style is no longer installed. Fields the reader has touched survive it.
  useEffect(() => {
    if (!family) return
    const current = store.get()
    if (current.familyId === family.def.id && current.model === family.model) return
    store.set(applyDefaults(current, defaultsOf(family)))
  }, [family])

  const mode = composition.mode === 'i2v' ? 'i2v' : 't2v'
  const fps = composition.fps ?? (family ? defaultsFor(family.def, family.model).fps : 24) ?? 24
  const frames = composition.length ?? (family ? defaultsFor(family.def, family.model).length : 81) ?? 81
  const houseNegative = family ? defaultsFor(family.def, family.model).negative : ''
  const shapes = useMemo(() => {
    if (!family) return []
    const list = shapesFor(family)
    if (list.some((s) => s.width === composition.width && s.height === composition.height)) return list
    // A size typed into the expert margin still shows as the chosen chip.
    return [
      ...list,
      {
        width: composition.width,
        height: composition.height,
        label: `${orientation(composition.width, composition.height)} ${times(composition.width, composition.height)}`,
      },
    ].sort((a, b) => a.width * a.height - b.width * b.height)
  }, [family, composition.width, composition.height])

  const lengths = useMemo(() => {
    if (!family) return []
    const list = lengthsFor(family, fps)
    return list.includes(frames) ? list : [...list, frames].sort((a, b) => a - b)
  }, [family, fps, frames])
  const notes = useMemo(() => (family ? marginaliaOf(family) : []), [family])

  const myJobs = allJobs
  const live = useMemo(() => myJobs.filter(unfinished), [myJobs])
  const runningJob = live[live.length - 1] ?? null
  const now = useNow(live.length > 0)

  // The plate shows the newest finished clip unless the reader is reading an
  // older one, which we never yank away from them.
  const newestDone = useMemo(
    () => myJobs.find((j) => j.status === 'done' && j.files.length > 0),
    [myJobs],
  )
  const failed = useMemo(
    () => myJobs.find((j) => j.status === 'error' || j.status === 'cancelled'),
    [myJobs],
  )

  const clips = useMemo(() => records.filter((e) => e.kind === 'video').slice(0, 12), [records])
  const sources = useMemo(() => records.filter((e) => e.kind === 'image').slice(0, 6), [records])

  const shown: { file: FileRef; entry: HistoryEntry | null; frames: number; fps: number } | null = useMemo(() => {
    if (viewing) {
      return { file: viewing.file, entry: viewing, frames: viewing.length ?? 0, fps: viewing.fps ?? 0 }
    }
    if (newestDone) {
      const file = newestDone.files.find((f) => f.kind === 'video') ?? newestDone.files[0]
      return {
        file,
        entry: newestDone.entryId ? (records.find((r) => r.id === newestDone.entryId) ?? null) : null,
        frames: newestDone.frames,
        fps: newestDone.fps,
      }
    }
    const latest = clips[0]
    if (latest) {
      return { file: latest.file, entry: latest, frames: latest.length ?? 0, fps: latest.fps ?? 0 }
    }
    return null
  }, [viewing, newestDone, clips, records])

  const estimate = useMemo(
    () =>
      measuredEstimate(records, {
        familyId: composition.familyId,
        length: frames,
        width: composition.width,
        height: composition.height,
        mode: composition.mode,
      }),
    [records, composition.familyId, composition.mode, composition.width, composition.height, frames],
  )

  /**
   * The estimate belonging to the job on the press, which is not the estimate
   * for what the rail is composing now. A reader who changes the shape while a
   * clip runs must not be told the running clip will take longer than it will.
   */
  const runningEstimate = useMemo(
    () => (runningJob ? measuredEstimate(records, runningJob.composition) : null),
    [records, runningJob],
  )

  // A finished clip develops in, the way a print comes up in the tray. It is
  // one of the three movements in the whole application.
  const [developed, setDeveloped] = useState(false)
  const showing = shown ? `${shown.file.subfolder}/${shown.file.filename}` : null
  useEffect(() => {
    if (!showing) return
    setDeveloped(false)
    const frame = requestAnimationFrame(() => setDeveloped(true))
    return () => cancelAnimationFrame(frame)
  }, [showing])

  // --- source picture ------------------------------------------------------

  /**
   * Set the start frame, revoking the object URL of the one it replaces. A
   * five-minute session of swapping start frames otherwise leaks every picture
   * the reader looked at.
   */
  const ownedUrl = useRef<string | null>(null)
  const setSource = useCallback((source: SourceRef | null) => {
    if (ownedUrl.current && ownedUrl.current !== source?.previewUrl) {
      URL.revokeObjectURL(ownedUrl.current)
      ownedUrl.current = null
    }
    if (source?.previewUrl?.startsWith('blob:')) ownedUrl.current = source.previewUrl
    store.patch({ source })
  }, [])

  const acceptFile = useCallback(
    async (file: File) => {
      if (!file.type.startsWith('image/')) {
        setNotice({
          kind: 'error',
          title: 'We could not use that file',
          body: 'A start frame has to be a picture. Try a PNG or a JPEG.',
        })
        return
      }
      setUploading(true)
      setNotice(null)
      const previewUrl = URL.createObjectURL(file)
      try {
        const name = await uploadImage(file)
        const size = await measure(previewUrl)
        setSource({
          name,
          previewUrl,
          label: file.name,
          bytes: file.size,
          width: size?.width,
          height: size?.height,
        })
        store.patch({ mode: 'i2v' })
      } catch (err) {
        URL.revokeObjectURL(previewUrl)
        setNotice({
          kind: 'error',
          title: 'We could not upload that picture',
          body: `ComfyUI refused the file. ${(err as Error).message}`,
        })
      } finally {
        setUploading(false)
      }
    },
    [setSource],
  )

  const adoptFromArchive = useCallback(
    async (entry: HistoryEntry) => {
      setUploading(true)
      setNotice(null)
      try {
        const res = await fetch(fileUrl(entry.file))
        if (!res.ok) throw new Error(`the file is no longer on disk (HTTP ${res.status})`)
        const blob = await res.blob()
        const name = await uploadImage(blob, entry.file.filename)
        setSource({
          name,
          ref: entry.file,
          previewUrl: fileUrl(entry.file),
          label: entry.file.filename,
          width: entry.width ?? undefined,
          height: entry.height ?? undefined,
          fromEntryId: entry.id,
        })
        store.patch({ mode: 'i2v' })
      } catch (err) {
        setNotice({
          kind: 'error',
          title: 'We could not use that picture',
          body: `${(err as Error).message}. The record is still in your archive.`,
        })
      } finally {
        setUploading(false)
      }
    },
    [setSource],
  )

  // A source handed over by the archive arrives with a ref and no input name;
  // LoadImage reads the *input* folder, so it has to be uploaded once. The ref
  // guard keeps the upload from restarting every time this effect re-runs.
  const adopting = useRef<string | null>(null)
  useEffect(() => {
    const src = composition.source
    const ref = src?.ref
    if (!src || src.name || !ref) return
    const key = `${ref.subfolder}/${ref.filename}`
    if (adopting.current === key) return
    adopting.current = key
    setUploading(true)
    ;(async () => {
      try {
        const res = await fetch(fileUrl(ref))
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const blob = await res.blob()
        const name = await uploadImage(blob, ref.filename)
        setSource({ ...src, name, previewUrl: src.previewUrl ?? fileUrl(ref) })
      } catch {
        setNotice({
          kind: 'correction',
          title: 'Correction',
          body: 'That start frame could not be loaded into ComfyUI. Choose another picture.',
        })
        setSource(null)
      } finally {
        adopting.current = null
        setUploading(false)
      }
    })()
  }, [composition.source, setSource])

  // Paste anywhere on the desk.
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const file = e.clipboardData?.files?.[0]
      if (!file) return
      e.preventDefault()
      void acceptFile(file)
    }
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
  }, [acceptFile])

  // --- running -------------------------------------------------------------

  const buildGraph = useCallback(
    (fam: VideoFamily, c: Composition, seed: number): ApiWorkflow | null => {
      const shaped = c.mode === 'i2v' ? deriveImageToVideo(fam.def) : fam.def
      if (!shaped) return null
      const params = toParams({ ...c, seed }, { negative: defaultsFor(fam.def, fam.model).negative })
      if (c.mode !== 'i2v') delete params.image
      const graph = instantiate(shaped, params)
      applyShift(graph, c.shift)
      return graph
    },
    [],
  )

  const blockedReason = useMemo(() => reasonFor(family, composition), [family, composition])

  /**
   * Queue the clip — or several, with successive seeds.
   *
   * Everything is read from the store rather than from this render's props,
   * because `Make another` loads a record into the store and runs in the same
   * tick, before React has re-rendered with it.
   */
  const make = useCallback(() => {
    if (uploading) return
    const c = store.get()
    const fam = cat?.families.find((f) => f.def.id === c.familyId) ?? cat?.families[0] ?? null
    if (!fam || reasonFor(fam, c)) return

    const runs = c.runs ?? 1
    const first = c.seedLocked ? c.seed : randomSeed()

    for (let i = 0; i < runs; i++) {
      const seed = first + i
      const snapshot: Composition = { ...c, seed }
      const graph = buildGraph(fam, snapshot, seed)
      if (!graph) {
        setNotice({
          kind: 'error',
          title: 'That shape is not available',
          body: `${fam.label} cannot take a start frame. Work from words, or choose another style.`,
        })
        return
      }
      startJob({
        composition: snapshot,
        graph,
        familyLabel: fam.def.label,
        modelLabel: modelLabelOf(fam),
      })
    }

    store.patch({ seed: first })
    setViewing(null)
    setNotice(null)
  }, [cat, uploading, buildGraph])

  /**
   * Put a record back on the desk. `Use these settings` restores everything and
   * runs nothing, so the reader sees exactly what they are about to make;
   * `Make another` does the same with a fresh seed and then runs.
   *
   * A draft that would be overwritten is never simply lost: the notice offers
   * both ways out, and the undo is the store's own previous state.
   */
  const reuse = useCallback(
    (entry: HistoryEntry, run: boolean) => {
      const installedModels = cat?.families.map((f) => f.model).filter(Boolean)
      const applied = reuseIntoDesk(entry, {
        freshSeed: run,
        installedModels,
        availableSamplers: cat?.samplers,
        availableSchedulers: cat?.schedulers,
      })
      setViewing(null)
      setReuseNotice(applied.clobbered || applied.notes.length ? { applied, no: entry.no } : null)
      if (run) make()
      else promptRef.current?.focus()
    },
    [cat, make],
  )

  // Ctrl/⌘+Enter runs, from inside the prompt too — the one deliberate
  // exception to "every single-key shortcut is dead while a field has focus".
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault()
        make()
        return
      }
      if (isTyping(e) || e.ctrlKey || e.metaKey || e.altKey) return
      if (e.key === 'u') {
        e.preventDefault()
        fileInput.current?.click()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [make])

  // The run button briefly wears the elapsed time, so you learn the card's
  // rhythm without ever opening a log.
  useEffect(() => {
    const done = myJobs.find((j) => j.status === 'done' && j.finishedAt)
    if (!done || !done.finishedAt) return
    if (Date.now() - done.finishedAt > 4000) return
    setJustFinished({ id: done.id, ms: done.finishedAt - done.startedAt })
    const t = setTimeout(() => setJustFinished(null), 4000)
    return () => clearTimeout(t)
  }, [myJobs])

  // --- expert focus --------------------------------------------------------

  const revealExpert = useCallback((fieldId: string) => {
    settings.patch({ expert: true })
    pendingFocus.current = fieldId
  }, [])

  // The margin fills in; nothing else moves. `prefers-reduced-motion` is
  // honoured globally in index.css.
  const [marginIn, setMarginIn] = useState(false)
  useEffect(() => {
    if (!prefs.expert) {
      setMarginIn(false)
      return
    }
    const frame = requestAnimationFrame(() => setMarginIn(true))
    return () => cancelAnimationFrame(frame)
  }, [prefs.expert])

  useEffect(() => {
    if (!prefs.expert || !pendingFocus.current) return
    const id = pendingFocus.current
    pendingFocus.current = null
    const frame = requestAnimationFrame(() => {
      const el = document.getElementById(id)
      el?.scrollIntoView({ block: 'center' })
      el?.querySelector<HTMLElement>('input,select,textarea')?.focus()
    })
    return () => cancelAnimationFrame(frame)
  }, [prefs.expert])

  // --- drag and drop -------------------------------------------------------

  const onDrop = (e: ReactDragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer?.files?.[0]
    if (file) void acceptFile(file)
  }

  // --- render --------------------------------------------------------------

  if (catError && !cat) return <ServerDown onRetry={() => setAttempt((a) => a + 1)} detail={catError} />

  const progress = runningJob
    ? runningJob.max > 1
      ? clamp(runningJob.value / runningJob.max, 0, 0.97)
      : 0.03
    : 0

  const elapsed = runningJob ? now - runningJob.startedAt : 0

  const remaining = (() => {
    if (!runningJob) return null
    if (runningEstimate) {
      const left = runningEstimate.ms - elapsed
      if (left <= 0) return 'Running long. Still working.'
      return `About ${duration(left)} left, from your last ${runningEstimate.runs} runs.`
    }
    if (runningJob.samplingAt && runningJob.value >= 2 && runningJob.max > 1) {
      const perStep = (now - runningJob.samplingAt) / runningJob.value
      const left = perStep * (runningJob.max - runningJob.value)
      if (left <= 0) return null
      return `About ${duration(left)} left of the drawing, at this run's pace. Developing and encoding follow.`
    }
    return null
  })()

  return (
    <main
      className="relative min-h-full px-6 py-6"
      onDragOver={(e) => {
        e.preventDefault()
        if (!dragging) setDragging(true)
      }}
      onDragLeave={(e) => {
        if (e.currentTarget === e.target) setDragging(false)
      }}
      onDrop={onDrop}
    >
      {dragging && (
        <div className="pointer-events-none absolute inset-4 z-20 grid place-items-center border-2 border-dashed border-burgundy-900 bg-newsprint/80">
          <span className="text-[0.625rem] font-semibold uppercase tracking-[0.18em] text-burgundy-900">
            Drop to use as your start frame
          </span>
        </div>
      )}

      <div className="mb-5 border-b-2 border-burgundy-900 pb-2">
        <h2 className="text-[0.625rem] font-semibold uppercase tracking-[0.18em] text-burgundy-900">
          The Video Desk
        </h2>
        <p className="mt-1 text-small italic text-grey-700">
          Five seconds at a time. This takes minutes, not seconds.
        </p>
      </div>

      <div className="grid gap-8 lg:grid-cols-[24rem_minmax(0,1fr)] xl:grid-cols-[17rem_24rem_1fr]">
        {/* ---------------------------------------------------------------- */}
        {/* The expert margin. Empty in simple mode; nothing else moves.      */}
        {/* ---------------------------------------------------------------- */}
        <aside className="order-3 lg:col-span-2 lg:border-t lg:border-grey-300 lg:pt-6 xl:order-1 xl:col-span-1 xl:border-t-0 xl:pt-0">
          {prefs.expert && family ? (
            <div
              style={{
                opacity: marginIn ? 1 : 0,
                transform: marginIn ? 'translateX(0)' : 'translateX(-8px)',
                transition: 'opacity 180ms ease, transform 180ms ease',
              }}
            >
              <Head title="All controls" />

              <ExpertField label="Frames" hint={`${clipLength(frames, fps)}`} id="field-frames">
                <NumberField
                  label="Frames"
                  value={frames}
                  min={family.frames.min}
                  max={family.frames.max}
                  step={family.frames.step}
                  commit={(n) => store.edit({ length: snap(n, family.frames) }, 'length')}
                />
                <p className="mt-1 text-caption italic text-grey-500">
                  Snapped to{' '}
                  {family.frames.step === 4 ? 'four frames plus one' : `steps of ${family.frames.step}`}. The
                  latent node accepts nothing else.
                </p>
              </ExpertField>

              <ExpertField label="Frames per second" id="field-fps">
                <NumberField
                  label="Frames per second"
                  value={fps}
                  min={1}
                  max={120}
                  step={1}
                  commit={(n) => store.edit({ fps: clamp(n, 1, 120) }, 'fps')}
                />
                <p className="mt-1 text-caption italic text-grey-500">
                  The encoder's frame rate. It changes the pace of the clip, not how long it takes to make.
                </p>
              </ExpertField>

              <div className="grid grid-cols-2 gap-2">
                <ExpertField label="Width" id="field-width">
                  <NumberField
                    label="Width"
                    value={composition.width}
                    min={family.width.min}
                    max={family.width.max}
                    step={family.width.step}
                    commit={(n) => store.edit({ width: snap(n, family.width) }, 'width')}
                  />
                </ExpertField>
                <ExpertField label="Height">
                  <NumberField
                    label="Height"
                    value={composition.height}
                    min={family.height.min}
                    max={family.height.max}
                    step={family.height.step}
                    commit={(n) => store.edit({ height: snap(n, family.height) }, 'height')}
                  />
                </ExpertField>
              </div>
              <p className="-mt-2 mb-3 text-caption italic text-grey-500 tabular-nums">
                {((composition.width * composition.height) / 1e6).toFixed(2)} megapixels a frame, in steps of{' '}
                {family.width.step}.
              </p>

              {family.def.dualModel ? (
                <p className="mb-3 text-caption italic text-grey-700">
                  This family runs a matched high-noise and low-noise pass. Its step count comes with the recipe and is
                  not adjustable here.
                </p>
              ) : (
                <ExpertField label="Steps" id="field-steps">
                  <NumberField
                    label="Steps"
                    value={composition.steps}
                    min={1}
                    max={200}
                    step={1}
                    commit={(n) => store.edit({ steps: clamp(Math.round(n), 1, 200) }, 'steps')}
                  />
                </ExpertField>
              )}

              <ExpertField label="CFG" id="field-cfg">
                <NumberField
                  label="CFG"
                  value={composition.cfg}
                  min={0}
                  max={30}
                  step={0.1}
                  commit={(n) => store.edit({ cfg: clamp(n, 0, 30) }, 'cfg')}
                />
              </ExpertField>

              <ExpertField label="Sampler" id="field-sampler">
                <select
                  className="field"
                  value={composition.sampler}
                  onChange={(e) => store.edit({ sampler: e.target.value }, 'sampler')}
                >
                  {(cat?.samplers.length ? cat.samplers : [composition.sampler]).map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </ExpertField>

              {family.def.bindings.scheduler ? (
                <ExpertField label="Scheduler" id="field-scheduler">
                  <select
                    className="field"
                    value={composition.scheduler}
                    onChange={(e) => store.edit({ scheduler: e.target.value }, 'scheduler')}
                  >
                    {(cat?.schedulers.length ? cat.schedulers : [composition.scheduler]).map((s) => (
                      <option key={s} value={s}>
                        {s}
                      </option>
                    ))}
                  </select>
                </ExpertField>
              ) : null}

              <ExpertField
                label="Seed"
                hint={composition.seedLocked ? 'fixed' : 'random each run'}
                id="field-seed"
              >
                <div className="flex gap-2">
                  <NumberField
                    label="Seed"
                    italic={!composition.seedLocked}
                    value={composition.seed}
                    min={0}
                    max={Number.MAX_SAFE_INTEGER}
                    step={1}
                    commit={(n) => {
                      if (n < 0) return
                      store.edit({ seed: Math.floor(n) }, 'seed')
                      store.patch({ seedLocked: true })
                    }}
                  />
                  <button
                    type="button"
                    className={`shrink-0 border px-2 text-[0.625rem] font-semibold uppercase tracking-[0.16em] ${
                      composition.seedLocked
                        ? 'border-ink bg-ink text-newsprint'
                        : 'border-grey-300 text-grey-700 hover:bg-newsprint-aged'
                    }`}
                    aria-pressed={composition.seedLocked}
                    onClick={() => store.patch({ seedLocked: !composition.seedLocked })}
                  >
                    {composition.seedLocked ? 'Fixed' : 'Random'}
                  </button>
                </div>
              </ExpertField>

              {graphShift(family.def) !== null ? (
                <ExpertField label="Shift" hint="model sampling" id="field-shift">
                  <NumberField
                    label="Shift"
                    value={composition.shift ?? graphShift(family.def) ?? 0}
                    min={0}
                    max={20}
                    step={0.5}
                    commit={(n) => store.edit({ shift: clamp(n, 0, 20) }, 'shift')}
                  />
                </ExpertField>
              ) : null}

              {family.def.bindings.negative ? (
                <ExpertField label="Negative" id="field-negative">
                  <textarea
                    className="field h-24 text-caption"
                    value={composition.negative ?? houseNegative ?? ''}
                    onChange={(e) => store.edit({ negative: e.target.value }, 'negative')}
                  />
                  {composition.negative !== null ? (
                    <button
                      type="button"
                      className="mt-1 text-caption text-burgundy-900 underline"
                      onClick={() => store.set(clearTouched({ ...store.get(), negative: null }, 'negative'))}
                    >
                      Reset to the house wording
                    </button>
                  ) : (
                    <p className="mt-1 text-caption italic text-grey-500">The family's own wording, unchanged.</p>
                  )}
                </ExpertField>
              ) : null}

              <ExpertField label="Runs" hint="sequential, successive seeds">
                <Chips
                  ariaLabel="How many clips to make"
                  value={composition.runs}
                  onChange={(v) => store.patch({ runs: v })}
                  options={[
                    { value: 1 as const, label: '×1' },
                    { value: 2 as const, label: '×2' },
                    { value: 4 as const, label: '×4' },
                  ]}
                />
              </ExpertField>

              <button
                type="button"
                className="mb-4 text-caption text-burgundy-900 underline"
                onClick={() => setShowWorkflow((v) => !v)}
              >
                {showWorkflow ? 'Hide the workflow' : 'Show the workflow'}
              </button>

              {showWorkflow ? (
                <WorkflowPeek build={() => buildGraph(family, store.get(), composition.seed)} />
              ) : null}

              {notes.length ? (
                <div className="mt-4 border-t border-grey-300 pt-3">
                  <Head title="From the model's card" />
                  {notes.slice(0, 3).map((n, i) => (
                    <Marginalia key={i} text={n} />
                  ))}
                </div>
              ) : null}

              {cat?.vramFree != null ? (
                <p className="mt-2 text-caption italic text-grey-500 tabular-nums">
                  The card has {gb(cat.vramFree)} free right now.
                </p>
              ) : null}

              <div className="mt-4 border-t border-grey-300 pt-3">
                <CataloguePanel
                  modes={['video']}
                  onInstalled={() => {
                    resetCatalogue()
                    setAttempt((a) => a + 1)
                  }}
                />
              </div>
            </div>
          ) : null}
        </aside>

        {/* ---------------------------------------------------------------- */}
        {/* The composing rail                                               */}
        {/* ---------------------------------------------------------------- */}
        <section className="order-1 xl:order-2">
          {!cat ? (
            <p className="text-small italic text-grey-500">Reading what this machine has…</p>
          ) : !family ? (
            <Notice tone="correction" title="Correction" >
              No video model is installed. {cat.blocked.length ? `${cat.blocked[0].label} ${cat.blocked[0].why}.` : ''}
            </Notice>
          ) : (
            <>
              {/* Source tabs */}
              <div className="mb-5 flex border border-grey-300" role="group" aria-label="Where the clip comes from">
                {(['t2v', 'i2v'] as const).map((m, i) => {
                  const active = mode === m
                  const disabled = m === 'i2v' && !family.canStartFromPicture
                  return (
                    <button
                      key={m}
                      type="button"
                      disabled={disabled}
                      aria-pressed={active}
                      onClick={() => store.patch({ mode: m })}
                      className={[
                        'flex-1 px-3 py-2 text-[0.625rem] font-semibold uppercase tracking-[0.18em] transition-colors',
                        RING,
                        i > 0 ? 'border-l border-grey-300' : '',
                        active ? 'border-t-2 border-t-burgundy-900 text-ink' : 'text-grey-500 hover:text-ink',
                        disabled ? 'cursor-not-allowed text-grey-400 hover:text-grey-400' : '',
                      ].join(' ')}
                    >
                      {m === 't2v' ? 'From words' : 'From a picture'}
                    </button>
                  )
                })}
              </div>

              {mode === 'i2v' && !family.canStartFromPicture ? (
                <div className="mb-4">
                  <Notice tone="correction" title="Correction">
                    {family.label} works from words only. Nothing was carried over from your picture.
                  </Notice>
                </div>
              ) : null}

              {mode === 'i2v' ? (
                <div className="mb-5">
                  <Head title="Start frame">
                    {composition.source ? (
                      <button
                        type="button"
                        className="text-caption text-burgundy-900 underline"
                        onClick={() => setSource(null)}
                      >
                        Clear
                      </button>
                    ) : null}
                  </Head>

                  <input
                    ref={fileInput}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={(e) => {
                      const f = e.target.files?.[0]
                      e.target.value = ''
                      if (f) void acceptFile(f)
                    }}
                  />

                  {composition.source ? (
                    <div className="flex gap-3 border border-grey-300 p-2">
                      {composition.source.previewUrl ? (
                        <img
                          src={composition.source.previewUrl}
                          alt=""
                          className="h-16 w-16 border border-grey-300 object-cover"
                        />
                      ) : (
                        <div className="h-16 w-16 border border-grey-300 bg-newsprint-aged" />
                      )}
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-small">{composition.source.label ?? composition.source.name}</p>
                        <p className="text-caption italic text-grey-500 tabular-nums">
                          {composition.source.width && composition.source.height
                            ? `${times(composition.source.width, composition.source.height)} · `
                            : ''}
                          {composition.source.bytes
                            ? `${(composition.source.bytes / 1024 ** 2).toFixed(1)} MB · `
                            : ''}
                          {uploading ? 'uploading…' : 'ready'}
                        </p>
                        <button
                          type="button"
                          className="mt-1 text-caption text-burgundy-900 underline"
                          onClick={() => fileInput.current?.click()}
                        >
                          Replace
                        </button>
                        {composition.source.fromFrame !== undefined ? (
                          <span className="ml-2 text-caption italic text-grey-500 tabular-nums">
                            lifted from frame {composition.source.fromFrame}
                          </span>
                        ) : null}
                      </div>
                    </div>
                  ) : (
                    <button
                      type="button"
                      onClick={() => fileInput.current?.click()}
                      className="w-full border border-dashed border-grey-300 px-4 py-6 text-center hover:bg-newsprint-aged"
                    >
                      <span className="block text-small italic text-grey-700">
                        {uploading ? 'Uploading…' : 'Drop a picture here, paste one, or click to choose.'}
                      </span>
                      <span className="mt-1 block text-caption text-grey-500">
                        The clip starts on this frame and moves from it.
                      </span>
                    </button>
                  )}

                  {sources.length ? (
                    <div className="mt-2">
                      <p className="mb-1 text-caption italic text-grey-500">Or take one from the archive:</p>
                      <div className="flex gap-1">
                        {sources.map((e) => (
                          <button
                            key={e.id}
                            type="button"
                            title={e.prompt || e.file.filename}
                            aria-label={`Use ${e.prompt || e.file.filename} as the start frame`}
                            onClick={() => void adoptFromArchive(e)}
                            className="h-12 w-12 border border-grey-300 hover:border-burgundy-900"
                          >
                            <img
                              src={fileUrl(e.file)}
                              alt=""
                              loading="lazy"
                              className="h-full w-full object-cover"
                            />
                          </button>
                        ))}
                      </div>
                    </div>
                  ) : null}
                </div>
              ) : null}

              {/* Prompt */}
              <div className="mb-5">
                <Head title="Describe the shot" />
                <textarea
                  ref={promptRef}
                  aria-label="Describe the shot"
                  className="field"
                  style={{ fontSize: '1.125rem', lineHeight: 1.6, minHeight: '9rem', maxWidth: '62ch' }}
                  placeholder="A tram crosses a wet junction at dusk, headlights smearing in the rain"
                  value={composition.prompt}
                  onChange={(e) => store.patch({ prompt: e.target.value })}
                />
                <p className="mt-1 text-caption italic text-grey-500">
                  Say what moves. Wan follows motion better than it follows adjectives.
                </p>
              </div>

              {/* Style — shown in simple mode only when there is a choice */}
              {(cat.families.length > 1 || prefs.expert) && (
                <div className="mb-5">
                  <Head title="Style" />
                  <select
                    aria-label="Style"
                    className="field"
                    value={family.def.id}
                    onChange={(e) => {
                      const next = cat.families.find((f) => f.def.id === e.target.value)
                      if (next) store.set(applyDefaults(store.get(), defaultsOf(next)))
                    }}
                  >
                    {cat.families.map((f) => (
                      <option key={f.def.id} value={f.def.id}>
                        {f.label}
                      </option>
                    ))}
                  </select>
                  {prefs.expert ? (
                    <p className="mt-1 text-caption italic text-grey-500">
                      {family.model || 'a matched pair of weight files'}
                      {family.def.verified ? '' : ' † settings from the model card, not checked against a live run'}
                    </p>
                  ) : null}
                  {family.verdict && family.verdict.level !== 'ok' ? (
                    <p className="mt-1 text-caption italic text-grey-700">{family.verdict.reason}</p>
                  ) : null}
                  {family.verdict?.offloads && family.verdict.level === 'ok' ? (
                    <p className="mt-1 text-caption italic text-grey-500 tabular-nums">
                      Largest file {gb(family.verdict.footprint.largestBytes)}. It streams from RAM, so it runs slower.
                    </p>
                  ) : null}
                </div>
              )}

              {/* Length */}
              <div className="mb-5">
                <Head title="Length">
                  <span className="text-caption italic text-grey-500 tabular-nums">
                    {frames} frames at {fps} fps
                  </span>
                </Head>
                <Chips
                  ariaLabel="How long the clip runs"
                  value={frames}
                  onChange={(v) => store.edit({ length: v }, 'length')}
                  options={lengths.map((f) => ({
                    value: f,
                    label: clipLength(f, fps),
                    caption: `${f} frames`,
                  }))}
                />
              </div>

              {/* Shape */}
              <div className="mb-5">
                <Head title="Shape" />
                <Chips
                  ariaLabel="The shape of the frame"
                  value={`${composition.width}x${composition.height}`}
                  onChange={(v) => {
                    const [w, h] = String(v).split('x').map(Number)
                    if (Number.isFinite(w) && Number.isFinite(h)) store.edit({ width: w, height: h }, 'width', 'height')
                  }}
                  options={shapes.map((s) => ({
                    value: `${s.width}x${s.height}`,
                    label: s.label.split(' ')[0],
                    caption: times(s.width, s.height),
                    title: s.note,
                  }))}
                />
                {(() => {
                  const chosen = shapes.find((s) => s.width === composition.width && s.height === composition.height)
                  const smallest = shapes[0]
                  if (!chosen || !smallest || chosen === smallest) return null
                  const ratio = (chosen.width * chosen.height) / (smallest.width * smallest.height)
                  return (
                    <p className="mt-1 text-caption italic text-grey-700 tabular-nums">
                      {ratio.toFixed(1)}× the pixels of {times(smallest.width, smallest.height)}, so expect it to take
                      longer.
                    </p>
                  )
                })()}
              </div>

              {/* The button. It stays live while a clip runs: there is one GPU
                  and one queue, and a clip sent now really does wait its turn —
                  which we say, rather than pretending to run two at once. */}
              <div className="mb-3">
                <button type="button" className="press" disabled={!!blockedReason || uploading} onClick={make}>
                  {justFinished
                    ? duration(justFinished.ms)
                    : live.length
                      ? 'Make the clip · next in line'
                      : composition.runs > 1
                        ? `Make ${composition.runs} clips`
                        : 'Make the clip'}
                </button>

                {blockedReason ? (
                  <p className="mt-1 text-caption italic text-grey-700">{blockedReason}</p>
                ) : live.length ? (
                  <p className="mt-1 text-caption italic text-grey-700">
                    The press is busy. This one starts when the clip in front of it finishes.
                  </p>
                ) : null}

                {runningJob ? (
                  <div className="mt-2">
                    <HoldToStop jobId={runningJob.id} />
                  </div>
                ) : null}
              </div>

              {/* What simple mode chose, said out loud */}
              {!prefs.expert ? (
                <p className="text-caption italic leading-relaxed text-grey-700">
                  Using the maker's settings:{' '}
                  <button className="underline" onClick={() => revealExpert('field-steps')}>
                    {composition.steps} steps
                  </button>{' '}
                  ·{' '}
                  <button className="underline" onClick={() => revealExpert('field-cfg')}>
                    CFG {composition.cfg}
                  </button>{' '}
                  ·{' '}
                  <button className="underline" onClick={() => revealExpert('field-sampler')}>
                    {composition.sampler} / {composition.scheduler}
                  </button>{' '}
                  ·{' '}
                  <button className="underline" onClick={() => revealExpert('field-seed')}>
                    seed {composition.seedLocked ? composition.seed : 'random'}
                  </button>
                  .{' '}
                  <button className="text-burgundy-900 underline" onClick={() => settings.patch({ expert: true })}>
                    Show all controls →
                  </button>
                </p>
              ) : null}

              {/* Cost, only when we have measured it or the registry states it */}
              {(() => {
                const registry = registryMinutes(family)
                if (estimate) {
                  return (
                    <p className="mt-2 text-caption italic text-grey-700 tabular-nums">
                      About {duration(estimate.ms)} a clip at this size, from your last {estimate.runs} runs.
                    </p>
                  )
                }
                if (registry) {
                  return (
                    <p className="mt-2 text-caption italic text-grey-700 tabular-nums">
                      About {registry} minutes a clip, from the model's own notes for this card.
                    </p>
                  )
                }
                return (
                  <p className="mt-2 text-caption italic text-grey-500">
                    We have not timed this size yet, so we will not guess. After three clips we will tell you.
                  </p>
                )
              })()}

              {reuseNotice ? (
                <div className="mt-4">
                  <Notice tone="correction" title="Settings loaded">
                    from No. {reuseNotice.no.toLocaleString('en-GB')}. Nothing has run yet.{' '}
                    {reuseNotice.applied.notes.map((n) => n.reason).join(' ')}{' '}
                    <button
                      className="underline"
                      onClick={() => {
                        reuseNotice.applied.undo()
                        setReuseNotice(null)
                      }}
                    >
                      Undo this
                    </button>{' '}
                    ·{' '}
                    <button className="underline" onClick={() => setReuseNotice(null)}>
                      Keep these settings
                    </button>
                  </Notice>
                </div>
              ) : null}

              {notice ? (
                <div className="mt-4">
                  <Notice tone={notice.kind} title={notice.title}>
                    {notice.body}
                  </Notice>
                </div>
              ) : null}

              {connection === 'closed' && live.length ? (
                <div className="mt-4">
                  <Notice tone="correction" title="Correction">
                    We lost the connection to ComfyUI. Your clip may still be running. We will reconnect and pick it up.
                  </Notice>
                </div>
              ) : null}
            </>
          )}
        </section>

        {/* ---------------------------------------------------------------- */}
        {/* The plate                                                        */}
        {/* ---------------------------------------------------------------- */}
        <section className="order-2 xl:order-3">
          <Head title="The plate">
            {clips.length && onNavigate ? (
              <button
                type="button"
                className="text-caption text-burgundy-900 underline"
                onClick={() => onNavigate('#/archive?q=is%3Avideo')}
              >
                All clips in the archive →
              </button>
            ) : null}
          </Head>

          {/* Running jobs */}
          {live.map((job) => (
            <div key={job.id} className="mb-4 border border-grey-300 bg-newsprint-aged p-3">
              <div className="flex items-baseline justify-between gap-3">
                <span className="text-[0.625rem] font-semibold uppercase tracking-[0.18em] text-grey-700">
                  {job.max > 1 && job.stage === 'Drawing'
                    ? `Drawing · step ${job.value} of ${job.max}`
                    : job.stage === 'Loading the model'
                      ? 'Loading the model · about a minute the first time'
                      : job.stage}
                </span>
                <span className="text-caption tabular-nums text-grey-700">{stopwatch(now - job.startedAt)}</span>
              </div>

              <div className="mt-2 h-[2px] w-full bg-grey-300">
                <div
                  className="h-full bg-burgundy-900"
                  style={{
                    width: `${Math.round(
                      (job.id === runningJob?.id ? progress : job.max > 1 ? clamp(job.value / job.max, 0, 0.97) : 0.03) *
                        100,
                    )}%`,
                    transition: 'width 200ms linear',
                  }}
                />
              </div>

              <p className="mt-2 truncate text-small italic text-grey-700">{job.prompt}</p>

              {job.id === runningJob?.id && remaining ? (
                <p className="mt-1 text-caption italic text-grey-700 tabular-nums">{remaining}</p>
              ) : null}

              {job.queuePos !== null && job.queuePos > 0 ? (
                <p className="mt-1 text-caption italic text-grey-700 tabular-nums">
                  The press is busy. This clip is {ordinal(job.queuePos + 1)} in line.
                </p>
              ) : null}

              {job.previewUrl ? (
                <img
                  src={job.previewUrl}
                  alt="Preview of the frame being drawn"
                  className="mt-3 w-full border border-grey-300 object-contain"
                />
              ) : null}

              <div className="mt-2">
                <HoldToStop jobId={job.id} label={live.length > 1 ? 'Hold to stop this one' : 'Hold to stop'} />
              </div>
            </div>
          ))}

          {/* Failure */}
          {failed ? (
            <div className="mb-4">
              <Notice
                tone={failed.status === 'cancelled' ? 'correction' : 'error'}
                title={faultTitle(failed.fault ?? faultOf(new Error(failed.error ?? '')))}
              >
                {faultBody(failed.fault ?? faultOf(new Error(failed.error ?? '')))}{' '}
                {failed.fault && faultWhere(failed.fault) ? (
                  <span className="block text-caption">{faultWhere(failed.fault)}</span>
                ) : null}
                <button className="underline" onClick={() => dismissJob(failed.id)}>
                  Dismiss
                </button>
              </Notice>
            </div>
          ) : null}

          {/* The clip */}
          {shown ? (
            <figure className="m-0">
              <div
                className="border border-grey-300 bg-newsprint-aged"
                style={{
                  filter: developed ? 'blur(0px)' : 'blur(8px)',
                  transition: 'filter 420ms ease',
                }}
              >
                {renderPlayer ? (
                  renderPlayer({
                    src: fileUrl(shown.file),
                    file: shown.file,
                    fps: shown.fps,
                    frames: shown.frames,
                    entry: shown.entry,
                  })
                ) : (
                  <video
                    key={fileUrl(shown.file)}
                    src={fileUrl(shown.file)}
                    controls
                    loop={prefs.loop}
                    playsInline
                    autoPlay
                    muted
                    className="block max-h-[60vh] w-full object-contain"
                  />
                )}
              </div>

              <figcaption className="mt-2 border-t border-grey-300 pt-2">
                {shown.entry ? (
                  <p className="border-l-4 border-burgundy-900 pl-4 text-h3 italic leading-snug">
                    {shown.entry.prompt}
                  </p>
                ) : null}
                <p className="mt-2 text-caption text-grey-700 tabular-nums">
                  {shown.entry
                    ? `Made by ${shown.entry.modelLabel} · ${dateline(shown.entry.at)} · ${duration(shown.entry.durationMs)}`
                    : 'From the archive'}
                </p>
                <p className="mt-1 text-caption text-grey-500 tabular-nums">
                  {shown.frames ? `${shown.frames} frames · ${clipLength(shown.frames, shown.fps || 1)} · ` : ''}
                  {shown.fps ? `${shown.fps} fps · ` : ''}
                  {shown.entry?.width ? `${times(shown.entry.width, shown.entry.height ?? 0)} · ` : ''}
                  {shown.entry ? `seed ${shown.entry.seed} · ` : ''}
                  {shown.file.filename} · silent · no audio track
                </p>

                <div className="mt-2 flex flex-wrap gap-3 text-caption">
                  <button
                    className="text-burgundy-900 underline"
                    onClick={() =>
                      void saveAs(fileUrl(shown.file), downloadName(shown.file, shown.entry)).catch((err: Error) =>
                        setNotice({ kind: 'error', title: 'We could not save that clip', body: err.message }),
                      )
                    }
                  >
                    Save the clip
                  </button>
                  <button
                    className="text-burgundy-900 underline"
                    onClick={() => {
                      void continueFrom(shown.file, shown.frames, shown.fps, setSource, setNotice)
                    }}
                  >
                    Continue from the last frame
                  </button>
                  {shown.entry ? (
                    <button className="text-burgundy-900 underline" onClick={() => reuse(shown.entry as HistoryEntry, false)}>
                      Use these settings
                    </button>
                  ) : null}
                  {shown.entry ? (
                    <button className="text-burgundy-900 underline" onClick={() => reuse(shown.entry as HistoryEntry, true)}>
                      Make another
                    </button>
                  ) : null}
                </div>
              </figcaption>
            </figure>
          ) : (
            !live.length && (
              <div className="border border-grey-300 bg-newsprint-aged px-6 py-8">
                <p className="dropcap max-w-[52ch] text-body leading-relaxed">
                  Describe a shot and press Make the clip. It will take a few minutes, and you can carry on working
                  while it runs. The progress follows you around the app.
                </p>
                <div className="mt-6">
                  <h4 className="mb-1 text-[0.625rem] font-semibold uppercase tracking-[0.18em] text-grey-700">
                    Try one of these
                  </h4>
                  <ul className="border-t border-grey-300">
                    {EXAMPLES.map((ex) => (
                      <li key={ex.prompt} className="border-b border-grey-300">
                        <button
                          type="button"
                          className="flex w-full items-baseline justify-between gap-4 py-2 text-left hover:bg-newsprint"
                          onClick={() => {
                            store.patch({ prompt: ex.prompt })
                            promptRef.current?.focus()
                          }}
                        >
                          <span className="text-small">{ex.prompt}</span>
                          <span className="shrink-0 text-caption italic text-grey-500">{ex.note}</span>
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            )
          )}

          {/* Recent clips — a ruled index, never a grid of video elements */}
          {clips.length ? (
            <div className="mt-6">
              <Head title="Recent clips" />
              <ul className="border-t border-grey-300">
                {clips.map((e) => (
                  <li key={e.id} className="border-b border-grey-300">
                    <button
                      type="button"
                      className="flex w-full items-baseline justify-between gap-4 py-1.5 text-left hover:bg-newsprint-aged"
                      onClick={() => setViewing(e)}
                    >
                      <span className="truncate text-small">{e.prompt || 'Untitled'}</span>
                      <span className="shrink-0 text-caption text-grey-500 tabular-nums">
                        No. {e.no.toLocaleString('en-GB')} ·{' '}
                        {e.length && e.fps ? clipLength(e.length, e.fps) : '—'}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
        </section>
      </div>
    </main>
  )
}

// ---------------------------------------------------------------------------
// Pieces that need a little logic of their own
// ---------------------------------------------------------------------------

function WorkflowPeek({ build }: { build: () => ApiWorkflow | null }) {
  const built = build()
  const [copied, setCopied] = useState(false)
  if (!built) {
    return <p className="mb-4 text-caption italic text-grey-500">Nothing to show until a style is chosen.</p>
  }
  const text = JSON.stringify(built, null, 2)
  return (
    <div className="mb-4">
      <button
        type="button"
        className="mb-1 text-caption text-burgundy-900 underline"
        onClick={() => {
          void navigator.clipboard?.writeText(text).then(
            () => {
              setCopied(true)
              setTimeout(() => setCopied(false), 2000)
            },
            () => setCopied(false),
          )
        }}
      >
        {copied ? 'Copied' : 'Copy the JSON'}
      </button>
      <pre className="max-h-64 overflow-auto border border-grey-300 bg-newsprint-aged p-2 font-mono text-[0.65rem] leading-tight">
        {text}
      </pre>
    </div>
  )
}

/** Read a picture's real dimensions, for the well's caption. */
/**
 * Lift the last frame out of a finished clip and stand it up as the next start
 * frame. Shot follows shot, which is the whole reason to have both modes in
 * one room.
 */
async function continueFrom(
  file: FileRef,
  frames: number,
  fps: number,
  setSource: (s: SourceRef | null) => void,
  notify: (n: { kind: 'error'; title: string; body: string } | null) => void,
): Promise<void> {
  try {
    const video = document.createElement('video')
    video.src = fileUrl(file)
    video.muted = true
    video.preload = 'auto'
    await new Promise<void>((resolve, reject) => {
      video.onloadeddata = () => resolve()
      video.onerror = () => reject(new Error('the clip could not be decoded in this browser'))
    })
    const last = frames > 0 && fps > 0 ? (frames - 0.5) / fps : Math.max(0, video.duration - 0.05)
    await new Promise<void>((resolve, reject) => {
      video.onseeked = () => resolve()
      video.onerror = () => reject(new Error('the clip could not be seeked'))
      video.currentTime = Math.min(last, Math.max(0, video.duration - 0.01))
    })
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('this browser would not give us a canvas')
    ctx.drawImage(video, 0, 0)
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/png'))
    if (!blob) throw new Error('the frame could not be written')
    const name = await uploadImage(blob, file.filename.replace(/\.[^.]+$/, '') + '_last.png')
    setSource({
      name,
      previewUrl: URL.createObjectURL(blob),
      label: `last frame of ${file.filename}`,
      width: canvas.width,
      height: canvas.height,
      bytes: blob.size,
      fromFrame: frames > 0 ? frames - 1 : undefined,
    })
    store.patch({ mode: 'i2v' })
    notify(null)
  } catch (err) {
    notify({
      kind: 'error',
      title: 'We could not lift that frame',
      body: `${(err as Error).message}. Save the clip and choose a frame by hand instead.`,
    })
  }
}

function downloadName(file: FileRef, entry: HistoryEntry | null): string {
  if (!entry) return file.filename
  const ext = file.filename.split('.').pop() ?? 'webm'
  return `switchgen-${entry.familyId}-${entry.seed}.${ext}`
}

