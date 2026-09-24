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
 *
 * Where the SwitchGen server runs its queue, the desk builds each clip's graph
 * and record as before and hands the lot to the server in one request; the
 * server then waits, releases, sends and files, and the desk only watches
 * (see "The queue on the server" below). Everything the page does itself, the
 * lane, the tab's saved clips and the wake lock, is kept unchanged for a
 * server without the queue.
 */

import { OfferList } from '../components/result/ResultActions'
import { videoOffersFor } from '../components/result/videoOffers'
import { LoraRack } from '../components/video/LoraRack'
import { drawingLeft, drawnFraction, nextPace, passOf, refusalsFor, type SamplingPass } from '../components/video/progress'
import { EMPTY_LIBRARY, loadLoraLibrary, loadStack, missingTriggers, resolveStack, saveStack, targetFor, type LoraLibrary, type LoraStack } from '../lib/loras'
import { chainVideoStack, restoreRack, videoLorasToRun } from '../lib/videoLoras'
import { clipMemory, releaseComfyMemory, releaseIfOthersAhead, waitForIdleComfy } from '../lib/clipMemory'
import { WAITS_IN_PAGE, WAITS_ON_SERVER, holdAwake, wakeLockAvailable } from '../lib/wakeLock'
import {
  HELD_AFTER_PAUSE,
  HELD_AFTER_RESTART,
  deviceId,
  dismiss as dismissOnServer,
  fallbackLine,
  follow,
  forgetGivenUp,
  givenUpBatches,
  laneWord,
  outboxPending,
  recordTemplate,
  reportedOf,
  runnerAvailable,
  runnerFault,
  runnerStore,
  stage as stageHandOver,
  stopJob as stopOnServer,
  submitGroup,
  waitLine,
  withdraw,
  type FollowEvent,
  type FollowResult,
  type GivenUp,
  type RunnerJob,
  type RunnerLane,
  type RunnerProgress,
  type RunnerSnapshot,
  type SubmitBody,
  type SubmitResult,
} from '../lib/runner'
import { copyText } from '../lib/clipboard'
import { thumbUrl } from '../lib/thumbs'
import { annotatedRef } from '../lib/continuation'
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
  type ReactNode,
} from 'react'
import { useHoldToConfirm } from '../components/shell/hotkeys'
import { heldCounts } from '../components/shell/RunnerHold'
import { onPlanLanded } from '../lib/downloads'
import { useServerCapabilities, type ServerCapabilities } from '../lib/capabilities'

import {
  ComfyError,
  LostJob,
  cancelJob,
  connect,
  connectionState,
  fetchPastRun,
  fileUrl,
  followPrompt,
  listJobs,
  newPromptId,
  objectInfo,
  run,
  systemStats,
  uploadImage,
  watchConnection,
  VIDEO_EXT,
  type ApiWorkflow,
  type ConnectionState,
  type FileRef,
  type Followed,
  type OutputFile,
  type ProgressEvent as ComfyProgress,
  type ServerJob,
} from '../lib/comfy'

import {
  FAMILIES,
  defaultsFor,
  instantiate,
  type FamilyDef,
} from '../lib/workflows'

import {
  feasibility,
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
  compositionFromEntry,
  deskStore,
  needsSource,
  newComposition,
  randomSeed,
  recordOf,
  reuseIntoDesk,
  settings,
  tabStore,
  toParams,
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

/** The stage a node of this class is, by its class alone. */
function stageFor(classType: string | null | undefined): string {
  return (classType && STAGES[classType]) || 'Working'
}

function stageOf(graph: ApiWorkflow, nodeId: string | null): string {
  if (!nodeId) return 'Working'
  return stageFor(graph[nodeId]?.class_type)
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

/**
 * The node that writes the clip, and whether the family's fps binding reaches
 * it. A binding that points at a node the graph does not have, or only at the
 * sampler's conditioning, leaves the encoder at its own rate whatever the
 * reader types, and every length the desk prints would then be computed at a
 * rate the file does not play at. With no encoder that carries a rate there
 * is nothing to check, and the field is taken at its word.
 */
function encoderOf(def: FamilyDef): { bound: boolean; fps: number | null } {
  const save = Object.entries(def.graph).find(
    ([, n]) => n.class_type.startsWith('Save') && typeof n.inputs.fps === 'number',
  )
  if (!save) return { bound: true, fps: null }
  const bound = (def.bindings.fps ?? []).some(([id, input]) => id === save[0] && input === 'fps')
  return { bound, fps: save[1].inputs.fps as number }
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
  /**
   * True when the family's own graph binds a start frame (the 14B I2V pair).
   * It has no text-to-video graph: sent from words, its LoadImage would keep
   * the placeholder `example.png` and ComfyUI would refuse the prompt.
   */
  needsStartFrame: boolean
  /** True when the fps binding reaches the node that writes the clip. */
  fpsReachesEncoder: boolean
  /** The frame rate the family's encoder node is set to in its own graph. */
  encoderFps: number | null
  width: NumSpec
  height: NumSpec
  frames: NumSpec
}

type Catalogue = {
  /**
   * Families whose every file is installed. Whether the machine can hold one
   * is a question of memory, which moves, so it is asked of a fresh reading
   * by {@link priced} rather than settled here.
   */
  families: VideoFamily[]
  /** Families with a recipe but a missing file, so the absence is explicable. */
  blocked: { label: string; why: string }[]
  samplers: string[]
  schedulers: string[]
  /** Weight files on disk and their sizes, for pricing a family against a reading. */
  sizes: Map<string, ModelFile>
  /** The reading taken with the catalogue. The desk takes fresher ones. */
  machine: Machine
}

/** One reading of the machine's memory, and when it was taken. */
type Machine = {
  hardware: Hardware | null
  /** Free VRAM as ComfyUI reports it, or nvidia-smi's figure when ComfyUI gave none. */
  vramFree: number | null
  at: number
}

async function readMachine(): Promise<Machine> {
  const [hardware, stats] = await Promise.all([
    probeHardware().catch(() => null),
    systemStats().catch(() => null),
  ])
  const devices = (stats as { devices?: { vram_free?: number }[] } | null)?.devices
  const vramFree = Array.isArray(devices) && typeof devices[0]?.vram_free === 'number'
    ? devices[0].vram_free
    : (hardware?.gpu?.vramFree ?? null)
  return { hardware, vramFree, at: Date.now() }
}

/**
 * The families this machine can hold, each with its memory verdict, priced
 * against one reading. A family the machine cannot hold at all is moved to
 * the blocked list with the verdict's own sentence. Without a reading nothing
 * is priced, as availabilityOf does.
 */
function priced(cat: Catalogue, hardware: Hardware | null): Pick<Catalogue, 'families' | 'blocked'> {
  if (!hardware) return cat
  const families: VideoFamily[] = []
  const blocked = [...cat.blocked]
  for (const family of cat.families) {
    const verdict = feasibility(family.def, cat.sizes, hardware)
    if (verdict.selectable) families.push({ ...family, verdict })
    else blocked.push({ label: family.def.label, why: verdict.reason })
  }
  return { families, blocked }
}

/** `14:05`, for saying when a figure was measured. */
const clock = (at: number) => new Date(at).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })

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
  const [info, sizes, machine] = await Promise.all([
    objectInfo(),
    modelFiles().catch(() => new Map<string, ModelFile>()),
    readMachine(),
  ])

  const inv = inventoryFrom(info)
  const weights = inv.weights

  const families: VideoFamily[] = []
  const blocked: Catalogue['blocked'] = []

  for (const def of FAMILIES) {
    if (def.mode !== 'video') continue

    // Files only. Memory is priced against the desk's latest reading.
    const avail = availabilityOf(def, inv, null, sizes)
    if (!avail.ok) {
      blocked.push({ label: def.label, why: avail.why })
      continue
    }

    const model = def.dualModel ? '' : (def.models.find((m) => weights.has(m)) ?? def.models[0] ?? '')

    const latent = latentClassOf(def)
    const encoder = encoderOf(def)
    families.push({
      def,
      model,
      label: def.label,
      verdict: null,
      canStartFromPicture: deriveImageToVideo(def) !== null,
      needsStartFrame: !!def.bindings.image,
      fpsReachesEncoder: encoder.bound,
      encoderFps: encoder.fps,
      width: readSpec(info, latent, 'width', SPEC_FALLBACK.size),
      height: readSpec(info, latent, 'height', SPEC_FALLBACK.size),
      frames: readSpec(info, latent, 'length', SPEC_FALLBACK.frames),
    })
  }

  return {
    families,
    blocked,
    samplers: inv.samplers,
    schedulers: inv.schedulers,
    sizes,
    machine,
  }
}

let cataloguePromise: Promise<Catalogue> | null = null

/** Forget the cached catalogue, so the next read sees files that just landed. */
export function resetCatalogue(): void {
  cataloguePromise = null
}

// A family fetched from the catalogue can land while the reader is in another
// room, or with the margin that holds the catalogue panel closed. The kept
// reading is dropped here so the next visit reads what is installed now; the
// desk on screen hears of it through its own listener.
onPlanLanded(() => resetCatalogue())

/**
 * What is installed is read once per page load, not once per visit to the
 * desk. Memory is not: see the reading in the desk. A failed read is
 * forgotten, so the retry is a real retry and not the same rejected promise.
 */
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

/** The frame rate the clip will actually be written at. */
function fpsOf(family: VideoFamily, c: Composition): number {
  if (!family.fpsReachesEncoder && family.encoderFps) return family.encoderFps
  return c.fps ?? defaultsFor(family.def, family.model).fps ?? 24
}

/** The size of the clip as the memory check reads it. */
function clipOf(family: VideoFamily, c: Composition): { width: number; height: number; frames: number } {
  return {
    width: c.width,
    height: c.height,
    frames: c.length ?? defaultsFor(family.def, family.model).length ?? 0,
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
  /** When the clip was made on the desk. The card's clock counts from here, wait and all. */
  startedAt: number
  /**
   * When ComfyUI began running it, as the page heard it. What the record says
   * the clip took is counted from here and not from the press: a clip that
   * waited behind others, in the lane or in ComfyUI's queue, did not take the
   * wait to make, and the desk's estimates are medians of these figures.
   */
  ranAt: number | null
  /** The first step of the drawing, so the plate knows the drawing has begun. */
  samplingAt: number | null
  /**
   * The pass drawing now, on a family that samples in two (Wan 2.2 14B). Each
   * pass is reported by ComfyUI as its own step 1 of N, so without it the bar
   * ran to the end, fell back to a tenth, and the time left grew several
   * times over. Null for a family that samples once.
   */
  pass: SamplingPass | null
  /** When the current pass's steps were first read, and at which step: its pace. */
  pace: { at: number; step: number } | null
  finishedAt: number | null
  /** How long ComfyUI took over the clip, as filed. Null when that was not measured. */
  tookMs: number | null
  error: string | null
  /** The classified failure, with ComfyUI's node and per-input detail when it gave them. */
  fault: Fault | null
  /** The add-ons chained into the graph, for the record. */
  loras?: HistoryEntry['loras']
  files: OutputFile[]
  entryId: string | null
  /**
   * The record that already names the file, when ComfyUI answered from its
   * cache with a clip it had made before. Nothing new is filed for it.
   */
  repeatOf: string | null
  composition: Composition
  graph: ApiWorkflow
  frames: number
  fps: number
  /** Position in ComfyUI's own queue; 0 means it is the one running. */
  queuePos: number | null
  cancelRequested: boolean
  /**
   * True once ComfyUI said it took the clip out of its hands at a Stop. Only
   * then can a stop explain ComfyUI having no record of the clip: a stop that
   * never reached it (ComfyUI down, or back up and knowing nothing of the id)
   * explains nothing, and the clip was lost with ComfyUI, not stopped.
   */
  stopLanded: boolean
  /**
   * The id the clip's prompt is going under, from the moment the page made it
   * until ComfyUI says it has the prompt; null otherwise. The clip is kept
   * for the tab under this id before the prompt goes (SENT_KEY), so a page
   * that goes before ComfyUI answers leaves the next one a clip to follow.
   * Without it the clip rendered unfollowed and a second Make rendered it
   * twice. On a clip taken up after a reload it stays set until ComfyUI
   * shows it has the prompt, since the page that sent it may never have
   * got it there.
   */
  sendingAs: string | null
  /** True when ComfyUI releases its cached models before this clip runs. */
  release: boolean
  /** What a clip that has not been sent yet is waiting for, in a sentence. */
  waitNote: string | null
  /**
   * True for a clip an earlier page in this tab sent, followed here through
   * ComfyUI's queue. ComfyUI reports steps and previews only to the page that
   * sent a prompt, so this one has neither.
   */
  resumed: boolean
  /**
   * True for a clip the queue on the SwitchGen server sends and files. The
   * page only watches it: none of the lane, the tab's saved clips, the wake
   * lock or the filing below applies, and the section bar hears of it from
   * the queue, not from this desk.
   */
  runner?: boolean
  /** For a queued clip, whether this browser sent it (by its device id) or another did. */
  sentHere?: boolean
  /**
   * For a queued clip this page did not make: taken up from the server's
   * list, after a reload or from another device. Its settings and graph are
   * on the server, not here, so its composition is only a placeholder.
   */
  adopted?: boolean
  /**
   * For a clip an earlier page in this tab handed over without hearing back,
   * shown from the tab's outbox until the server lists it: only its label is
   * known here, and its settings are nowhere this page can read them.
   */
  handedBefore?: boolean
}

let jobs: VideoJob[] = []
const jobListeners = new Set<() => void>()
let pollTimer: ReturnType<typeof setInterval> | null = null

/**
 * Clips that need a release go to the press one at a time.
 *
 * ComfyUI applies a release to the next prompt its worker takes, or at once
 * when it is idle, so a release is only this clip's when nothing is queued
 * and nothing else is sent between it and the clip. Each such clip waits here
 * for the one before it to settle, then for ComfyUI's queue to empty
 * (waitForIdleComfy), then releases and submits. Sent together, a batch of
 * them spent every release on the first clip, and the rest ran on whatever
 * the one before had left resident.
 */
let releaseLane: Promise<void> = Promise.resolve()

/**
 * Take a place at the back of the lane: the turn to wait for, and the
 * function that says this clip has settled.
 *
 * The lane's promise stands for every heavy clip so far, never for the last
 * one alone. A clip stopped while it waits settles before the clip ahead of
 * it, and a clip followed after a reload never waits at all; either one
 * standing for the whole lane let the next clip go while an older one still
 * waited for an empty queue, and the two then released and sent together.
 */
function joinLane(): { turn: Promise<void>; leave: () => void } {
  const turn = releaseLane
  let leave = () => {}
  const settled = new Promise<void>((resolve) => {
    leave = resolve
  })
  releaseLane = Promise.all([turn, settled]).then(() => undefined)
  return { turn, leave }
}

/** Clips still waiting their turn, so Stop can call the wait off. */
const waiting = new Map<string, AbortController>()

/**
 * The lane held after a heavy clip was lost.
 *
 * A heavy clip that ComfyUI forgets part way went down with ComfyUI, and the
 * likeliest cause on this machine is earlyoom killing it for memory. The
 * clips waiting behind it need as much, and they found the restarted ComfyUI
 * empty and went straight in, towards the same end, with nobody asked. The
 * picture batch and the reel stop in this case; the lane now waits for the
 * reader's word instead.
 */
let laneHold: { promise: Promise<void>; release: () => void } | null = null

/**
 * The lane, kept for this tab.
 *
 * A clip waiting in the lane has not reached ComfyUI, so nothing on the
 * server knows about it; one waiting here would vanish with a reload, without
 * a trace. So from the moment a clip joins the lane until the moment before
 * its prompt goes, it is written to this tab's session storage, with the
 * graph as it was built and everything its record will say, and the next page
 * in the tab puts it back in the lane in the same order. As its prompt goes
 * it is kept under SENT_KEY instead, until it settles.
 *
 * Session storage, not local storage, because it belongs to the tab: no other
 * tab reads it, so two tabs can never both send one clip. The one way a tab
 * gets another's copy is by being duplicated from it, while that tab is
 * still sending. So a page takes a saved lane up by itself only when the page
 * that wrote it said, as it went, that it was going (pagehide), or the browser
 * discarded the tab. Anything else, such as a copied tab or a page that
 * crashed, is shown on the desk for the reader to send or forget, never sent
 * on a guess and never dropped without a word.
 *
 * Closing the tab still loses what waits, so while anything waits the page
 * asks the browser to check with the reader before it goes (a browser may
 * skip that on a page nobody has pressed anything on yet), asks for the
 * screen to stay on where the browser allows it unless the lane is held, and
 * the desk says where waiting clips live and that a hidden or locked page
 * sends nothing.
 */
const LANE_KEY = 'switchgen.videolane.v1'
/** Clips an earlier page left that this page would not send by itself. */
const LEFT_KEY = 'switchgen.videolane.v1.left'
/**
 * Clips whose prompts ComfyUI has, or that are on their way to it, kept for
 * this tab until they settle.
 *
 * The socket that reports a clip belongs to the page that sent it, and a
 * reload or a tab the phone threw away in the background took that page with
 * it. The clip went on rendering, but the desk forgot it: no progress, no
 * Stop, the section bar calling it somebody else's, and nothing filed when it
 * landed. So the next page in the tab takes each one up again, by its prompt
 * id, follows it through ComfyUI's queue and files it the same way. The same
 * rules as the lane decide which page does that.
 */
const SENT_KEY = 'switchgen.videosent.v1'
/** Sent clips an earlier page left that this page would not follow by itself. */
const LEFT_SENT_KEY = 'switchgen.videosent.v1.left'
/** This page, so a page back from the browser's cache can tell whether a later one took its clips. */
const PAGE = globalThis.crypto?.randomUUID?.() ?? `page_${Date.now()}_${Math.random().toString(36).slice(2)}`

/** One clip waiting in the lane, with what a later page needs to send and file it. */
export type LaneClip = {
  id: string
  startedAt: number
  composition: Composition
  graph: ApiWorkflow
  familyLabel: string
  modelLabel: string
  loras?: HistoryEntry['loras']
}

/** One clip in ComfyUI's hands, with what a later page needs to follow and file it. */
export type SentClip = LaneClip & {
  promptId: string
  /** When it began running, if the page that sent it heard. */
  ranAt: number | null
  release: boolean
  /**
   * True when it was kept on its way, before ComfyUI said it had it: the
   * page that sent it may have gone before the prompt got there.
   */
  sending?: boolean
}

/** This page's waiting clips, oldest first. */
let laneClips: LaneClip[] = []
/** What an earlier page left and this one did not take up by itself. */
let leftOver: LaneClip[] = []
/** False when the tab would not keep the last write, so a reload would lose the lane. */
let laneKept = true
/** Sent clips an earlier page left and this one did not take up by itself. */
let leftSent: SentClip[] = []
/** False when the tab would not keep the last write of the sent clips. */
let sentKept = true
/**
 * Set when this page came back from the browser's cache to find that another
 * page in the tab had taken its clips over. It reloads at once; until then it
 * files nothing, since the other page does.
 */
let handedOver = false

const isObj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v)

function isLaneClip(v: unknown): v is LaneClip {
  return (
    isObj(v) &&
    typeof v.id === 'string' &&
    typeof v.startedAt === 'number' &&
    isObj(v.composition) &&
    isObj(v.graph) &&
    typeof v.familyLabel === 'string' &&
    typeof v.modelLabel === 'string' &&
    (v.loras === undefined || Array.isArray(v.loras))
  )
}

function isSentClip(v: unknown): v is SentClip {
  return (
    isLaneClip(v) &&
    typeof (v as Record<string, unknown>).promptId === 'string' &&
    ((v as Record<string, unknown>).ranAt === null || typeof (v as Record<string, unknown>).ranAt === 'number') &&
    typeof (v as Record<string, unknown>).release === 'boolean' &&
    ((v as Record<string, unknown>).sending === undefined || typeof (v as Record<string, unknown>).sending === 'boolean')
  )
}

/** A saved lane as written, or null when there is none or it cannot be read. */
function readLane(raw: string | null): { writer: string; released: boolean; held: boolean; clips: LaneClip[] } | null {
  let v: unknown
  try {
    v = JSON.parse(raw ?? 'null')
  } catch {
    return null
  }
  if (!isObj(v) || !Array.isArray(v.clips)) return null
  return {
    writer: typeof v.writer === 'string' ? v.writer : '',
    released: v.released === true,
    held: v.held === true,
    clips: v.clips.filter(isLaneClip),
  }
}

/** Saved sent clips as written, or null when there are none or they cannot be read. */
function readSent(raw: string | null): { writer: string; released: boolean; jobs: SentClip[] } | null {
  let v: unknown
  try {
    v = JSON.parse(raw ?? 'null')
  } catch {
    return null
  }
  if (!isObj(v) || !Array.isArray(v.jobs)) return null
  return {
    writer: typeof v.writer === 'string' ? v.writer : '',
    released: v.released === true,
    jobs: v.jobs.filter(isSentClip),
  }
}

/** The id a clip is in ComfyUI's hands under, or on its way there under; null before it goes. */
const sentUnder = (j: VideoJob): string | null => j.promptId ?? j.sendingAs

/**
 * True for a clip sent, or on its way, and not settled: one a later page
 * would have to follow. Never a clip of the queue on the server: the server
 * follows and files it, and a later page following it too would file it a
 * second time.
 */
const outThere = (j: VideoJob): boolean => !j.runner && sentUnder(j) !== null && unfinished(j)

/** The clips a page has in ComfyUI's hands or on their way there, oldest first: sent and not settled. */
function sentClipsOf(list: readonly VideoJob[]): SentClip[] {
  return list
    .filter(outThere)
    .map((j) => ({
      id: j.id,
      promptId: sentUnder(j)!,
      startedAt: j.startedAt,
      ranAt: j.ranAt,
      composition: j.composition,
      graph: j.graph,
      familyLabel: j.familyLabel,
      modelLabel: j.modelLabel,
      release: j.release,
      ...(j.sendingAs !== null ? { sending: true } : {}),
      ...(j.loras ? { loras: j.loras } : {}),
    }))
    .reverse()
}

function stayPut(e: BeforeUnloadEvent): void {
  e.preventDefault()
}

let holding = false
/** Lets the screen lock again; set while clips wait in the lane to be sent. */
let letSleep: (() => void) | null = null
/** Hold the page for as long as clips wait in the lane, as the lane now stands. */
function holdPage(): void {
  if (typeof window === 'undefined') return
  const waits = laneClips.length > 0
  if (waits !== holding) {
    holding = waits
    if (waits) window.addEventListener('beforeunload', stayPut)
    else window.removeEventListener('beforeunload', stayPut)
  }
  // A locked phone suspends the page, and nothing more leaves the lane until
  // it wakes. Where the browser allows it (a secure page only), the screen is
  // asked to stay on while clips wait to be sent; the desk says so either
  // way. Not while the lane is held: nothing goes until the reader says, and
  // a phone left after the first lost clip kept its screen on for nothing.
  const sends = waits && laneHold === null
  if (sends && !letSleep) letSleep = holdAwake('Clips waiting in the Video desk lane')
  else if (!sends && letSleep) {
    letSleep()
    letSleep = null
  }
}

/** Write this page's lane, or clear it when nothing waits. `released` says the page is going. */
function saveLane(released = false): void {
  if (laneClips.length) {
    laneKept = tabStore.set(
      LANE_KEY,
      JSON.stringify({ writer: PAGE, released, held: laneHold !== null, clips: laneClips }),
    )
  } else {
    tabStore.remove(LANE_KEY)
    laneKept = true
  }
  holdPage()
}

function saveLeftOver(): void {
  if (leftOver.length) tabStore.set(LEFT_KEY, JSON.stringify({ clips: leftOver }))
  else tabStore.remove(LEFT_KEY)
}

/** Write this page's sent clips, or clear them when none is out. `released` says the page is going. */
function saveSent(released = false): void {
  if (handedOver) return
  const sent = sentClipsOf(jobs)
  if (sent.length) sentKept = tabStore.set(SENT_KEY, JSON.stringify({ writer: PAGE, released, jobs: sent }))
  else {
    tabStore.remove(SENT_KEY)
    sentKept = true
  }
}

function saveLeftSent(): void {
  if (leftSent.length) tabStore.set(LEFT_SENT_KEY, JSON.stringify({ jobs: leftSent }))
  else tabStore.remove(LEFT_SENT_KEY)
}

function joinSavedLane(clip: LaneClip): void {
  laneClips = [...laneClips, clip]
  saveLane()
}

/** Sent, stopped or failed: nothing is left for a later page to pick up. */
function leaveSavedLane(id: string): void {
  if (!laneClips.some((c) => c.id === id)) return
  laneClips = laneClips.filter((c) => c.id !== id)
  // Nothing is left to hold once the last waiting clip has gone.
  if (!laneClips.length && laneHold) {
    laneHold.release()
    laneHold = null
  }
  saveLane()
}

/** Hold the clips waiting in the lane until the reader says (see laneHold). */
function holdLane(): void {
  if (laneHold || !laneClips.length) return
  let release = () => {}
  const promise = new Promise<void>((resolve) => {
    release = resolve
  })
  laneHold = { promise, release }
  saveLane()
  announce()
}

/**
 * The reader's word: send the held clips after all. The page's own hold
 * first; with none, the hold on the server's queue, which is said there, and
 * which goes for everything it keeps back, of this desk or not.
 */
function sendHeld(): void {
  if (!laneHold && serverHold) {
    sendHeldOnServer(serverHold.since)
    return
  }
  const hold = laneHold
  laneHold = null
  hold?.release()
  saveLane()
  announce()
}

/** The reader's word: call the held clips off, the page's own or, with none, the server's. */
function stopHeld(): void {
  if (!laneHold && !laneClips.length && serverHold) {
    stopHeldOnServer(serverHold.since)
    return
  }
  for (const c of laneClips) void stopJob(c.id)
}

/** Put a saved clip back in the lane, as it was when it was made. */
function resumeClip(c: LaneClip): void {
  startJob({
    id: c.id,
    startedAt: c.startedAt,
    composition: c.composition,
    graph: c.graph,
    familyLabel: c.familyLabel,
    modelLabel: c.modelLabel,
    loras: c.loras,
    release: true,
    note: 'Picked up from the page before this one.',
  })
}

const wasDiscarded = () =>
  typeof document !== 'undefined' && (document as Document & { wasDiscarded?: boolean }).wasDiscarded === true

/**
 * Take in what the last page in this tab left, once, when the module loads.
 * The saved lane is cleared either way: taken up, it is written again as
 * this page's own; left alone, it moves to the desk's question, so a copy
 * that another tab is sending can never be taken up later by accident.
 */
function restoreLane(): void {
  const saved = readLane(tabStore.get(LANE_KEY))
  const left = readLane(tabStore.get(LEFT_KEY))
  tabStore.remove(LANE_KEY)
  leftOver = left?.clips ?? []
  if (saved?.clips.length) {
    if (saved.released || wasDiscarded()) {
      for (const c of saved.clips) resumeClip(c)
      // Held when the page went, so held still: nothing was said since.
      if (saved.held) holdLane()
    } else leftOver = [...leftOver, ...saved.clips.filter((c) => !leftOver.some((l) => l.id === c.id))]
  }
  saveLeftOver()
}

/**
 * The same for clips the last page had sent: followed from here when that
 * page handed them on, otherwise put to the reader. Following one twice
 * would not render it twice, but it could file it twice.
 */
function restoreSent(): void {
  const saved = readSent(tabStore.get(SENT_KEY))
  const left = readSent(tabStore.get(LEFT_SENT_KEY))
  tabStore.remove(SENT_KEY)
  leftSent = left?.jobs ?? []
  if (saved?.jobs.length) {
    if (saved.released || wasDiscarded()) for (const s of saved.jobs) followSent(s)
    else leftSent = [...leftSent, ...saved.jobs.filter((s) => !leftSent.some((l) => l.id === s.id))]
  }
  saveLeftSent()
}

/** Send what an earlier page left, from this page, at the reader's word. */
function sendLeftOver(): void {
  const clips = leftOver
  leftOver = []
  saveLeftOver()
  for (const c of clips) resumeClip(c)
  announce()
}

function forgetLeftOver(): void {
  leftOver = []
  saveLeftOver()
  announce()
}

/** Follow what an earlier page sent, from this page, at the reader's word. */
function followLeftSent(): void {
  const sent = leftSent
  leftSent = []
  saveLeftSent()
  for (const s of sent) followSent(s)
  announce()
}

/** Stop following them here. ComfyUI goes on making them. */
function forgetLeftSent(): void {
  leftSent = []
  saveLeftSent()
  announce()
}

/** Resolves true when `p` settles, or false as soon as `signal` aborts. */
function unlessStopped(p: Promise<void>, signal: AbortSignal): Promise<boolean> {
  return new Promise((resolve) => {
    if (signal.aborted) return resolve(false)
    const stop = () => resolve(false)
    signal.addEventListener('abort', stop, { once: true })
    void p.then(() => {
      signal.removeEventListener('abort', stop)
      resolve(!signal.aborted)
    })
  })
}

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
  /** Clips an earlier page in this tab left waiting, which this page will not send by itself. */
  leftOver: (): LaneClip[] => leftOver,
  sendLeftOver,
  forgetLeftOver,
  /** Clips an earlier page in this tab sent and did not hand on, which this page will not follow by itself. */
  leftSent: (): SentClip[] => leftSent,
  followLeftSent,
  forgetLeftSent,
  /** False when this tab would not keep the waiting clips, so a reload would lose them. */
  laneKept: (): boolean => laneKept,
  /** Clips in this page's lane that have not been sent yet. */
  waitingCount: (): number => laneClips.length,
  /** True while the lane waits for the reader's word after a heavy clip was lost. */
  held: (): boolean => laneHold !== null,
  sendHeld,
  stopHeld,
  /** The queue on the server holding clips of this desk until the reader says; null when it holds none. */
  heldOnServer: (): ServerHold | null => serverHold,
  sendHeldOnServer,
  stopHeldOnServer,
  /**
   * "This page sends the work itself: …", when the last clips made here went
   * by the page's own lane for a reason the server or the page gave; null
   * otherwise.
   */
  sendsItself: (): string | null => fellBack,
  /** Hand-overs of this desk an earlier page in the tab left, which were never sent, each with why. */
  givenUp: (): GivenUp[] => givenUpHere,
  forgetGivenUp: forgetGivenUpHere,
}

/**
 * Stop one clip from outside the desk, as the section bar does. It is the
 * desk's own stop, so a clip still waiting in the lane is called off before
 * it is ever sent, where cancelling a prompt could not reach it: it has none.
 * A clip of the queue on the server is stopped there, whatever it is doing.
 */
export function stopVideoJob(id: string): void {
  void stopJob(id)
}

function dismissJob(id: string): void {
  const job = jobById(id)
  if (job?.runner) {
    dismissedHere.add(id)
    unfollowRunner(id)
    // Put away on the server too, so no other page shows it again. Only a
    // clip that has ended can be, and only one the server has.
    if (!unfinished(job) && listedRunner.has(id)) void dismissOnServer([id]).catch(() => false)
  }
  jobs = jobs.filter((j) => j.id !== id)
  announce()
}

async function stopJob(id: string): Promise<void> {
  const job = jobById(id)
  if (!job || !unfinished(job)) return
  if (job.runner) {
    patchJob(id, { cancelRequested: true, stage: 'Stopping' })
    // One the server has not listed is still in the tab's outbox, waiting its
    // turn or its answer. Taken out, it is never handed over again, by this
    // page or by the next one in the tab.
    const handing = listedRunner.has(id) ? undefined : outboxPending().find((g) => g.jobIds.includes(id))
    if (handing) withdraw(handing.groupId, [id])
    // No request ever carried it, so nothing is left to stop.
    if (handing && !handing.sent) {
      failFromRunner(id, new ComfyError(STOPPED_UNSENT, { cancelled: true }))
      return
    }
    // The server does the rest in whatever state the clip is in: never sent
    // when it still waits, cancelled the moment ComfyUI has it otherwise.
    // Its ending arrives through the queue like any other. Until the stop
    // lands it is kept in the tab, and asked again as soon as the server
    // lists the clip: one handed over without an answer may be there.
    keepStop(id)
    if (await stopOnServer(id).catch(() => false)) {
      forgetStop(id)
      return
    }
    // This page waits on no request for it (its answer came back pending, or
    // an earlier page in the tab sent it), so nothing else would settle it.
    if (handing && unanswered.has(handing.groupId) && !listedRunner.has(id)) stoppedUnanswered(id)
    return
  }
  if (!job.promptId) {
    // Still in flight to the queue. Mark it, and the queued handler stops it
    // the moment ComfyUI hands us an id. One still waiting its turn is never
    // sent at all.
    patchJob(id, { cancelRequested: true, stage: 'Stopping' })
    waiting.get(id)?.abort()
    return
  }
  patchJob(id, { cancelRequested: true, stage: 'Stopping' })
  try {
    // `false` means it had already finished, and the terminal event settles
    // it, or that ComfyUI does not know the id, which a restart does to
    // every job it had. Only a stop ComfyUI took is kept, and a second press
    // that finds the clip already gone does not undo the first (see fail).
    if (await cancelJob(job.promptId)) patchJob(id, { stopLanded: true })
  } catch (err) {
    patchJob(id, { error: (err as Error).message })
  }
}

/**
 * A clip of the queue stopped while its hand-over had no answer, which no
 * request of this page carries any more: shown stopped. Should the server
 * list it after all, its view takes the card's place (see fromOutbox), and
 * the stop kept for it is asked there.
 */
function stoppedUnanswered(id: string): void {
  fromOutbox.add(id)
  failFromRunner(id, new ComfyError(STOPPED_UNANSWERED, { cancelled: true }))
}

type StartOptions = {
  composition: Composition
  graph: ApiWorkflow
  familyLabel: string
  modelLabel: string
  /** The add-ons chained into the graph, for the record. */
  loras?: HistoryEntry['loras']
  /** Release ComfyUI's cached models immediately before queueing. See lib/clipMemory.ts. */
  release?: boolean
  /** A clip put back in the lane by a later page keeps its id and the time it was made. */
  id?: string
  startedAt?: number
  /** Said under the clip until its wait says something of its own. */
  note?: string
}

/** A clip that has not been heard of by ComfyUI yet. */
function newJob(c: LaneClip & { release: boolean }): VideoJob {
  return {
    id: c.id,
    promptId: null,
    status: 'submitting',
    familyLabel: c.familyLabel,
    modelLabel: c.modelLabel,
    prompt: c.composition.prompt,
    value: 0,
    max: 0,
    stage: 'Sending it to the press',
    previewUrl: null,
    startedAt: c.startedAt,
    ranAt: null,
    samplingAt: null,
    pass: null,
    pace: null,
    finishedAt: null,
    tookMs: null,
    error: null,
    fault: null,
    loras: c.loras,
    files: [],
    entryId: null,
    repeatOf: null,
    composition: c.composition,
    graph: c.graph,
    frames: c.composition.length ?? 0,
    fps: c.composition.fps ?? 0,
    queuePos: null,
    cancelRequested: false,
    stopLanded: false,
    sendingAs: null,
    release: c.release,
    waitNote: null,
    resumed: false,
  }
}

const sameFile = (a: FileRef, b: FileRef) =>
  a.filename === b.filename && (a.subfolder ?? '') === (b.subfolder ?? '') && (a.type || 'output') === (b.type || 'output')

/** The record that already names a file, if one does. */
function recordNaming(file: FileRef): HistoryEntry | null {
  return history.all().find((e) => sameFile(e.file, file) || !!e.files?.some((f) => sameFile(f, file))) ?? null
}

/** True for a file ComfyUI answered from its cache: the one an earlier run with the same settings wrote. */
const fromCache = (f: OutputFile) => f.cached === true

/**
 * When ComfyUI says it began and ended a run, from its own record. Given up
 * after a few seconds, so a slow answer cannot hold the lane.
 */
async function timesOfRun(promptId: string): Promise<{ startedAt: number | null; finishedAt: number | null } | null> {
  let timer: ReturnType<typeof setTimeout> | undefined
  const read = fetchPastRun(promptId).then(
    (run) => (run ? { startedAt: run.startedAt, finishedAt: run.finishedAt } : null),
    () => null,
  )
  const late = new Promise<null>((resolve) => {
    timer = setTimeout(() => resolve(null), 5000)
  })
  try {
    return await Promise.race([read, late])
  } finally {
    clearTimeout(timer)
  }
}

/** File a clip that landed, the one way for a clip sent here and for one followed after a reload. */
async function finish(id: string, files: OutputFile[]): Promise<void> {
  // When the page heard of the ending, which is when it ended only if the
  // page was awake to hear it. See below.
  let finishedAt = Date.now()
  const before = jobById(id)
  if (!before || handedOver) return
  const file = files.find((f) => f.kind === 'video') ?? files[0] ?? null
  // Asked again with the same settings, ComfyUI hands back the file it made
  // the first time without drawing anything, and filing it again would put a
  // second record on one file.
  const known = file && fromCache(file) ? recordNaming(file) : null
  let entryId: string | null = known?.id ?? null
  let tookMs: number | null = null
  if (file && !known) {
    // The end is ComfyUI's, from its record. A clip that landed while the
    // phone was locked is heard of when the page wakes, and one taken up
    // after a reload when the page comes back, which can be long after it
    // landed. Filed with the page's clock, such a clip took the whole wait
    // to make and was dated to the moment it was read, and the desk's
    // estimates, medians of these figures, crept up with each one.
    const past = before.promptId ? await timesOfRun(before.promptId) : null
    const current = jobById(id)
    if (!current || handedOver) return
    if (past?.finishedAt != null) finishedAt = past.finishedAt
    const ranAt = before.ranAt ?? past?.startedAt ?? null
    // The start may be the page's, as it heard it, and the end ComfyUI's; an
    // end before the start is the two clocks disagreeing and measures nothing.
    tookMs = ranAt !== null && finishedAt >= ranAt ? finishedAt - ranAt : null
    try {
      const entry = history.add(
        recordOf(current.composition, {
          file,
          files: files.length > 1 ? files : undefined,
          kind: file.kind,
          promptId: current.promptId ?? '',
          // 0 when it was not measured, which the estimates leave out.
          durationMs: tookMs ?? 0,
          seed: current.composition.seed,
          familyLabel: current.familyLabel,
          modelLabel: current.modelLabel,
          at: finishedAt,
          loras: current.loras,
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
    repeatOf: known?.id ?? null,
    finishedAt,
    tookMs,
    stage: 'Done',
    previewUrl: null,
    queuePos: null,
  })
  saveSent()
}

/**
 * Settle a clip that did not land. The same classification the Pictures desk
 * uses, so a clip that failed on a bad frame names the node instead of saying
 * something went wrong.
 */
function fail(id: string, err: unknown): void {
  if (handedOver) return
  const f = faultOf(err)
  const job = jobById(id)
  // A job the reader stopped and ComfyUI then forgot was stopped, not lost:
  // a dequeued prompt leaves no record, and blaming a restart for the
  // reader's own stop would be false. But only a stop that can explain it:
  // one ComfyUI took (stopLanded) while the clip waited. A stop pressed on a
  // clip frozen by a restart never reached a live job, and calling the loss
  // a stop hid the restart and skipped the hold below, so the next heavy
  // clip went into the restarted ComfyUI with nobody asked. A running clip
  // that is stopped ends on its interrupt, never as lost, and one ComfyUI
  // says has ended may have left its file (mayExist), which is for the
  // reader to look for.
  if (f.lost && !f.mayExist && job?.cancelRequested && job.stopLanded && job.status !== 'running') {
    patchJob(id, {
      status: 'cancelled',
      error: null,
      fault: { ...f, lost: false, cancelled: true },
      finishedAt: Date.now(),
      previewUrl: null,
      stage: 'Stopped',
      queuePos: null,
    })
    saveSent()
    return
  }
  // A clip taken up after a reload that ComfyUI never showed it had: the
  // page that sent it went while it was on its way (see NOT_SENT).
  const notSent = f.lost && !!job?.sendingAs
  patchJob(id, {
    status: f.cancelled ? 'cancelled' : 'error',
    error: f.message || 'Something went wrong.',
    fault: f,
    finishedAt: Date.now(),
    previewUrl: null,
    stage: f.cancelled ? 'Stopped' : notSent ? 'Not sent' : f.lost ? 'Lost' : 'Failed',
    queuePos: null,
  })
  saveSent()
  // Before the lane lets the next heavy clip go: see laneHold. Not for a clip
  // that may never have reached ComfyUI, which says nothing of its memory,
  // and never for a clip of the server's queue, which holds its own lane in
  // the same step as it records the loss.
  if (f.lost && job?.release && !notSent && !job.runner) holdLane()
}

function startJob(opts: StartOptions): string {
  const id = opts.id ?? globalThis.crypto?.randomUUID?.() ?? `job_${Date.now()}_${Math.random().toString(36).slice(2)}`
  const job: VideoJob = {
    ...newJob({
      id,
      startedAt: opts.startedAt ?? Date.now(),
      composition: opts.composition,
      graph: opts.graph,
      familyLabel: opts.familyLabel,
      modelLabel: opts.modelLabel,
      loras: opts.loras,
      release: !!opts.release,
    }),
    waitNote: opts.note ?? null,
  }

  // Taken in the same tick the job is made, so clips made together keep
  // their order in the lane, and saved for this tab before the desk hears of
  // the clip (see LANE_KEY).
  let leaveLane = () => {}
  let turn: Promise<void> = Promise.resolve()
  if (opts.release) {
    const place = joinLane()
    turn = place.turn
    leaveLane = place.leave
    joinSavedLane({
      id,
      startedAt: job.startedAt,
      composition: opts.composition,
      graph: opts.graph,
      familyLabel: opts.familyLabel,
      modelLabel: opts.modelLabel,
      ...(opts.loras ? { loras: opts.loras } : {}),
    })
  }

  jobs = [job, ...jobs]
  announce()

  const onEvent = (e: ComfyProgress) => {
    const current = jobById(id)
    if (!current) return
    if (e.phase === 'queued') {
      patchJob(id, { promptId: e.promptId, sendingAs: null, status: 'queued', stage: 'Queued' })
      // Kept for the tab as ComfyUI's now, so the next page follows it as a
      // clip that got there (SENT_KEY).
      saveSent()
      if (current.cancelRequested) {
        void cancelJob(e.promptId).then(
          (landed) => {
            if (landed) patchJob(id, { stopLanded: true })
          },
          () => undefined,
        )
      }
      // A release is only this clip's if nothing ran between it and the
      // prompt, and a reel shot or a picture sent from another desk or tab
      // in that moment spends it. ComfyUI applies a release sent while other
      // work is ahead after that work, so one sent now reaches this clip.
      else if (current.release) void releaseIfOthersAhead(e.promptId)
    } else if (e.phase === 'running') {
      // A heavy clip releases for itself on an empty queue, but something of
      // ours sent in the moment between its release and its prompt runs first
      // and leaves its models behind. ComfyUI applies a release after the job
      // it is running, so one sent now lands between this job and that clip.
      // A spare one costs only a reload. The server's clips are not counted:
      // the server releases for its own.
      if (
        current.status !== 'running' &&
        jobs.some((j) => j.id !== id && !j.runner && j.release && (j.status === 'queued' || j.status === 'submitting'))
      ) {
        void releaseComfyMemory()
      }
      const now = Date.now()
      const stage = stageOf(current.graph, e.node)
      const drawing = stage === 'Drawing' && e.max > 1
      const nodePass = passOf(current.graph, e.node)
      patchJob(id, {
        status: 'running',
        value: e.value,
        max: e.max,
        stage,
        pass: nodePass ?? current.pass,
        pace: nextPace(current, { pass: nodePass, drawing, value: e.value }, now),
        ranAt: current.ranAt ?? now,
        samplingAt: drawing && current.samplingAt === null ? now : current.samplingAt,
        queuePos: 0,
      })
      // Its start, for the page that may have to file it after a reload.
      if (current.ranAt === null) saveSent()
    } else if (e.phase === 'preview') {
      patchJob(id, { previewUrl: e.url })
    }
  }

  const queue = async (): Promise<OutputFile[]> => {
    if (opts.release) {
      // Its turn, then an empty queue, then the release, then the prompt, with
      // nothing in between: see releaseLane.
      const stop = new AbortController()
      waiting.set(id, stop)
      const stopped = () => new ComfyError('Stopped before it was sent.', { cancelled: true })
      try {
        if (jobs.some((j) => j.id !== id && !j.runner && j.release && unfinished(j))) {
          patchJob(id, {
            stage: 'Waiting its turn',
            waitNote: 'Waits for the clip before it to finish, so that one’s memory can be released before this starts.',
          })
        }
        if (!(await unlessStopped(turn, stop.signal))) throw stopped()
        // A hold can also land while this clip waits for the queue, when a
        // clip followed after a reload is lost, so it is asked again after.
        for (;;) {
          while (laneHold) {
            patchJob(id, {
              stage: 'Held',
              waitNote: 'Held until you say, because the heavy clip before it was lost.',
            })
            if (!(await unlessStopped(laneHold.promise, stop.signal))) throw stopped()
          }
          const idle = await waitForIdleComfy(stop.signal, (ahead) =>
            patchJob(id, {
              stage: ahead < 0 ? 'Waiting for ComfyUI' : 'Waiting for the press',
              waitNote:
                ahead < 0
                  ? 'ComfyUI is not answering; it may be restarting. This clip waits until it answers, then releases its memory and goes.'
                  : `ComfyUI has ${ahead} ${ahead === 1 ? 'job' : 'jobs'} to finish first. This clip waits for them, so the memory they hold can be released before it starts.`,
            }),
          )
          if (!idle) throw stopped()
          if (!laneHold) break
        }
        patchJob(id, { stage: 'Releasing memory', waitNote: null })
        await releaseComfyMemory()
        if (stop.signal.aborted) throw stopped()
      } finally {
        waiting.delete(id)
        // Out of the saved lane before the prompt goes, never after: a page
        // that went away in between would otherwise send it a second time.
        leaveSavedLane(id)
      }
      patchJob(id, { stage: 'Sending it to the press' })
    }
    // The prompt's id is made here and the clip kept for the tab under it
    // before the prompt goes, in the same turn it left the saved lane, so no
    // page can go in between. A page that goes before ComfyUI answers (the
    // phone throws the tab away just after Make, or a reload) then leaves the
    // next page an id to follow the clip by, where it had none: the clip
    // rendered unfollowed with no Stop, and a second Make rendered it twice.
    const promptId = newPromptId()
    patchJob(id, { sendingAs: promptId })
    saveSent()
    return run(opts.graph, onEvent, { promptId })
  }

  void queue()
    .then(
      (files) => finish(id, files),
      (err: unknown) => fail(id, err),
    )
    // Settled either way, so the next heavy clip may take its turn.
    .finally(() => leaveLane())

  return id
}

/** Said of a clip followed after a reload that ComfyUI no longer knows. */
const LOST_AFTER_RELOAD =
  'We lost track of this clip after the page reloaded. ComfyUI has no record of it any more, which usually means it restarted. If it finished first, its file may be on disk: open the Archive and press “Look for files with no record”.'

/**
 * Said of a clip taken up after a reload that was kept on its way and that
 * ComfyUI has nothing under. Whether it never got there or ComfyUI has
 * restarted since, no answer tells, so neither is claimed.
 */
const NOT_SENT =
  'This clip may never have reached ComfyUI. The page went away while it was being sent, and ComfyUI has nothing under its number, so nothing is running for it, and the desk will not send it again by itself. If ComfyUI did get it and has restarted since, its file may be on disk if it finished first: open the Archive and press “Look for files with no record”.'

/**
 * A followed prompt's ending, as run() would have ended: files, or the error
 * it rejects with. `onItsWay` is true while ComfyUI has not yet shown it has
 * the prompt at all.
 */
function outcomeOf(promptId: string, r: Followed, onItsWay: boolean): OutputFile[] {
  if (r.status === 'done') return r.files
  if (r.status === 'cancelled') throw new ComfyError('Job stopped. Nothing was saved.', { cancelled: true, promptId })
  if (r.status === 'error') throw new ComfyError(r.message, { promptId, node: r.node, nodeType: r.nodeType })
  throw new LostJob(onItsWay ? NOT_SENT : LOST_AFTER_RELOAD, promptId)
}

/**
 * Take up a clip an earlier page in this tab sent. It is a live job again,
 * with its prompt id, so the section bar counts it as this tab's and Stop
 * reaches it, and it is filed the way a clip sent from here is.
 */
function followSent(s: SentClip): void {
  if (jobs.some((j) => j.id === s.id || sentUnder(j) === s.promptId)) return
  // A heavy clip holds its place in the lane as it did on the page that sent
  // it, so clips taken up behind it wait for it to settle, and are held if it
  // is lost, rather than go the moment a restarted ComfyUI reads as empty.
  // It is in ComfyUI's queue already, so it takes no turn of its own.
  let leaveLane = () => {}
  if (s.release) leaveLane = joinLane().leave
  // Kept on its way: the page that sent it may have gone before the prompt
  // got there, so it is not called queued until ComfyUI shows it has it.
  const onItsWay = s.sending === true
  jobs = [
    {
      ...newJob(s),
      promptId: s.promptId,
      sendingAs: onItsWay ? s.promptId : null,
      status: onItsWay ? 'submitting' : 'queued',
      stage: onItsWay ? 'Asking ComfyUI whether it has it' : 'Queued',
      ranAt: s.ranAt,
      resumed: true,
    },
    ...jobs,
  ]
  saveSent()
  announce()
  // Seen waiting by this page, so the moment it is seen running is its start,
  // to within one ask of the queue. One already running when this page
  // loaded started earlier, and its start is read from ComfyUI's record.
  let seenWaiting = false
  void followPrompt(s.promptId, {
    onState: (state) => {
      const current = jobById(s.id)
      if (state === 'queued') seenWaiting = true
      if (!current || !unfinished(current)) return
      // Queued or running, ComfyUI has it: it got there.
      const arrived = current.sendingAs !== null
      const moved = current.status !== state
      if (moved) {
        patchJob(s.id, {
          status: state,
          sendingAs: null,
          stage: current.cancelRequested ? 'Stopping' : state === 'running' ? 'Running' : 'Queued',
          ...(state === 'running'
            ? { queuePos: 0, ranAt: current.ranAt ?? (seenWaiting ? Date.now() : null) }
            : {}),
        })
      } else if (arrived) patchJob(s.id, { sendingAs: null })
      if (arrived || (moved && state === 'running' && current.ranAt === null && seenWaiting)) saveSent()
    },
  })
    .then((r) => outcomeOf(s.promptId, r, !!jobById(s.id)?.sendingAs))
    .then(
      (files) => finish(s.id, files),
      (err: unknown) => fail(s.id, err),
    )
    .finally(() => leaveLane())
}

/** A clip the page sent, or will send, itself and that has not settled: not one of the server's queue. */
const sentByPage = (j: VideoJob) => !j.runner && unfinished(j)

/**
 * Read ComfyUI's own queue every five seconds while a clip is unfinished, so
 * the desk can say honestly how many clips are ahead of this one.
 *
 * Finding a job the server has forgotten is not done here. run() follows every
 * prompt it queued for that itself, settles one that finished from its
 * /history record so the clip is still filed, and rejects with a lost job when
 * there is no record at all, which the catch in startJob reports; followPrompt
 * does the same for a clip taken up after a reload. A second watch here raced
 * them and called a finished clip lost.
 *
 * Only for clips the page sent itself (sentByPage). The server says where
 * each of its own clips waits, and a phone showing only those has no reason
 * to keep asking.
 */
function managePoll(): void {
  const live = jobs.some(sentByPage)
  if (live && !pollTimer) pollTimer = setInterval(() => void reconcile(), 5000)
  if (!live && pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

/**
 * True while a read of the queue is out. A ComfyUI that stalls without
 * closing its socket answers nothing for as long as it stalls, and a new read
 * every five seconds on top of the last filled the browser's few connections
 * to this server, so the archive, the thumbnails and the plate stopped too.
 */
let reconciling = false

async function reconcile(): Promise<void> {
  if (reconciling || !jobs.some(sentByPage)) return
  reconciling = true
  try {
    let listed: ServerJob[]
    try {
      const page = await listJobs({ status: ['pending', 'in_progress'], limit: 100 })
      listed = page.jobs
    } catch {
      return // the connection notice covers an unreachable server
    }

    // Read again after the await: a job may have settled while the list was on
    // its way, and its place in line is then nobody's business. A clip that is
    // drawing has no place in line; its progress events keep it at 0.
    for (const job of jobs.filter(sentByPage)) {
      if (!job.promptId || job.status === 'running') continue
      const ahead = jobsAhead(listed, job.promptId)
      if (ahead !== null && ahead !== job.queuePos) patchJob(job.id, { queuePos: ahead })
    }
  } finally {
    reconciling = false
  }
}

/**
 * How many jobs ComfyUI will run before this one, or null when it does not
 * list it. /api/jobs sorts newest first, so a job's index in the list counts
 * the jobs queued after it. ComfyUI takes the lowest priority number first
 * (its queue is a heap on that number), and whatever is running goes first.
 */
function jobsAhead(listed: readonly ServerJob[], promptId: string): number | null {
  const mine = listed.find((j) => j.id === promptId)
  if (!mine) return null
  if (mine.status === 'in_progress') return 0
  const order = (j: ServerJob) => j.priority ?? j.create_time ?? 0
  return listed.filter(
    (j) => j.id !== promptId && (j.status === 'in_progress' || (j.status === 'pending' && order(j) < order(mine))),
  ).length
}

// ---------------------------------------------------------------------------
// The queue on the server
// ---------------------------------------------------------------------------
//
// A clip waiting in the page's lane is sent only while the page is awake to
// send it, and a phone that locks suspends the page. Where the SwitchGen
// server runs its queue, Make hands the clips to it in one request instead,
// and the page only watches. The server keeps one heavy lane for every
// device and every desk, waits for ComfyUI's queue to empty, releases, sends
// and files, whatever the page is doing. It files each clip once, under the
// clip's own id, and the archive brings the record to every page, so nothing
// here files anything.
//
// Every clip of this desk the queue lists is shown here, from any device and
// after any reload: the page that made a clip may be long gone, and the clip
// is not lost from sight with it.

/**
 * The queue's hold, as this desk shows it: why it holds, the clip whose loss
 * set it, and how many waiting jobs it keeps back, of this desk and of the
 * others (one hold covers the reel's heavy shots too, and after a reboot, or
 * a spell with the queue off, everything). Counted as the queue counts what
 * a word on the hold answers for, and as the shell's hold notice counts it.
 */
export type ServerHold = {
  why: NonNullable<RunnerLane['held']>['why']
  jobId: string | null
  clips: number
  others: number
  /**
   * When the server set this hold. The word given on it names it, so a word
   * given on a notice that is out of date is refused rather than taken for a
   * newer hold the reader has not seen.
   */
  since: number
}

/** One clip as Make built it, ready to go by either road. */
export type PlannedClip = {
  /** The composition as sent, with the positive prompt the graph carries. */
  composition: Composition
  graph: ApiWorkflow
  familyLabel: string
  modelLabel: string
  loras?: HistoryEntry['loras']
  /** Needs ComfyUI's memory released before it starts: see lib/clipMemory.ts. */
  release: boolean
}

/** Where one press of Make went. `pending` says the server has not answered yet. */
export type SentClips =
  | { road: 'server'; ids: string[]; pending: boolean }
  | { road: 'page'; ids: string[] }
  | { road: 'refused'; error: string }

/** The stage a clip shows while it is being handed over. */
const HANDING = 'Handing it to the server'

const NOT_ANSWERED =
  'The SwitchGen server has not answered yet, so this clip may not have reached it. It shows here as soon as the server lists it.'

/** Under a clip an earlier page in this tab handed over without hearing back, until the server lists it. */
const HANDED_BEFORE =
  'An earlier page in this tab handed this clip over and did not hear back, so this page asks the server again. It shows here as soon as the server lists it.'

/**
 * Under a clip Make made on an earlier page in this tab, which went before its
 * turn to be handed over came: nothing of it was sent.
 */
const PRESSED_BEFORE =
  'Make was pressed on an earlier page in this tab, which went before it handed this clip over, so this page hands it over now. It shows here as soon as the server lists it.'

/** A clip stopped before any request carried it to the server. */
const STOPPED_UNSENT = 'Stopped before it was sent.'

/**
 * A clip stopped while its hand-over had no answer: the server may have it,
 * and then it is asked to stop it as soon as it lists it (see stopAgain).
 */
const STOPPED_UNANSWERED = 'Stopped before the server answered for it.'

/**
 * Why a press made while ComfyUI was not answering, on the word that the
 * queue on the server would take it, is refused after all: the page would
 * have to send it itself, and cannot until ComfyUI answers.
 */
const OFFLINE_NO_QUEUE = 'ComfyUI is not answering, and the queue on the server is not taking these clips, so nothing can be queued.'

/** Why clips made while the page's lane has work go by that lane, as fallbackLine takes a reason. */
const BEHIND_HERE = 'the clips already waiting in this page go first, and new ones wait behind them here'

/** How a clip on the server can end. Every other status is under way. */
const SERVER_ENDED: ReadonlySet<RunnerJob['status']> = new Set(['done', 'failed', 'stopped', 'lost', 'unsent', 'skipped'])

/** Clips of the queue followed from here, so each is followed once and can be let go. */
const followingRunner = new Map<string, AbortController>()
/**
 * Clips the queue has listed. A clip it has stopped listing is then told
 * from one being handed over, which it has not listed yet.
 */
const listedRunner = new Set<string>()
/** Clips put away here, kept off the desk while the server's list catches up. */
const dismissedHere = new Set<string>()
/**
 * Stops that did not reach the server, asked again as soon as it lists the
 * clip. Kept in the tab, so a page the browser throws away before the server
 * lists a clip the reader stopped leaves the stop to the next page: the
 * server may have the clip even when it never answered for it.
 */
const STOPS_KEY = 'switchgen.videostops.v1'

function readStops(): Set<string> {
  try {
    const list: unknown = JSON.parse(tabStore.get(STOPS_KEY) ?? '[]')
    return new Set(Array.isArray(list) ? list.filter((id): id is string => typeof id === 'string' && id.length <= 64) : [])
  } catch {
    return new Set()
  }
}

let stopAgain = readStops()

function keepStops(): void {
  // A handful at most: each is a clip stopped while the server had not listed it.
  if (stopAgain.size > 50) stopAgain = new Set([...stopAgain].slice(-50))
  if (stopAgain.size) tabStore.set(STOPS_KEY, JSON.stringify([...stopAgain]))
  else tabStore.remove(STOPS_KEY)
}

function keepStop(id: string): void {
  if (stopAgain.has(id)) return
  stopAgain.add(id)
  keepStops()
}

/** @returns whether a stop was waiting for the clip. */
function forgetStop(id: string): boolean {
  if (!stopAgain.delete(id)) return false
  keepStops()
  return true
}

/** Ask the server to stop a clip it lists, keeping the stop for the next change should it not land. */
function stopListed(id: string): void {
  forgetStop(id)
  void stopOnServer(id).then(
    (landed) => {
      if (!landed) keepStop(id)
    },
    () => keepStop(id),
  )
}

/**
 * Clips shown before the server listed them, whose cards the server's view of
 * them replaces once it does: those shown from the tab's outbox
 * (takeUpOutbox), and those stopped while their hand-over had no answer
 * (stoppedUnanswered), which the server may have after all.
 */
const fromOutbox = new Set<string>()
/**
 * Hand-overs the server has not answered yet, by group, with the clips shown
 * for each. The tab keeps the request and sends it again when the server is
 * next heard from; if that is given up, these clips say so.
 */
const unanswered = new Map<string, string[]>()
/** The queue's hold over clips of this desk, or null. Replaced only when it changes, for useSyncExternalStore. */
let serverHold: ServerHold | null = null
/** Hand-overs of this desk from an earlier page in the tab that were never sent. Replaced only when they change. */
let givenUpHere: GivenUp[] = []
/** "This page sends the work itself: …", when the last clips made here went by the page's lane for a reason. */
let fellBack: string | null = null
/** One hand-over at a time, so clips made by two quick presses reach the server in the order they were made. */
let handing: Promise<unknown> = Promise.resolve()

let device: string | null = null
const thisDevice = (): string => (device ??= deviceId())

const cap = (s: string, n: number) => (s.length > n ? s.slice(0, n) : s)

/**
 * A clip waiting on the server to be sent. Not one still being handed over,
 * which the server may not have yet.
 */
const waitsOnServer = (j: VideoJob): boolean => !!j.runner && unfinished(j) && !j.promptId && listedRunner.has(j.id)

function runnerJobOf(id: string): RunnerJob | undefined {
  return runnerStore.snapshot().jobs.find((j) => j.id === id)
}

/** A pass worth naming: one of two or more, never "pass 1 of 1". */
function passFrom(p: { pass?: SamplingPass | null } | undefined): SamplingPass | null {
  const pass = p?.pass
  return pass && pass.count > 1 ? { index: pass.index, count: pass.count } : null
}

/** What a clip on the server is doing, in the desk's words, until it ends. */
function stageOnServer(
  rj: RunnerJob,
  p: RunnerProgress | undefined,
  stopping: boolean,
): Pick<VideoJob, 'stage' | 'waitNote'> {
  // Filing goes on after a late stop: the file was made, and it is kept. So
  // it says Filing, where waitLine would say Stopping.
  if (rj.status === 'filing') return { stage: 'Filing', waitNote: null }
  if (stopping) return { stage: 'Stopping', waitNote: null }
  if (rj.status === 'running') return { stage: p?.classType ? stageFor(p.classType) : 'Running', waitNote: null }
  const line = waitLine(rj)
  return { stage: line.stage, waitNote: rj.status === 'queued' ? null : line.note }
}

/** A clip that did not land, as fail() settles one. */
function endedBy(rj: RunnerJob | undefined, f: Fault): Pick<VideoJob, 'status' | 'error' | 'fault' | 'stage'> {
  return {
    status: f.cancelled ? 'cancelled' : 'error',
    error: f.message || 'Something went wrong.',
    fault: f,
    stage: f.cancelled ? 'Stopped' : rj?.status === 'unsent' ? 'Not sent' : f.lost ? 'Lost' : 'Failed',
  }
}

/** A clip of the queue this page did not make, as the desk shows its clips. */
function jobFromRunner(rj: RunnerJob, p: RunnerProgress | undefined): VideoJob {
  const meta = isObj(rj.meta) ? rj.meta : {}
  const frames = typeof meta.frames === 'number' ? meta.frames : 0
  const fps = typeof meta.fps === 'number' ? meta.fps : 0
  const shown = reportedOf(rj, p)
  const job: VideoJob = {
    ...newJob({
      id: rj.id,
      startedAt: rj.createdAt,
      composition: newComposition('video', { prompt: rj.prompt, length: frames || null, fps: fps || null }),
      graph: {},
      familyLabel: rj.label,
      modelLabel: '',
      release: rj.heavy,
    }),
    promptId: rj.promptId,
    status: shown.status,
    value: shown.value,
    max: shown.max,
    pass: passFrom(p),
    ranAt: rj.ranAt,
    files: rj.files,
    entryId: rj.entryId,
    repeatOf: rj.repeatOf,
    frames,
    fps,
    cancelRequested: rj.stopRequested,
    stopLanded: rj.stopLanded,
    runner: true,
    sentHere: rj.device === thisDevice(),
    adopted: true,
  }
  if (rj.status === 'done') {
    return {
      ...job,
      stage: 'Done',
      finishedAt: rj.finishedAt ?? rj.endedAt,
      // 0 is a time the server did not measure, not a clip made in no time.
      tookMs: rj.durationMs > 0 ? rj.durationMs : null,
    }
  }
  if (SERVER_ENDED.has(rj.status)) {
    return { ...job, ...endedBy(rj, faultOf(runnerFault(rj))), finishedAt: rj.endedAt ?? rj.createdAt }
  }
  return { ...job, ...stageOnServer(rj, p, rj.stopRequested) }
}

/** Patch only what differs, so the queue's frequent changes do not redraw the desk for nothing. */
function patchIfChanged(id: string, patch: Partial<VideoJob>): void {
  const current = jobById(id)
  if (!current) return
  const keys = Object.keys(patch) as (keyof VideoJob)[]
  if (keys.some((k) => current[k] !== patch[k])) patchJob(id, patch)
}

/** How far along a clip that has not ended is, so what the queue says never moves one back. */
const ALONG: Record<'submitting' | 'queued' | 'running', number> = { submitting: 0, queued: 1, running: 2 }

/**
 * Bring a clip under way into line with the queue's list. Progress, stage
 * and preview of a running clip come from follow(), report by report; its
 * ending too, which is why nothing here settles a clip.
 */
function applyRunner(current: VideoJob, rj: RunnerJob, p: RunnerProgress | undefined): void {
  if (SERVER_ENDED.has(rj.status) || !unfinished(current)) return
  const cancelRequested = current.cancelRequested || rj.stopRequested
  const patch: Partial<VideoJob> = {
    cancelRequested,
    stopLanded: rj.stopLanded,
    // The server may count a clip heavier than the page did, never lighter.
    release: rj.heavy,
    sentHere: rj.device === thisDevice(),
  }
  if (rj.promptId !== null) patch.promptId = rj.promptId
  // As ComfyUI said it began, when this page did not hear it itself.
  if (current.ranAt === null && rj.ranAt !== null) patch.ranAt = rj.ranAt
  const status = rj.status === 'running' || rj.status === 'filing' ? 'running' : rj.status === 'queued' ? 'queued' : 'submitting'
  const from = ALONG[current.status as keyof typeof ALONG]
  // A progress report can arrive before the list says the clip runs.
  if (ALONG[status] >= from) {
    patch.status = status
    if (status === 'running') patch.queuePos = 0
    if (status !== 'running' || current.status !== 'running' || rj.status === 'filing') {
      Object.assign(patch, stageOnServer(rj, p, cancelRequested))
    }
  }
  patchIfChanged(current.id, patch)
}

/** Take a clip off the desk without a word to the server. */
function dropRunnerJob(id: string): void {
  unfollowRunner(id)
  if (!jobById(id)) return
  jobs = jobs.filter((j) => j.id !== id)
  announce()
}

function unfollowRunner(id: string): void {
  followingRunner.get(id)?.abort()
  followingRunner.delete(id)
}

/** Put clips into the desk's list, newest first, where they belong by when they were made. */
function placeByAge(list: readonly VideoJob[], add: readonly VideoJob[]): VideoJob[] {
  const out = [...list]
  for (const job of add) {
    const at = out.findIndex((j) => j.startedAt < job.startedAt)
    if (at < 0) out.push(job)
    else out.splice(at, 0, job)
  }
  return out
}

/**
 * The queue's hold as this desk shows it, or null when it keeps nothing back.
 *
 * Counted by the shell's own count (heldCounts), which counts as the queue's
 * covers() does and so as Send and Stop answer: every waiting job when the
 * hold is on everything, every waiting heavy one otherwise, whatever its wait
 * says. Its wait is no guide: the next job of a group says it waits for the
 * one before, and the job at the head of the lane keeps its own wait, yet the
 * hold keeps both back and Stop stops both.
 */
function holdOf(snap: RunnerSnapshot): ServerHold | null {
  const held = snap.lane.held
  if (!held) return null
  const counts = heldCounts(snap)
  const clips = counts.video
  const others = Object.values(counts).reduce((sum, n) => sum + n, 0) - clips
  return clips + others > 0 ? { why: held.why, jobId: held.jobId, clips, others, since: held.since } : null
}

const sameHold = (a: ServerHold | null, b: ServerHold | null) =>
  a === b ||
  (!!a &&
    !!b &&
    a.why === b.why &&
    a.jobId === b.jobId &&
    a.clips === b.clips &&
    a.others === b.others &&
    a.since === b.since)

/**
 * Mirror the queue's clips of this desk into the desk's list: take up the
 * ones it has not shown yet, from any device, keep the ones under way in line
 * with it, and let go of those the queue no longer lists or that were put
 * away on some device.
 */
function syncRunner(): void {
  const snap = runnerStore.snapshot()
  const listed = new Set<string>()
  const taken: VideoJob[] = []
  for (const rj of snap.jobs) {
    if (rj.desk !== 'video') continue
    listed.add(rj.id)
    // A stop pressed in this tab before the server listed the clip, by this
    // page or one before it. One that has ended wants none. Not while the
    // queue is off: it answers no stop then, and one asked again at every
    // change would only be refused. The change that says it runs again asks it.
    const stopAsked = stopAgain.has(rj.id)
    if (SERVER_ENDED.has(rj.status)) forgetStop(rj.id)
    else if (stopAsked && !queueOff(snap)) stopListed(rj.id)
    const current = jobById(rj.id)
    if (rj.dismissed || dismissedHere.has(rj.id)) {
      // Put away on some device, which the server allows only once a clip
      // has ended. A card here still under way is one whose ending never
      // reached this page: one it handed over without hearing back, or one
      // shown from the tab's outbox. Nothing else would settle it (the list
      // names it, so it is not given up, and a clip put away is not
      // followed), so it goes with the rest.
      if (current?.runner && (!unfinished(current) || SERVER_ENDED.has(rj.status))) {
        fromOutbox.delete(rj.id)
        dropRunnerJob(rj.id)
      }
      continue
    }
    listedRunner.add(rj.id)
    if (!current || fromOutbox.delete(rj.id)) {
      // A clip taken up from the list, or one whose card was shown before
      // the server listed it and knew only the label, or said stopped before
      // an answer came: the server's view takes its place, keeping a Stop
      // pressed on it meanwhile. Even one said not sent: the server has it
      // after all.
      const job = jobFromRunner(rj, snap.progress[rj.id])
      const stopping = unfinished(job) && (stopAsked || !!current?.cancelRequested)
      const shown = stopping ? { ...job, cancelRequested: true, ...stageOnServer(rj, snap.progress[rj.id], true) } : job
      if (current) {
        jobs = jobs.map((j) => (j.id === rj.id ? shown : j))
        announce()
      } else taken.push(shown)
      if (unfinished(job)) followRunner(rj.id)
      continue
    }
    // Ids are random, so one the page gave a clip of its own is never the server's.
    if (!current.runner || !unfinished(current)) continue
    applyRunner(current, rj, snap.progress[rj.id])
    followRunner(rj.id)
  }
  // Only a list the server answered with is taken at its word about what is
  // missing from it; one not in yet, or from a server whose queue is not
  // running, says nothing. An ended clip it no longer lists was put away or
  // cleared out on the server. One under way is follow()'s to settle, which
  // asks the server once more before it calls the clip gone.
  if (snap.connected && snap.available) {
    for (const j of [...jobs]) {
      if (j.runner && !unfinished(j) && !listed.has(j.id) && listedRunner.has(j.id)) dropRunnerJob(j.id)
    }
  }
  if (taken.length) {
    jobs = placeByAge(jobs, taken)
    announce()
  }
  for (const [group, ids] of unanswered) {
    if (ids.every((id) => listed.has(id))) unanswered.delete(group)
  }
  settleGivenUp()
  const hold = holdOf(snap)
  if (!sameHold(hold, serverHold)) {
    serverHold = hold
    announce()
  }
}

/**
 * Hand-overs of this desk the tab sent again and then gave up on, as too old
 * or refused. A clip still shown from this page says so itself; one from an
 * earlier page in the tab, which this page never showed, is kept for the
 * desk's notice.
 */
function settleGivenUp(): void {
  const given = givenUpBatches().filter((g) => g.desk === 'video')
  const handled = new Set<string>()
  const onServer = new Set(runnerStore.snapshot().jobs.map((j) => j.groupId))
  for (const g of given) {
    const ids = unanswered.get(g.groupId)
    // One the server lists after all took the hand-over and lost only its
    // answer: its clips show from the list, and nothing is said of it.
    if (!ids && !onServer.has(g.groupId)) continue
    unanswered.delete(g.groupId)
    handled.add(g.groupId)
    for (const id of ids ?? []) {
      const current = jobById(id)
      if (!current || !unfinished(current) || listedRunner.has(id)) continue
      patchJob(id, {
        status: 'error',
        error: g.line,
        fault: faultOf(new ComfyError(g.line)),
        stage: 'Not sent',
        finishedAt: Date.now(),
        previewUrl: null,
        queuePos: null,
        waitNote: null,
      })
    }
  }
  // Said on the clips, so not again in the notice. Put away once the store
  // has finished telling of this change, since doing so tells of another.
  if (handled.size) queueMicrotask(() => handled.forEach((g) => forgetGivenUp(g)))
  const rest = given.filter((g) => !handled.has(g.groupId))
  const same = rest.length === givenUpHere.length && rest.every((g, i) => g.groupId === givenUpHere[i]?.groupId)
  if (!same) {
    givenUpHere = rest
    announce()
  }
}

/** The reader has read that a hand-over from an earlier page was never sent. */
function forgetGivenUpHere(groupId: string): void {
  forgetGivenUp(groupId)
}

/**
 * Clips an earlier page in this tab handed over without hearing back, or
 * made and went before handing over, which the tab's outbox keeps and sends
 * as the page loads. Each is shown as being handed over, as the page that
 * made it showed it, until the server lists it (syncRunner then puts the
 * server's view in its place) or it is given up (settleGivenUp then says so
 * on it). Until then only the label is known here: the rest of what was
 * handed over is in the outbox.
 */
function takeUpOutbox(): void {
  const shown: VideoJob[] = []
  for (const g of outboxPending()) {
    if (g.desk !== 'video') continue
    const ids = g.jobIds.filter((id) => !jobById(id) && !runnerJobOf(id))
    if (!ids.length) continue
    unanswered.set(g.groupId, [...(unanswered.get(g.groupId) ?? []), ...ids])
    // The newest first, as handOver shows one press.
    for (const id of [...ids].reverse()) {
      fromOutbox.add(id)
      shown.push({
        ...newJob({
          id,
          startedAt: g.at,
          composition: newComposition('video'),
          graph: {},
          familyLabel: g.label,
          modelLabel: g.label,
          release: false,
        }),
        stage: HANDING,
        waitNote: g.sent ? HANDED_BEFORE : PRESSED_BEFORE,
        runner: true,
        sentHere: true,
        handedBefore: true,
      })
    }
  }
  if (!shown.length) return
  jobs = placeByAge(jobs, shown)
  announce()
}

/**
 * Follow one clip of the queue to its ending. Started once the store has
 * finished telling its listeners of the change that listed the clip, never
 * from inside that telling.
 */
function followRunner(id: string): void {
  if (followingRunner.has(id)) return
  const stop = new AbortController()
  followingRunner.set(id, stop)
  const done = () => {
    if (followingRunner.get(id) === stop) followingRunner.delete(id)
  }
  void Promise.resolve()
    .then(() => (stop.signal.aborted ? null : follow(id, (e) => onRunnerEvent(id, e), { signal: stop.signal })))
    .then(
      (ending) => {
        done()
        if (ending) finishFromRunner(id, ending)
      },
      (err: unknown) => {
        done()
        if (stop.signal.aborted) return
        // A watch that broke while the server still has the clip in hand
        // says nothing of the clip: the queue's next change follows it again.
        const rj = runnerJobOf(id)
        if (rj && !SERVER_ENDED.has(rj.status)) return
        failFromRunner(id, err)
      },
    )
}

/**
 * The page's own onEvent, for a clip the server sent: the same stage, pass
 * and pace. The stage and the pass come from the class of the node the
 * server reports, since a clip made on another device has no graph here.
 * Nothing is cancelled or released from here; the server does both.
 */
function onRunnerEvent(id: string, e: FollowEvent): void {
  const current = jobById(id)
  if (!current || !unfinished(current)) return
  if (e.phase === 'queued') {
    patchJob(id, {
      promptId: e.promptId,
      sendingAs: null,
      // A report of progress can come before the word that it was queued.
      ...(current.status === 'running' ? {} : { status: 'queued' as const, stage: current.cancelRequested ? 'Stopping' : 'Queued' }),
      waitNote: null,
    })
  } else if (e.phase === 'running') {
    const now = Date.now()
    const stage = e.classType ? stageFor(e.classType) : stageOf(current.graph, e.node)
    const drawing = stage === 'Drawing' && e.max > 1
    const nodePass = passFrom(e) ?? passOf(current.graph, e.node)
    patchJob(id, {
      status: 'running',
      value: e.value,
      max: e.max,
      stage,
      pass: nodePass ?? current.pass,
      pace: nextPace(current, { pass: nodePass, drawing, value: e.value }, now),
      ranAt: current.ranAt ?? now,
      samplingAt: drawing && current.samplingAt === null ? now : current.samplingAt,
      queuePos: 0,
      waitNote: null,
    })
  } else if (e.phase === 'preview') {
    patchJob(id, { previewUrl: e.url })
  }
}

/**
 * A clip the server filed. Its record reaches every page through the
 * archive, so nothing is added here. The time is ComfyUI's own, from the
 * start of the run to its end; 0 means it was not measured, and is shown as
 * no time at all.
 */
function finishFromRunner(id: string, r: FollowResult): void {
  const current = jobById(id)
  if (!current || !unfinished(current)) return
  patchJob(id, {
    status: 'done',
    files: r.files,
    entryId: r.entryId,
    repeatOf: r.repeatOf,
    finishedAt: r.finishedAt,
    tookMs: r.durationMs || null,
    stage: 'Done',
    previewUrl: null,
    queuePos: null,
    waitNote: null,
  })
}

/**
 * A clip of the queue that did not land. Never holds the page's lane: the
 * server holds its own, in the same step as it records the loss, for every
 * device at once.
 */
function failFromRunner(id: string, err: unknown): void {
  unfollowRunner(id)
  const current = jobById(id)
  if (!current || !unfinished(current)) return
  const rj = runnerJobOf(id)
  patchJob(id, {
    ...endedBy(rj, faultOf(err)),
    finishedAt: rj?.endedAt ?? Date.now(),
    previewUrl: null,
    queuePos: null,
    waitNote: null,
  })
}

/**
 * The reader's word on the queue's hold, from this desk. It answers for every
 * job the hold covers, and names the hold the desk showed (by default the one
 * it shows now), so the server refuses it if another hold has taken that
 * one's place meanwhile; the page then reads the server's state again, and
 * the desk shows the new hold for the reader's word on it.
 */
function sendHeldOnServer(since: number | undefined = serverHold?.since): void {
  if (since === undefined) return
  void laneWord('send', since).catch(() => false)
}

function stopHeldOnServer(since: number | undefined = serverHold?.since): void {
  if (since === undefined) return
  void laneWord('stop', since).catch(() => false)
}

/**
 * Send the clips one press of Make built: to the queue on the server where
 * it runs, else by the page's own lane, exactly as without a queue.
 *
 * Clips the page is still sending itself go first, and clips made behind
 * them wait behind them in the same lane. A lane in the page and a lane on
 * the server would each wait for an empty queue, release and send, and two
 * heavy clips could then run together.
 *
 * Make calls it; it is exported for the desk's tests, which cannot mount the
 * desk.
 */
// oxlint-disable-next-line react/only-export-components -- exported for the desk's tests, which cannot mount the desk
export async function sendClips(clips: readonly PlannedClip[], opts: SendOptions = {}): Promise<SentClips> {
  if (!clips.length) return { road: 'page', ids: [] }
  let offer: { ok: boolean; reason: string | null }
  try {
    offer = await runnerAvailable('video')
  } catch {
    offer = { ok: false, reason: null }
  }
  const behindHere = laneClips.length > 0 || jobs.some(sentByPage)
  if (!offer.ok || behindHere) {
    if (pageCannotSend(opts)) return { road: 'refused', error: OFFLINE_NO_QUEUE }
    const why = !offer.ok ? offer.reason : BEHIND_HERE
    fellBack = why ? fallbackLine(why) : null
    return { road: 'page', ids: clips.map((c) => startJob(c)) }
  }
  fellBack = null
  return handOver(clips, opts)
}

/** How Make pressed. */
type SendOptions = {
  /**
   * Pressed while ComfyUI was not answering, on the word that the queue on
   * the server would take the clips (queueTakesClips). Should they go by the
   * page after all, they are refused as the press would have been.
   */
  offline?: boolean
}

/** A press made while ComfyUI was not answering that would go by the page, while it still is not. */
const pageCannotSend = (opts: SendOptions): boolean => !!opts.offline && connectionState() === 'closed'

/**
 * Whether the queue on the server takes the next clips made here, as far as
 * the page can tell without asking (sendClips asks): the server said its
 * queue takes this desk's clips, the queue's own state, as the page last
 * heard it, says it is running, and nothing the page sends itself is ahead of
 * them. Then a clip made while ComfyUI is not answering waits on the server,
 * which sends it once ComfyUI answers, and Make is not refused for it. Not
 * sure is no.
 */
function queueTakesClips(caps: ServerCapabilities | null): boolean {
  if (!caps?.runner || !caps.runnerDesks.includes('video')) return false
  return queueRunning() && !laneClips.length && !jobs.some(sentByPage)
}

/** The queue's state, as the page last heard it, says it is running. */
function queueRunning(): boolean {
  const s = runnerStore.snapshot()
  return s.connected && s.available && s.boot !== ''
}

/**
 * The server has said where its queue stands, and it is not running (turned
 * off, standing back for another server, or stopped after faults). It still
 * lists the clips it keeps, as they were last saved, but takes no word on
 * them: a stop, or a word on its hold, is refused until it runs again.
 */
const queueOff = (s: RunnerSnapshot): boolean => s.boot !== '' && !s.available

/** Why the queue on the server is not running, in its own words; null while it runs, or before it has said. */
function queueOffReason(): string | null {
  const s = runnerStore.snapshot()
  return queueOff(s) ? (s.reason ?? QUEUE_NOT_RUNNING) : null
}

const QUEUE_NOT_RUNNING = 'The queue on the server is not running.'

/**
 * A clip the queue on the server keeps while it is off: shown as it was last
 * saved, which nothing here can move on or stop. Not one this page is still
 * handing over, which the server has not listed.
 */
const parkedOnServer = (j: VideoJob): boolean => !!j.runner && unfinished(j) && listedRunner.has(j.id)

/** One request to the queue for the clips of one press, shown on the desk from the moment of the press. */
async function handOver(clips: readonly PlannedClip[], opts: SendOptions = {}): Promise<SentClips> {
  runnerStore.start()
  const startedAt = Date.now()
  const placed = clips.map((clip) => ({ clip, id: newPromptId() }))
  const ids = placed.map((p) => p.id)
  // Shown at once, as a clip the page sends is: the answer can take a while,
  // and the reader has pressed Make. The newest first, as startJob puts them.
  const shown = placed.map(
    ({ clip, id }): VideoJob => ({ ...newJob({ id, startedAt, ...clip }), stage: HANDING, runner: true, sentHere: true }),
  )
  jobs = [...shown.reverse(), ...jobs]
  announce()

  const first = clips[0]!
  const group = newPromptId()
  const body: SubmitBody = {
    v: 1,
    group: {
      id: group,
      desk: 'video',
      kind: 'clips',
      label: cap(first.modelLabel || first.familyLabel, 200),
      device: thisDevice(),
    },
    jobs: placed.map(({ clip, id }) => ({
      id,
      label: cap(clip.modelLabel || clip.familyLabel, 200),
      prompt: cap(clip.composition.prompt, 4000),
      kind: 'video',
      primary: 'video',
      // A family that writes its clip as frames has no video file; its first
      // file is what it made.
      orFirst: true,
      // A clip that wrote no file is done with nothing to show, as a clip the
      // page sent is.
      noFile: 'done',
      heavy: clip.release,
      graph: clip.graph,
      record: recordTemplate(clip.composition, {
        seed: clip.composition.seed,
        familyLabel: clip.familyLabel,
        modelLabel: clip.modelLabel,
        ...(clip.loras ? { loras: clip.loras } : {}),
      }),
      meta: { frames: clip.composition.length ?? 0, fps: clip.composition.fps ?? 0 },
    })),
  }

  // Kept in the tab's outbox now, before any request goes: a press made while
  // an earlier hand-over still waits for its answer (which can take the whole
  // time a phone sleeps) waits its turn below, and a tab the browser throws
  // away meanwhile would otherwise lose it without a word. The next page in
  // the tab then shows it and hands it over, in the order the presses were
  // made. A clip stopped before its turn is taken out of it (see stopJob),
  // and a press whose every clip was stopped is not handed over at all.
  stageHandOver(body)
  const turn = handing.then(() => (ids.every((id) => jobById(id)?.cancelRequested) ? null : submitGroup(body)))
  handing = turn.catch(() => undefined)
  let r: SubmitResult | null
  try {
    r = await turn
  } catch {
    // Never sent by the page instead: the request may have reached the
    // server, and a clip the server has would then run twice.
    r = { ok: false, fallback: false, pending: true }
  }
  if (!r) return { road: 'server', ids: [], pending: false }

  if (r.ok) {
    for (const answered of r.jobs) {
      const current = jobById(answered.id)
      if (current) applyRunner(current, runnerJobOf(answered.id) ?? answered, undefined)
    }
    // The server's group names every clip it keeps of the press. One it does
    // not name was stopped and taken out before any request that reached it.
    const taken = new Set(r.group.jobIds)
    for (const id of ids) {
      if (taken.has(id) || !jobById(id)?.cancelRequested) continue
      forgetStop(id)
      failFromRunner(id, new ComfyError(STOPPED_UNSENT, { cancelled: true }))
    }
    if (![...taken].every((id) => runnerJobOf(id))) void runnerStore.refresh().catch(() => undefined)
    return { road: 'server', ids, pending: false }
  }

  if (r.fallback) {
    // Nothing was stored. A press the page could not have sent itself is
    // refused as it would have been, and the clips go.
    if (pageCannotSend(opts)) {
      for (const id of ids) forgetStop(id)
      jobs = jobs.filter((j) => !ids.includes(j.id))
      announce()
      return { road: 'refused', error: OFFLINE_NO_QUEUE }
    }
    // Otherwise the page sends them itself, as with no queue, under the same
    // ids and from the same press. One stopped meanwhile is never sent.
    fellBack = fallbackLine(r.reason)
    const stopped = new Set(ids.filter((id) => jobById(id)?.cancelRequested))
    jobs = jobs.filter((j) => !ids.includes(j.id) || stopped.has(j.id))
    announce()
    const sent: string[] = []
    for (const { clip, id } of placed) {
      forgetStop(id)
      if (stopped.has(id)) failFromRunner(id, new ComfyError(STOPPED_UNSENT, { cancelled: true }))
      else sent.push(startJob({ ...clip, id, startedAt }))
    }
    return { road: 'page', ids: sent }
  }

  if ('pending' in r) {
    unanswered.set(group, ids)
    for (const id of ids) {
      const current = jobById(id)
      if (!current || !unfinished(current) || listedRunner.has(id)) continue
      // Stopped while the request was out: the tab no longer hands it over,
      // so nothing else would settle it.
      if (current.cancelRequested) stoppedUnanswered(id)
      else patchIfChanged(id, { waitNote: NOT_ANSWERED })
    }
    return { road: 'server', ids, pending: true }
  }

  // Refused, and nothing stored: the desk says why, and the clips go.
  for (const id of ids) forgetStop(id)
  jobs = jobs.filter((j) => !ids.includes(j.id))
  announce()
  return { road: 'refused', error: r.error }
}

/**
 * The settings a clip of the queue was made with, read back from the record
 * the server keeps for it, as a record from the archive is read back: for a
 * clip this page did not make, whose settings are not here.
 */
async function runnerSettingsOf(id: string): Promise<Pick<VideoJob, 'composition' | 'loras'> | null> {
  try {
    const res = await fetch(`/api/runner/jobs/${encodeURIComponent(id)}`, { headers: { Accept: 'application/json' } })
    if (!res.ok || !(res.headers.get('content-type') ?? '').includes('json')) return null
    const record = ((await res.json()) as { record?: unknown }).record
    if (!isObj(record) || typeof record.familyId !== 'string' || typeof record.prompt !== 'string') return null
    // The record the server files is this one with its file added; none is needed to read the settings.
    const entry = {
      ...record,
      id,
      no: 0,
      at: 0,
      kind: 'video',
      file: { filename: '', subfolder: '', type: 'output' },
      promptId: '',
      durationMs: 0,
    } as unknown as HistoryEntry
    return {
      composition: compositionFromEntry(entry).composition,
      loras: Array.isArray(record.loras) ? (record.loras as HistoryEntry['loras']) : undefined,
    }
  } catch {
    return null
  }
}

if (typeof window !== 'undefined') {
  // A reload or a close says it is going, so the next page in the tab takes
  // the lane and the sent clips up at once instead of asking.
  window.addEventListener('pagehide', () => {
    if (laneClips.length) saveLane(true)
    if (jobs.some(outThere)) saveSent(true)
  })
  // Back from the browser's cache. If another page ran in this tab meanwhile,
  // it took this page's lane and sent clips up, may have sent or filed some,
  // and saved what is left, so this page's copy is out of date and must never
  // be sent or filed. The page starts again from what was saved instead, as a
  // reload does. Otherwise they are simply this page's again.
  window.addEventListener('pageshow', (e) => {
    if (!e.persisted) return
    const outHere = jobs.some(outThere)
    const laneTaken = laneClips.length > 0 && laneKept && readLane(tabStore.get(LANE_KEY))?.writer !== PAGE
    const sentTaken = outHere && sentKept && readSent(tabStore.get(SENT_KEY))?.writer !== PAGE
    if (laneTaken || sentTaken) {
      handedOver = true
      const stale = laneClips
      laneClips = []
      for (const c of stale) waiting.get(c.id)?.abort()
      holdPage()
      window.location.reload()
      return
    }
    if (laneClips.length) saveLane()
    if (outHere) saveSent()
  })
}

// Last in the engine, so everything it calls is defined. Sent clips first, so
// the lane's clips see them ahead in ComfyUI's queue.
restoreSent()
restoreLane()
// The queue's clips of this desk, whichever device sent them, for as long as
// the page lives. Listening starts nothing: the shell starts the queue's
// stream, and until it has, the list is empty.
runnerStore.subscribe(syncRunner)
syncRunner()
// Then what this tab still has to hand over, which the server has not listed.
takeUpOutbox()

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
 *
 * The hold is the shell's own, so it behaves as every other stop does: a key
 * held past the hold does not start another through auto-repeat, and a screen
 * reader, which presses with a bare click and can never hold, arms the stop
 * and confirms it with a second press.
 */
function HoldToStop({ jobId, label = 'Hold to stop' }: { jobId: string; label?: string }) {
  const hold = useHoldToConfirm(() => {
    void stopJob(jobId)
  }, 600)
  return (
    <>
      <button
        type="button"
        {...hold.bind}
        aria-label={hold.armed ? 'Press again to stop this clip' : 'Hold to stop this clip'}
        className={`sg-hold relative flex w-full items-center justify-center overflow-hidden border border-burgundy-900 px-4 py-2 text-[0.625rem] font-semibold uppercase tracking-[0.16em] text-burgundy-900 [@media(pointer:coarse)]:min-h-11 ${RING}`}
      >
        <span
          aria-hidden
          className="absolute inset-y-0 left-0 bg-burgundy-900"
          style={{ width: `${Math.round(hold.progress * 100)}%` }}
        />
        <span className="relative" style={{ color: hold.progress > 0.5 ? 'var(--color-newsprint)' : undefined }}>
          {hold.armed ? 'Press again to stop' : label}
        </span>
      </button>
      <p aria-live="polite" className="sr-only">
        {hold.armed ? 'Stop armed. Press the button again within three seconds to stop the clip.' : ''}
      </p>
    </>
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

/**
 * Why the press cannot run yet, in the reader's words. Null means it can.
 * `addOns` is how many rack files the clip would chain; see clipMemory.
 * `offline` is true while ComfyUI's socket is closed and the page would send
 * the clip itself: a clip the queue on the server takes waits there until
 * ComfyUI answers (queueTakesClips), so it is not refused for this.
 */
function reasonFor(
  family: VideoFamily | null,
  c: Composition,
  hardware: Hardware | null,
  addOns: number,
  offline: boolean,
): string | null {
  if (!family) return 'No video model is installed.'
  // First, because it is the one the reader cannot guess: a clip this size
  // samples for minutes and is then killed in its final decode.
  const memory = clipMemory(family.def, clipOf(family, c), hardware, addOns)
  if (memory.level === 'refuse') return memory.reason
  if (!c.prompt.trim()) return 'Describe the shot first.'
  if (family.needsStartFrame && (c.mode !== 'i2v' || !c.source?.name)) {
    return `${family.label} starts from a picture. Add a start frame.`
  }
  if (c.mode === 'i2v' && !family.canStartFromPicture) return `${family.label} works from words only.`
  if (c.mode === 'i2v' && !c.source?.name) return 'Add a start frame, or work from words.'
  // Uploaded by an older player, which sent the whole clip rather than a frame.
  // An output handed over in place carries its folder after the name.
  if (c.mode === 'i2v' && c.source && VIDEO_EXT.test(c.source.ref?.filename ?? c.source.name.replace(/ \[\w+\]$/, ''))) {
    return 'The start frame is a whole clip, not one frame of it. Clear it and use one frame.'
  }
  // While ComfyUI restarts, a clip the page sends is pressed in vain: it
  // fails at once, unless it is heavy and waits in the lane for ComfyUI to
  // answer. Last, because it passes by itself and the rest still stand when
  // it does. Said as the Pictures desk and the reel say it.
  if (offline) return 'ComfyUI is not answering, so nothing can be queued.'
  return null
}

/**
 * The add-ons a rack will actually chain on a family: installed, fitting, on,
 * pairs expanded.
 */
function rackToRun(fam: VideoFamily, l: LoraLibrary, stack: LoraStack) {
  const target = targetFor(fam.def, fam.model)
  const resolved = resolveStack(stack, l, target)
  const installed = new Set(l.all.filter((i) => i.installed).map((i) => i.file))
  // Rows on the rack that cannot run: switched off, or dropped as missing or
  // made for another size. The other half of a pair is never pulled back in
  // from among these, so its partner goes to both halves.
  const dropped = new Set(resolved.dropped.map((d) => d.file))
  const excluded = new Set(stack.filter((e) => !e.enabled || dropped.has(e.file)).map((e) => e.file))
  // A row at 0 is not one of those. It says its half gets nothing, and on a
  // two-half family it has to reach the expansion to say so, or its partner
  // would take that half as well. resolveStack leaves it out of its specs.
  const byName = new Map(resolved.specs.map((sp) => [sp.name, sp]))
  const specs = fam.def.dualModel
    ? stack.flatMap((e) => {
        const sp = byName.get(e.file)
        if (sp) return [sp]
        return e.enabled && e.strength === 0 && !dropped.has(e.file) ? [{ name: e.file, strength: 0 }] : []
      })
    : resolved.specs
  // What goes into the graph, each file at the strength it runs at, which is
  // also what the record says ran. A half at 0 ran nothing.
  const ran = videoLorasToRun(fam.def, specs, installed, excluded).filter((s) => s.strength !== 0)
  return { target, stack, specs, installed, excluded, ran }
}

/** How many add-on files a clip chains from the rack, for the memory check. */
const addOnCount = (ran: readonly { name: string }[]) => new Set(ran.map((s) => s.name)).size

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
  const leftOver = useSyncExternalStore(videoJobs.subscribe, videoJobs.leftOver)
  const leftSent = useSyncExternalStore(videoJobs.subscribe, videoJobs.leftSent)
  const laneHeld = useSyncExternalStore(videoJobs.subscribe, videoJobs.held)
  const serverHold = useSyncExternalStore(videoJobs.subscribe, videoJobs.heldOnServer)
  // Whether the queue on the server would take a clip made now (queueTakesClips).
  const caps = useServerCapabilities()
  const capsRef = useRef(caps)
  useEffect(() => {
    capsRef.current = caps
  }, [caps])
  const queueUp = useSyncExternalStore(runnerStore.subscribe, queueRunning, queueRunning)
  // Why the queue on the server is not running, when it has said it is not.
  // Its clips then show as it keeps them, with no Stop and no word on its
  // hold, which it would refuse, and the press is free for clips of the page's own.
  const queueOffWhy = useSyncExternalStore(runnerStore.subscribe, queueOffReason, queueOffReason)
  const givenUp = useSyncExternalStore(videoJobs.subscribe, videoJobs.givenUp)

  const [cat, setCat] = useState<Catalogue | null>(null)
  const [catError, setCatError] = useState<string | null>(null)
  const [connection, setConnection] = useState<ConnectionState>('connecting')
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [notice, setNotice] = useState<{ kind: 'info' | 'correction' | 'error'; title: string; body: string } | null>(null)
  const [showWorkflow, setShowWorkflow] = useState(false)
  const [viewing, setViewing] = useState<HistoryEntry | null>(null)
  const [justFinished, setJustFinished] = useState<{ id: string; ms: number } | null>(null)
  const [reuseNotice, setReuseNotice] = useState<{
    /** Where the settings came from, as the notice says it: `No. 1,204`. */
    from: string
    /** Everything the reader should know about what was and was not carried over. */
    notes: string[]
    /** True when `Make another` queued a clip straight after loading. */
    ran: boolean
    /** The desk and its add-on rack, exactly as they were before. */
    undo: () => void
  } | null>(null)

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

  // The margin's catalogue panel says when a fetch lands, but only while it
  // is open. A family that lands with it closed is heard of here.
  useEffect(() => onPlanLanded(() => setAttempt((a) => a + 1)), [])

  useEffect(() => watchConnection(setConnection), [])

  // --- free memory, read afresh -------------------------------------------
  /**
   * The catalogue is read once per page load, and the page is an installed app
   * that stays open for hours. Its memory reading went stale with it: the
   * verdicts compared the weights against what was free when the tab was
   * opened, and the margin called that figure free "right now". So memory is
   * read again on every visit to the desk, whenever the page comes back into
   * view, and after every clip, which is when the most memory changes hands.
   * A reading that fails keeps the last good one, and its time.
   */
  const [freshMachine, setFreshMachine] = useState<Machine | null>(null)
  const [machineAsk, setMachineAsk] = useState(0)
  const lastEnded = useMemo(() => allJobs.reduce((t, j) => Math.max(t, j.finishedAt ?? 0), 0), [allJobs])
  useEffect(() => {
    let alive = true
    void readMachine().then((m) => {
      if (alive && m.hardware) setFreshMachine(m)
    })
    return () => {
      alive = false
    }
  }, [machineAsk, lastEnded])
  useEffect(() => {
    const onShow = () => {
      if (document.visibilityState === 'visible') setMachineAsk((n) => n + 1)
    }
    document.addEventListener('visibilitychange', onShow)
    return () => document.removeEventListener('visibilitychange', onShow)
  }, [])
  const machine = freshMachine ?? cat?.machine ?? null
  const hardware = machine?.hardware ?? null
  const offer = useMemo(() => (cat ? priced(cat, hardware) : null), [cat, hardware])
  const families = useMemo(() => offer?.families ?? [], [offer])

  const family = useMemo<VideoFamily | null>(() => {
    if (!families.length) return null
    return families.find((f) => f.def.id === composition.familyId) ?? families[0]
  }, [families, composition.familyId])

  // Load the family's verified recipe the first time, and whenever the chosen
  // style is no longer installed. Fields the reader has touched survive it.
  useEffect(() => {
    if (!family) return
    const current = store.get()
    if (current.familyId === family.def.id && current.model === family.model) return
    store.set(applyDefaults(current, defaultsOf(family)))
  }, [family])

  // A family that only starts from a picture has no words-only graph, so a
  // draft that arrives on it in `From words`, saved or restored, is moved over.
  useEffect(() => {
    if (family?.needsStartFrame && composition.mode !== 'i2v') store.patch({ mode: 'i2v' })
  }, [family, composition.mode])

  const mode = composition.mode === 'i2v' ? 'i2v' : 't2v'
  // The rate the file will be written at, which is not always the one typed:
  // see encoderOf. Every length on the desk is computed from this one.
  const fps = family ? fpsOf(family, composition) : (composition.fps ?? 24)
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
  // Clips the queue on the server keeps while it is off (parkedOnServer): shown, never stopped from here.
  const parked = (j: VideoJob): boolean => queueOffWhy !== null && parkedOnServer(j)
  // What the press is busy with: not those, which wait for the queue on the server and hold nothing up here.
  const moving = useMemo(
    () => (queueOffWhy === null ? live : live.filter((j) => !parkedOnServer(j))),
    [live, queueOffWhy],
  )
  // Clips in the lane that ComfyUI has not been handed yet. Read from the
  // lane itself, which a clip leaves just before its prompt goes; every change
  // to it comes with a change to the jobs, which renders this again.
  const waitingHere = videoJobs.waitingCount()
  // The one drawing, when one is. A heavy clip waiting for the queue to empty
  // can be older than a light clip that went straight in and is drawing now.
  const runningJob = moving.find((j) => j.status === 'running') ?? moving[moving.length - 1] ?? null
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

  // An output is handed to LoadImage where it lies, by its annotated name, as
  // the reel does. It used to be fetched to the browser and uploaded straight
  // back into ComfyUI's input folder: a full-size picture down and up again
  // over the phone's link, every time one was chosen.
  const adoptFromArchive = useCallback(
    (entry: HistoryEntry) => {
      setNotice(null)
      setSource({
        name: annotatedRef(entry.file),
        ref: entry.file,
        previewUrl: thumbUrl(entry.file, 256),
        label: entry.file.filename,
        width: entry.width ?? undefined,
        height: entry.height ?? undefined,
        fromEntryId: entry.id,
      })
      store.patch({ mode: 'i2v' })
    },
    [setSource],
  )

  // A source handed over by the archive arrives with a ref and no name. It is
  // given its annotated name here, the same way, with no copy.
  useEffect(() => {
    const src = composition.source
    const ref = src?.ref
    if (!src || src.name || !ref) return
    // A clip is not a start frame. LoadImage decodes every frame of one, and
    // the next clip would take all of them as its opening. An older player
    // sent the whole clip this way, and a draft saved then may still hold it.
    if (VIDEO_EXT.test(ref.filename)) {
      setSource(null)
      setNotice({
        kind: 'correction',
        title: 'Correction',
        body: 'The start frame waiting here was a whole clip, not one frame of it, so it was taken off. Open the clip and use one frame.',
      })
      return
    }
    setSource({ ...src, name: annotatedRef(ref), previewUrl: src.previewUrl ?? thumbUrl(ref, 256) })
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

  // --- add-ons -------------------------------------------------------------
  //
  // The rack is per family and persisted the way the picture rack is. Both
  // are read through refs by buildGraph, which is deliberately stable: the
  // press reads the store, not the render, so a clip queued from `Make
  // another` in the same tick sees the same rack the reader sees.
  const [lib, setLib] = useState<LoraLibrary>(EMPTY_LIBRARY)
  const [stack, setStack] = useState<LoraStack>([])
  const [stackFor, setStackFor] = useState<string | null>(null)
  const libRef = useRef(lib)
  /** The rack as the press sees it, tagged with its family so a stale one is never chained. */
  const stackRef = useRef<{ familyId: string; stack: LoraStack }>({ familyId: '', stack: [] })
  const takeLib = useCallback((l: LoraLibrary) => {
    libRef.current = l
    setLib(l)
  }, [])
  useEffect(() => {
    void loadLoraLibrary().then(takeLib, () => {})
  }, [takeLib])
  // A different family is a different rack. Adjusted during render, as React
  // asks for state that follows a prop, so the old family's rack is never
  // painted under the new family's name.
  if (family && stackFor !== family.def.id) {
    setStackFor(family.def.id)
    setStack(loadStack(family.def.id))
  }
  const updateStack = useCallback(
    (next: LoraStack, familyId = family?.def.id ?? '') => {
      setStack(next)
      stackRef.current = { familyId, stack: next }
      if (familyId) saveStack(familyId, next)
    },
    [family],
  )

  /**
   * The rack the press would chain for a family. The ref holds what the reader
   * last set; for a family it has not been set for yet, the saved rack is the
   * rack.
   */
  const rackOf = useCallback(
    (familyId: string): LoraStack =>
      stackRef.current.familyId === familyId ? stackRef.current.stack : loadStack(familyId),
    [],
  )

  /** The add-ons that will actually be chained, as the press reads them. */
  const resolvedLoras = useCallback(
    (fam: VideoFamily) => rackToRun(fam, libRef.current, rackOf(fam.def.id)),
    [rackOf],
  )

  /** The graph one run sends, and the positive prompt in it, trigger words and all. */
  const buildGraph = useCallback(
    (fam: VideoFamily, c: Composition, seed: number): { graph: ApiWorkflow; positive: string } | null => {
      // Sent from words, the start-frame-only family would go out with its
      // LoadImage still holding the placeholder, which ComfyUI refuses.
      if (fam.needsStartFrame && c.mode !== 'i2v') return null
      const shaped = c.mode === 'i2v' ? deriveImageToVideo(fam.def) : fam.def
      if (!shaped) return null
      const { target, stack: rack, specs, installed, excluded } = resolvedLoras(fam)
      const chained = chainVideoStack(shaped, specs, installed, excluded)
      const params = toParams({ ...c, seed }, { negative: defaultsFor(fam.def, fam.model).negative })
      if (c.mode !== 'i2v') delete params.image
      // An add-on without its trigger words runs at a fraction of itself.
      const words = missingTriggers(rack, libRef.current, params.positive, target)
      if (words.length) params.positive = params.positive.trim() ? `${params.positive}, ${words.join(', ')}` : words.join(', ')
      const graph = instantiate(chained ?? shaped, params)
      applyShift(graph, c.shift)
      return { graph, positive: params.positive }
    },
    [resolvedLoras],
  )

  // From the rack and library this render shows, which are the ones the
  // press will read: `stack` follows `family` (see above).
  const addOns = useMemo(() => (family ? addOnCount(rackToRun(family, lib, stack).ran) : 0), [family, lib, stack])
  // ComfyUI not answering refuses a clip the page would send itself, not one
  // the queue on the server takes and sends once ComfyUI answers. Read each
  // render: the page's own lane and clips decide it too, and they come with
  // the desk's jobs.
  const onQueue = queueUp && queueTakesClips(caps)
  const blockedReason = useMemo(
    () => reasonFor(family, composition, hardware, addOns, connection === 'closed' && !onQueue),
    [family, composition, hardware, addOns, connection, onQueue],
  )
  const memory = useMemo(
    () => (family ? clipMemory(family.def, clipOf(family, composition), hardware, addOns) : null),
    [family, composition, hardware, addOns],
  )

  // The verdict each length would get at this shape, and each shape at this
  // length, with this rack, asked the way the press asks it. A chip the press
  // would refuse used to take the tap and then grey the button out, leaving
  // the reader to try the others one by one.
  const refusedLength = useMemo(() => {
    if (!family) return []
    return refusalsFor(lengths, (f) => {
      const v = clipMemory(family.def, { width: composition.width, height: composition.height, frames: f }, hardware, addOns)
      return v.level === 'refuse' ? v.reason : null
    })
  }, [family, lengths, composition.width, composition.height, hardware, addOns])
  const refusedShape = useMemo(() => {
    if (!family) return []
    return refusalsFor(shapes, (sh) => {
      const v = clipMemory(family.def, { width: sh.width, height: sh.height, frames }, hardware, addOns)
      return v.level === 'refuse' ? v.reason : null
    })
  }, [family, shapes, frames, hardware, addOns])

  /**
   * Queue the clip — or several, with successive seeds.
   *
   * Everything is read from the store rather than from this render's props,
   * because `Make another` loads a record into the store and runs in the same
   * tick, before React has re-rendered with it.
   *
   * @returns true when at least one clip was queued.
   */
  const make = useCallback((): boolean => {
    if (uploading) return false
    const c = store.get()
    // The family the desk names and no other. Falling back to the first one
    // would queue a record's size, length and steps on a model it was never
    // made with, and file the result under the wrong name. A blank desk has
    // no family yet, and takes the first.
    const fam = families.find((f) => f.def.id === c.familyId) ?? (c.familyId ? null : families[0]) ?? null
    if (!fam) {
      if (cat && c.familyId) {
        setNotice({
          kind: 'error',
          title: 'That style is not installed',
          body: 'These settings name a style that is no longer on this machine. Choose another style before you run this.',
        })
      }
      return false
    }
    const chained = addOnCount(resolvedLoras(fam).ran)
    // Read afresh, not from this render: `Make another` and the shortcut run
    // from callbacks that may predate the last change of connection. Not
    // answering refuses only clips the page would send itself.
    if (reasonFor(fam, c, hardware, chained, connectionState() === 'closed' && !queueTakesClips(capsRef.current))) return false
    // Pressed while ComfyUI is not answering, on the word that the queue takes them.
    const offline = connectionState() === 'closed'

    const runs = c.runs ?? 1
    const first = c.seedLocked ? c.seed : randomSeed()
    const release = clipMemory(fam.def, clipOf(fam, c), hardware, chained).release

    // Every clip is built before any goes, so one press goes as one: to the
    // queue on the server in one request where it runs, otherwise into the
    // page's own lane in the order made. Which road is asked of the server,
    // so the clips go a moment after the press rather than within it.
    const planned: PlannedClip[] = []
    const send = () => {
      void sendClips(planned, { offline }).then((sent) => {
        if (sent.road === 'refused') setNotice({ kind: 'error', title: 'Nothing was sent', body: sent.error })
      })
    }
    for (let i = 0; i < runs; i++) {
      const seed = first + i
      // A start frame left on the desk after switching to words never reaches
      // the graph, so it is not carried into the record either; and the rate
      // is the one the encoder will actually write.
      const snapshot: Composition = {
        ...c,
        seed,
        fps: fpsOf(fam, c),
        source: needsSource(c.mode) ? c.source : null,
      }
      const built = buildGraph(fam, snapshot, seed)
      if (!built) {
        // The ones built before it still go, as they always did.
        send()
        setNotice({
          kind: 'error',
          title: 'That shape is not available',
          body:
            fam.needsStartFrame && c.mode !== 'i2v'
              ? `${fam.label} starts from a picture. Add a start frame.`
              : `${fam.label} cannot take a start frame. Work from words, or choose another style.`,
        })
        return i > 0
      }
      const { ran } = resolvedLoras(fam)
      planned.push({
        // The prompt exactly as sent, so the record carries the trigger words
        // the rack added and not only the words the reader typed.
        composition: { ...snapshot, positive: built.positive },
        graph: built.graph,
        familyLabel: fam.def.label,
        modelLabel: modelLabelOf(fam),
        // Where each add-on ran, so reuse can tell a pair's partner that was
        // switched off from one that was simply not on the rack.
        loras: ran.length
          ? ran.map((s) => ({ name: s.name, strength: s.strength, ...(s.half ? { half: s.half } : {}) }))
          : undefined,
        release,
      })
    }
    send()

    store.patch({ seed: first })
    setViewing(null)
    setNotice(null)
    return true
  }, [cat, families, hardware, uploading, buildGraph, resolvedLoras])

  /**
   * Put a record back on the desk. `Use these settings` restores everything and
   * runs nothing, so the reader sees exactly what they are about to make;
   * `Make another` does the same with a fresh seed and then runs.
   *
   * A draft that would be overwritten is never simply lost: the notice offers
   * both ways out, and the undo puts back the store's own previous state and
   * the add-on rack the record replaced.
   */
  const reuse = useCallback(
    (entry: HistoryEntry, run: boolean) => {
      const fam = families.find((f) => f.def.id === entry.familyId) ?? null
      const installedModels = cat ? families.map((f) => f.model).filter(Boolean) : undefined
      const l = libRef.current
      // The rack as the desk holds it, which outlives a browser that cannot
      // save one; the undo puts back this, not the saved copy.
      const priorRack = rackOf(entry.familyId)
      const applied = reuseIntoDesk(entry, {
        freshSeed: run,
        installedModels,
        availableSamplers: cat?.samplers,
        availableSchedulers: cat?.schedulers,
        // Not known until the library has loaded, which the rack then allows for.
        installedLoras: l === EMPTY_LIBRARY ? undefined : new Set(l.all.filter((i) => i.installed).map((i) => i.file)),
      })
      // reuseIntoDesk has put the record's add-ons back on the family's saved
      // rack, the same as it does from the Archive. The desk's own copy
      // follows, because `Make another` queues in this same tick and reads it.
      const rack = applied.rack
      if (rack) updateStack(rack.next, rack.familyId)

      const notes = applied.notes.map((n) => n.reason)
      // A two-model family records no single model file, so its absence has
      // to be read off the catalogue rather than off the record.
      const gone = !!cat && !fam
      if (gone && !applied.notes.some((n) => n.field === 'model')) {
        notes.unshift(`${entry.familyLabel} is no longer installed. Choose another style before you run this.`)
      }

      setViewing(null)
      const ran = run && !gone && !applied.notes.some((n) => n.field === 'model') ? make() : false
      setReuseNotice(
        applied.clobbered || notes.length
          ? {
              from: `No. ${entry.no.toLocaleString('en-GB')}`,
              notes,
              ran,
              undo: () => {
                applied.undo()
                if (rack) updateStack(priorRack, rack.familyId)
              },
            }
          : null,
      )
      if (!run) promptRef.current?.focus()
    },
    [cat, families, make, rackOf, updateStack],
  )

  /**
   * Put a clip that failed or was lost back on the desk, as it was sent, with
   * its add-on rack. It has no record to reuse, and by the time a long clip
   * fails the draft usually holds the next idea, so this is the one way back
   * to it short of rebuilding it from memory. Nothing runs: the reader
   * presses Make, so the memory check and the lane apply again. A draft it
   * replaces can be put back, as with a record.
   */
  const putBack = useCallback(
    (job: VideoJob) => {
      const prior = store.get()
      const familyId = job.composition.familyId
      const priorRack = rackOf(familyId)
      // The prompt as typed. The one sent carries the add-ons' words, which
      // the rack adds again on the next run.
      const typed: Composition = { ...job.composition }
      delete typed.positive
      store.set(typed)
      const l = libRef.current
      const rack = restoreRack(
        { familyId, loras: job.loras },
        l === EMPTY_LIBRARY ? null : new Set(l.all.filter((i) => i.installed).map((i) => i.file)),
      )
      updateStack(rack.next, familyId)
      dismissJob(job.id)
      setViewing(null)
      const clobbered = prior.prompt.trim().length > 0 && prior.prompt.trim() !== typed.prompt.trim()
      setReuseNotice(
        clobbered || rack.note
          ? {
              from: 'the clip that did not finish',
              notes: rack.note ? [rack.note] : [],
              ran: false,
              undo: () => {
                store.set(prior)
                updateStack(priorRack, familyId)
              },
            }
          : null,
      )
      promptRef.current?.focus()
    },
    [rackOf, updateStack],
  )

  /**
   * The same for a clip of the queue this page did not make: after a reload,
   * or from another device. Its settings are on the server, in the record it
   * would have filed, and are read from there first.
   */
  const putBackAny = useCallback(
    (job: VideoJob) => {
      if (!job.adopted) {
        putBack(job)
        return
      }
      void runnerSettingsOf(job.id).then((settings) => {
        if (settings) putBack({ ...job, ...settings })
        else {
          setNotice({
            kind: 'error',
            title: 'We could not read those settings',
            body: 'The SwitchGen server did not send the settings this clip was made with. Try again in a moment.',
          })
        }
      })
    },
    [putBack],
  )

  // Ctrl/⌘+Enter runs, from inside the prompt too — the one deliberate
  // exception to "every single-key shortcut is dead while a field has focus".
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // A key somebody nearer the event already claimed is not ours.
      if (e.defaultPrevented) return
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault()
        make()
        return
      }
      if (isTyping(e) || e.ctrlKey || e.metaKey || e.altKey) return
      if (e.key === 'u') {
        // The player claims `u` for its frame menu while it has focus or fills
        // the screen. Both listeners sit on window, and this one is added again
        // whenever `make` changes, so it can run before the player's and the
        // claim above cannot be relied on to have been made yet.
        const active = document.activeElement as HTMLElement | null
        if (document.fullscreenElement || active?.closest('[data-player]')) return
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
    // What ComfyUI took, as filed; a clip whose start was never heard has no
    // such figure, and its wait is not its making.
    if (!done || !done.finishedAt || done.tookMs === null) return
    if (Date.now() - done.finishedAt > 4000) return
    setJustFinished({ id: done.id, ms: done.tookMs })
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

  // From when ComfyUI began the clip: a wait in the lane or its queue is not
  // part of how long the clip takes, and the estimate below is not either.
  const elapsed = runningJob?.ranAt != null ? now - runningJob.ranAt : 0

  const remaining = (() => {
    if (!runningJob || runningJob.status !== 'running') return null
    if (runningEstimate && runningJob.ranAt !== null) {
      const left = runningEstimate.ms - elapsed
      if (left <= 0) return 'Running long. Still working.'
      return `About ${duration(left)} left, from your last ${runningEstimate.runs} runs.`
    }
    const left = drawingLeft(runningJob, now)
    if (left === null) return null
    const pass = runningJob.pass
    return pass
      ? `About ${duration(left)} left of the drawing, at this pass's pace${
          pass.index < pass.count ? ', not counting the change of model between passes' : ''
        }. Developing and encoding follow.`
      : `About ${duration(left)} left of the drawing, at this run's pace. Developing and encoding follow.`
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

              {family.fpsReachesEncoder ? (
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
              ) : (
                <p className="mb-3 text-caption italic text-grey-700 tabular-nums" id="field-fps">
                  Written at {fps} frames per second. This family's recipe does not pass a frame rate to its
                  encoder, so the rate cannot be changed here.
                </p>
              )}

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

              <LoraRack
                def={family.def}
                model={family.model}
                lib={lib}
                stack={stack}
                onStack={(next) => updateStack(next)}
                onLibraryReload={() => void loadLoraLibrary().then(takeLib, () => {})}
              />

              <button
                type="button"
                className="mb-4 text-caption text-burgundy-900 underline"
                onClick={() => setShowWorkflow((v) => !v)}
              >
                {showWorkflow ? 'Hide the workflow' : 'Show the workflow'}
              </button>

              {showWorkflow ? (
                <WorkflowPeek build={() => buildGraph(family, store.get(), composition.seed)?.graph ?? null} />
              ) : null}

              {notes.length ? (
                <div className="mt-4 border-t border-grey-300 pt-3">
                  <Head title="From the model's card" />
                  {notes.slice(0, 3).map((n, i) => (
                    <Marginalia key={i} text={n} />
                  ))}
                </div>
              ) : null}

              {machine?.vramFree != null ? (
                <p className="mt-2 text-caption italic text-grey-500 tabular-nums">
                  The card had {gb(machine.vramFree)} free when it was last checked, at {clock(machine.at)}.
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
            <>
              <Notice tone="correction" title="Correction">
                No video model is installed.{' '}
                {offer?.blocked.length ? `${offer.blocked[0].label} ${offer.blocked[0].why.replace(/\.$/, '')}.` : ''}
              </Notice>
              {/* The first family has to come from somewhere. The catalogue
                  used to sit only in the margin of all controls, which needs
                  a family to show at all, so with none installed there was no
                  way to it but placing files by hand. */}
              <div className="mt-4 border-t border-grey-300 pt-3">
                <CataloguePanel
                  modes={['video']}
                  onInstalled={() => {
                    resetCatalogue()
                    setAttempt((a) => a + 1)
                  }}
                />
              </div>
            </>
          ) : (
            <>
              {/* Source tabs */}
              <div className="mb-5 flex border border-grey-300" role="group" aria-label="Where the clip comes from">
                {(['t2v', 'i2v'] as const).map((m, i) => {
                  const active = mode === m
                  const disabled =
                    (m === 'i2v' && !family.canStartFromPicture) || (m === 't2v' && family.needsStartFrame)
                  return (
                    <button
                      key={m}
                      type="button"
                      disabled={disabled}
                      title={
                        disabled
                          ? m === 't2v'
                            ? `${family.label} starts from a picture.`
                            : `${family.label} works from words only.`
                          : undefined
                      }
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
                            onClick={() => adoptFromArchive(e)}
                            className="h-12 w-12 border border-grey-300 hover:border-burgundy-900"
                          >
                            {/* A 48 px tile: one 256 px thumbnail covers it on any screen. */}
                            <img
                              src={thumbUrl(e.file, 256)}
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
              {(families.length > 1 || prefs.expert) && (
                <div className="mb-5">
                  <Head title="Style" />
                  <select
                    aria-label="Style"
                    className="field"
                    value={family.def.id}
                    onChange={(e) => {
                      const next = families.find((f) => f.def.id === e.target.value)
                      if (!next) return
                      const loaded = applyDefaults(store.get(), defaultsOf(next))
                      store.set(next.needsStartFrame ? { ...loaded, mode: 'i2v' } : loaded)
                    }}
                  >
                    {families.map((f) => (
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
                    <p className="mt-1 text-caption italic text-grey-700 tabular-nums">
                      {family.verdict.reason}
                      {machine ? ` Memory last checked at ${clock(machine.at)}.` : ''}
                    </p>
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
                  options={lengths.map((f, i) => {
                    const refused = refusedLength[i] ?? null
                    return {
                      value: f,
                      label: clipLength(f, fps),
                      // Said on the chip, since a title never shows on a phone.
                      caption: refused ? `${f} frames, too long` : `${f} frames`,
                      title: refused ?? undefined,
                      // The chosen one stays pressable, so its refusal can be read.
                      disabled: refused !== null && f !== frames,
                    }
                  })}
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
                  options={shapes.map((s, i) => {
                    const refused = refusedShape[i] ?? null
                    const chosen = s.width === composition.width && s.height === composition.height
                    return {
                      value: `${s.width}x${s.height}`,
                      label: s.label.split(' ')[0],
                      caption: refused ? `${times(s.width, s.height)}, too large` : times(s.width, s.height),
                      title: refused ?? s.note,
                      disabled: refused !== null && !chosen,
                    }
                  })}
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
                {/* The length and the shape are what decide whether a 14B pair
                    survives its final decode. A refusal is said under the button. */}
                {memory?.level === 'caution' && memory.reason ? (
                  <p className="mt-1 text-caption italic text-grey-700 tabular-nums">{memory.reason}</p>
                ) : null}
              </div>

              {/* The button. It stays live while a clip runs: there is one GPU
                  and one queue, and a clip sent now really does wait its turn —
                  which we say, rather than pretending to run two at once. */}
              <div className="mb-3">
                <button type="button" className="press" disabled={!!blockedReason || uploading} onClick={make}>
                  {justFinished
                    ? duration(justFinished.ms)
                    : moving.length
                      ? 'Make the clip · next in line'
                      : composition.runs > 1
                        ? `Make ${composition.runs} clips`
                        : 'Make the clip'}
                </button>

                {blockedReason ? (
                  <p className="mt-1 text-caption italic text-grey-700">{blockedReason}</p>
                ) : moving.length ? (
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
                    from {reuseNotice.from}.{' '}
                    {reuseNotice.ran ? 'A clip is queued with them, on a new seed.' : 'Nothing has run yet.'}{' '}
                    {reuseNotice.notes.join(' ')}{' '}
                    <button
                      className="underline"
                      onClick={() => {
                        reuseNotice.undo()
                        setReuseNotice(null)
                      }}
                    >
                      {reuseNotice.ran ? 'Put back what was on the desk' : 'Undo this'}
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

              {/* Only for clips the page is following itself: the server
                  follows its own whatever this page's connection does. */}
              {connection === 'closed' && live.some((j) => !j.runner) ? (
                <div className="mt-4">
                  <Notice tone="correction" title="Correction">
                    We lost the connection to ComfyUI. If your clip is still running, we will pick it up when ComfyUI
                    answers again. If ComfyUI restarted, the clip is gone and the desk will say so.
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

          {/* Clips an earlier page left, which this page will not send on a guess */}
          {leftOver.length ? (
            <div className="mb-4">
              <Notice tone="warning" title={leftOver.length === 1 ? 'A clip left waiting' : 'Clips left waiting'}>
                {leftOver.length === 1 ? 'One clip was' : `${leftOver.length} clips were`} waiting in this tab when the
                page before this one went away without handing {leftOver.length === 1 ? 'it' : 'them'} on, as happens
                when a page crashes or a tab is copied. If this tab was copied from one that is still open, that one
                is sending {leftOver.length === 1 ? 'it' : 'them'}, and sending from here as well would make{' '}
                {leftOver.length === 1 ? 'it' : 'them'} twice.
                <ul className="my-1 list-none p-0">
                  {leftOver.map((c) => (
                    <li key={c.id} className="truncate italic">
                      {c.composition.prompt || 'No words'} · {c.modelLabel || c.familyLabel}
                    </li>
                  ))}
                </ul>
                <button className="underline" onClick={() => videoJobs.sendLeftOver()}>
                  Send {leftOver.length === 1 ? 'it' : 'them'} from here
                </button>{' '}
                ·{' '}
                <button className="underline" onClick={() => videoJobs.forgetLeftOver()}>
                  Forget {leftOver.length === 1 ? 'it' : 'them'}
                </button>
              </Notice>
            </div>
          ) : null}

          {/* Clips an earlier page handed to the server, which the server never took */}
          {givenUp.length ? (
            <div className="mb-4">
              <Notice tone="warning" title="Clips not sent">
                A page before this one in this tab handed clips to the SwitchGen server and went before the server
                answered. When this page offered them again, they were not taken:
                <ul className="my-1 list-none p-0">
                  {givenUp.map((g) => (
                    <li key={g.groupId} className="italic">
                      {g.label || 'Clips'}, made at {clock(g.at)}: {g.line}
                    </li>
                  ))}
                </ul>
                <button
                  className="inline-flex items-center underline [@media(pointer:coarse)]:min-h-11"
                  onClick={() => {
                    for (const g of givenUp) videoJobs.forgetGivenUp(g.groupId)
                  }}
                >
                  Dismiss
                </button>
              </Notice>
            </div>
          ) : null}

          {/* Clips an earlier page sent, which this page will not follow on a guess */}
          {leftSent.length ? (
            <div className="mb-4">
              <Notice tone="warning" title={leftSent.length === 1 ? 'A clip sent before this page' : 'Clips sent before this page'}>
                {leftSent.length === 1 ? 'One clip was' : `${leftSent.length} clips were`} sent to ComfyUI from this
                tab by the page before this one, which went away without handing {leftSent.length === 1 ? 'it' : 'them'}{' '}
                on, as happens when the phone closes a tab in the background, a page crashes or a tab is copied.{' '}
                {leftSent.some((c) => c.sending)
                  ? 'One marked as on its way was still being sent when the page went, so it may never have got there: following it finds out, and shows it as not sent if it did not. '
                  : null}
                If
                this tab was copied from one that is still open, that one is following {leftSent.length === 1 ? 'it' : 'them'}{' '}
                already, and following from here as well could file {leftSent.length === 1 ? 'it' : 'them'} twice.
                <ul className="my-1 list-none p-0">
                  {leftSent.map((c) => (
                    <li key={c.id} className="truncate italic">
                      {c.composition.prompt || 'No words'} · {c.modelLabel || c.familyLabel}
                      {c.sending ? ' · on its way' : ''}
                    </li>
                  ))}
                </ul>
                <button
                  className="inline-flex items-center underline [@media(pointer:coarse)]:min-h-11"
                  onClick={() => videoJobs.followLeftSent()}
                >
                  Follow {leftSent.length === 1 ? 'it' : 'them'} from here
                </button>{' '}
                ·{' '}
                <button
                  className="inline-flex items-center underline [@media(pointer:coarse)]:min-h-11"
                  onClick={() => videoJobs.forgetLeftSent()}
                >
                  Forget {leftSent.length === 1 ? 'it' : 'them'}
                </button>{' '}
                Forgetting does not stop ComfyUI making {leftSent.length === 1 ? 'it' : 'any'} that got there; the
                Archive’s “Look for files with no record” files {leftSent.length === 1 ? 'it once it has' : 'them once they have'}{' '}
                landed.
              </Notice>
            </div>
          ) : null}

          {/* The lane, held after a heavy clip was lost */}
          {laneHeld && waitingHere ? (
            <div className="mb-4">
              <Notice tone="warning" title={waitingHere === 1 ? 'A clip held back' : 'Clips held back'}>
                The heavy clip before {waitingHere === 1 ? 'this one' : 'these'} was lost: ComfyUI no longer knew it,
                which usually means ComfyUI restarted, as it does when memory runs out.{' '}
                {waitingHere === 1 ? 'The clip waiting behind it needs' : `The ${waitingHere} clips waiting behind it need`}{' '}
                as much memory, so the desk holds {waitingHere === 1 ? 'it' : 'them'} rather than send{' '}
                {waitingHere === 1 ? 'it' : 'them'} the same way without a word from you.{' '}
                <button
                  className="inline-flex items-center underline [@media(pointer:coarse)]:min-h-11"
                  onClick={() => videoJobs.sendHeld()}
                >
                  Send {waitingHere === 1 ? 'it' : 'them'} anyway
                </button>{' '}
                ·{' '}
                <button
                  className="inline-flex items-center underline [@media(pointer:coarse)]:min-h-11"
                  onClick={() => videoJobs.stopHeld()}
                >
                  Stop {waitingHere === 1 ? 'it' : 'them'}
                </button>
              </Notice>
            </div>
          ) : null}

          {/* Where waiting clips live: nothing outside this tab knows about them yet */}
          {waitingHere ? (
            <p className="mb-3 text-caption italic text-grey-700">
              {waitingHere === 1 ? 'The clip waiting its turn lives' : `The ${waitingHere} clips waiting their turn live`}{' '}
              in this tab until {waitingHere === 1 ? 'it is' : 'they are'} sent to ComfyUI.{' '}
              {videoJobs.laneKept()
                ? `A reload picks ${waitingHere === 1 ? 'it' : 'them'} up again; closing the tab loses ${waitingHere === 1 ? 'it' : 'them'}.`
                : `This browser will not let the desk keep ${waitingHere === 1 ? 'it' : 'them'}, so a reload or closing the tab loses ${waitingHere === 1 ? 'it' : 'them'}.`}{' '}
              {/* Why the page is sending them itself when the server has a queue, when the server said. */}
              {videoJobs.sendsItself() ? `${videoJobs.sendsItself()} ` : ''}
              {WAITS_IN_PAGE}
              {/* A held lane lets the screen lock: nothing goes until the reader says. */}
              {wakeLockAvailable() && !laneHeld ? ' The page asks for the screen to stay on while clips wait.' : ''}
            </p>
          ) : null}

          {/* The queue on the server, holding clips after a heavy one was lost, the machine restarted, or the
              queue was off while they waited. The shell's hold notice shows the same hold on every room,
              whatever it covers, with the same word; this one says it where the held clips are. */}
          {serverHold && serverHold.clips > 0 ? (
            <div className="mb-4">
              <Notice tone="warning" title={serverHold.clips === 1 ? 'A clip held back' : 'Clips held back'}>
                {serverHold.why === 'restart' ? (
                  <>{HELD_AFTER_RESTART} </>
                ) : serverHold.why === 'paused' ? (
                  <>{HELD_AFTER_PAUSE} </>
                ) : (
                  <>
                    {serverHold.why === 'unsent'
                      ? `The heavy clip before ${serverHold.clips === 1 ? 'this one' : 'these'} may never have reached ComfyUI: the server could not tell whether it got there, and ComfyUI had no record of it. That usually means ComfyUI restarted, as it does when memory runs out. `
                      : `The heavy clip before ${serverHold.clips === 1 ? 'this one' : 'these'} was lost: ComfyUI no longer knew it, which usually means ComfyUI restarted, as it does when memory runs out. `}
                    {serverHold.clips === 1
                      ? 'The clip waiting behind it needs'
                      : `The ${serverHold.clips} clips waiting behind it need`}{' '}
                    as much memory, so the desk holds {serverHold.clips === 1 ? 'it' : 'them'} rather than send{' '}
                    {serverHold.clips === 1 ? 'it' : 'them'} the same way without a word from you.{' '}
                  </>
                )}
                {/* One hold covers every desk's waiting work it keeps back, and so does the word given here. */}
                {serverHold.others > 0
                  ? `It also holds ${serverHold.others} waiting ${serverHold.others === 1 ? 'job' : 'jobs'} from the other desks, and your word here goes for ${serverHold.others === 1 ? 'that one' : 'those'} too. `
                  : null}
                {/* The word names the hold shown here, so one given on a notice out of date is refused. The
                    queue takes no word while it is off, so none is offered then. */}
                {queueOffWhy !== null ? (
                  `${queueOffWhy} The server takes your word on ${serverHold.clips + serverHold.others === 1 ? 'it' : 'them'} once its queue is running again.`
                ) : (
                  <>
                    <button
                      className="inline-flex items-center underline [@media(pointer:coarse)]:min-h-11"
                      onClick={() => videoJobs.sendHeldOnServer(serverHold.since)}
                    >
                      Send {serverHold.clips + serverHold.others === 1 ? 'it' : 'them'}
                      {serverHold.why === 'lost' || serverHold.why === 'unsent' ? ' anyway' : ''}
                    </button>{' '}
                    ·{' '}
                    <button
                      className="inline-flex items-center underline [@media(pointer:coarse)]:min-h-11"
                      onClick={() => videoJobs.stopHeldOnServer(serverHold.since)}
                    >
                      Call {serverHold.clips + serverHold.others === 1 ? 'it' : 'them'} off
                    </button>
                  </>
                )}
              </Notice>
            </div>
          ) : null}

          {/* Where the server's waiting clips live: on the server, so the page may close. Not while its queue
              is off, when nothing is sent in turn; each card says why instead. */}
          {moving.some(waitsOnServer) ? (
            <p className="mb-3 text-caption italic text-grey-700">{WAITS_ON_SERVER}</p>
          ) : null}

          {/* Running jobs */}
          {live.map((job) => (
            <div key={job.id} className="mb-4 border border-grey-300 bg-newsprint-aged p-3">
              <div className="flex items-baseline justify-between gap-3">
                <span className="text-[0.625rem] font-semibold uppercase tracking-[0.18em] text-grey-700">
                  {job.max > 1 && job.stage === 'Drawing'
                    ? `Drawing${job.pass ? `, pass ${job.pass.index} of ${job.pass.count}` : ''} · step ${job.value} of ${job.max}`
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
                    width: `${Math.round(drawnFraction(job) * 100)}%`,
                    transition: 'width 200ms linear',
                  }}
                />
              </div>

              {/* A clip from the tab's outbox has only its label here until the server lists it. */}
              <p className="mt-2 truncate text-small italic text-grey-700">
                {job.handedBefore ? job.familyLabel : job.prompt}
              </p>

              {job.id === runningJob?.id && remaining ? (
                <p className="mt-1 text-caption italic text-grey-700 tabular-nums">{remaining}</p>
              ) : null}

              {job.queuePos !== null && job.queuePos > 0 ? (
                <p className="mt-1 text-caption italic text-grey-700 tabular-nums">
                  The press is busy. This clip is {ordinal(job.queuePos + 1)} in line.
                </p>
              ) : null}

              {job.waitNote && !job.promptId ? (
                <p className="mt-1 text-caption italic text-grey-700 tabular-nums">{job.waitNote}</p>
              ) : null}

              {job.resumed ? (
                <p className="mt-1 text-caption italic text-grey-700">
                  Picked up after the page reloaded. ComfyUI reports steps and previews only to the page that sent a
                  clip, so none show here; it is filed when it lands.
                </p>
              ) : null}

              {/* The server's clips show on every device, so each says whose it is. */}
              {job.runner ? (
                <p className="mt-1 text-caption italic text-grey-700">
                  {job.sentHere ? 'Sent from this browser.' : 'Sent from another browser.'}
                </p>
              ) : null}

              {job.previewUrl ? (
                <img
                  src={job.previewUrl}
                  alt="Preview of the frame being drawn"
                  className="mt-3 w-full border border-grey-300 object-contain"
                />
              ) : null}

              {/* The queue on the server refuses a stop while it is off, so none is offered. */}
              {parked(job) ? (
                <p className="mt-1 text-caption italic text-grey-700">
                  {job.waitNote && !job.promptId ? '' : `${queueOffWhy} `}
                  {job.cancelRequested
                    ? 'The stop goes through once the queue on the server is running again.'
                    : 'It can be stopped here once the queue on the server is running again.'}
                </p>
              ) : (
                <div className="mt-2">
                  <HoldToStop jobId={job.id} label={live.length > 1 ? 'Hold to stop this one' : 'Hold to stop'} />
                </div>
              )}
            </div>
          ))}

          {/* Failure */}
          {failed ? (
            <div className="mb-4">
              <Notice
                tone={failed.status === 'cancelled' ? 'correction' : 'error'}
                title={faultTitle(failed.fault ?? faultOf(new Error(failed.error ?? '')))}
              >
                {/* A lost heavy clip holds the lane behind it (see laneHold), so
                    the desk is not free again, and a second go is not the offer.
                    On the server, the hold names the clip whose loss set it, and
                    is said held only while it keeps something back: the shell's
                    hold notice then shows on every room with Send and Call off,
                    even when what it holds is all another desk's. */}
                {faultBody(failed.fault ?? faultOf(new Error(failed.error ?? '')), failed.runner ? { held: serverHold?.jobId === failed.id } : {
                  held: failed.release && laneHeld,
                })}{' '}
                {failed.fault && faultWhere(failed.fault) ? (
                  <span className="block text-caption">{faultWhere(failed.fault)}</span>
                ) : null}
                {/* A clip the reader stopped wants no second go offered, and one
                    from the tab's outbox that was never sent has no settings to
                    put back: they were in the hand-over, and went with it. */}
                {failed.status === 'error' && !failed.handedBefore ? (
                  <>
                    <button
                      className="inline-flex items-center underline [@media(pointer:coarse)]:min-h-11"
                      onClick={() => putBackAny(failed)}
                    >
                      Put these settings back
                    </button>{' '}
                    ·{' '}
                  </>
                ) : null}
                <button
                  className="inline-flex items-center underline [@media(pointer:coarse)]:min-h-11"
                  onClick={() => dismissJob(failed.id)}
                >
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
                    ? `Made by ${shown.entry.modelLabel} · ${dateline(shown.entry.at)}${
                        // 0 is a time nobody measured, not a clip made in no time.
                        shown.entry.durationMs > 0 ? ` · ${duration(shown.entry.durationMs)}` : ''
                      }`
                    : 'From the archive'}
                </p>
                {!viewing && newestDone?.repeatOf && shown.entry?.id === newestDone.repeatOf ? (
                  <p className="mt-1 text-caption italic text-grey-700">
                    ComfyUI had made this exact clip before, so it sent the same file back without drawing it again.
                    Nothing new was filed.
                  </p>
                ) : null}
                <p className="mt-1 text-caption text-grey-500 tabular-nums">
                  {shown.frames ? `${shown.frames} frames · ${clipLength(shown.frames, shown.fps || 1)} · ` : ''}
                  {shown.fps ? `${shown.fps} fps · ` : ''}
                  {shown.entry?.width ? `${times(shown.entry.width, shown.entry.height ?? 0)} · ` : ''}
                  {shown.entry ? `seed ${shown.entry.seed} · ` : ''}
                  {shown.file.filename} · silent · no audio track
                </p>

                <div className="mt-3">
                  <p className="mb-2 text-caption leading-snug text-grey-700">
                    <span className="mr-2 text-overline font-semibold uppercase tracking-[0.18em] text-grey-700">What next</span>
                    Nothing on a clip has been measured the way the picture passes were, so no detail pass is
                    offered here. These carry the work forward.
                  </p>
                  <OfferList
                    offers={videoOffersFor(shown.entry)}
                    held={false}
                    onAction={(id) => {
                      if (id === 'save') {
                        void saveAs(fileUrl(shown.file), downloadName(shown.file, shown.entry)).catch((err: Error) =>
                          setNotice({ kind: 'error', title: 'We could not save that clip', body: err.message }),
                        )
                      } else if (id === 'continue') {
                        void continueFrom(shown.file, shown.frames, shown.fps, setSource, setNotice)
                      } else if (id === 'settings' && shown.entry) {
                        reuse(shown.entry, false)
                      } else if (id === 'again' && shown.entry) {
                        reuse(shown.entry, true)
                      } else if (id === 'toPictures') {
                        void continueFrom(
                          shown.file,
                          shown.frames,
                          shown.fps,
                          (s) => {
                            if (!s) return
                            deskStore('images').patch({ source: s, mode: 'i2i' })
                            if (onNavigate) onNavigate('#/pictures')
                            else window.location.hash = '#/pictures'
                          },
                          setNotice,
                          false,
                        )
                      }
                    }}
                  />
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
  // Over plain http the browser has no clipboard API, so copyText falls back
  // and can fail; the word says which.
  const [copied, setCopied] = useState<'yes' | 'no' | null>(null)
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
          void copyText(text).then((ok) => {
            setCopied(ok ? 'yes' : 'no')
            setTimeout(() => setCopied(null), 2000)
          })
        }}
      >
        {copied === 'yes' ? 'Copied' : copied === 'no' ? 'Could not copy' : 'Copy the JSON'}
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
  /** Switch this desk to a start frame. False when the frame is bound elsewhere. */
  andMode = true,
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
    if (andMode) store.patch({ mode: 'i2v' })
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

