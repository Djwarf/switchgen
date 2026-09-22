/**
 * The Pictures desk.
 *
 * One room, three intents: from words (text to image), from a picture
 * (image to image) and change a picture (instruction editing). The intent is a
 * tab, not a hidden mode, and each one reconfigures the rail rather than
 * revealing a different screen.
 *
 * Simple mode shows five controls and runs the registry's verified recipe.
 * Expert mode fills the left margin with the same values, already carrying
 * whatever simple mode chose, so the margin doubles as the explanation.
 *
 * The job engine lives at module scope, below the imports. A desk that keeps
 * its progress while you read the archive is the difference between a tool and
 * a demo, and there is no press.ts in this build to do it for us.
 */
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
  type RefObject,
} from 'react'
import {
  ComfyError,
  cancelJob,
  connectionState,
  fileUrl,
  getJob,
  listJobs,
  objectInfo,
  optionsFor,
  run,
  uploadImage,
  watchConnection,
  type ApiWorkflow,
  type ProgressEvent,
} from '../lib/comfy'
import {
  BY_ID,
  IMG2IMG,
  defaultsFor,
  instantiate,
  modelsOf,
  sidecarsOf,
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
import {
  add as fileRecord,
  all as allRecords,
  search as searchRecords,
  star as starRecord,
  subscribe as subscribeRecords,
  type HistoryEntry,
} from '../lib/history'
import {
  MODE_LABEL,
  adoptValue,
  applyDefaults,
  deskStore,
  needsSource,
  randomSeed,
  recordOf,
  settings,
  toParams,
  type Composition,
  type CompositionParams,
  type FamilyDefaults,
  type Mode,
  type SourceRef,
} from '../lib/session'
import {
  capabilitiesOf,
  deriveAutoDetail,
  deriveHiresFix,
  deriveRefine,
  hiresSize,
  hiresStepsFor,
  instantiateRefine,
  withLoras,
  writeExtras,
  type Capabilities,
  type DerivedDef,
  type LoraSpec,
} from '../lib/refine'
import { resolveStack } from '../lib/loras'
import { LoraRack, useLoraRack } from '../components/loras'
import { RegionRefine, type RefineRequest } from '../components/refine'
import {
  INTENTS,
  anatomyNote,
  briefFrom,
  intentReport,
  mismatchHint,
  promptStyleNote,
  type Brief,
  type Intent,
  type IntentReport,
  type Recommendation,
} from '../lib/intent'

// ---------------------------------------------------------------------------
// Vocabulary
// ---------------------------------------------------------------------------

const DESK = 'images' as const
const DESK_MODES: Mode[] = ['t2i', 'i2i', 'edit']

type Stop = { key: string; label: string; denoise: number; help: string }

/** "How much to change", in the order a person thinks about it. */
const STOPS: Stop[] = [
  {
    key: 'touch',
    label: 'Touch up',
    denoise: 0.25,
    help: 'Keeps your picture. Changes the surface and the detail.',
  },
  {
    key: 'rework',
    label: 'Rework',
    denoise: 0.45,
    help: 'Keeps the composition and the colours.',
  },
  {
    key: 'reimagine',
    label: 'Reimagine',
    denoise: 0.65,
    help: 'Keeps the rough layout only.',
  },
  {
    key: 'over',
    label: 'Start over',
    denoise: 0.85,
    help: 'Uses your picture as a loose hint.',
  },
]
const DEFAULT_DENOISE = 0.65

/** Output budget for image-to-image, where the source dictates the shape. */
const MEGAPIXELS = [0.6, 1.0, 1.4, 2.0]

type ShapeSpec = { key: string; label: string; ratio: number }
const SHAPES: ShapeSpec[] = [
  { key: 'portrait', label: 'Portrait', ratio: 2 / 3 },
  { key: 'square', label: 'Square', ratio: 1 },
  { key: 'landscape', label: 'Landscape', ratio: 3 / 2 },
  { key: 'wide', label: 'Wide', ratio: 16 / 9 },
]

/**
 * Plain names for the weight files. A reader chooses a style, not a quant.
 * Anything not named here is derived from the filename, so a new checkpoint
 * still reads sensibly on the day it lands.
 */
const PLAIN_NAMES: Record<string, string> = {
  'Z-Image-Turbo-fp8mix.safetensors': 'Z-Image Turbo',
  'Z-Image-Base-bf16.safetensors': 'Z-Image Base',
  'flux-2-klein-4b-fp8.safetensors': 'Flux.2 Klein',
  'moodyCutieMixKrea2_v50_int8.safetensors': 'Krea 2',
  'qwen_image_2.1_int8_convrot.safetensors': 'Qwen-Image 2.1',
  'qwen-image-edit-2511-Q4_K_M.gguf': 'Qwen Image Edit',
  'miaomiaoHarem_29BBETA10.safetensors': 'MiaoMiao Harem',
  'miaomiaoRealskin_anima13.safetensors': 'MiaoMiao Realskin',
  'oneObsession_anima29BV1.safetensors': 'One Obsession',
}

/** class_type → what is actually happening, so a 60-second load is not a hang. */
const STAGES: [RegExp, string][] = [
  [/^(UnetLoaderGGUF|UNETLoader|CheckpointLoaderSimple|VAELoader)$/, 'Loading the model'],
  [/^(CLIPLoader|CLIPSetLastLayer|CLIPTextEncode|TextEncodeQwen|ConditioningZeroOut)/, 'Reading the prompt'],
  [/^LoraLoaderModelOnly$/, 'Loading the LoRA'],
  [/^(LoadImage|VAEEncode|ImageScaleToTotalPixels|FluxKontext)/, 'Preparing your picture'],
  [/^(KSampler|KSamplerAdvanced|SamplerCustomAdvanced)/, 'Drawing'],
  [/^(VAEDecode|VAEDecodeTiled)$/, 'Developing the picture'],
  [/^SaveImage$/, 'Writing the file'],
]

const EXAMPLES: { prompt: string; familyId: string; shape: string }[] = [
  {
    prompt: 'A rain-slicked tram stop at dusk, neon in the puddles',
    familyId: 'krea2',
    shape: 'square',
  },
  {
    prompt: 'A portrait of a woman in a red coat, low winter sun',
    familyId: 'sdxl-illustrious',
    shape: 'portrait',
  },
  {
    prompt: 'Steam rising off wet asphalt at first light',
    familyId: 'z-image',
    shape: 'wide',
  },
]

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

const THIN = ' '
const times = (w: number, h: number) => `${w}${THIN}×${THIN}${h}`

function seconds(ms: number): string {
  if (ms < 10_000) return `${(ms / 1000).toFixed(1)}${THIN}s`
  if (ms < 90_000) return `${Math.round(ms / 1000)}${THIN}s`
  const m = Math.floor(ms / 60_000)
  const s = Math.round((ms % 60_000) / 1000)
  return `${m} min ${s}${THIN}s`
}

function bytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(0)} KB`
  return `${(n / 1024 ** 2).toFixed(1)} MB`
}

const DATE = new Intl.DateTimeFormat('en-GB', {
  day: 'numeric',
  month: 'long',
  year: 'numeric',
  hour: '2-digit',
  minute: '2-digit',
})

/**
 * How long a picture has actually taken here, read out of the archive.
 *
 * Nothing is stated until three runs of this exact model have been measured.
 * The line that used to sit in this slot was a hardcoded five-to-thirty-second
 * band that nothing measured, on a desk whose own arithmetic knows the detail
 * passes can multiply a run by 4.35 and whose slowest offered recipe is 30
 * steps. Video.tsx and the progress slug both refuse to state a time on less
 * evidence than this; so does this.
 */
function timingNote(records: readonly HistoryEntry[], model: string): string | null {
  const ms: number[] = []
  for (const r of records) {
    if (r.desk !== DESK || r.kind !== 'image' || r.model !== model) continue
    if (!(r.durationMs > 0)) continue
    ms.push(r.durationMs)
    if (ms.length === 20) break
  }
  if (ms.length < 3) return null
  ms.sort((a, b) => a - b)
  const lo = ms[0]
  const hi = ms[ms.length - 1]
  if (lo === hi) return `Your last ${ms.length} pictures with this model each took ${seconds(lo)}.`
  return `Your last ${ms.length} pictures with this model took ${seconds(lo)} to ${seconds(hi)}.`
}

/** A fragment from elsewhere, made into a sentence. */
function sentence(text: string): string {
  const t = text.trim().replace(/[.\s]+$/, '')
  return t ? `${t.charAt(0).toUpperCase()}${t.slice(1)}.` : ''
}

const round2 = (n: number) => Math.round(n * 100) / 100
const snap16 = (n: number) => Math.max(16, Math.round(n / 16) * 16)
const clamp = (n: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, n))

function titleFromFilename(file: string): string {
  const stem = file.replace(/\.[^.]+$/, '')
  const cleaned = stem
    .replace(/[_.]/g, ' ')
    .replace(/\b(fp8|fp16|bf16|int8|gguf|Q\d+_K_[MS]|safetensors|mix|convrot)\b/gi, '')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/\s+/g, ' ')
    .trim()
  return cleaned.charAt(0).toUpperCase() + cleaned.slice(1)
}

/** The family's name without its parenthetical, for the picker's groups. */
const groupName = (def: FamilyDef) => def.label.split('(')[0].replace(/[—-]\s*$/, '').trim()

function stageFor(classType: string | null | undefined, value: number, max: number): string {
  if (!classType) return 'Working'
  for (const [test, name] of STAGES) {
    if (!test.test(classType)) continue
    if (name === 'Drawing' && max > 1) return `Drawing · step ${value} of ${max}`
    return name
  }
  return 'Working'
}

// ---------------------------------------------------------------------------
// The catalogue — what this machine can actually run, right now
// ---------------------------------------------------------------------------

export type Style = {
  model: string
  def: FamilyDef
  /** Plain name, e.g. "Krea 2". */
  label: string
  /** Family name, for the picker's groups. */
  group: string
  verdict: Verdict | null
  /** The author-card guidance the registry carries for this file. */
  note: string
  clipSkip: number | null
  positivePrefix: string | null
  shift: number | null
  maxSide: number | null
  alt: { sampler: string; scheduler: string; steps: number; cfg: number; note?: string } | null
}

type Catalogue = {
  styles: Style[]
  unavailable: { name: string; why: string }[]
  samplers: string[]
  schedulers: string[]
  hardware: Hardware | null
  /** Every weight file ComfyUI reports, family or no family. */
  installed: string[]
  /** Real file sizes, so the intent ranking can weigh a model against the RAM. */
  sizes: Map<string, ModelFile>
}

function readPerModel(def: FamilyDef, model: string) {
  return (def.perModel?.[model] ?? {}) as Record<string, unknown>
}

function num(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function styleOf(def: FamilyDef, model: string, verdict: Verdict | null): Style {
  const per = readPerModel(def, model)
  const altRaw = per.altSampler as Record<string, unknown> | undefined
  const alt =
    altRaw && typeof altRaw.sampler === 'string'
      ? {
          sampler: String(altRaw.sampler),
          scheduler: String(altRaw.scheduler ?? def.defaults.scheduler),
          steps: num(altRaw.steps) ?? def.defaults.steps,
          cfg: num(altRaw.cfg) ?? def.defaults.cfg,
          note: typeof altRaw.note === 'string' ? altRaw.note : undefined,
        }
      : null
  const note =
    (typeof per.note === 'string' && per.note) || (typeof per.notes === 'string' && per.notes) || ''
  return {
    model,
    def,
    label: typeof per.label === 'string' ? per.label : (PLAIN_NAMES[model] ?? titleFromFilename(model)),
    group: groupName(def),
    verdict,
    note,
    clipSkip: num(per.clipSkip),
    positivePrefix: typeof per.positivePrefix === 'string' ? per.positivePrefix : null,
    shift: num(per.shift),
    maxSide: num(per.maxSide),
    alt,
  }
}

async function readCatalogue(): Promise<Catalogue> {
  const [info, hardware, sizes] = await Promise.all([
    objectInfo(),
    probeHardware().catch(() => null),
    modelFiles().catch(() => new Map<string, ModelFile>()),
  ])

  const clips = optionsFor(info, 'CLIPLoader', 'clip_name')
  const vaes = optionsFor(info, 'VAELoader', 'vae_name')
  const loras = optionsFor(info, 'LoraLoaderModelOnly', 'lora_name')
  const installed = new Set([
    ...optionsFor(info, 'CheckpointLoaderSimple', 'ckpt_name'),
    ...optionsFor(info, 'UNETLoader', 'unet_name'),
    ...optionsFor(info, 'UnetLoaderGGUF', 'unet_name'),
  ])

  const styles: Style[] = []
  const unavailable: { name: string; why: string }[] = []

  for (const model of installed) {
    const def = familyOwning(model)
    if (!def || (def.mode !== 'image' && def.mode !== 'edit')) continue

    // Every file the graph references must exist, or the run fails with an
    // opaque backend error. Name the missing file instead.
    const { clip, vae } = sidecarsOf(def)
    const missing = [
      ...clip.filter((c) => !clips.includes(c)),
      ...(vae && !vaes.includes(vae) ? [vae] : []),
      ...modelsOf(def).filter((m) => !installed.has(m)),
      ...Object.values(def.graph)
        .map((n) => n.inputs['lora_name'])
        .filter((l): l is string => typeof l === 'string' && !loras.includes(l)),
    ]
    if (missing.length) {
      unavailable.push({ name: model, why: `needs ${missing.join(', ')}` })
      continue
    }

    const verdict = hardware ? feasibility(def, sizes, hardware) : null
    if (verdict && !verdict.selectable) {
      unavailable.push({ name: model, why: verdict.reason })
      continue
    }
    styles.push(styleOf(def, model, verdict))
  }

  styles.sort((a, b) => a.group.localeCompare(b.group) || a.label.localeCompare(b.label))

  return {
    styles,
    unavailable,
    samplers: optionsFor(info, 'KSampler', 'sampler_name'),
    schedulers: optionsFor(info, 'KSampler', 'scheduler'),
    hardware,
    installed: [...installed],
    sizes,
  }
}

/** The family that lists this exact file. Hints are not good enough for a picker. */
function familyOwning(model: string): FamilyDef | null {
  for (const def of Object.values(BY_ID)) if (def.models.includes(model)) return def
  return null
}

let cataloguePromise: Promise<Catalogue> | null = null
function catalogue(reload = false): Promise<Catalogue> {
  if (reload || !cataloguePromise) cataloguePromise = readCatalogue()
  return cataloguePromise
}

// ---------------------------------------------------------------------------
// The job engine — module scope, so a job outlives a route change
// ---------------------------------------------------------------------------

type DeskJob = {
  id: string
  promptId: string | null
  status: 'submitting' | 'queued' | 'running' | 'done' | 'error' | 'cancelled'
  stage: string
  value: number
  max: number
  /** 0…0.97 while running; 1 when done. Never jumps backwards. */
  pct: number
  previewUrl: string | null
  label: string
  index: number
  total: number
  startedAt: number
  finishedAt: number | null
}

type DeskFault = {
  message: string
  cancelled: boolean
  /** True when ComfyUI no longer has any record of the job. See watchLost. */
  lost: boolean
  node: string | null
  nodeType: string | null
  detail: string | null
}

/**
 * A job ComfyUI has forgotten, as opposed to one it refused.
 *
 * Distinct from ComfyError so the desk can say which of the two happened: a
 * rejected graph is the reader's settings, a lost job is the server.
 */
class LostJob extends Error {
  readonly promptId: string
  constructor(message: string, promptId: string) {
    super(message)
    this.name = 'LostJob'
    this.promptId = promptId
  }
}

type PressState = {
  job: DeskJob | null
  /** Everything this desk has made since the page loaded, newest first. */
  results: HistoryEntry[]
  current: HistoryEntry | null
  fault: DeskFault | null
  /** Duration of the last finished run, for the button's quiet receipt. */
  lastMs: number | null
}

type RunPlan = {
  graph: ApiWorkflow
  composition: Composition
  seed: number
  familyLabel: string
  modelLabel: string
  variant: HistoryEntry['variant']
  label: string
  /**
   * What the graph was derived with. Neither lives on the Composition, so the
   * plan has to carry them or the record files a plain single pass render for
   * a picture that had two passes, a face detailer and a LoRA chain on it.
   */
  passes: Passes
  loras: LoraSpec[]
}

let press: PressState = { job: null, results: [], current: null, fault: null, lastMs: null }
const pressListeners = new Set<() => void>()

function emit(patch: Partial<PressState>) {
  press = { ...press, ...patch }
  for (const fn of [...pressListeners]) {
    try {
      fn()
    } catch {
      /* one broken subscriber must not stop the rest */
    }
  }
}

function patchJob(patch: Partial<DeskJob>) {
  if (!press.job) return
  emit({ job: { ...press.job, ...patch } })
}

export const subscribePress = (fn: () => void) => {
  pressListeners.add(fn)
  return () => {
    pressListeners.delete(fn)
  }
}
export const pressSnapshot = () => press

let queue: RunPlan[] = []
let driving = false
let stopped = false

function busy(state: PressState): boolean {
  const s = state.job?.status
  return s === 'submitting' || s === 'queued' || s === 'running'
}

/** How often to ask the server whether it still knows about our prompt. */
const LOST_POLL_MS = 5000
/** Consecutive absences from /api/jobs before the direct lookup. */
const LOST_MISSES = 2
/** A prompt accepted seconds ago may not be listed yet. */
const LOST_GRACE_MS = 20_000
/** Ticks to wait for a socket event the server says has already happened. */
const LOST_TERMINAL_WAITS = 3

/**
 * Reject when ComfyUI has forgotten the prompt this run is following.
 *
 * `run()` settles only on a terminal socket event carrying its own prompt id,
 * and a ComfyUI that restarts mid generation never sends one: comfy.ts's own
 * reconciler bails when /history has no record of the id, settling nothing.
 * Without this the await in drive() never returns, so `driving` stays true and
 * every later press of the run button is refused in silence, with no way back
 * but a page reload.
 *
 * The evidence is the same evidence Video.tsx and shell/jobs.tsx already use:
 * two consecutive absences from the server's own queue, confirmed by a direct
 * lookup that finds no record either. A terminal status with no socket event
 * gets a few more ticks first, because the socket is the authority and usually
 * lands within one.
 */
function watchLost(): { promise: Promise<never>; stop: () => void } {
  let timer: ReturnType<typeof setInterval> | null = null
  const stop = () => {
    if (timer !== null) clearInterval(timer)
    timer = null
  }

  const promise = new Promise<never>((_resolve, reject) => {
    const startedAt = Date.now()
    let sighted = false
    let misses = 0
    let terminalWaits = 0
    let checking = false

    const giveUp = (message: string, promptId: string) => {
      stop()
      reject(new LostJob(message, promptId))
    }

    const tick = async () => {
      if (checking || timer === null) return
      checking = true
      try {
        const id = press.job?.promptId
        if (!id) return // still submitting: there is nothing to look for yet

        let ids: string[]
        try {
          const page = await listJobs({ status: ['pending', 'in_progress'], limit: 100 })
          ids = page.jobs.map((j) => j.id)
        } catch {
          return // an unreachable server is the offline notice's business
        }
        if (timer === null) return

        if (ids.includes(id)) {
          sighted = true
          misses = 0
          return
        }
        if (!sighted && Date.now() - startedAt < LOST_GRACE_MS) return

        misses += 1
        if (misses < LOST_MISSES) return

        const server = await getJob(id).catch(() => null)
        if (timer === null) return

        if (server && (server.status === 'pending' || server.status === 'in_progress')) {
          sighted = true
          misses = 0
          return
        }
        if (server) {
          misses = 0
          terminalWaits += 1
          if (terminalWaits < LOST_TERMINAL_WAITS) return
          giveUp(
            'ComfyUI says this job has ended but never sent the result. Look in the archive: the picture may be on disk anyway.',
            id,
          )
          return
        }
        giveUp(
          'We lost track of this job. ComfyUI has no record of it any more, which usually means it restarted. Nothing was saved.',
          id,
        )
      } finally {
        checking = false
      }
    }

    timer = setInterval(() => void tick(), LOST_POLL_MS)
  })

  return { promise, stop }
}

function startRuns(plans: RunPlan[]) {
  if (driving || !plans.length) return
  queue = [...plans]
  stopped = false
  void drive()
}

async function drive() {
  driving = true
  let index = 0
  const total = queue.length

  try {
    while (queue.length && !stopped) {
      const plan = queue.shift()!
      index += 1
      const startedAt = Date.now()
      emit({
        fault: null,
        job: {
          id: `${startedAt.toString(36)}-${index}`,
          promptId: null,
          status: 'submitting',
          stage: 'Sending the job',
          value: 0,
          max: 0,
          pct: 0,
          previewUrl: null,
          label: plan.label,
          index,
          total,
          startedAt,
          finishedAt: null,
        },
      })

      // The run is raced against a watch on the server's own queue, because
      // run() settles only on a socket event for this prompt and a restarted
      // ComfyUI never sends one.
      const lost = watchLost()
      try {
        const files = await Promise.race([
          run(plan.graph, (ev) => onProgress(ev, plan)),
          lost.promise,
        ])
        const picture = files.find((f) => f.kind === 'image') ?? files[0] ?? null
        const durationMs = Date.now() - startedAt
        if (!picture) {
          emit({
            fault: {
              message: 'The job finished but wrote no file. Check ComfyUI’s own log for the reason.',
              cancelled: false,
              lost: false,
              node: null,
              nodeType: null,
              detail: null,
            },
          })
          patchJob({ status: 'error', finishedAt: Date.now() })
          continue
        }
        const record = recordOf(plan.composition, {
          file: picture,
          files: files.length > 1 ? files : undefined,
          kind: picture.kind,
          promptId: press.job?.promptId ?? '',
          durationMs,
          seed: plan.seed,
          familyLabel: plan.familyLabel,
          modelLabel: plan.modelLabel,
          variant: plan.variant,
          passes: plan.passes,
          loras: plan.loras,
        })
        // A full or unwritable archive must never present as a failed picture:
        // the file is on disk either way, so show it and carry on.
        let entry: HistoryEntry
        try {
          entry = fileRecord(record)
        } catch {
          entry = {
            ...record,
            id: `unfiled-${startedAt.toString(36)}`,
            no: 0,
            at: record.at ?? Date.now(),
          }
        }
        emit({
          results: [entry, ...press.results],
          current: entry,
          lastMs: durationMs,
        })
        patchJob({ status: 'done', pct: 1, stage: 'Done', finishedAt: Date.now() })
      } catch (err) {
        const fault = faultOf(err)
        emit({ fault })
        patchJob({ status: fault.cancelled ? 'cancelled' : 'error', finishedAt: Date.now() })
        // A rejected queue or a stopped job ends the whole batch: three more of
        // the same mistake helps nobody.
        break
      } finally {
        lost.stop()
      }
    }
  } finally {
    // Every exit releases the desk. Releasing only on the happy path is what
    // used to wedge the run button for the rest of the page's life.
    queue = []
    driving = false
    stopped = false
  }
}

function onProgress(ev: ProgressEvent, plan: RunPlan) {
  if (!press.job) return
  if (ev.phase === 'queued') {
    patchJob({ promptId: ev.promptId, status: 'queued', stage: 'Queued' })
    return
  }
  if (ev.phase === 'preview') {
    patchJob({ previewUrl: ev.url })
    return
  }
  if (ev.phase !== 'running') return

  const cls = ev.node ? plan.graph[ev.node]?.class_type : null
  const stage = stageFor(cls, ev.value, ev.max)
  const sampling = ev.max > 1
  const pct = sampling
    ? clamp(ev.value / ev.max, press.job.pct, 0.97)
    : Math.max(press.job.pct, 0.02)
  patchJob({ status: 'running', stage, value: ev.value, max: ev.max, pct })
}

function faultOf(err: unknown): DeskFault {
  if (err instanceof LostJob) {
    return {
      message: err.message,
      cancelled: false,
      lost: true,
      node: null,
      nodeType: null,
      detail: null,
    }
  }
  if (err instanceof ComfyError) {
    const detail = err.nodeErrors
      ? Object.values(err.nodeErrors)
          .flatMap((n) => {
            const errs = (n as { errors?: { message?: string; details?: string }[] })?.errors ?? []
            return errs.map((e) => [e.message, e.details].filter(Boolean).join(': '))
          })
          .filter(Boolean)
          .join('; ')
      : null
    return {
      message: err.message,
      cancelled: err.cancelled,
      lost: false,
      node: err.node,
      nodeType: err.nodeType,
      detail: detail || null,
    }
  }
  return {
    message: err instanceof Error ? err.message : String(err),
    cancelled: false,
    lost: false,
    node: null,
    nodeType: null,
    detail: null,
  }
}

async function stopRun() {
  stopped = true
  queue = []
  const id = press.job?.promptId
  if (!id) {
    patchJob({ status: 'cancelled', finishedAt: Date.now() })
    return
  }
  try {
    await cancelJob(id)
  } catch {
    /* the watcher reports the outcome; a failed cancel is not a second error */
  }
}

/** Show a finished picture on the plate without re-running anything. */
function showResult(entry: HistoryEntry) {
  emit({ current: entry })
}

// ---------------------------------------------------------------------------
// Small shared pieces
// ---------------------------------------------------------------------------

const store = deskStore(DESK)

function useComposition(): Composition {
  return useSyncExternalStore(store.subscribe, store.get, store.get)
}

function useExpert(): boolean {
  const s = useSyncExternalStore(settings.subscribe, settings.get, settings.get)
  return s.expert
}

function usePress(): PressState {
  return useSyncExternalStore(subscribePress, pressSnapshot, pressSnapshot)
}

function useRecords(): readonly HistoryEntry[] {
  return useSyncExternalStore(subscribeRecords, allRecords, allRecords)
}

function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(
    () => typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches,
  )
  useEffect(() => {
    if (typeof matchMedia !== 'function') return
    const mq = matchMedia('(prefers-reduced-motion: reduce)')
    const on = () => setReduced(mq.matches)
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [])
  return reduced
}

type NoticeKind = 'info' | 'correction' | 'error' | 'warning'

const NOTICE_STYLE: Record<NoticeKind, string> = {
  info: 'bg-burgundy-50 border-burgundy-900 text-ink',
  correction: 'bg-[#F5F5F5] border-ink text-ink italic',
  error: 'bg-[#FEF2F2] border-error text-[#7F1D1D]',
  warning: 'bg-[#FFFBEB] border-warning text-[#78350F]',
}

function Notice({
  kind = 'info',
  title,
  children,
}: {
  kind?: NoticeKind
  title?: string
  children?: ReactNode
}) {
  return (
    <div
      role={kind === 'error' ? 'alert' : 'status'}
      className={`border-l-4 px-4 py-3 text-small leading-normal ${NOTICE_STYLE[kind]}`}
    >
      {title && (
        <strong className="mr-2 text-small font-bold uppercase not-italic tracking-[0.05em]">
          {title}
        </strong>
      )}
      {children}
    </div>
  )
}

function Kicker({ children, className = '' }: { children: ReactNode; className?: string }) {
  return (
    <span
      className={`text-overline font-semibold uppercase tracking-[0.18em] text-grey-700 ${className}`}
    >
      {children}
    </span>
  )
}

function Label({ children, hint }: { children: ReactNode; hint?: string }) {
  return (
    <span className="mb-1.5 block text-overline font-semibold uppercase tracking-[0.18em] text-grey-700">
      {children}
      {hint && (
        <span className="ml-2 normal-case tracking-normal text-caption italic text-grey-500">
          {hint}
        </span>
      )}
    </span>
  )
}

/** An editorial link: burgundy, underlined, no button chrome. */
function Link({
  onClick,
  children,
  className = '',
  title,
}: {
  onClick: () => void
  children: ReactNode
  className?: string
  title?: string
}) {
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      className={`cursor-pointer text-burgundy-900 underline decoration-burgundy-900/60 underline-offset-2 hover:decoration-burgundy-900 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900 ${className}`}
    >
      {children}
    </button>
  )
}

// ---------------------------------------------------------------------------
// Shapes
// ---------------------------------------------------------------------------

type Shape = ShapeSpec & { width: number; height: number; maker: boolean }

/**
 * Four proportional rectangles, sized against the family's own pixel budget.
 * The bucket nearest the family's default ratio carries the maker's exact
 * numbers, so "Portrait" on Illustrious really is 832 × 1216.
 */
function shapesFor(style: Style | null): Shape[] {
  const d = style ? defaultsFor(style.def, style.model) : null
  const dw = d?.width || 1024
  const dh = d?.height || 1024
  const area = dw * dh
  const ratio = dw / dh
  const maxSide = style?.maxSide ?? null

  let nearest = 0
  let best = Infinity
  SHAPES.forEach((s, i) => {
    const gap = Math.abs(Math.log(s.ratio) - Math.log(ratio))
    if (gap < best) {
      best = gap
      nearest = i
    }
  })

  return SHAPES.map((s, i) => {
    if (i === nearest) return { ...s, width: dw, height: dh, maker: true }
    let w = snap16(Math.sqrt(area * s.ratio))
    let h = snap16(Math.sqrt(area / s.ratio))
    if (maxSide && Math.max(w, h) > maxSide) {
      const k = maxSide / Math.max(w, h)
      w = snap16(w * k)
      h = snap16(h * k)
    }
    return { ...s, width: w, height: h, maker: false }
  })
}

// ---------------------------------------------------------------------------
// Quality passes
//
// Three derivations from lib/refine.ts that ride on the ordinary render. Order
// matters and is not arbitrary:
//
//   hires first, because it changes the frame the picture is drawn in,
//   then the detailers, which work on the finished pixels of that frame,
//   then the LoRAs, which patch the model every one of those passes samples on.
//
// Each derivation returns null when the family cannot carry it, and the fall
// back here is the graph as it was. The toggles are gated on capabilitiesOf()
// so a null is never reached from the UI, but a silent fall back beats a crash
// if a registry change ever makes one of them impossible.
// ---------------------------------------------------------------------------

type Passes = { face: boolean; hand: boolean; hires: boolean }

/**
 * A finished picture, copied into ComfyUI's input folder and measured, ready
 * for a refine pass. Both halves are needed before anything can be queued:
 * LoadImage reads the input folder, and the crop arithmetic needs real pixels.
 */
type RefineSource = {
  /** Filename in ComfyUI's input folder, for LoadImage. */
  name: string
  /** Where the browser shows it from, which is still the output folder. */
  url: string
  width: number
  height: number
  entryId: string
  prompt: string
}

const NO_PASSES: Passes = { face: false, hand: false, hires: false }

function withPasses(def: FamilyDef, passes: Passes, loras: LoraSpec[]): FamilyDef {
  let out: FamilyDef | DerivedDef = def
  if (passes.hires) out = deriveHiresFix(out) ?? out
  if (passes.face) out = deriveAutoDetail(out, 'face') ?? out
  if (passes.hand) out = deriveAutoDetail(out, 'hand') ?? out
  if (loras.length) out = withLoras(out, loras) ?? out
  return out
}

/**
 * What the passes cost, as a multiple of one plain generation.
 *
 * A detailer runs once per detection, and nobody knows how many faces are in a
 * picture before it is drawn, so the figure assumes one of each. Two people in
 * frame is two face passes. The copy says "at least" for that reason.
 */
function passCost(passes: Passes): number {
  let cost = 1
  if (passes.hires) cost += 1.35
  if (passes.face) cost += 1
  if (passes.hand) cost += 1
  return cost
}

// ---------------------------------------------------------------------------
// Two settings the bindings cannot carry
//
// Shift and clip skip have no binding in registry.ts and instantiate() writes
// neither, so both expert rows used to be collected, dropped on the floor, and
// then filed in the archive as though they had been applied. They are written
// here by class_type instead, on the instantiated graph, because registry.ts
// is generated and must not be hand edited. Video.tsx does the shift half the
// same way, for the same reason.
// ---------------------------------------------------------------------------

/** ModelSampling* shift, written by class_type rather than by a binding. */
function applyShift(wf: ApiWorkflow, shift: number | null | undefined): void {
  if (shift === null || shift === undefined || !Number.isFinite(shift)) return
  for (const node of Object.values(wf)) {
    if (node.class_type.startsWith('ModelSampling') && typeof node.inputs.shift === 'number') {
      node.inputs.shift = shift
    }
  }
}

/**
 * CLIPSetLastLayer, likewise.
 *
 * ComfyUI counts from the end, so the value is negative: -1 is the last layer,
 * -2 the one before it, which is what the Illustrious and Pony checkpoints
 * were trained against. Anything else would be written straight through, so it
 * is clamped to the range the node accepts.
 */
function applyClipSkip(wf: ApiWorkflow, clipSkip: number | null | undefined): void {
  if (clipSkip === null || clipSkip === undefined || !Number.isFinite(clipSkip)) return
  const layer = clamp(Math.round(clipSkip), -24, -1)
  for (const node of Object.values(wf)) {
    if (
      node.class_type === 'CLIPSetLastLayer' &&
      typeof node.inputs.stop_at_clip_layer === 'number'
    ) {
      node.inputs.stop_at_clip_layer = layer
    }
  }
}

/**
 * The graph exactly as it will be queued.
 *
 * One function, called by the run button and by "Show the workflow" alike, so
 * the JSON on screen cannot drift from the JSON that goes to the server. The
 * second pass runs fewer steps than the first; see hiresStepsFor.
 */
function buildQueued(
  def: FamilyDef | DerivedDef,
  params: CompositionParams,
  hires: boolean,
): ApiWorkflow {
  const wf = instantiate(def, params)
  if (hires && 'derived' in def) {
    writeExtras(wf, def as DerivedDef, { hiresSteps: hiresStepsFor(params.steps) })
  }
  applyShift(wf, params.shift)
  applyClipSkip(wf, params.clipSkip)
  return wf
}

// ---------------------------------------------------------------------------
// The desk
// ---------------------------------------------------------------------------

/** Which picture style the reader was on before they went to edit a picture. */
let lastPictureStyle: { familyId: string; model: string } | null = null

export function Pictures() {
  const c = useComposition()
  const expert = useExpert()
  const state = usePress()
  const records = useRecords()
  const reduced = useReducedMotion()

  const [cat, setCat] = useState<Catalogue | null>(null)
  const [catError, setCatError] = useState<string | null>(null)
  const [correction, setCorrection] = useState<string | null>(null)
  const [sourceError, setSourceError] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [picking, setPicking] = useState(false)
  const [adopted, setAdopted] = useState<string | null>(null)
  const [focusField, setFocusField] = useState<string | null>(null)
  const [offline, setOffline] = useState(() => connectionState() === 'closed')
  const [ahead, setAhead] = useState(0)
  const [retryIn, setRetryIn] = useState(5)

  // Quality passes, the region refine surface and the reader's override of the
  // brief the prompt implies. None of these belong in Composition: session.ts
  // is another agent's file and a desk-local preference does not need filing.
  const [passes, setPasses] = useState<Passes>(NO_PASSES)
  const [intentPick, setIntentPick] = useState<Intent | null>(null)
  const [explicitPick, setExplicitPick] = useState<boolean | null>(null)
  const [refining, setRefining] = useState(false)
  const [refineSource, setRefineSource] = useState<RefineSource | null>(null)
  const [refineResult, setRefineResult] = useState<HistoryEntry | null>(null)
  const [refineFault, setRefineFault] = useState<string | null>(null)
  const [openingRefine, setOpeningRefine] = useState(false)
  const refineToken = useRef(0)
  /** True between queueing a refine and its result landing on the plate. */
  const awaitingRefine = useRef(false)

  const fileInput = useRef<HTMLInputElement | null>(null)
  const promptRef = useRef<HTMLTextAreaElement | null>(null)
  const dragDepth = useRef(0)

  const running = busy(state)

  // --- the catalogue ------------------------------------------------------
  const load = useCallback((reload = false) => {
    setCatError(null)
    catalogue(reload).then(
      (next) => setCat(next),
      (err: unknown) =>
        setCatError(err instanceof Error ? err.message : 'ComfyUI did not answer.'),
    )
  }, [])

  useEffect(() => {
    load()
  }, [load])

  // The desk recovers on its own when ComfyUI comes back — no reload.
  useEffect(() => {
    if (!catError) return
    let left = 5
    setRetryIn(left)
    const timer = setInterval(() => {
      left -= 1
      if (left <= 0) {
        left = 5
        load(true)
      }
      setRetryIn(left)
    }, 1000)
    return () => clearInterval(timer)
  }, [catError, load])

  useEffect(() => watchConnection((s) => setOffline(s === 'closed')), [])

  // --- what is in front of us in the single queue -------------------------
  useEffect(() => {
    let alive = true
    const poll = async () => {
      try {
        const page = await listJobs({ status: ['pending', 'in_progress'], limit: 20 })
        if (!alive) return
        const mine = press.job?.promptId
        setAhead(page.jobs.filter((j) => j.id !== mine).length)
      } catch {
        if (alive) setAhead(0)
      }
    }
    void poll()
    const timer = setInterval(poll, 5000)
    return () => {
      alive = false
      clearInterval(timer)
    }
  }, [])

  // --- styles -------------------------------------------------------------
  const styles = useMemo(() => cat?.styles ?? [], [cat])
  const pictureStyles = useMemo(() => styles.filter((s) => s.def.mode === 'image'), [styles])
  const editStyle = useMemo(() => styles.find((s) => s.def.mode === 'edit') ?? null, [styles])
  const style = useMemo(
    () => styles.find((s) => s.model === c.model && s.def.id === c.familyId) ?? null,
    [styles, c.model, c.familyId],
  )

  const applyStyle = useCallback((next: Style) => {
    const d = defaultsFor(next.def, next.model)
    const current = store.get()
    const fd: FamilyDefaults = {
      familyId: next.def.id,
      model: next.model,
      steps: d.steps || current.steps,
      cfg: typeof d.cfg === 'number' ? d.cfg : current.cfg,
      width: d.width || current.width,
      height: d.height || current.height,
      sampler: d.sampler || current.sampler,
      scheduler: d.scheduler || current.scheduler,
      negative: d.negative || undefined,
      clipSkip: next.clipSkip ?? undefined,
      positivePrefix: next.positivePrefix ?? undefined,
    }
    const applied = applyDefaults(current, fd)
    const touched = new Set(applied.touched)
    store.set({
      ...applied,
      shift: touched.has('shift') ? applied.shift : next.shift,
      megapixels: touched.has('megapixels')
        ? applied.megapixels
        : round2(((d.width || 1024) * (d.height || 1024)) / 1e6),
    })
  }, [])

  // Settle on a style that exists. A draft naming an uninstalled file is a
  // correction, not a silent swap.
  useEffect(() => {
    if (!cat || !cat.styles.length) return
    const known = cat.styles.some((s) => s.model === c.model && s.def.id === c.familyId)
    if (known) return
    const want = c.mode === 'edit' ? editStyle : (pictureStyles[0] ?? editStyle)
    if (!want) return
    if (c.model) {
      setCorrection(
        `${PLAIN_NAMES[c.model] ?? titleFromFilename(c.model)} is not installed any more. We loaded ${want.label} instead. Your prompt and settings are untouched.`,
      )
    }
    applyStyle(want)
    // Deliberately keyed on the catalogue alone: this runs when the model list
    // lands, not every time the reader changes style.

  }, [cat])

  // An intent the installed style cannot serve is corrected out loud, never by
  // quietly running something else.
  useEffect(() => {
    if (!cat || !cat.styles.length) return
    if (c.mode === 'i2i' && c.familyId && !IMG2IMG[c.familyId]) store.patch({ mode: 't2i' })
    if (c.mode === 'edit' && !editStyle) store.patch({ mode: 't2i' })
  }, [cat, c.mode, c.familyId, editStyle])

  // --- capability ---------------------------------------------------------
  const canI2I = !!(c.familyId && IMG2IMG[c.familyId])
  /**
   * Whether the *tab* can be offered, which is a different question while the
   * reader is on the edit family: leaving edit restores their picture style,
   * and that is the style the tab would run.
   */
  const tabI2I =
    c.mode === 'edit'
      ? !!pictureStyles.find(
          (s) =>
            s.def.id === (lastPictureStyle?.familyId ?? pictureStyles[0]?.def.id) &&
            IMG2IMG[s.def.id],
        )
      : canI2I
  const i2iAlternative = useMemo(
    () => pictureStyles.find((s) => IMG2IMG[s.def.id]) ?? null,
    [pictureStyles],
  )

  const activeDef: FamilyDef | null = useMemo(() => {
    if (!c.familyId) return null
    if (c.mode === 'i2i') return IMG2IMG[c.familyId] ?? null
    return BY_ID[c.familyId] ?? null
  }, [c.familyId, c.mode])

  const hasNegative = !!activeDef?.bindings.negative
  const hasScheduler = !!activeDef?.bindings.scheduler
  const hasSize = !!activeDef?.bindings.width
  // Both rows are gated on a node that can actually receive the value, and on
  // the input existing on it: ModelSamplingDiscrete is a ModelSampling* node
  // with no shift at all. The test is the same one applyShift and applyClipSkip
  // use, so the control, the graph and the archive record cannot disagree.
  const hasShift = !!activeDef && Object.values(activeDef.graph).some(
    (n) => n.class_type.startsWith('ModelSampling') && typeof n.inputs.shift === 'number',
  )
  const hasClipSkip = !!activeDef && Object.values(activeDef.graph).some(
    (n) => n.class_type === 'CLIPSetLastLayer' && typeof n.inputs.stop_at_clip_layer === 'number',
  )

  const shapes = useMemo(() => shapesFor(style), [style])
  const houseNegative = style ? (defaultsFor(style.def, style.model).negative ?? '') : ''

  // --- quality passes, LoRAs and the brief --------------------------------

  /** The plain text-to-image graph, which is what a refine crop is drawn with. */
  const baseDef: FamilyDef | null = useMemo(
    () => (c.familyId ? (BY_ID[c.familyId] ?? null) : null),
    [c.familyId],
  )

  /** What the graph in front of us can actually carry. Never guessed. */
  const caps: Capabilities | null = useMemo(
    () => (activeDef ? capabilitiesOf(activeDef) : null),
    [activeDef],
  )
  /**
   * Which style draws the region.
   *
   * Normally the one in the picker. But the edit family, which is the "change a
   * picture" the user relies on, cannot carry a refine pass at all: its
   * conditioning carries reference latents that a detached crop would misread,
   * so deriveRefine returns null for it. Refusing to refine an edited picture
   * would leave exactly the pictures with the worst anatomy unfixable, so the
   * bench falls back to the picture style the reader was last on and says so in
   * print. Rendering a region with a different checkpoint from the frame around
   * it is ordinary inpainting practice; doing it without saying so is not.
   */
  const refineStyle = useMemo(() => {
    if (style && deriveRefine(style.def)) return style
    const back = pictureStyles.find(
      (s) => s.def.id === lastPictureStyle?.familyId && s.model === lastPictureStyle?.model,
    )
    if (back && deriveRefine(back.def)) return back
    return pictureStyles.find((s) => deriveRefine(s.def) !== null) ?? null
  }, [style, pictureStyles])

  const canRefine = !!refineStyle

  /** True when the region will be drawn by something other than the picker's style. */
  const refineBorrows =
    !!refineStyle && !!style && (refineStyle.def.id !== style.def.id || refineStyle.model !== style.model)

  // A pass the style cannot carry must not stay switched on behind the scenes.
  useEffect(() => {
    if (!caps) return
    setPasses((p) => {
      const next = {
        face: p.face && caps.faceDetail,
        hand: p.hand && caps.handDetail,
        hires: p.hires && caps.hires,
      }
      return next.face === p.face && next.hand === p.hand && next.hires === p.hires ? p : next
    })
  }, [caps])

  const rack = useLoraRack(baseDef, c.model)

  /**
   * The stack is resolved twice, once per pass. A LoRA catalogued as close
   * framing work does almost nothing across a whole body and does its whole
   * job inside the crop, so resolveStack holds it back from the first render
   * and hands it over for the refine. `noLora` is the reader saying: none.
   */
  const basePick = useMemo(
    () => resolveStack(c.noLora ? [] : rack.stack, rack.lib, rack.target, 'base'),
    [c.noLora, rack.stack, rack.lib, rack.target],
  )
  const refinePick = useMemo(
    () => resolveStack(c.noLora ? [] : rack.stack, rack.lib, rack.target, 'refine'),
    [c.noLora, rack.stack, rack.lib, rack.target],
  )

  /**
   * The family graph with the passes derived and the LoRAs patched in: what
   * the button will queue, and what "Show the workflow" must show. Built once
   * so the two cannot disagree.
   */
  const queuedDef: FamilyDef | DerivedDef | null = useMemo(
    () => (activeDef ? withPasses(activeDef, passes, basePick.specs) : null),
    [activeDef, passes, basePick],
  )

  /**
   * The refine graph, LoRAs and all. Null means: do not offer the pass.
   *
   * The stack is held back when the region is drawn by a borrowed style: the
   * rack resolved it against the picker's architecture, and an SDXL LoRA on a
   * Qwen base loads without error and changes nothing. Sending it anyway would
   * be a silent lie about what patched the weights.
   */
  const refineDef: DerivedDef | null = useMemo(() => {
    if (!refineStyle) return null
    const derived = deriveRefine(refineStyle.def)
    if (!derived) return null
    if (refineBorrows || !refinePick.specs.length) return derived
    return withLoras(derived, refinePick.specs) ?? derived
  }, [refineStyle, refineBorrows, refinePick])

  /**
   * What the reader is asking for, read out of the prompt and overridable by
   * hand. Reading it is a hint, never a switch: nothing changes style on its
   * own because a word matched.
   */
  const brief: Brief = useMemo(() => {
    const guessed = briefFrom(c.prompt, { intent: 'photoreal', explicit: false, mode: 'image' })
    return {
      intent: intentPick ?? guessed.intent,
      explicit: explicitPick ?? guessed.explicit,
      mode: c.mode === 'edit' ? 'edit' : 'image',
    }
  }, [c.prompt, intentPick, explicitPick, c.mode])

  /**
   * Files this brief could possibly use.
   *
   * A model owned by a family of the WRONG mode is filtered out rather than
   * passed in, because intentReport marks a file as routed only while walking
   * families of the mode it was asked about. Hand it the edit checkpoint during
   * an image brief and it reports a perfectly wired, currently running model as
   * having no verified graph, which is not a small wrong: the whole point of
   * that list is to name the real gaps.
   */
  const installedForBrief = useMemo(() => {
    if (!cat) return []
    return cat.installed.filter((m) => {
      const owner = familyOwning(m)
      return !owner || owner.mode === (brief.mode ?? 'image')
    })
  }, [cat, brief.mode])

  const report = useMemo(() => {
    if (!cat) return null
    return intentReport(brief, {
      installed: installedForBrief,
      sizes: cat.sizes,
      hardware: cat.hardware,
      limit: 4,
    })
  }, [brief, cat, installedForBrief])

  /**
   * Whether the top ranked base can carry the pass that actually fixes small
   * anatomy. Ranking weighs what a base draws, not what can be done to it
   * afterwards, so a base can win the list and still be the wrong tool for the
   * job the user came here with.
   */
  const topRefines = useMemo(() => {
    const top = report?.ranked[0]
    return top ? deriveRefine(top.def) !== null : true
  }, [report])

  const mismatch = useMemo(
    () => (style ? mismatchHint(style.model, brief, style.def) : null),
    [style, brief],
  )

  const timing = useMemo(() => timingNote(records, c.model), [records, c.model])

  // --- mode ---------------------------------------------------------------
  const setMode = useCallback(
    (mode: Mode) => {
      const current = store.get()
      if (mode === current.mode) return
      if (mode === 'edit') {
        if (!editStyle) return
        lastPictureStyle = { familyId: current.familyId, model: current.model }
        store.patch({ mode, denoise: null, megapixels: null })
        applyStyle(editStyle)
        return
      }
      if (current.mode === 'edit') {
        const back =
          pictureStyles.find(
            (s) => s.def.id === lastPictureStyle?.familyId && s.model === lastPictureStyle?.model,
          ) ?? pictureStyles[0]
        store.patch({ mode })
        if (back) applyStyle(back)
      } else {
        store.patch({ mode })
      }
      if (mode === 'i2i') {
        const now = store.get()
        store.patch({
          denoise: now.denoise ?? DEFAULT_DENOISE,
          megapixels:
            now.megapixels ??
            (style ? round2((defaultsFor(style.def, style.model).width * defaultsFor(style.def, style.model).height) / 1e6) : 1),
        })
      }
    },
    [applyStyle, editStyle, pictureStyles, style],
  )

  // --- the source picture -------------------------------------------------
  const clearSource = useCallback(() => {
    const s = store.get().source
    if (s?.previewUrl?.startsWith('blob:')) URL.revokeObjectURL(s.previewUrl)
    store.patch({ source: null })
    setSourceError(null)
  }, [])

  const takeFile = useCallback(
    async (file: File) => {
      if (!file.type.startsWith('image/')) {
        setSourceError('That file is not a picture. Try a PNG or a JPEG.')
        return
      }
      setSourceError(null)
      setUploading(true)
      const previewUrl = URL.createObjectURL(file)
      try {
        const name = await uploadImage(file)
        const size = await measure(previewUrl)
        const previous = store.get().source
        if (previous?.previewUrl?.startsWith('blob:')) URL.revokeObjectURL(previous.previewUrl)
        const source: SourceRef = {
          name,
          previewUrl,
          label: file.name,
          width: size?.width,
          height: size?.height,
          bytes: file.size,
        }
        const now = store.get()
        const mode: Mode =
          now.mode === 'edit' ? 'edit' : canI2I ? 'i2i' : now.mode
        store.patch({ source, mode })
        if (mode === 'i2i') {
          store.patch({ denoise: store.get().denoise ?? DEFAULT_DENOISE })
        }
      } catch (err) {
        URL.revokeObjectURL(previewUrl)
        const status = err instanceof Error ? err.message : String(err)
        setSourceError(`ComfyUI refused the file (${status}). Try a PNG or JPEG under 50 MB.`)
      } finally {
        setUploading(false)
      }
    },
    [canI2I],
  )

  const takeRecord = useCallback((entry: HistoryEntry) => {
    const previous = store.get().source
    if (previous?.previewUrl?.startsWith('blob:')) URL.revokeObjectURL(previous.previewUrl)
    store.patch({
      source: {
        name: '',
        ref: entry.file,
        previewUrl: fileUrl(entry.file),
        label: entry.file.filename,
        fromEntryId: entry.id,
        width: entry.width ?? undefined,
        height: entry.height ?? undefined,
      },
    })
  }, [])

  // A picture adopted from the archive lives in the *output* folder; LoadImage
  // reads the input folder, so it is copied across once, here.
  useEffect(() => {
    const s = c.source
    if (!s || s.name || !s.ref) return
    let alive = true
    setUploading(true)
    ;(async () => {
      try {
        const res = await fetch(fileUrl(s.ref!))
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const blob = await res.blob()
        const name = await uploadImage(blob, s.ref!.filename)
        if (alive) store.patch({ source: { ...store.get().source, ...s, name } })
      } catch {
        if (alive) {
          setSourceError('We could not copy that picture into ComfyUI’s input folder. Try picking it again.')
        }
      } finally {
        if (alive) setUploading(false)
      }
    })()
    return () => {
      alive = false
    }
  }, [c.source])

  // --- drag, drop, paste --------------------------------------------------
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const file = [...(e.clipboardData?.files ?? [])].find((f) => f.type.startsWith('image/'))
      if (!file) return
      e.preventDefault()
      void takeFile(file)
    }
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
  }, [takeFile])

  const onDragEnter = (e: ReactDragEvent) => {
    if (![...e.dataTransfer.types].includes('Files')) return
    dragDepth.current += 1
    setDragging(true)
  }
  const onDragLeave = () => {
    dragDepth.current = Math.max(0, dragDepth.current - 1)
    if (!dragDepth.current) setDragging(false)
  }
  const onDrop = (e: ReactDragEvent) => {
    e.preventDefault()
    dragDepth.current = 0
    setDragging(false)
    const file = [...e.dataTransfer.files].find((f) => f.type.startsWith('image/'))
    if (file) void takeFile(file)
  }

  // --- running ------------------------------------------------------------
  const ready = useMemo(() => {
    if (!activeDef || !style) return { ok: false, why: 'Waiting for the model list.' }
    if (!c.prompt.trim()) {
      return {
        ok: false,
        why: c.mode === 'edit' ? 'Describe the change first.' : 'Write a line first.',
      }
    }
    if (needsSource(c.mode)) {
      if (!c.source) return { ok: false, why: 'Add a picture to work from.' }
      if (!c.source.name) return { ok: false, why: 'Still copying your picture across.' }
    }
    return { ok: true, why: '' }
  }, [activeDef, style, c.prompt, c.mode, c.source])

  const start = useCallback(() => {
    if (!ready.ok || !queuedDef || !style || busy(press)) return
    const base = store.get()
    const seed0 = base.seedLocked ? Math.floor(base.seed) : randomSeed()
    const plans: RunPlan[] = []

    for (let i = 0; i < base.runs; i += 1) {
      const seed = seed0 + i
      const composition: Composition = {
        ...base,
        seed,
        // Only carry what this mode actually used into the record.
        source: needsSource(base.mode) ? base.source : null,
        // And only what the graph can actually receive. The archive is the
        // record of what made the picture: a shift filed against a family
        // with no ModelSampling node reached nothing, and offering it back
        // as the way to reproduce the picture would be a second lie.
        shift: hasShift ? base.shift : null,
        clipSkip: hasClipSkip ? base.clipSkip : null,
        denoise: base.mode === 'i2i' ? (base.denoise ?? DEFAULT_DENOISE) : null,
        megapixels: base.mode === 'i2i' ? (base.megapixels ?? 1) : null,
        length: null,
        fps: null,
      }
      const params = toParams(composition, { negative: houseNegative })
      const graph = buildQueued(queuedDef, params, passes.hires)
      plans.push({
        graph,
        composition,
        seed,
        familyLabel: style.group,
        modelLabel: style.label,
        variant: base.mode === 'i2i' ? 'img2img' : base.noLora ? 'nolora' : null,
        label: style.label,
        passes,
        loras: basePick.specs,
      })
    }

    store.patch({ seed: seed0 })
    startRuns(plans)
  }, [ready.ok, queuedDef, style, houseNegative, passes, basePick.specs, hasShift, hasClipSkip])

  // Ctrl/⌘+Enter runs, and is the one shortcut that works inside the prompt.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // A key somebody nearer the event already claimed is not ours. The
      // player and the mask canvas both preventDefault on keys this desk also
      // binds, and firing both opens two things at once.
      if (e.defaultPrevented) return
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault()
        start()
        return
      }
      const target = e.target as HTMLElement | null
      const typing =
        !!target && (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName))
      if (typing || e.ctrlKey || e.metaKey || e.altKey) return
      if (e.key === 'u') {
        e.preventDefault()
        fileInput.current?.click()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [start])

  // Expert mode opens where the reader clicked.
  useEffect(() => {
    if (!focusField || !expert) return
    const el = document.getElementById(focusField)
    el?.scrollIntoView({ block: 'center', behavior: reduced ? 'auto' : 'smooth' })
    ;(el as HTMLElement | null)?.focus?.()
    setFocusField(null)
  }, [focusField, expert, reduced])

  const openExpert = useCallback((field: string) => {
    settings.patch({ expert: true })
    setFocusField(field)
  }, [])

  const flash = useCallback((what: string) => {
    setAdopted(what)
    setTimeout(() => setAdopted((v) => (v === what ? null : v)), 2000)
  }, [])

  // The plate always reads the live record, so a star lands without a reload.
  const current = useMemo(() => {
    const cur = state.current
    if (!cur) return null
    return records.find((r) => r.id === cur.id) ?? cur
  }, [state, records])

  const adopt = useCallback(
    (field: Parameters<typeof adoptValue>[1], value: number | string, label: string) => {
      adoptValue(DESK, field, value as never)
      flash(label)
    },
    [flash],
  )

  // --- region refine ------------------------------------------------------

  /**
   * Open the refine surface on a finished picture.
   *
   * Two things have to be true before a mask can be drawn. The picture must be
   * in ComfyUI's INPUT folder, because LoadImage will not read an output, so it
   * is copied across here exactly as the source well does it. And its real
   * pixel size must be known, because every crop number is computed from it.
   *
   * The size is always measured off the file, never read out of the record.
   * The record files the size the composer ASKED for, and a two pass render
   * upscales the latent by 1.5 on the way to disk with nothing in the record
   * to say so. Trusting it put the crop, the mask and the composite in the
   * pre upscale space and pasted a correct patch into the wrong part of the
   * frame, burning a whole generation to produce a visible graft. The record
   * is kept only as a fallback for a file the browser cannot decode.
   */
  const openRefine = useCallback(async (entry: HistoryEntry) => {
    const token = (refineToken.current += 1)
    awaitingRefine.current = false
    setRefining(true)
    setRefineSource(null)
    setRefineResult(null)
    setRefineFault(null)
    setOpeningRefine(true)
    const url = fileUrl(entry.file)
    try {
      const measured =
        (await measure(url)) ??
        (entry.width && entry.height ? { width: entry.width, height: entry.height } : null)
      if (!measured) throw new Error('the size could not be read')
      const res = await fetch(url)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const blob = await res.blob()
      const name = await uploadImage(blob, entry.file.filename)
      if (refineToken.current !== token) return
      setRefineSource({
        name,
        url,
        width: measured.width,
        height: measured.height,
        entryId: entry.id,
        prompt: entry.prompt,
      })
    } catch (err) {
      if (refineToken.current !== token) return
      const why = err instanceof Error ? err.message : String(err)
      setRefineFault(
        `That picture could not be copied into ComfyUI’s input folder (${why}). A refine pass reads the source from there, so nothing can run until it lands.`,
      )
    } finally {
      if (refineToken.current === token) setOpeningRefine(false)
    }
  }, [])

  const closeRefine = useCallback(() => {
    refineToken.current += 1
    awaitingRefine.current = false
    setRefining(false)
    setRefineSource(null)
    setRefineResult(null)
    setRefineFault(null)
    setOpeningRefine(false)
  }, [])

  /**
   * Queue one refine pass.
   *
   * The mask arrives as a source-sized PNG, white on black and opaque, and is
   * uploaded exactly as handed over: re-encoding it with an alpha channel would
   * make LoadImageMask read it as empty and the pass would finish having
   * changed nothing.
   */
  const runRefine = useCallback(
    async (req: RefineRequest) => {
      if (!refineDef || !refineSource || !refineStyle || busy(press)) return
      setRefineFault(null)
      try {
        const mask = await uploadImage(req.mask, req.maskName)
        const base = store.get()
        const seed = req.seed ?? Math.floor(base.seed)
        // A borrowed style brings its own recipe. Carrying the edit family's
        // CFG of 2.5 onto an SDXL checkpoint would wash the region out, and it
        // would look like the refine pass had failed rather than like the wrong
        // numbers had been handed to it.
        const rd = defaultsFor(refineStyle.def, refineStyle.model)
        const recipe = refineBorrows
          ? {
              steps: rd.steps,
              cfg: rd.cfg,
              sampler: rd.sampler,
              scheduler: rd.scheduler,
              negative: null,
              positivePrefix: refineStyle.positivePrefix,
              shift: refineStyle.shift,
              clipSkip: refineStyle.clipSkip,
            }
          : {}
        const composition: Composition = {
          ...base,
          // Filed as image-to-image, which is what it is: a partial denoise of
          // an existing picture. That also files no width or height, which is
          // right, because the output is the source's size, not the composer's.
          mode: 'i2i',
          familyId: refineStyle.def.id,
          model: refineStyle.model,
          ...recipe,
          prompt: req.prompt,
          seed,
          source: {
            name: refineSource.name,
            previewUrl: refineSource.url,
            label: `region of ${refineSource.entryId}`,
            width: refineSource.width,
            height: refineSource.height,
            fromEntryId: refineSource.entryId,
          },
          denoise: req.denoise,
          megapixels: null,
          length: null,
          fps: null,
        }
        const params = toParams(composition, { negative: rd.negative ?? houseNegative })
        const graph = instantiateRefine(refineDef, params, {
          image: refineSource.name,
          mask,
          crop: req.plan.crop,
          target: req.plan.target,
          denoise: req.denoise,
          grow: req.grow,
          feather: req.feather,
          prompt: req.prompt,
          seed,
        })
        // The same two settings the bindings cannot carry. A borrowed style
        // brings its own shift and clip skip in `recipe` above; without these
        // the region would be drawn at the picker style's numbers.
        applyShift(graph, params.shift)
        applyClipSkip(graph, params.clipSkip)
        awaitingRefine.current = true
        startRuns([
          {
            graph,
            composition,
            seed,
            familyLabel: refineStyle.group,
            modelLabel: refineStyle.label,
            variant: 'img2img',
            label: `${refineStyle.label}, region refine`,
            // A refine pass carries no detail passes of its own; it IS the
            // detail pass. The rack is held back entirely for a borrowed
            // style, which is what refineDef was built with.
            passes: NO_PASSES,
            loras: refineBorrows ? [] : refinePick.specs,
          },
        ])
      } catch (err) {
        setRefineFault(
          err instanceof Error ? err.message : 'The refine pass could not be queued.',
        )
      }
    },
    [refineDef, refineSource, refineStyle, refineBorrows, refinePick.specs, houseNegative],
  )

  // The result is whatever the press produced for the pass we queued, and only
  // that: the composer still works while the bench is open, so an ordinary
  // picture made in the meantime must not be presented as the refined one. The
  // result is deliberately not made the source either. Swapping it in would
  // turn a before and after into two copies of the same frame.
  useEffect(() => {
    if (!awaitingRefine.current) return
    const cur = state.current
    if (!cur || cur.id === refineSource?.entryId) return
    awaitingRefine.current = false
    setRefineResult(cur)
  }, [state, refineSource])

  // A style that cannot carry a refine, or a press already busy with something
  // else, blocks the button in print rather than failing at queue time.
  const refineBlocked =
    refineFault ??
    (!refineDef
      ? 'No installed style can re render a region. Every one of them samples through a custom schedule with no denoise control.'
      : openingRefine
        ? 'Copying the picture into ComfyUI’s input folder.'
        : null)

  // --- render -------------------------------------------------------------
  if (catError) {
    return (
      <main className="mx-auto max-w-2xl px-6 py-16">
        <h1 className="mb-1 text-h3 font-semibold">The server is not answering</h1>
        <p className="mb-4 text-body text-grey-700">
          We could not read the model list from ComfyUI on port 8188. It may not be running.
        </p>
        <pre className="mb-4 border border-grey-300 bg-newsprint-aged px-3 py-2 text-caption">
          systemctl --user start comfyui
        </pre>
        <p className="text-small text-grey-700">
          <Link onClick={() => load(true)}>Try now</Link>
          <span className="px-2 text-grey-400" aria-hidden>
            ·
          </span>
          <span className="italic text-grey-500">
            Trying again in <span className="tabular-nums">{retryIn}</span>s.
          </span>
        </p>
        <p className="mt-6 text-caption italic text-grey-500">{catError}</p>
      </main>
    )
  }

  return (
    <main
      className="relative grid grid-cols-1 items-start lg:grid-cols-[24rem_minmax(0,1fr)] xl:h-full xl:grid-cols-[17rem_24rem_1fr]"
      onDragEnter={onDragEnter}
      onDragOver={(e) => {
        if ([...e.dataTransfer.types].includes('Files')) e.preventDefault()
      }}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      {dragging && (
        <div className="pointer-events-none absolute inset-3 z-30 grid place-items-center border-2 border-dashed border-burgundy-900 bg-newsprint/80">
          <Kicker className="text-burgundy-900">Drop to use as your source</Kicker>
        </div>
      )}

      {/* ---- the margin: empty in simple, the whole apparatus in expert ---- */}
      <aside className="order-2 px-6 py-6 lg:order-3 lg:col-span-2 lg:border-t lg:border-grey-300 xl:order-1 xl:col-span-1 xl:border-t-0 xl:h-full xl:overflow-y-auto">
        {expert && style && activeDef && cat && (
          <ExpertMargin
            c={c}
            style={style}
            cat={cat}
            hasNegative={hasNegative}
            hasScheduler={hasScheduler}
            hasShift={hasShift}
            hasClipSkip={hasClipSkip}
            houseNegative={houseNegative}
            onClose={() => settings.patch({ expert: false })}
            graph={activeDef}
            queued={queuedDef ?? activeDef}
            hires={passes.hires}
            faultNode={state.fault?.node ?? null}
          />
        )}
      </aside>

      {/* ---- the composer ---- */}
      <div className="order-1 border-grey-300 px-6 py-6 lg:order-1 lg:border-r xl:order-2 xl:h-full xl:overflow-y-auto xl:border-x">
        <h2 className="mb-1 border-b-2 border-burgundy-900 pb-1.5 text-overline font-semibold uppercase tracking-[0.18em] text-burgundy-900">
          The Pictures Desk
        </h2>
        <p className="mb-5 text-caption italic text-grey-500">
          {timing ??
            'Nothing timed for this model yet. After three pictures this line says how long one takes here.'}
        </p>

        {offline && (
          <div className="mb-4">
            <Notice kind="correction" title="Correction">
              We have lost the connection to ComfyUI. Anything already running will be picked up
              when it comes back.
            </Notice>
          </div>
        )}

        {correction && (
          <div className="mb-4">
            <Notice kind="correction" title="Correction">
              {correction}{' '}
              <Link onClick={() => setCorrection(null)}>Dismiss</Link>
            </Notice>
          </div>
        )}

        <ModeTabs mode={c.mode} canI2I={tabI2I} canEdit={!!editStyle} onPick={setMode} />

        {c.mode === 't2i' && !canI2I && c.source && (
          <div className="mt-3">
            <Notice kind="correction" title="Correction">
              {style?.label ?? 'This style'} works from words only, so your picture is standing by
              unused.{' '}
              {i2iAlternative && (
                <Link
                  onClick={() => {
                    applyStyle(i2iAlternative)
                    setMode('i2i')
                  }}
                >
                  {i2iAlternative.label} works from a picture
                </Link>
              )}
            </Notice>
          </div>
        )}

        {!canI2I && c.mode !== 'edit' && !c.source && (
          <p className="mt-2 text-caption italic text-grey-700">
            {style?.label ?? 'This style'} works from words only.{' '}
            {i2iAlternative && (
              <Link onClick={() => applyStyle(i2iAlternative)}>
                {i2iAlternative.label} works from a picture.
              </Link>
            )}
          </p>
        )}

        {(c.mode === 'i2i' || c.mode === 'edit') && (
          <div className="mt-5">
            <SourceWell
              source={c.source}
              busy={uploading}
              required={c.mode === 'edit'}
              error={sourceError}
              onPick={() => fileInput.current?.click()}
              onArchive={() => setPicking(true)}
              onClear={clearSource}
            />
          </div>
        )}

        <div className="mt-5">
          <PromptField
            textRef={promptRef}
            mode={c.mode}
            value={c.prompt}
            prefix={expert ? null : c.positivePrefix}
            onChange={(prompt) => store.patch({ prompt })}
          />
        </div>

        {c.mode !== 'edit' && (
          <div className="mt-5">
            <StylePicker
              styles={pictureStyles}
              style={style}
              expert={expert}
              onPick={applyStyle}
            />
            <IntentRail
              brief={brief}
              report={report}
              unavailable={cat?.unavailable ?? []}
              topRefines={topRefines}
              mismatch={mismatch}
              current={style}
              styles={pictureStyles}
              onIntent={(i) => setIntentPick((was) => (was === i ? null : i))}
              onExplicit={(v) => setExplicitPick(v)}
              onPick={(rec) => {
                const next = pictureStyles.find(
                  (s) => s.def.id === rec.familyId && s.model === rec.model,
                )
                if (next) applyStyle(next)
              }}
            />
          </div>
        )}

        {c.mode === 'edit' && editStyle && (
          <p className="mt-4 text-caption italic text-grey-700">
            Changing a picture always uses {editStyle.label}; it is the only model here that follows
            an instruction. {editStyle.verdict && editStyle.verdict.level !== 'ok' && (
              <span className="not-italic"> {editStyle.verdict.reason}</span>
            )}
          </p>
        )}

        {c.mode === 't2i' && hasSize && (
          <div className="mt-6">
            <ShapePicker
              shapes={shapes}
              width={c.width}
              height={c.height}
              expert={expert}
              onPick={(s) => store.edit({ width: s.width, height: s.height }, 'width', 'height')}
              onSize={(w, h) => store.edit({ width: w, height: h }, 'width', 'height')}
            />
          </div>
        )}

        {c.mode === 'i2i' && (
          <div className="mt-6">
            <SizeFollows
              megapixels={c.megapixels ?? 1}
              expert={expert}
              onPick={(mp) => store.edit({ megapixels: mp }, 'megapixels')}
            />
          </div>
        )}

        {c.mode === 'i2i' && (
          <div className="mt-6">
            <Strength
              denoise={c.denoise ?? DEFAULT_DENOISE}
              expert={expert}
              steps={c.steps}
              modelLabel={style?.label ?? 'This model'}
              onChange={(d) => store.edit({ denoise: d }, 'denoise')}
            />
          </div>
        )}

        {caps && (caps.faceDetail || caps.handDetail || caps.hires) && (
          <div className="mt-6">
            <QualityPasses
              caps={caps}
              value={passes}
              onChange={setPasses}
              size={{ width: c.width, height: c.height }}
              sized={c.mode === 't2i' && hasSize}
              steps={c.steps}
            />
          </div>
        )}

        {baseDef && (
          <div className="mt-6">
            <LoraRack
              rack={rack}
              prompt={c.prompt}
              expert={expert}
              onAddTriggers={(tokens) => {
                const now = store.get().prompt
                const joined = tokens.join(', ')
                store.patch({ prompt: now.trim() ? `${joined}, ${now}` : joined })
              }}
            />
            {refinePick.deferred.length > 0 && (
              <p className="mt-2 text-caption leading-snug text-grey-700">
                <Kicker className="block">Held for the refine pass</Kicker>
                {refinePick.deferred.map((d) => d.label).join(', ')}
                {refinePick.deferred.length === 1 ? ' is' : ' are'} trained on close framing, so
                {refinePick.deferred.length === 1 ? ' it is' : ' they are'} applied when you refine a
                region rather than on the first render.
              </p>
            )}
          </div>
        )}

        <div className="mt-7">
          <RunButton
            label={c.mode === 'edit' ? 'Make the change' : 'Make the picture'}
            disabled={!ready.ok}
            why={ready.why}
            running={running}
            queuedAhead={ahead}
            lastMs={state.lastMs}
            runs={c.runs}
            job={state.job}
            onRun={start}
            onStop={() => void stopRun()}
            reduced={reduced}
          />
          <SettingLine
            c={c}
            expert={expert}
            hasScheduler={hasScheduler}
            onOpen={openExpert}
          />
        </div>

        {state.fault && (
          <div className="mt-5">
            <Fault fault={state.fault} onDismiss={() => emit({ fault: null })} />
          </div>
        )}

        <input
          ref={fileInput}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0]
            e.target.value = ''
            if (file) void takeFile(file)
          }}
        />
      </div>

      {/* ---- the plate, or the refine bench standing in for it ---- */}
      <section className="order-3 px-6 py-6 lg:order-2 xl:order-3 xl:h-full xl:overflow-y-auto">
        {refining ? (
          <div className="flex h-full flex-col">
            <div className="mb-4 flex items-baseline justify-between gap-4 border-b-2 border-burgundy-900 pb-1.5">
              <Kicker className="text-burgundy-900">The Refine Bench</Kicker>
              <Link onClick={closeRefine}>Back to the plate</Link>
            </div>
            {refineStyle && (
              <p className="mb-3 text-caption leading-snug text-grey-700">
                <Kicker className="block">Drawn by {refineStyle.label}</Kicker>
                {refineBorrows
                  ? `${style?.label ?? 'The style in the picker'} cannot re render a region, so the region is drawn by ${refineStyle.label} at its own settings. The rest of the picture is untouched, and any LoRAs in the rack are held back because they were resolved against a different architecture.`
                  : 'The region is drawn by the style in the picker, at its own settings, with the LoRAs in the rack applied.'}
              </p>
            )}
            {refineResult && (
              <p className="mb-4 text-caption text-grey-700">
                <Kicker className="block">The pass landed</Kicker>
                The comparison below is the picture you started from against the refined one. To
                refine a second region of the result,{' '}
                <Link onClick={() => void openRefine(refineResult)}>carry on from it</Link>. The mask
                resets, because the picture underneath has changed.
              </p>
            )}
            <RegionRefine
              source={refineSource}
              parentPrompt={refineSource?.prompt ?? ''}
              result={refineResult ? { url: fileUrl(refineResult.file) } : null}
              expert={expert}
              busy={running}
              progress={running ? (state.job?.pct ?? null) : null}
              blocked={refineBlocked}
              onRun={(req) => void runRefine(req)}
              onStop={() => void stopRun()}
            />
            {state.fault && (
              <div className="mt-5">
                <Fault fault={state.fault} onDismiss={() => emit({ fault: null })} />
              </div>
            )}
          </div>
        ) : (
        <Plate
          state={state}
          entry={current}
          c={c}
          style={style}
          reduced={reduced}
          adopted={adopted}
          examples={c.prompt.trim() ? [] : exampleLines(pictureStyles)}
          onExample={(ex) => {
            const s = pictureStyles.find((p) => p.def.id === ex.familyId)
            if (s) applyStyle(s)
            const shape = shapesFor(s ?? style).find((sh) => sh.key === ex.shape)
            store.patch({ prompt: ex.prompt })
            if (shape) store.edit({ width: shape.width, height: shape.height }, 'width', 'height')
            promptRef.current?.focus()
          }}
          onAdopt={adopt}
          onShow={showResult}
          onWorkFrom={(entry) => {
            takeRecord(entry)
            setMode('i2i')
          }}
          onChangeThis={(entry) => {
            takeRecord(entry)
            setMode('edit')
          }}
          canI2I={tabI2I}
          canEdit={!!editStyle}
          canRefine={canRefine}
          onRefine={(entry) => void openRefine(entry)}
        />
        )}
      </section>

      {picking && (
        <ArchivePicker
          records={records}
          onClose={() => setPicking(false)}
          onPick={(entry) => {
            takeRecord(entry)
            setPicking(false)
          }}
        />
      )}
    </main>
  )
}

export default Pictures

function exampleLines(styles: Style[]) {
  return EXAMPLES.filter((ex) => styles.some((s) => s.def.id === ex.familyId))
}

async function measure(url: string): Promise<{ width: number; height: number } | null> {
  return new Promise((resolve) => {
    const img = new Image()
    img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight })
    img.onerror = () => resolve(null)
    img.src = url
  })
}

// ---------------------------------------------------------------------------
// Mode tabs
// ---------------------------------------------------------------------------

function ModeTabs({
  mode,
  canI2I,
  canEdit,
  onPick,
}: {
  mode: Mode
  canI2I: boolean
  canEdit: boolean
  onPick: (m: Mode) => void
}) {
  const enabled = (m: Mode) => (m === 'i2i' ? canI2I : m === 'edit' ? canEdit : true)
  return (
    <div role="tablist" aria-label="What are you making from?" className="flex border border-grey-300">
      {DESK_MODES.map((m, i) => {
        const on = m === mode
        const live = enabled(m)
        return (
          <button
            key={m}
            role="tab"
            type="button"
            aria-selected={on}
            disabled={!live}
            onClick={() => onPick(m)}
            title={live ? undefined : 'This style cannot do that'}
            className={[
              'flex-1 px-2 py-2 text-overline font-semibold uppercase tracking-[0.16em] transition-colors',
              i > 0 ? 'border-l border-grey-300' : '',
              on
                ? 'border-t-2 border-t-burgundy-900 bg-newsprint text-ink'
                : live
                  ? 'cursor-pointer text-grey-500 hover:text-ink'
                  : 'cursor-not-allowed text-grey-300',
              'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900',
            ].join(' ')}
          >
            {MODE_LABEL[m]}
          </button>
        )
      })}
    </div>
  )
}

// ---------------------------------------------------------------------------
// The source well
// ---------------------------------------------------------------------------

function SourceWell({
  source,
  busy: uploading,
  required,
  error,
  onPick,
  onArchive,
  onClear,
}: {
  source: SourceRef | null
  busy: boolean
  required: boolean
  error: string | null
  onPick: () => void
  onArchive: () => void
  onClear: () => void
}) {
  const caption = source
    ? [
        source.label ?? source.name,
        source.width && source.height ? times(source.width, source.height) : null,
        source.bytes ? bytes(source.bytes) : null,
      ]
        .filter(Boolean)
        .join(' · ')
    : ''

  return (
    <div>
      <Label hint={required ? 'required' : undefined}>Your picture</Label>
      {source ? (
        <div className="flex items-start gap-3 border border-grey-300 bg-newsprint-aged p-2">
          <div className="h-16 w-16 shrink-0 overflow-hidden border border-grey-300 bg-newsprint">
            {source.previewUrl ? (
              <img
                src={source.previewUrl}
                alt=""
                className="h-full w-full object-cover"
              />
            ) : (
              <div className="grid h-full place-items-center text-caption text-grey-400">—</div>
            )}
          </div>
          <div className="min-w-0 flex-1">
            <p className="truncate text-caption italic text-grey-700">{caption}</p>
            {uploading && <p className="text-caption italic text-grey-500">Copying it across…</p>}
            {source.fromEntryId && (
              <p className="text-caption italic text-grey-500">From your archive.</p>
            )}
            <p className="mt-1 text-caption">
              <Link onClick={onPick}>Replace</Link>
              <span className="text-grey-400"> · </span>
              <Link onClick={onArchive}>From the archive</Link>
              <span className="text-grey-400"> · </span>
              <Link onClick={onClear}>Clear</Link>
            </p>
          </div>
        </div>
      ) : (
        <button
          type="button"
          onClick={onPick}
          className="w-full cursor-pointer border border-dashed border-grey-400 bg-newsprint-aged px-3 py-5 text-center transition-colors hover:border-burgundy-900 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900"
        >
          <span className="block text-overline font-semibold uppercase tracking-[0.18em] text-grey-700">
            Choose a picture
          </span>
          <span className="mt-1 block text-caption italic text-grey-500">
            Drop one here, paste it, or press u
          </span>
        </button>
      )}
      {!source && (
        <p className="mt-1 text-caption italic text-grey-500">
          You can also <Link onClick={onArchive}>take one from the archive</Link>.
        </p>
      )}
      {error && (
        <div className="mt-2">
          <Notice kind="error" title="We could not use that picture">
            {error}
          </Notice>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Prompt
// ---------------------------------------------------------------------------

function PromptField({
  textRef,
  mode,
  value,
  prefix,
  onChange,
}: {
  textRef: RefObject<HTMLTextAreaElement | null>
  mode: Mode
  value: string
  prefix: string | null
  onChange: (v: string) => void
}) {
  const edit = mode === 'edit'
  return (
    <label className="block">
      <Label>{edit ? 'Describe the change' : 'Describe the picture'}</Label>
      <textarea
        ref={textRef}
        value={value}
        rows={edit ? 3 : 5}
        spellCheck
        onChange={(e) => onChange(e.target.value)}
        placeholder={
          edit ? 'Make the jacket red' : 'A rain-slicked tram stop at dusk, neon in the puddles'
        }
        className="field"
        style={{ fontSize: '1.125rem', lineHeight: 1.6, maxWidth: '62ch' }}
      />
      <span className="mt-1 block text-caption italic text-grey-500">
        {prefix
          ? `The maker’s quality words are added for you: “${prefix.trim().replace(/,$/, '')}”. `
          : ''}
        Ctrl+Enter runs it.
      </span>
    </label>
  )
}

// ---------------------------------------------------------------------------
// Style
// ---------------------------------------------------------------------------

function fitTag(v: Verdict | null): string | null {
  if (!v) return null
  if (v.level === 'risky') return 'not enough free memory'
  if (v.level === 'tight') return 'tight fit'
  if (v.offloads) return 'heavy'
  return null
}

function StylePicker({
  styles,
  style,
  expert,
  onPick,
}: {
  styles: Style[]
  style: Style | null
  expert: boolean
  onPick: (s: Style) => void
}) {
  const groups = useMemo(() => {
    const map = new Map<string, Style[]>()
    for (const s of styles) map.set(s.group, [...(map.get(s.group) ?? []), s])
    return [...map.entries()]
  }, [styles])

  const verdict = style?.verdict ?? null
  const tag = fitTag(verdict)

  return (
    <label className="block">
      <Label>Style</Label>
      <select
        className="field"
        value={style ? `${style.def.id}::${style.model}` : ''}
        onChange={(e) => {
          const [id, model] = e.target.value.split('::')
          const next = styles.find((s) => s.def.id === id && s.model === model)
          if (next) onPick(next)
        }}
      >
        {!style && <option value="">Reading the model list…</option>}
        {groups.map(([group, list]) => (
          <optgroup key={group} label={group}>
            {list.map((s) => {
              const t = fitTag(s.verdict)
              return (
                <option key={s.model} value={`${s.def.id}::${s.model}`}>
                  {s.label}
                  {s.def.verified ? '' : ' †'}
                  {t ? ` · ${t}` : ''}
                </option>
              )
            })}
          </optgroup>
        ))}
      </select>

      {verdict && verdict.level !== 'ok' && (
        <span className="mt-1 block text-caption italic text-grey-700">{verdict.reason}</span>
      )}
      {verdict && verdict.level === 'ok' && verdict.offloads && (
        <span className="mt-1 block text-caption italic text-grey-700">
          <Kicker className="block not-italic">Heavy</Kicker>
          The largest file is {gb(verdict.footprint.largestBytes)}, more than the card holds, so it
          streams from memory and runs slower.
        </span>
      )}
      {tag === null && verdict && verdict.level === 'ok' && !verdict.offloads && (
        <span className="mt-1 block text-caption italic text-grey-500">{verdict.reason}</span>
      )}
      {style && !style.def.verified && (
        <span className="mt-1 block text-caption italic text-grey-700">
          † These settings come from the model’s card and have not been checked against a live run.
        </span>
      )}
      {expert && style && (
        <span className="mt-1 block truncate text-caption text-grey-500" title={style.model}>
          {style.model}
        </span>
      )}
    </label>
  )
}

// ---------------------------------------------------------------------------
// Intent routing
//
// The picker above lists every installed style alphabetically, which is the
// right order for finding a name and the wrong order for choosing one. This
// rail reads the prompt, ranks the installed weights against what it asks for,
// and says which one to use and why.
//
// It never switches style on its own. A keyword match is a hint; overriding a
// deliberate choice because someone typed "photo" would be worse than the
// mismatch it was trying to prevent.
// ---------------------------------------------------------------------------

function IntentRail({
  brief,
  report,
  unavailable,
  topRefines,
  mismatch,
  current,
  styles,
  onIntent,
  onExplicit,
  onPick,
}: {
  brief: Brief
  report: IntentReport | null
  /** Why each detected file did not make it into the picker. */
  unavailable: { name: string; why: string }[]
  /** False when the top ranked base cannot carry a region refine pass. */
  topRefines: boolean
  mismatch: string | null
  current: Style | null
  styles: Style[]
  onIntent: (i: Intent) => void
  onExplicit: (v: boolean) => void
  onPick: (rec: Recommendation) => void
}) {
  const [open, setOpen] = useState(false)
  if (!report) return null

  const top = report.ranked[0] ?? null
  const isCurrent = (r: Recommendation) =>
    !!current && current.def.id === r.familyId && current.model === r.model
  const blurb = INTENTS.find((i) => i.id === brief.intent)?.blurb ?? ''
  const reachable = (r: Recommendation) =>
    styles.some((s) => s.def.id === r.familyId && s.model === r.model)

  /**
   * Why a ranked model is not in the picker.
   *
   * The ranking runs over every installed weight; the picker drops anything
   * whose graph names a CLIP, VAE or LoRA that is not on disk, or that the
   * card cannot hold. The reason is already known by then, so a headline that
   * names a model the reader cannot select has no excuse to withhold it.
   */
  const whyUnavailable = (r: Recommendation): string =>
    sentence(
      unavailable.find((u) => u.name === r.model)?.why ??
        report.unrouted.find((u) => u.model === r.model)?.why ??
        'It is on disk, but the picker has no verified graph for it',
    )

  return (
    <div className="mt-3 border-t border-grey-300 pt-3">
      <Kicker>What are you after</Kicker>

      <div className="mt-1.5 flex flex-wrap gap-1.5">
        {INTENTS.map((opt) => {
          const on = brief.intent === opt.id
          return (
            <button
              key={opt.id}
              type="button"
              aria-pressed={on}
              onClick={() => onIntent(opt.id)}
              title={opt.blurb}
              className={`cursor-pointer border px-2 py-1 text-overline font-semibold uppercase tracking-[0.14em] transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900 ${
                on
                  ? 'border-burgundy-900 bg-burgundy-900 text-newsprint'
                  : 'border-grey-300 text-grey-500 hover:border-ink hover:text-ink'
              }`}
            >
              {opt.label}
            </button>
          )
        })}
      </div>

      {blurb && <p className="mt-1.5 text-caption italic text-grey-500">{blurb}</p>}

      <label className="mt-2 flex cursor-pointer items-start gap-2 text-caption text-grey-700">
        <input
          type="checkbox"
          className="mt-0.5 cursor-pointer accent-burgundy-900"
          checked={brief.explicit}
          onChange={(e) => onExplicit(e.target.checked)}
        />
        <span>
          Explicit anatomy in frame. Weighs what a base was actually trained to draw as heavily as
          how it looks, which reorders the list completely.
        </span>
      </label>

      <p className="mt-2 text-caption leading-snug text-grey-700">{report.note}</p>

      {top && !isCurrent(top) && reachable(top) && (
        <p className="mt-1 text-caption">
          <Link onClick={() => onPick(top)}>Use {top.label}</Link>
          <span className="text-grey-400"> · </span>
          <Link onClick={() => setOpen((v) => !v)}>{open ? 'hide the ranking' : 'see the ranking'}</Link>
        </p>
      )}
      {top && (isCurrent(top) || !reachable(top)) && (
        <p className="mt-1 text-caption">
          {isCurrent(top) && <span className="italic text-grey-500">You are on it. </span>}
          <Link onClick={() => setOpen((v) => !v)}>{open ? 'hide the ranking' : 'see the ranking'}</Link>
        </p>
      )}

      {top && !reachable(top) && (
        <p className="mt-1.5 text-caption leading-snug text-warning">
          <Kicker className="block text-warning">Ranks first, not in the picker</Kicker>
          {top.label} cannot be selected here. {whyUnavailable(top)}
        </p>
      )}

      {open && (
        <ol className="mt-2 border-t border-grey-300">
          {report.ranked.map((r) => (
            <li key={`${r.familyId}:${r.model}`} className="border-b border-grey-300 py-1.5">
              <div className="flex items-baseline justify-between gap-3">
                <span className="text-small font-semibold">
                  {r.rank}. {r.label}
                  {isCurrent(r) ? ' ·  in use' : ''}
                </span>
                <span className="shrink-0 text-caption tabular-nums text-grey-500">{r.score}</span>
              </div>
              <p className="mt-0.5 text-caption leading-snug text-grey-700">{r.why}</p>
              {r.warning && (
                <p className="mt-0.5 text-caption leading-snug text-warning">{r.warning}</p>
              )}
              {r.caveat && (
                <p className="mt-0.5 text-caption italic leading-snug text-grey-500">{r.caveat}</p>
              )}
              {!isCurrent(r) && reachable(r) && (
                <p className="mt-0.5 text-caption">
                  <Link onClick={() => onPick(r)}>Use it</Link>
                </p>
              )}
              {!reachable(r) && (
                <p className="mt-0.5 text-caption italic leading-snug text-grey-500">
                  Not in the picker. {whyUnavailable(r)}
                </p>
              )}
            </li>
          ))}
        </ol>
      )}

      {top && !topRefines && (
        <p className="mt-2 text-caption leading-snug text-warning">
          <Kicker className="block text-warning">Ranks first, cannot be refined</Kicker>
          {top.label} samples through a custom schedule with no denoise control, so neither a region
          refine nor a face or hand detail pass can run on it. It is ranked on what it draws, not on
          what can be done to it afterwards. For anatomy at small scale, pick a base one row down
          that can carry the passes.
        </p>
      )}

      {mismatch && (
        <p className="mt-2 text-caption leading-snug text-warning">
          <Kicker className="block text-warning">Mismatch</Kicker>
          {mismatch}
        </p>
      )}

      {top && (
        <p className="mt-2 text-caption leading-snug text-grey-700">
          <Kicker className="block">How to write it</Kicker>
          {promptStyleNote(isCurrentStyleRec(current, report) ?? top)}
        </p>
      )}

      <p className="mt-2 text-caption leading-snug text-grey-700">
        <Kicker className="block">Anatomy</Kicker>
        {anatomyNote(brief)}
      </p>

      {report.unrouted.length > 0 && (
        <p className="mt-2 text-caption leading-snug text-error">
          <Kicker className="block text-error">On disk, not wired up</Kicker>
          {report.unrouted.map((u) => u.why).join(' ')} Adding it means a verified graph in
          src/lib/registry.ts, which is generated rather than written by hand.
        </p>
      )}

      {report.blocked.length > 0 && (
        <p className="mt-2 text-caption leading-snug text-grey-500">
          <Kicker className="block">Too large for this machine</Kicker>
          {report.blocked.map((b) => `${b.label}: ${b.why}`).join(' ')}
        </p>
      )}
    </div>
  )
}

/** The ranking row for the style actually selected, when it has one. */
function isCurrentStyleRec(current: Style | null, report: IntentReport): Recommendation | null {
  if (!current) return null
  return (
    report.ranked.find((r) => r.familyId === current.def.id && r.model === current.model) ?? null
  )
}

// ---------------------------------------------------------------------------
// Quality passes
//
// Three switches, each one a real derivation of the graph and each one costing
// real GPU time. The arithmetic is printed rather than implied: a card that is
// busy for three minutes because of one unlabelled checkbox is a bad tool.
//
// There is no detector for breasts, nipples, vulvas or penises. Faces and hands
// are the two regions YOLO can find on its own, which is why they are the only
// two offered here. Every other region needs a drawn mask and the refine bench.
// ---------------------------------------------------------------------------

function QualityPasses({
  caps,
  value,
  onChange,
  size,
  sized,
  steps,
}: {
  caps: Capabilities
  value: Passes
  onChange: (p: Passes) => void
  size: { width: number; height: number }
  /** False when the frame size is not the composer's to set, as in image to image. */
  sized: boolean
  steps: number
}) {
  const cost = passCost(value)
  const big = hiresSize(size)

  const toggle = (key: keyof Passes) => onChange({ ...value, [key]: !value[key] })

  return (
    <div>
      <Label hint="Each pass re renders part of the picture at a higher resolution. That is the only thing that adds real detail, and it costs a full pass of GPU time.">
        Detail passes
      </Label>

      <ul className="mt-1 border-t border-grey-300">
        {caps.faceDetail && (
          <PassRow
            on={value.face}
            onToggle={() => toggle('face')}
            title="Detail every face"
            note="Finds faces with a detector, crops each one, re renders it at up to 1024px and pastes it back. A face 80px across has 100 latent cells and cannot hold two eyes and a mouth; at 1024 it has nine thousand."
            cost="about one extra pass per face found"
          />
        )}
        {caps.handDetail && (
          <PassRow
            on={value.hand}
            onToggle={() => toggle('hand')}
            title="Detail every hand"
            note="The same pass, on the hand detector, at a higher strength. Hands come out wrong rather than merely soft, so this one is allowed to rebuild rather than sharpen."
            cost="about one extra pass per hand found"
          />
        )}
        {caps.hires && (
          <PassRow
            on={value.hires}
            onToggle={() => toggle('hires')}
            title="Two pass render"
            note={`Composes at the size the model was trained on, upscales the latent by 1.5, then redraws at ${hiresStepsFor(steps)} steps and 0.45 strength. Every region, breasts and hands and faces included, gets 2.25 times the cells to resolve in.${sized ? ` Output ${times(big.width, big.height)}.` : ''}`}
            cost="about 1.35 extra passes, and noticeably more VRAM"
          />
        )}
      </ul>

      {cost > 1 && (
        <p className="mt-1.5 text-caption tabular-nums text-grey-700">
          Roughly {cost.toFixed(2)}x the time of a plain picture, at least. A detector that finds
          two faces runs the face pass twice.
        </p>
      )}
    </div>
  )
}

function PassRow({
  on,
  onToggle,
  title,
  note,
  cost,
}: {
  on: boolean
  onToggle: () => void
  title: string
  note: string
  cost: string
}) {
  return (
    <li className="border-b border-grey-300 py-2">
      <label className="flex cursor-pointer items-start gap-2">
        <input
          type="checkbox"
          className="mt-1 cursor-pointer accent-burgundy-900"
          checked={on}
          onChange={onToggle}
        />
        <span className="min-w-0">
          <span className="block text-small font-semibold">{title}</span>
          <span className="mt-0.5 block text-caption leading-snug text-grey-700">{note}</span>
          <span className="mt-0.5 block text-caption italic text-grey-500">Costs {cost}.</span>
        </span>
      </label>
    </li>
  )
}

// ---------------------------------------------------------------------------
// Shape
// ---------------------------------------------------------------------------

function ShapePicker({
  shapes,
  width,
  height,
  expert,
  onPick,
  onSize,
}: {
  shapes: Shape[]
  width: number
  height: number
  expert: boolean
  onPick: (s: Shape) => void
  onSize: (w: number, h: number) => void
}) {
  const active = shapes.find((s) => s.width === width && s.height === height) ?? null
  const mp = round2((width * height) / 1e6)

  return (
    <div>
      <Label>Shape</Label>
      <div className="flex items-end gap-2">
        {shapes.map((s) => {
          const on = active?.key === s.key
          const w = 46
          const scale = Math.min(w / Math.max(s.width, s.height), 1)
          return (
            <button
              key={s.key}
              type="button"
              aria-pressed={on}
              onClick={() => onPick(s)}
              title={`${s.label} · ${times(s.width, s.height)}`}
              className="group flex w-16 cursor-pointer flex-col items-center gap-1 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900"
            >
              <span className="grid h-12 w-full place-items-center">
                <span
                  className={`block border ${on ? 'border-ink bg-ink' : 'border-grey-400 bg-transparent group-hover:border-ink'}`}
                  style={{
                    width: Math.max(10, Math.round(s.width * scale)),
                    height: Math.max(10, Math.round(s.height * scale)),
                  }}
                />
              </span>
              <span
                className={`text-overline font-semibold uppercase tracking-[0.14em] ${on ? 'text-ink' : 'text-grey-500'}`}
              >
                {s.label}
              </span>
            </button>
          )
        })}
      </div>
      <p className="mt-1.5 text-caption italic text-grey-700 tabular-nums">
        {times(width, height)} · {mp.toFixed(2)} megapixels
        {active?.maker ? ' · the maker’s shape' : ''}
      </p>

      {expert && (
        <div className="mt-3 flex items-end gap-3">
          <Stepper
            id="sg-width"
            label="Width"
            value={width}
            step={16}
            min={256}
            max={4096}
            onChange={(v) => onSize(snap16(v), height)}
          />
          <span className="pb-2 text-grey-400">×</span>
          <Stepper
            id="sg-height"
            label="Height"
            value={height}
            step={16}
            min={256}
            max={4096}
            onChange={(v) => onSize(width, snap16(v))}
          />
        </div>
      )}
    </div>
  )
}

function Stepper({
  id,
  label,
  value,
  step,
  min,
  max,
  onChange,
}: {
  id: string
  label: string
  value: number
  step: number
  min: number
  max: number
  onChange: (v: number) => void
}) {
  return (
    <label className="block w-24">
      <Label>{label}</Label>
      <input
        id={id}
        type="number"
        className="field tabular-nums"
        value={value}
        step={step}
        min={min}
        max={max}
        onChange={(e) => {
          const v = Number(e.target.value)
          if (Number.isFinite(v)) onChange(clamp(v, min, max))
        }}
      />
    </label>
  )
}

// ---------------------------------------------------------------------------
// Size, in image-to-image
// ---------------------------------------------------------------------------

function SizeFollows({
  megapixels,
  expert,
  onPick,
}: {
  megapixels: number
  expert: boolean
  onPick: (mp: number) => void
}) {
  if (!expert) {
    return (
      <p className="text-caption text-grey-700">
        <Kicker>Size</Kicker>{' '}
        <span className="italic">
          follows your picture. About {megapixels.toFixed(1)} megapixels, edges rounded to 16.
          Nothing is cropped or stretched.
        </span>
      </p>
    )
  }
  return (
    <label className="block">
      <Label hint="aspect ratio is always kept">Output size</Label>
      <div className="flex border border-grey-300">
        {MEGAPIXELS.map((mp, i) => {
          const on = Math.abs(mp - megapixels) < 0.001
          return (
            <button
              key={mp}
              id={i === 0 ? 'sg-megapixels' : undefined}
              type="button"
              aria-pressed={on}
              onClick={() => onPick(mp)}
              className={`flex-1 cursor-pointer px-2 py-1.5 text-caption tabular-nums ${i > 0 ? 'border-l border-grey-300' : ''} ${
                on ? 'bg-ink text-newsprint' : 'text-grey-700 hover:text-ink'
              } focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900`}
            >
              {mp.toFixed(1)} MP
            </button>
          )
        })}
      </div>
      <span className="mt-1 block text-caption italic text-grey-700">
        Your picture is scaled to this budget, aspect kept, edges rounded to 16.
      </span>
    </label>
  )
}

// ---------------------------------------------------------------------------
// Strength
// ---------------------------------------------------------------------------

function Strength({
  denoise,
  expert,
  steps,
  modelLabel,
  onChange,
}: {
  denoise: number
  expert: boolean
  steps: number
  modelLabel: string
  onChange: (d: number) => void
}) {
  const railRef = useRef<HTMLDivElement | null>(null)
  const min = 0.05
  const max = 0.95
  const nearest = STOPS.reduce((best, s) =>
    Math.abs(s.denoise - denoise) < Math.abs(best.denoise - denoise) ? s : best,
  )
  const onStop = Math.abs(nearest.denoise - denoise) < 0.005
  const pos = (denoise - min) / (max - min)

  const commit = (value: number) => {
    if (expert) onChange(round2(clamp(value, min, max)))
    else {
      const stop = STOPS.reduce((best, s) =>
        Math.abs(s.denoise - value) < Math.abs(best.denoise - value) ? s : best,
      )
      onChange(stop.denoise)
    }
  }

  const fromPointer = (clientX: number) => {
    const rect = railRef.current?.getBoundingClientRect()
    if (!rect || rect.width === 0) return
    commit(min + ((clientX - rect.left) / rect.width) * (max - min))
  }

  const onKey = (e: ReactKeyboardEvent) => {
    const i = STOPS.indexOf(nearest)
    if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') {
      e.preventDefault()
      if (expert) onChange(round2(clamp(denoise - 0.01, min, max)))
      else onChange(STOPS[Math.max(0, i - 1)].denoise)
    } else if (e.key === 'ArrowRight' || e.key === 'ArrowUp') {
      e.preventDefault()
      if (expert) onChange(round2(clamp(denoise + 0.01, min, max)))
      else onChange(STOPS[Math.min(STOPS.length - 1, i + 1)].denoise)
    } else if (e.key === 'Home') {
      e.preventDefault()
      onChange(STOPS[0].denoise)
    } else if (e.key === 'End') {
      e.preventDefault()
      onChange(STOPS[STOPS.length - 1].denoise)
    }
  }

  const advise = steps <= 10 && nearest.key === 'touch'

  return (
    <div>
      <div className="flex items-end justify-between">
        <Label>How much to change</Label>
        {expert && (
          <label className="mb-1.5 flex items-center gap-1.5 text-caption text-grey-700">
            <span className="uppercase tracking-[0.14em]">denoise</span>
            <input
              id="sg-denoise"
              type="number"
              className="field w-20 tabular-nums"
              value={denoise}
              min={min}
              max={max}
              step={0.01}
              onChange={(e) => {
                const v = Number(e.target.value)
                if (Number.isFinite(v)) onChange(round2(clamp(v, min, max)))
              }}
            />
          </label>
        )}
      </div>

      <div
        ref={railRef}
        role="slider"
        tabIndex={0}
        aria-label="How much to change"
        aria-valuemin={min}
        aria-valuemax={max}
        aria-valuenow={denoise}
        aria-valuetext={onStop ? nearest.label : `${nearest.label}, denoise ${denoise.toFixed(2)}`}
        onKeyDown={onKey}
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId)
          fromPointer(e.clientX)
        }}
        onPointerMove={(e) => {
          if (e.buttons) fromPointer(e.clientX)
        }}
        className="sg-tap relative mt-1 h-4 touch-none cursor-pointer select-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900"
      >
        <span className="absolute inset-x-0 top-1/2 block h-[2px] -translate-y-1/2 bg-grey-300" />
        <span
          className="absolute top-1/2 left-0 block h-[2px] -translate-y-1/2 bg-burgundy-900"
          style={{ width: `${clamp(pos, 0, 1) * 100}%` }}
        />
        {STOPS.map((s) => (
          <span
            key={s.key}
            aria-hidden
            className="absolute top-1/2 block h-2 w-px -translate-x-1/2 -translate-y-1/2 bg-grey-400"
            style={{ left: `${((s.denoise - min) / (max - min)) * 100}%` }}
          />
        ))}
        <span
          aria-hidden
          className="absolute top-1/2 block h-3.5 w-[2px] -translate-x-1/2 -translate-y-1/2 bg-burgundy-900"
          style={{ left: `${clamp(pos, 0, 1) * 100}%` }}
        />
      </div>

      <div className="relative mt-1 h-4">
        {STOPS.map((s) => {
          const on = nearest.key === s.key
          const left = ((s.denoise - min) / (max - min)) * 100
          return (
            <button
              key={s.key}
              type="button"
              onClick={() => onChange(s.denoise)}
              className={`absolute -translate-x-1/2 cursor-pointer whitespace-nowrap text-overline font-semibold uppercase tracking-[0.12em] ${
                on ? 'text-ink' : 'text-grey-500 hover:text-ink'
              } focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900`}
              style={{ left: `${clamp(left, 8, 92)}%` }}
            >
              {s.label}
            </button>
          )
        })}
      </div>

      <p className="mt-4 text-caption italic text-grey-700">{nearest.help}</p>
      {advise && (
        <p className="mt-1 text-caption italic text-grey-700">
          {modelLabel} works in {steps} steps. At Touch up only two of them are used. Rework will
          serve you better.
        </p>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// The run button
// ---------------------------------------------------------------------------

function RunButton({
  label,
  disabled,
  why,
  running,
  queuedAhead,
  lastMs,
  runs,
  job,
  onRun,
  onStop,
  reduced,
}: {
  label: string
  disabled: boolean
  why: string
  running: boolean
  queuedAhead: number
  lastMs: number | null
  runs: 1 | 2 | 4
  job: DeskJob | null
  onRun: () => void
  onStop: () => void
  reduced: boolean
}) {
  const [holding, setHolding] = useState(false)
  const [receipt, setReceipt] = useState<string | null>(null)
  const timer = useRef<number | null>(null)
  const seen = useRef<number | null>(null)

  useEffect(() => {
    if (lastMs == null || seen.current === lastMs) return
    seen.current = lastMs
    setReceipt(seconds(lastMs))
    const t = window.setTimeout(() => setReceipt(null), 2600)
    return () => window.clearTimeout(t)
  }, [lastMs])

  const beginHold = () => {
    setHolding(true)
    timer.current = window.setTimeout(() => {
      setHolding(false)
      onStop()
    }, 600)
  }
  const endHold = () => {
    setHolding(false)
    if (timer.current) window.clearTimeout(timer.current)
    timer.current = null
  }

  if (running) {
    return (
      <div>
        <button
          type="button"
          aria-label="Hold to stop this job"
          onPointerDown={beginHold}
          onPointerUp={endHold}
          onPointerLeave={endHold}
          onPointerCancel={endHold}
          onKeyDown={(e: ReactKeyboardEvent) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault()
              onStop()
            }
          }}
          className="press sg-hold relative overflow-hidden"
          style={{ backgroundColor: 'var(--color-newsprint)', color: 'var(--color-burgundy-900)' }}
        >
          <span className="relative">Hold to stop</span>
          <span
            aria-hidden
            className="absolute inset-0 grid place-items-center bg-burgundy-900 text-newsprint"
            style={{
              clipPath: holding ? 'inset(0 0 0 0)' : 'inset(0 100% 0 0)',
              transition: reduced ? 'none' : 'clip-path 600ms linear',
            }}
          >
            Hold to stop
          </span>
        </button>
        <p className="mt-1.5 text-caption italic text-grey-700 tabular-nums">
          {job?.total && job.total > 1 ? `Picture ${job.index} of ${job.total} · ` : ''}
          {job?.stage ?? 'Working'}
          {job && job.pct >= 0.97 && job.status === 'running' ? ' · running long, still working' : ''}
        </p>
      </div>
    )
  }

  return (
    <div>
      <button type="button" className="press" disabled={disabled} onClick={onRun}>
        {receipt ? (
          <span className="tabular-nums">{receipt}</span>
        ) : (
          <>
            {label}
            {runs > 1 ? ` · ×${runs}` : ''}
            {queuedAhead > 0 ? ' · next in line' : ''}
          </>
        )}
      </button>
      {disabled && why && <p className="mt-1.5 text-caption italic text-grey-500">{why}</p>}
      {!disabled && queuedAhead > 0 && (
        <p className="mt-1.5 text-caption italic text-grey-700">
          There is one 16 GB card and it is busy. Your picture starts when the job in front of it
          finishes.
        </p>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// What simple mode chose
// ---------------------------------------------------------------------------

function SettingLine({
  c,
  expert,
  hasScheduler,
  onOpen,
}: {
  c: Composition
  expert: boolean
  hasScheduler: boolean
  onOpen: (field: string) => void
}) {
  if (expert) return null
  return (
    <p className="mt-2 text-caption italic text-grey-700">
      Using the maker’s settings:{' '}
      <Link className="not-italic tabular-nums" onClick={() => onOpen('sg-steps')}>
        {c.steps} steps
      </Link>
      <span> · </span>
      <Link className="not-italic tabular-nums" onClick={() => onOpen('sg-cfg')}>
        CFG {c.cfg.toFixed(1)}
      </Link>
      <span> · </span>
      <Link className="not-italic" onClick={() => onOpen('sg-sampler')}>
        {c.sampler}
        {hasScheduler ? ` / ${c.scheduler}` : ''}
      </Link>
      <span> · </span>
      <Link className="not-italic" onClick={() => onOpen('sg-seed')}>
        seed {c.seedLocked ? c.seed : 'random'}
      </Link>
      . <Link onClick={() => onOpen('sg-steps')}>Show all controls →</Link>
    </p>
  )
}

// ---------------------------------------------------------------------------
// Faults
// ---------------------------------------------------------------------------

function Fault({ fault, onDismiss }: { fault: DeskFault; onDismiss: () => void }) {
  if (fault.cancelled) {
    return (
      <Notice kind="correction" title="Correction">
        Job stopped. Nothing was saved. <Link onClick={onDismiss}>Dismiss</Link>
      </Notice>
    )
  }

  if (fault.lost) {
    return (
      <Notice kind="warning" title="We lost track of that job">
        {fault.message} The desk is free again, so you can try again.{' '}
        <Link onClick={onDismiss}>Dismiss</Link>
      </Notice>
    )
  }

  const text = `${fault.message} ${fault.detail ?? ''}`.toLowerCase()
  if (text.includes('out of memory') || text.includes('cuda') || text.includes('alloc')) {
    return (
      <Notice kind="error" title="The card ran out of memory">
        This size needs more than the card has free. Try a smaller shape, or close anything else
        using the GPU. <Link onClick={onDismiss}>Dismiss</Link>
      </Notice>
    )
  }

  return (
    <Notice kind="error" title="That job was rejected">
      ComfyUI would not accept it: {fault.detail ?? fault.message}
      {fault.nodeType && (
        <span className="block text-caption">
          The trouble is in {fault.nodeType}
          {fault.node ? ` (node ${fault.node})` : ''}.
        </span>
      )}
      <span className="block">
        Check the highlighted setting and try again. <Link onClick={onDismiss}>Dismiss</Link>
      </span>
    </Notice>
  )
}

// ---------------------------------------------------------------------------
// The expert margin
// ---------------------------------------------------------------------------

function ExpertMargin({
  c,
  style,
  cat,
  hasNegative,
  hasScheduler,
  hasShift,
  hasClipSkip,
  houseNegative,
  graph,
  queued,
  hires,
  faultNode,
  onClose,
}: {
  c: Composition
  style: Style
  cat: Catalogue
  hasNegative: boolean
  hasScheduler: boolean
  hasShift: boolean
  hasClipSkip: boolean
  houseNegative: string
  /** The family graph, for the registry's own notes. */
  graph: FamilyDef
  /** The same graph with the passes and LoRAs on it, which is what gets sent. */
  queued: FamilyDef | DerivedDef
  hires: boolean
  faultNode: string | null
  onClose: () => void
}) {
  const [showJson, setShowJson] = useState(false)
  const reduced = useReducedMotion()
  const [filled, setFilled] = useState(false)

  // The margin fills in; nothing else on the desk moves.
  useEffect(() => {
    const id = requestAnimationFrame(() => setFilled(true))
    return () => cancelAnimationFrame(id)
  }, [])

  const shown = filled || reduced

  return (
    <div
      style={{
        opacity: shown ? 1 : 0,
        transform: shown ? 'translateX(0)' : 'translateX(-8px)',
        transition: reduced ? 'none' : 'opacity 180ms ease-out, transform 180ms ease-out',
      }}
      className="text-small"
    >
      <div className="mb-4 flex items-baseline justify-between border-b-2 border-burgundy-900 pb-1.5">
        <Kicker className="text-burgundy-900">All controls</Kicker>
        <Link className="text-caption" onClick={onClose}>
          ◂ Simple
        </Link>
      </div>

      <ExpertRow label="Steps" hint="more steps, more time">
        <input
          id="sg-steps"
          type="number"
          className="field tabular-nums"
          min={1}
          max={150}
          value={c.steps}
          onChange={(e) => {
            const v = Number(e.target.value)
            if (Number.isFinite(v)) store.edit({ steps: clamp(Math.round(v), 1, 150) }, 'steps')
          }}
        />
      </ExpertRow>

      <ExpertRow label="CFG" hint="how hard it follows the words">
        <input
          id="sg-cfg"
          type="number"
          className="field tabular-nums"
          min={0}
          max={30}
          step={0.1}
          value={c.cfg}
          onChange={(e) => {
            const v = Number(e.target.value)
            if (Number.isFinite(v)) store.edit({ cfg: clamp(v, 0, 30) }, 'cfg')
          }}
        />
      </ExpertRow>

      <ExpertRow label="Sampler">
        <select
          id="sg-sampler"
          className="field"
          value={c.sampler}
          onChange={(e) => store.edit({ sampler: e.target.value }, 'sampler')}
        >
          {!cat.samplers.includes(c.sampler) && <option value={c.sampler}>{c.sampler}</option>}
          {cat.samplers.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </ExpertRow>

      {hasScheduler && (
        <ExpertRow label="Scheduler">
          <select
            id="sg-scheduler"
            className="field"
            value={c.scheduler}
            onChange={(e) => store.edit({ scheduler: e.target.value }, 'scheduler')}
          >
            {!cat.schedulers.includes(c.scheduler) && (
              <option value={c.scheduler}>{c.scheduler}</option>
            )}
            {cat.schedulers.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </ExpertRow>
      )}

      {style.alt && (
        <button
          type="button"
          onClick={() =>
            store.edit(
              {
                sampler: style.alt!.sampler,
                scheduler: style.alt!.scheduler,
                steps: style.alt!.steps,
                cfg: style.alt!.cfg,
              },
              'sampler',
              'scheduler',
              'steps',
              'cfg',
            )
          }
          className="mb-4 w-full cursor-pointer border border-grey-300 px-2 py-1.5 text-left text-caption text-grey-700 hover:border-ink hover:text-ink focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900"
        >
          <span className="block text-overline font-semibold uppercase tracking-[0.16em]">
            Use the author’s alternative
          </span>
          <span className="tabular-nums">
            {style.alt.sampler} / {style.alt.scheduler}, {style.alt.steps} steps, CFG{' '}
            {style.alt.cfg.toFixed(1)}
          </span>
        </button>
      )}

      <ExpertRow label="Seed">
        <div className="flex items-center gap-2">
          <input
            id="sg-seed"
            type="number"
            className={`field tabular-nums ${c.seedLocked ? 'not-italic text-ink' : 'italic text-grey-500'}`}
            value={c.seed}
            min={0}
            onChange={(e) => {
              const v = Number(e.target.value)
              if (Number.isFinite(v)) store.edit({ seed: Math.max(0, Math.floor(v)), seedLocked: true }, 'seed')
            }}
          />
          <button
            type="button"
            onClick={() => store.patch({ seedLocked: !c.seedLocked })}
            aria-pressed={c.seedLocked}
            className={`shrink-0 cursor-pointer border px-2 py-1.5 text-overline font-semibold uppercase tracking-[0.14em] ${
              c.seedLocked ? 'border-ink bg-ink text-newsprint' : 'border-grey-300 text-grey-700'
            } focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900`}
          >
            {c.seedLocked ? 'Fixed' : 'Random'}
          </button>
        </div>
        <span className="mt-1 block text-caption italic text-grey-500">
          {c.seedLocked
            ? 'The same seed and the same settings make the same picture.'
            : 'A fresh seed every run. The last one used is shown above.'}
        </span>
      </ExpertRow>

      {hasNegative && (
        <ExpertRow label="Negative prompt" hint="what to keep out">
          <textarea
            id="sg-negative"
            className="field"
            rows={3}
            value={c.negative ?? houseNegative}
            onChange={(e) => store.edit({ negative: e.target.value }, 'negative')}
          />
          {c.negative !== null && (
            <Link
              className="mt-1 text-caption"
              onClick={() => store.set({ ...c, negative: null, touched: c.touched.filter((t) => t !== 'negative') })}
            >
              Reset to the house wording
            </Link>
          )}
        </ExpertRow>
      )}

      {style.positivePrefix !== null && (
        <ExpertRow label="Quality words" hint="added in front of your prompt">
          <input
            id="sg-prefix"
            className="field"
            value={c.positivePrefix ?? ''}
            onChange={(e) => store.edit({ positivePrefix: e.target.value }, 'positivePrefix')}
          />
        </ExpertRow>
      )}

      {hasClipSkip && (
        <ExpertRow label="Clip skip" hint="−2 on Illustrious checkpoints">
          <input
            id="sg-clipskip"
            type="number"
            className="field tabular-nums"
            min={-12}
            max={-1}
            value={c.clipSkip ?? -1}
            onChange={(e) => {
              const v = Number(e.target.value)
              if (Number.isFinite(v)) store.edit({ clipSkip: clamp(Math.round(v), -12, -1) }, 'clipSkip')
            }}
          />
        </ExpertRow>
      )}

      {hasShift && (
        <ExpertRow label="Shift" hint="sampling curve">
          <input
            id="sg-shift"
            type="number"
            className="field tabular-nums"
            step={0.1}
            min={0}
            max={12}
            value={c.shift ?? 0}
            onChange={(e) => {
              const v = Number(e.target.value)
              if (Number.isFinite(v)) store.edit({ shift: clamp(v, 0, 12) }, 'shift')
            }}
          />
        </ExpertRow>
      )}

      <ExpertRow label="How many" hint="one after another, never in one batch">
        <div className="flex border border-grey-300">
          {([1, 2, 4] as const).map((n, i) => (
            <button
              key={n}
              type="button"
              aria-pressed={c.runs === n}
              onClick={() => store.patch({ runs: n })}
              className={`flex-1 cursor-pointer px-2 py-1.5 text-caption tabular-nums ${i > 0 ? 'border-l border-grey-300' : ''} ${
                c.runs === n ? 'bg-ink text-newsprint' : 'text-grey-700 hover:text-ink'
              } focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900`}
            >
              ×{n}
            </button>
          ))}
        </div>
      </ExpertRow>

      {(style.note || graph.notes) && (
        <div className="mb-5 border-l-[3px] border-burgundy-900 pl-3">
          <p className="text-small italic leading-snug text-grey-700">
            {style.note || trim(graph.notes, 320)}
          </p>
          {style.note && graph.notes && (
            <details className="mt-2">
              <summary className="cursor-pointer text-caption text-grey-500">
                More on this family
              </summary>
              <p className="mt-1 text-caption italic leading-snug text-grey-700">
                {trim(graph.notes, 700)}
              </p>
            </details>
          )}
        </div>
      )}

      {cat.unavailable.length > 0 && (
        <details className="mb-5">
          <summary className="cursor-pointer text-caption text-grey-500">
            Detected but unavailable ({cat.unavailable.length})
          </summary>
          <ul className="mt-1 space-y-1">
            {cat.unavailable.map((u) => (
              <li key={u.name} className="text-caption text-grey-700">
                <span className="block truncate">{u.name}</span>
                <span className="italic text-grey-500">{u.why}</span>
              </li>
            ))}
          </ul>
        </details>
      )}

      <div>
        <Link className="text-caption" onClick={() => setShowJson((v) => !v)}>
          {showJson ? 'Hide the workflow' : 'Show the workflow'}
        </Link>
        {showJson && (
          <WorkflowPeek
            def={queued}
            hires={hires}
            c={c}
            houseNegative={houseNegative}
            faultNode={faultNode}
          />
        )}
      </div>
    </div>
  )
}

function ExpertRow({
  label,
  hint,
  children,
}: {
  label: string
  hint?: string
  children: ReactNode
}) {
  return (
    <div className="mb-4 border-b border-grey-300 pb-3">
      <Label hint={hint}>{label}</Label>
      {children}
    </div>
  )
}

function trim(text: string, n: number): string {
  if (text.length <= n) return text
  return `${text.slice(0, n).replace(/\s+\S*$/, '')}…`
}

/**
 * The graph, as JSON, with a Copy button.
 *
 * `def` is the derived graph the run button would send, not the bare family
 * one: with a detail pass, a two pass render or a LoRA switched on, the two
 * differ by several nodes and a rewired decode, and printing the bare graph
 * under a Copy button meant the JSON on screen reproduced none of the picture.
 */
function WorkflowPeek({
  def,
  hires,
  c,
  houseNegative,
  faultNode,
}: {
  def: FamilyDef | DerivedDef
  hires: boolean
  c: Composition
  houseNegative: string
  faultNode: string | null
}) {
  const [copied, setCopied] = useState(false)
  const json = useMemo(() => {
    try {
      // The same shaping start() does before it queues. Without it the preview
      // carries a denoise or a source left over from another mode, which the
      // real run drops.
      const shaped: Composition = {
        ...c,
        source: needsSource(c.mode) ? c.source : null,
        denoise: c.mode === 'i2i' ? (c.denoise ?? DEFAULT_DENOISE) : null,
        megapixels: c.mode === 'i2i' ? (c.megapixels ?? 1) : null,
        length: null,
        fps: null,
      }
      const params = toParams(shaped, { negative: houseNegative })
      return JSON.stringify(buildQueued(def, params, hires), null, 2)
    } catch (err) {
      return `Could not build the graph: ${err instanceof Error ? err.message : String(err)}`
    }
  }, [def, hires, c, houseNegative])

  return (
    <div className="mt-2">
      <p className="mb-1 text-caption text-grey-500">
        {Object.keys(def.graph).length} nodes
        {faultNode ? ` · node ${faultNode} is the one ComfyUI complained about` : ''}
        {' · '}
        <Link
          onClick={() => {
            void navigator.clipboard?.writeText(json).then(
              () => {
                setCopied(true)
                setTimeout(() => setCopied(false), 1500)
              },
              () => setCopied(false),
            )
          }}
        >
          {copied ? 'Copied' : 'Copy'}
        </Link>
      </p>
      <pre className="max-h-80 overflow-auto border border-grey-300 bg-newsprint-aged p-2 text-[11px] leading-snug">
        {json}
      </pre>
    </div>
  )
}

// ---------------------------------------------------------------------------
// The plate
// ---------------------------------------------------------------------------

function Plate({
  state,
  entry,
  c,
  style,
  reduced,
  adopted,
  examples,
  onExample,
  onAdopt,
  onShow,
  onWorkFrom,
  onChangeThis,
  canI2I,
  canEdit,
  canRefine,
  onRefine,
}: {
  state: PressState
  entry: HistoryEntry | null
  c: Composition
  style: Style | null
  reduced: boolean
  adopted: string | null
  examples: typeof EXAMPLES
  onExample: (ex: (typeof EXAMPLES)[number]) => void
  onAdopt: (field: Parameters<typeof adoptValue>[1], value: number | string, label: string) => void
  onShow: (e: HistoryEntry) => void
  onWorkFrom: (e: HistoryEntry) => void
  onChangeThis: (e: HistoryEntry) => void
  canI2I: boolean
  canEdit: boolean
  canRefine: boolean
  onRefine: (e: HistoryEntry) => void
}) {
  const job = state.job
  const running = busy(state)
  const [developed, setDeveloped] = useState(false)

  useEffect(() => {
    if (!entry) return
    if (reduced) {
      setDeveloped(true)
      return
    }
    setDeveloped(false)
    const id = requestAnimationFrame(() => setDeveloped(true))
    return () => cancelAnimationFrame(id)
  }, [entry, reduced])

  return (
    <div className="flex h-full flex-col">
      <div className="mb-4 flex items-baseline justify-between border-b-2 border-burgundy-900 pb-1.5">
        <Kicker className="text-burgundy-900">The Plate</Kicker>
        {adopted && <Kicker className="text-ink">{adopted} adopted</Kicker>}
      </div>

      <div className="relative flex min-h-[18rem] flex-1 items-center justify-center border border-grey-300 bg-newsprint-aged">
        {running && job?.previewUrl ? (
          <img
            src={job.previewUrl}
            alt="The picture as it develops"
            className="max-h-[70vh] max-w-full object-contain opacity-80"
          />
        ) : entry ? (
          <img
            key={entry.id}
            src={fileUrl(entry.file)}
            alt={entry.prompt || 'The finished picture'}
            className="max-h-[70vh] max-w-full object-contain"
            style={{
              filter: developed ? 'blur(0px)' : 'blur(8px)',
              transition: reduced ? 'none' : 'filter 420ms ease-out',
            }}
          />
        ) : (
          <EmptyPlate examples={examples} onExample={onExample} running={running} />
        )}

        {running && job && (
          <div className="absolute inset-x-0 bottom-0">
            <div className="h-[2px] w-full bg-grey-200">
              <div
                className="h-full bg-burgundy-900"
                style={{
                  width: `${Math.round(job.pct * 100)}%`,
                  transition: reduced ? 'none' : 'width 200ms linear',
                }}
              />
            </div>
            <p className="bg-newsprint/90 px-3 py-1 text-overline font-semibold uppercase tracking-[0.18em] text-grey-700 tabular-nums">
              {job.stage}
            </p>
          </div>
        )}
      </div>

      {entry && !running && (
        <Caption
          entry={entry}
          style={style}
          onAdopt={onAdopt}
          onWorkFrom={onWorkFrom}
          onChangeThis={onChangeThis}
          onRefine={onRefine}
          canI2I={canI2I}
          canEdit={canEdit}
          canRefine={canRefine}
        />
      )}

      {!entry && !running && c.prompt.trim() && (
        <p className="mt-3 text-small italic text-grey-500">
          Nothing yet. Press Make the picture on the left.
        </p>
      )}

      {state.results.length > 1 && (
        <TodayStrip results={state.results} current={entry} onShow={onShow} />
      )}
    </div>
  )
}

function EmptyPlate({
  examples,
  onExample,
  running,
}: {
  examples: typeof EXAMPLES
  onExample: (ex: (typeof EXAMPLES)[number]) => void
  running: boolean
}) {
  if (running) {
    return <p className="px-8 text-center text-small italic text-grey-500">Setting the press…</p>
  }
  return (
    <div className="max-w-xl px-8 py-10">
      <p className="dropcap text-body leading-relaxed text-grey-700">
        Type a line on the left and press Make the picture. Everything you make is filed in the
        archive with the settings that made it, so you can find it again and change one word.
      </p>
      {examples.length > 0 && (
        <div className="mt-6">
          <Kicker>Try one of these</Kicker>
          <ul className="mt-2 border-t border-grey-300">
            {examples.map((ex) => (
              <li key={ex.prompt} className="border-b border-grey-300">
                <button
                  type="button"
                  onClick={() => onExample(ex)}
                  className="flex w-full cursor-pointer items-baseline justify-between gap-4 py-2 text-left hover:text-burgundy-900 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900"
                >
                  <span className="text-body">{ex.prompt}</span>
                  <span className="shrink-0 text-caption italic text-grey-500">
                    {BY_ID[ex.familyId] ? groupName(BY_ID[ex.familyId]) : ex.familyId} · {ex.shape}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

function Caption({
  entry,
  style,
  onAdopt,
  onWorkFrom,
  onChangeThis,
  onRefine,
  canI2I,
  canEdit,
  canRefine,
}: {
  entry: HistoryEntry
  style: Style | null
  onAdopt: (field: Parameters<typeof adoptValue>[1], value: number | string, label: string) => void
  onWorkFrom: (e: HistoryEntry) => void
  onChangeThis: (e: HistoryEntry) => void
  onRefine: (e: HistoryEntry) => void
  canI2I: boolean
  canEdit: boolean
  canRefine: boolean
}) {
  const starred = !!entry.starred

  return (
    <div className="mt-3">
      {entry.prompt && (
        <p className="border-l-4 border-burgundy-900 pl-4 text-h3 italic leading-snug">
          {entry.prompt}
        </p>
      )}

      <p className="mt-2 text-small text-grey-700">
        Made by {entry.modelLabel} · {DATE.format(entry.at)} ·{' '}
        <span className="tabular-nums">{seconds(entry.durationMs)}</span>
      </p>

      <p className="mt-1 text-caption text-grey-700">
        {entry.width && entry.height ? (
          <>
            <Link
              className="not-italic tabular-nums"
              onClick={() => {
                onAdopt('width', entry.width!, 'Width')
                onAdopt('height', entry.height!, 'Size')
              }}
            >
              {times(entry.width, entry.height)}
            </Link>
            <span className="text-grey-400"> · </span>
          </>
        ) : null}
        <Link className="tabular-nums" onClick={() => onAdopt('steps', entry.steps, 'Steps')}>
          {entry.steps} steps
        </Link>
        <span className="text-grey-400"> · </span>
        <Link className="tabular-nums" onClick={() => onAdopt('cfg', entry.cfg, 'CFG')}>
          CFG {entry.cfg.toFixed(1)}
        </Link>
        <span className="text-grey-400"> · </span>
        <Link onClick={() => onAdopt('sampler', entry.sampler, 'Sampler')}>{entry.sampler}</Link>
        <span className="text-grey-400"> · </span>
        <Link className="tabular-nums" onClick={() => onAdopt('seed', entry.seed, 'Seed')}>
          seed {entry.seed}
        </Link>
        {entry.denoise != null && (
          <>
            <span className="text-grey-400"> · </span>
            <Link className="tabular-nums" onClick={() => onAdopt('denoise', entry.denoise!, 'Strength')}>
              denoise {entry.denoise.toFixed(2)}
            </Link>
          </>
        )}
        <span className="text-grey-400"> · </span>
        <span className="italic text-grey-500">{entry.file.filename}</span>
      </p>

      <p className="mt-2 text-caption">
        {canRefine && (
          <>
            <Link onClick={() => onRefine(entry)}>Refine a region</Link>
            <span className="text-grey-400"> · </span>
          </>
        )}
        {canI2I && <Link onClick={() => onWorkFrom(entry)}>Work from this</Link>}
        {canI2I && canEdit && <span className="text-grey-400"> · </span>}
        {canEdit && <Link onClick={() => onChangeThis(entry)}>Change this</Link>}
        <span className="text-grey-400"> · </span>
        <Link onClick={() => void download(fileUrl(entry.file), downloadName(entry))}>
          Save the picture
        </Link>
        <span className="text-grey-400"> · </span>
        <Link onClick={() => starRecord(entry.id, !starred)}>
          {starred ? 'Starred ★' : 'Star it'}
        </Link>
      </p>

      {style && !style.def.verified && (
        <p className="mt-1 text-caption italic text-grey-500">
          † Made with settings from the model’s card, not from a checked run.
        </p>
      )}
    </div>
  )
}

function downloadName(entry: HistoryEntry): string {
  const ext = entry.file.filename.split('.').pop() ?? 'png'
  return `switchgen-${entry.familyId}-${entry.seed}.${ext}`
}

/** ComfyUI's /view sets no Content-Disposition, so a plain link would open it. */
async function download(url: string, name: string): Promise<void> {
  try {
    const res = await fetch(url)
    const blob = await res.blob()
    const href = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = href
    a.download = name
    document.body.appendChild(a)
    a.click()
    a.remove()
    setTimeout(() => URL.revokeObjectURL(href), 1000)
  } catch {
    window.open(url, '_blank', 'noopener')
  }
}

function TodayStrip({
  results,
  current,
  onShow,
}: {
  results: HistoryEntry[]
  current: HistoryEntry | null
  onShow: (e: HistoryEntry) => void
}) {
  return (
    <div className="mt-5">
      <Kicker>This session</Kicker>
      <div className="mt-2 flex gap-2 overflow-x-auto pb-1">
        {results.map((r) => {
          const on = current?.id === r.id
          return (
            <button
              key={r.id}
              type="button"
              onClick={() => onShow(r)}
              title={r.prompt}
              className={`h-16 w-16 shrink-0 cursor-pointer overflow-hidden border ${
                on ? 'border-burgundy-900' : 'border-grey-300 hover:border-ink'
              } focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900`}
            >
              <img src={fileUrl(r.file)} alt="" className="h-full w-full object-cover" />
            </button>
          )
        })}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Taking a source out of the archive
// ---------------------------------------------------------------------------

function ArchivePicker({
  records,
  onClose,
  onPick,
}: {
  records: readonly HistoryEntry[]
  onClose: () => void
  onPick: (e: HistoryEntry) => void
}) {
  const [query, setQuery] = useState('')
  const panel = useRef<HTMLDivElement | null>(null)
  const pictures = useMemo(() => records.filter((r) => r.kind === 'image' && !r.missing), [records])
  const shown = useMemo(
    () => (query.trim() ? searchRecords(query, pictures) : [...pictures]).slice(0, 120),
    [query, pictures],
  )

  /**
   * A dialog that declares aria-modal has to behave like one.
   *
   * Escape closes, Tab cycles inside the panel, focus goes back where it came
   * from, and every other key stops here so the desk's own single key
   * bindings do not fire behind the scrim: u used to open the file dialog on
   * top of this one. Capture phase, the way DeleteDialog does it. Stopping
   * propagation is not stopping the default action, so typing in the search
   * box still works.
   */
  useEffect(() => {
    const restore = document.activeElement as HTMLElement | null
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault()
        e.stopPropagation()
        onClose()
        return
      }
      if (e.key === 'Tab') {
        const focusable = panel.current?.querySelectorAll<HTMLElement>(
          'button:not(:disabled), input:not(:disabled), [href]',
        )
        if (!focusable || !focusable.length) return
        const first = focusable[0]
        const last = focusable[focusable.length - 1]
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault()
          last.focus()
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault()
          first.focus()
        }
        return
      }
      e.stopPropagation()
    }
    document.addEventListener('keydown', onKey, true)
    return () => {
      document.removeEventListener('keydown', onKey, true)
      restore?.focus?.()
    }
  }, [onClose])

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Take a picture from the archive"
      className="fixed inset-0 z-40 grid place-items-center bg-ink/40 p-6"
      onClick={onClose}
    >
      <div
        ref={panel}
        className="flex max-h-[min(80dvh,calc(100dvh-3rem-var(--sg-safe-t)-var(--sg-safe-b)))] w-full max-w-3xl flex-col border border-grey-300 bg-newsprint p-5"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-3 flex items-baseline justify-between border-b-2 border-burgundy-900 pb-1.5">
          <Kicker className="text-burgundy-900">From the archive</Kicker>
          <Link className="text-caption" onClick={onClose}>
            Close
          </Link>
        </div>

        <input
          autoFocus
          className="field mb-3"
          placeholder="Search your pictures. Try rain neon, model:krea, is:starred"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />

        {shown.length === 0 ? (
          <p className="py-8 text-center text-small italic text-grey-500">
            {pictures.length
              ? 'Nothing matches that.'
              : 'Nothing in the archive yet. Make a picture first, then it can seed the next one.'}
          </p>
        ) : (
          <ul className="grid grid-cols-3 gap-3 overflow-y-auto sm:grid-cols-4 md:grid-cols-5">
            {shown.map((r) => (
              <li key={r.id}>
                <button
                  type="button"
                  onClick={() => onPick(r)}
                  className="group block w-full cursor-pointer text-left focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900"
                >
                  <span className="block aspect-square overflow-hidden border border-grey-300 group-hover:border-burgundy-900">
                    <img src={fileUrl(r.file)} alt="" className="h-full w-full object-cover" />
                  </span>
                  <span className="mt-1 block truncate text-caption text-grey-700">
                    {r.prompt || r.file.filename}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}
