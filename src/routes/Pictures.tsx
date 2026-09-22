/**
 * The Pictures desk.
 *
 * The default screen asks for three things: a prompt, a look, and how much
 * anatomy. Everything else is DECIDED, by lib/recipe.ts, from measurements:
 * which family, which weight file, which LoRAs at which strengths, the prompt
 * prefix that file was trained with, the sampler and the size. What was chosen
 * and why is printed in prose under the button, not offered as forty controls
 * above it.
 *
 * Nothing was removed to get there. Three destinations hold what used to sit on
 * this screen:
 *
 *   components/compose   the three answers, the button, the prose, the footnote
 *   components/advanced  every model, every LoRA, every strength, the sampler,
 *                        the size, the seed, the passes and the workflow JSON,
 *                        mounted only while More is open
 *   components/result    the quality passes, offered on the finished picture
 *                        where the reader can see whether they are needed
 *
 * This file is what remains: the catalogue, the job engine, the plate, the
 * refine bench, the archive picker, and the wiring between them.
 *
 * The job engine lives at module scope, below the imports. A desk that keeps
 * its progress while you read the archive is the difference between a tool and
 * a demo, and there is no press.ts in this build to do it for us.
 */
import { Reading } from '../components/result/Reading'
import type { ImageFacts, VisionReport } from '../lib/vision'
import { AddOnOffers } from '../components/compose/AddOnOffers'
import { RecipeProse } from '../components/compose/RecipeProse'
import { faultBody, faultOf, faultTitle, faultWhere, type Fault as DeskFault } from '../lib/faults'
import {
  NO_PASS_BLOCKS,
  availabilityOf,
  inventoryFrom,
  missingWhy,
  packNeededFor,
  passBlocks,
  type PassBlocks,
} from '../lib/availability'
import { measureImage as measure } from '../lib/images'
import { clamp } from '../lib/num'
import { ServerDown } from '../components/ServerDown'
import { onPlanLanded } from '../lib/downloads'
import { thumbSrcSet, thumbUrl } from '../lib/thumbs'
import { Kicker, Link, Notice } from '../components/type'
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
  type DragEvent as ReactDragEvent,
  type ReactNode,
  type RefObject,
} from 'react'
import {
  cancelJob,
  connectionState,
  fileUrl,
  listJobs,
  objectInfo,
  run,
  uploadImage,
  watchConnection,
  type ApiWorkflow,
  type ProgressEvent,
  relPath,
  VIDEO_EXT,
} from '../lib/comfy'
import {
  BY_ID,
  FAMILIES,
  IMG2IMG,
  defaultsFor,
  deriveImg2Img,
  instantiate,
  type FamilyDef,
  type Params,
  familyOwning
} from '../lib/workflows'
import {
  feasibility,
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
  update as updateRecord
} from '../lib/history'
import {
  adoptValue,
  compositionFromEntry,
  deskStore,
  needsSource,
  randomSeed,
  recordOf,
  settings,
  takeRegionRequest,
  toParams,
  type Composition,
  type Mode,
  type SourceRef,
} from '../lib/session'
import {
  capabilitiesOf,
  deriveAutoDetail,
  deriveHiresFix,
  deriveRefine,
  hiresStepsFor,
  instantiateRefine,
  rebuildable,
  regionOrigin,
  withLoras,
  writeExtras,
  type DerivedDef,
  type LoraSpec,
} from '../lib/refine'
import { EMPTY_LIBRARY, loadLoraLibrary, type LoraLibrary, defaultStrength, fitFor, targetFor, triggersFor, archFor } from '../lib/loras'
import {
  LOOKS,
  decide,
  passesFor,
  plainWords,
  positiveFor,
  regionAddOns,
  suggestAnatomy,
  type AnatomyLevel,
  type Look,
  type Recipe,
  type RecipeNote,
  type RecipeLora
} from '../lib/recipe'
import { RegionRefine, type RefineRequest } from '../components/refine'
import {
  ComposeDesk,
  MoreFootnote,
  PromptField,
  RunButton,
  SourceWell,
  Choice,
} from '../components/compose'
import {
  AdvancedPanel,
  NO_PASSES,
  buildGraph,
  settle,
  useOverrides,
  type Overrides,
  type Passes,
} from '../components/advanced'
import { ResultActions } from '../components/result'
import { INTENTS, intentReport, lookScore, strongestLook, type Intent } from '../lib/intent'

// ---------------------------------------------------------------------------
// Vocabulary
// ---------------------------------------------------------------------------

const DESK = 'images' as const

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
  [/^LoraLoaderModelOnly$/, 'Loading add-ons'],
  [/^(LoadImage|VAEEncode|ImageScaleToTotalPixels|FluxKontext)/, 'Preparing your picture'],
  [/^(KSampler|KSamplerAdvanced|SamplerCustomAdvanced)/, 'Drawing'],
  [/^(VAEDecode|VAEDecodeTiled)$/, 'Developing the picture'],
  [/^SaveImage$/, 'Writing the file'],
]

/** Openers for an empty plate. Each one sets the prompt and the look, nothing else. */
const EXAMPLES: { prompt: string; look: Look }[] = [
  { prompt: 'A rain-slicked tram stop at dusk, neon in the puddles', look: 'photoreal' },
  { prompt: 'A portrait of a woman in a red coat, low winter sun', look: 'photoreal' },
  { prompt: 'Steam rising off wet asphalt at first light', look: 'illustration' },
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
  /** What stops each quality pass from running on this ComfyUI, by name. */
  passBlocks: PassBlocks
}

/** The same weight file of the same family. */
function sameStyle(a: Style | null, b: Style | null): boolean {
  return !!a && !!b && a.def.id === b.def.id && a.model === b.model
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

  const inv = inventoryFrom(info)
  const installed = inv.weights

  const styles: Style[] = []
  const unavailable: { name: string; why: string }[] = []

  for (const model of installed) {
    const def = familyOwning(model)
    if (!def || (def.mode !== 'image' && def.mode !== 'edit')) continue

    // Every file the graph references must exist, or the run fails with an
    // opaque backend error. Name the missing file instead; then the memory,
    // priced on this file rather than the family's default.
    const avail = availabilityOf(def, inv, hardware, sizes, model)
    if (!avail.ok) {
      unavailable.push({ name: model, why: avail.why })
      continue
    }
    styles.push(styleOf(def, model, avail.verdict))
  }

  // A weight file ComfyUI cannot list because the node that reads it is not
  // installed never reaches the loop above: it is on disk, and it looked like
  // nothing at all. The instruction editing model is a .gguf, so without the
  // GGUF node pack "Change a picture" simply vanished. It is named here with
  // the pack it needs.
  for (const def of FAMILIES) {
    if (def.mode !== 'image' && def.mode !== 'edit') continue
    for (const model of def.models) {
      if (installed.has(model) || !sizes.has(model)) continue
      if (!packNeededFor(model, inv)) continue
      unavailable.push({ name: model, why: missingWhy([model], inv) })
    }
  }

  styles.sort((a, b) => a.group.localeCompare(b.group) || a.label.localeCompare(b.label))

  return {
    styles,
    unavailable,
    samplers: inv.samplers,
    schedulers: inv.schedulers,
    hardware,
    installed: [...installed],
    sizes,
    passBlocks: passBlocks(inv),
  }
}

let cataloguePromise: Promise<Catalogue> | null = null
function catalogue(reload = false): Promise<Catalogue> {
  if (reload || !cataloguePromise) cataloguePromise = readCatalogue()
  return cataloguePromise
}

// A family fetched from the catalogue changes what this machine can run, and
// the fetch can land while the reader is in another room. The desk reads the
// catalogue once and keeps it, so the kept reading goes here, and the next
// visit reads what is installed now.
onPlanLanded(() => {
  cataloguePromise = null
})

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

/** The one fault this desk writes itself. Fault gives it its own title. */
const NO_FILE = 'The job finished but wrote no file. Check ComfyUI’s own log for the reason.'

function busy(state: PressState): boolean {
  const s = state.job?.status
  return s === 'submitting' || s === 'queued' || s === 'running'
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

      // run() also rejects when ComfyUI forgets the prompt, as after a
      // restart mid job, so this await always returns and the desk is freed.
      try {
        const files = await run(plan.graph, (ev) => onProgress(ev, plan))
        const picture = files.find((f) => f.kind === 'image') ?? files[0] ?? null
        const durationMs = Date.now() - startedAt
        if (!picture) {
          emit({
            fault: {
              message: NO_FILE,
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
    // Stop was held while the job was still being sent, when there was no id
    // to cancel yet. This is the first moment there is one. Set to "Queued"
    // here, the job used to go on and render, and be filed, as if nobody had
    // asked; now it is cancelled, and run() settles it as a stop.
    if (stopped) {
      patchJob({ promptId: ev.promptId, stage: 'Stopping' })
      void cancelJob(ev.promptId).catch(() => undefined)
      return
    }
    patchJob({ promptId: ev.promptId, status: 'queued', stage: 'Queued' })
    return
  }
  if (ev.phase === 'preview') {
    patchJob({ previewUrl: ev.url })
    return
  }
  if (ev.phase !== 'running') return

  const cls = ev.node ? plan.graph[ev.node]?.class_type : null
  // A job asked to stop may still report a step or two before the cancel
  // lands. It says it is stopping until it has.
  const stage = stopped ? 'Stopping' : stageFor(cls, ev.value, ev.max)
  const sampling = ev.max > 1
  const pct = sampling
    ? clamp(ev.value / ev.max, press.job.pct, 0.97)
    : Math.max(press.job.pct, 0.02)
  patchJob({ status: 'running', stage, value: ev.value, max: ev.max, pct })
}

async function stopRun() {
  stopped = true
  queue = []
  const id = press.job?.promptId
  if (!id) {
    // Still being sent, so there is nothing to cancel yet: the "queued" event
    // cancels it the moment ComfyUI names it (see onProgress). The job stays
    // busy until then. Marked cancelled here, the Run button came back while
    // the send was still in flight, did nothing when pressed, and the job
    // went on to render.
    patchJob({ stage: 'Stopping' })
    return
  }
  patchJob({ stage: 'Stopping' })
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

/**
 * What the desk was set to when the reader last walked away from it.
 *
 * The look, the detail setting, the pinned model and everything set by hand
 * behind More are component state, and App mounts one room at a time, so a
 * glance at the archive used to put every one of them back to its default
 * while the prompt, which lives in the desk store, came back intact. "Use
 * these settings" was the worst of it: the pin and the hand set values are
 * taken once and the request they came from is cleared, so a visit to another
 * room dropped them and the next press made a different picture from the one
 * the reader had loaded, with nothing on screen to say so.
 *
 * Memory only, never localStorage. An override is a statement about this
 * session's picture (see useOverrides), so a reload still starts clean; a walk
 * to another room and back does not.
 */
type Held = {
  look: Intent
  anatomy: AnatomyLevel
  anatomySaid: boolean
  pinned: string | null
  overrides: Overrides
  seed0: number
  correction: string | null
  /**
   * The mode all of that was set in. The archive can change the mode while
   * the desk is away ("Use as source" puts the draft on image to image), and
   * a prompt typed by hand for one mode means something else in another.
   */
  mode: Mode
}

let held: Held | null = null

/**
 * The trained prefix to file on a record, when the prompt sent really opened
 * with it and the reader's own words did not already carry it.
 */
function prefixFiled(prefix: string | null, positive: string, words: string): string | null {
  if (!prefix) return null
  if (!positive.startsWith(prefix)) return null
  return words.trim().startsWith(prefix.trim()) ? null : prefix
}

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

// ---------------------------------------------------------------------------
// Shapes
// ---------------------------------------------------------------------------


/**
 * What the bench needs of a picture to open on it: the file, the size the
 * record says, and who made it. A record has all of it; a picture known only
 * by its file has the file.
 */
type RegionPicture = Pick<
  HistoryEntry,
  'file' | 'width' | 'height' | 'id' | 'prompt' | 'modelLabel' | 'model' | 'familyId'
>

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
  /**
   * The model that actually made this picture, when it is known.
   *
   * Null for an upload, which has no maker we know of. The bench used to state
   * the DESK's current model as the picture's provenance, which was true only
   * while the bench could be opened on a picture the desk had just produced.
   * Both new routes in - the archive and an uploaded file - break that.
   */
  madeBy: string | null
  /**
   * The family and weight file behind `madeBy`, when known. The bench draws
   * with the picture's own model when it can, and when it cannot, the reason
   * it gives has to be about that model rather than whatever the desk is on.
   */
  maker: { familyId: string; model: string } | null
}

// ---------------------------------------------------------------------------
// The edit family's plan
// ---------------------------------------------------------------------------

/**
 * decide() ranks image families against a look. Instruction editing is not a
 * look: exactly one installed model follows an instruction, and picking it is
 * not a decision anybody needs help with. So the edit desk is handed the same
 * `Plan` shape, filled from the registry, and every panel behind More reads it
 * exactly as it reads a decided one.
 */
function editRecipe(input: {
  style: Style
  prompt: string
  look: Look
  anatomy: AnatomyLevel
  source: string | null
  seed: number
  cat: Catalogue
  /** Free memory as read for this visit, which is fresher than the catalogue's. */
  hardware: Hardware | null
  lib: LoraLibrary
  addOns?: { accepted?: readonly string[]; declined?: readonly string[] }
}): Recipe {
  const { style, cat } = input
  const prompt = input.prompt.trim()
  const def = style.def
  const d = defaultsFor(def, style.model)
  const warnings: string[] = []

  // ADD-ONS FOR AN INSTRUCTION.
  //
  // suggest() scores a prompt's wording against each add-on's training
  // vocabulary, and "make the jacket red" has nothing for it to match. So the
  // edit desk offers every installed add-on that fits this model, the reader
  // decides, and an accepted one is chained into the graph exactly as the
  // compose desk chains its own. Declined ones stay declined across recomputes
  // for the same reason they do there: the decision arrives as input.
  const target = targetFor(def, style.model)
  const accepted = new Set(input.addOns?.accepted ?? [])
  const declined = new Set(input.addOns?.declined ?? [])
  const loras: RecipeLora[] = []
  const offers: RecipeLora[] = []
  for (const info of input.lib.all) {
    if (!info.installed || declined.has(info.file)) continue
    if (fitFor(info, target).level !== 'match') continue
    const row: RecipeLora = {
      file: info.file,
      label: info.label,
      strength: defaultStrength(info),
      measured: false,
      why: info.does,
    }
    if (accepted.has(info.file)) loras.push(row)
    else offers.push(row)
  }
  let planDef: FamilyDef | DerivedDef = def
  if (loras.length) {
    const chained = withLoras(def, loras.map((l) => ({ name: l.file, strength: l.strength })))
    if (chained) planDef = chained
    else warnings.push(`${style.label} cannot take add-ons, so none were applied.`)
  }
  const triggers = plainWords(
    triggersFor(
      loras.map((l) => ({ file: l.file, strength: l.strength, enabled: true })),
      input.lib,
      target,
    ),
  ).filter((t) => !prompt.toLowerCase().includes(t.toLowerCase()))
  const report = intentReport(
    { intent: input.look as Intent, explicit: input.anatomy !== 'off', mode: 'edit' },
    { installed: cat.installed, sizes: cat.sizes, hardware: input.hardware },
  )

  const positiveBase = style.positivePrefix ? `${style.positivePrefix}${prompt}` : prompt
  const params: Params = {
    model: style.model,
    positive: triggers.length ? `${positiveBase}, ${triggers.join(', ')}` : positiveBase,
    negative: d.negative ?? '',
    seed: input.seed,
    steps: d.steps,
    cfg: d.cfg,
    width: d.width,
    height: d.height,
    sampler: d.sampler,
    scheduler: d.scheduler,
  }
  if (style.clipSkip != null) params.clipSkip = style.clipSkip
  if (style.shift != null) params.shift = style.shift
  if (input.source) params.image = input.source

  const capabilities = capabilitiesOf(planDef)
  // The catalogue's verdict was taken when the page loaded. Judge the fit
  // against this visit's reading instead, and on the graph that will be
  // queued: the add-ons the reader took load on top of the model.
  const verdict = input.hardware
    ? feasibility(def, cat.sizes, input.hardware, instantiate(planDef, params))
    : style.verdict
  if (verdict && verdict.level !== 'ok') warnings.push(verdict.reason)
  // No word about the detail setting. This desk does not show it, so a
  // warning about a control the reader cannot see here would only confuse.

  const notes: RecipeNote[] = [
    {
      kind: 'model',
      text: `${style.label} is the only installed model that follows an instruction, so changing a picture always uses it.`,
      measured: false,
    },
    {
      kind: 'source',
      text: 'Your picture is the frame. Size, shape and composition come from it, not from a size control.',
      measured: false,
    },
    {
      kind: 'passes',
      text: 'Faces, hands, a masked region and a larger render are offered on the finished picture, not before it.',
      measured: false,
    },
  ]
  if (style.note) notes.push({ kind: 'model', text: style.note, measured: false })
  if (triggers.length) {
    notes.push({
      kind: 'prompt',
      text: `These words were added to your instruction so the add-ons work: ${triggers.join(', ')}.`,
      measured: false,
    })
  }

  return {
    ok: true,
    look: input.look,
    anatomy: input.anatomy,
    prompt,
    familyId: def.id,
    model: style.model,
    label: style.label,
    familyLabel: style.group,
    base: def,
    def: planDef,
    params,
    loras,
    sharpness: null,
    missingLoras: [],
    refineLoras: [],
    offers,
    passes: passesFor(
      capabilities,
      {
        face: 'Re renders every detected face at 768 and pastes it back.',
        hand: 'Re renders every detected hand with more freedom than a face. Not measured here.',
        refine: 'Draw a mask over a region and it is cropped, upscaled and rendered alone.',
        hires: 'Renders the same picture larger, at low denoise.',
      },
      cat.passBlocks,
    ),
    capabilities,
    verdict,
    notes,
    warnings,
    report,
  }
}

/** Waiting for the model list, in the shape the desk already knows how to print. */
function waitingRecipe(prompt: string, look: Look, anatomy: AnatomyLevel, why: string): Recipe {
  return {
    ok: false,
    look,
    anatomy,
    prompt,
    reason: why,
    report: intentReport({ intent: look as Intent, explicit: anatomy !== 'off', mode: 'image' }, {}),
    notes: [],
    warnings: [],
  }
}

/** The three modes. The first question on the page, above the desk. */
const MODE_CHOICES: readonly { id: Mode; label: string; blurb: string }[] = [
  { id: 't2i', label: 'From words', blurb: 'A picture from the prompt alone.' },
  {
    id: 'i2i',
    label: 'From a picture',
    blurb: 'Redraw a picture you hand it. Dropping or pasting one on the page does this without coming here.',
  },
  {
    id: 'edit',
    label: 'Change a picture',
    blurb: 'Say what to change and the edit model follows the instruction.',
  },
]

/** Overrides a caption figure can be adopted into. */
const ADOPTABLE = new Set<string>([
  'width',
  'height',
  'steps',
  'cfg',
  'sampler',
  'scheduler',
  'seed',
  'denoise',
  'megapixels',
  'shift',
  'clipSkip',
])

// ---------------------------------------------------------------------------
// The desk
// ---------------------------------------------------------------------------

export function Pictures() {
  const c = useComposition()
  const expert = useExpert()
  const state = usePress()
  const records = useRecords()
  const reduced = useReducedMotion()

  const [cat, setCat] = useState<Catalogue | null>(null)
  const [catError, setCatError] = useState<string | null>(null)
  const [correction, setCorrection] = useState<string | null>(() => held?.correction ?? null)
  const [sourceError, setSourceError] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [picking, setPicking] = useState(false)
  const [adopted, setAdopted] = useState<string | null>(null)
  const [offline, setOffline] = useState(() => connectionState() === 'closed')
  const [ahead, setAhead] = useState(0)
  const [retryIn, setRetryIn] = useState(5)

  // --- the three answers --------------------------------------------------
  /**
   * Held as `Intent`, not as `Look`.
   *
   * The look picker offers three, which is the right number for a screen with
   * three answers on it. intent.ts ranks against four: cartoon is a real
   * routing dimension, western toon styling against Japanese cel shading, and
   * dropping it would delete a capability rather than move one. So the fourth
   * lives on the full picker behind More, and the state is wide enough to hold
   * it.
   */
  const [look, setLook] = useState<Intent>(() => held?.look ?? 'photoreal')
  const [anatomy, setAnatomy] = useState<AnatomyLevel>(
    () => held?.anatomy ?? suggestAnatomy(store.get().prompt),
  )
  /** True once the reader has said what they want. Until then the prompt hints. */
  const [anatomySaid, setAnatomySaid] = useState(() => held?.anatomySaid ?? false)

  // --- everything else, mounted only while More is open -------------------
  const ov = useOverrides(held?.overrides)
  const [pinned, setPinned] = useState<string | null>(() => held?.pinned ?? null)
  const [lib, setLib] = useState<LoraLibrary>(EMPTY_LIBRARY)
  /**
   * Held rather than rolled inside decide(), so the seed on the sampling panel
   * does not change under the reader on every keystroke. A new one is drawn at
   * the press, unless they locked it.
   */
  const [seed0, setSeed0] = useState(() => held?.seed0 ?? randomSeed())

  // Kept for the next visit. See `held`.
  useEffect(() => {
    held = { look, anatomy, anatomySaid, pinned, overrides: ov.value, seed0, correction, mode: c.mode }
  }, [look, anatomy, anatomySaid, pinned, ov.value, seed0, correction, c.mode])

  /**
   * A prompt typed by hand behind More replaces the whole prompt sent. Typed
   * for a picture made from words it describes a picture; on "Change a
   * picture" the prompt is an instruction. Carried across that line, it was
   * sent in place of the instruction the reader then typed, with nothing on
   * the main screen to say so. So crossing into or out of changing a picture
   * hands the prompt back to the recipe, out loud. This covers a mode changed
   * while the desk was away as well: the held mode is where it starts from.
   *
   * Done while rendering, the way React adjusts state to a changed input, so
   * no render with the prompt in the wrong mode is ever committed.
   */
  const [seenMode, setSeenMode] = useState<Mode>(() => held?.mode ?? c.mode)
  if (seenMode !== c.mode) {
    setSeenMode(c.mode)
    if ((seenMode === 'edit') !== (c.mode === 'edit') && ov.value.positive !== undefined) {
      ov.clear('positive')
      setCorrection(
        c.mode === 'edit'
          ? 'The prompt you typed by hand behind More was for making a picture, so it is set aside. What you type as the change is what is sent.'
          : 'The prompt you typed by hand behind More was an instruction for changing a picture, so it is set aside. The desk builds the prompt from your words again.',
      )
    }
  }

  const [refining, setRefining] = useState(false)
  const [refineSource, setRefineSource] = useState<RefineSource | null>(null)
  const [refineResult, setRefineResult] = useState<HistoryEntry | null>(null)
  const [refineFault, setRefineFault] = useState<string | null>(null)
  const [openingRefine, setOpeningRefine] = useState(false)
  const refineToken = useRef(0)
  /** True between queueing a refine and its result landing on the plate. */
  const awaitingRefine = useRef(false)
  /**
   * The press job the region pass was queued as.
   *
   * The result is the picture THAT job produced, and nothing else. It used to
   * be 'the first picture that is not the one on screen when the pass was
   * queued', which held only while the pass succeeded: a rejected or stopped
   * pass left the flag up, the composer stays usable beside the bench, and the
   * next ordinary picture was then shown as the refined result, compared
   * against the source and offered as the next region to work on.
   */
  const refineJob = useRef<string | null>(null)
  /** The region add-ons the reader ticked on the bench, by filename. None until they do. */
  const [refinePicked, setRefinePicked] = useState<string[]>([])

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

  // The panel that re-reads after a fetch is mounted only while More is open,
  // so a family that lands with More closed is heard of here instead.
  useEffect(() => onPlanLanded(() => load(true)), [load])

  // The desk recovers on its own when ComfyUI comes back. No reload.
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

  // --- free memory, read once a visit --------------------------------------
  /**
   * The catalogue is read once per page load, and the page is an installed app
   * that stays open for hours, so its memory reading went stale with it. Free
   * memory is read again when the reader comes to the desk, and that one
   * reading ranks the models and words the warnings for the whole visit.
   *
   * It is not read again after a job, or when the tab comes back into view.
   * os.freemem() counts the weights ComfyUI keeps loaded after a job as used,
   * so a reading taken after one saw several gigabytes less free than before,
   * the model that had just run was marked down for its own cache, and the
   * next press of the same button, with nothing touched, could render on a
   * different model, or warn the reader to close other applications about
   * memory that model was holding. One reading per visit cannot move under
   * the reader between two presses. A visit that opens while ComfyUI is still
   * holding weights reads low for the same reason; that needs the server to
   * report what ComfyUI itself holds, which it does not yet.
   */
  const [visitHardware, setVisitHardware] = useState<Hardware | null>(null)
  useEffect(() => {
    let alive = true
    probeHardware().then(
      (hw) => {
        if (alive) setVisitHardware(hw)
      },
      () => {},
    )
    return () => {
      alive = false
    }
  }, [])
  const hardware = visitHardware ?? cat?.hardware ?? null

  // The LoRA catalogue, read once. Until it lands the recipe resolves nothing
  // and says so in print rather than pretending the stack was applied.
  useEffect(() => {
    let alive = true
    loadLoraLibrary().then(
      (next) => {
        if (alive) setLib(next)
      },
      () => {},
    )
    return () => {
      alive = false
    }
  }, [])

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

  /**
   * Why "Change a picture" is not on offer, when the model that would do it is
   * on disk and something else is missing: a file it loads, the node pack that
   * reads it, or the memory to hold it. Without this the choice simply was not
   * there, and nothing said what to install.
   */
  const editGap = useMemo(() => {
    if (!cat || editStyle) return null
    const gap = cat.unavailable.find((u) => familyOwning(u.name)?.mode === 'edit')
    if (!gap) return null
    const name = PLAIN_NAMES[gap.name] ?? titleFromFilename(gap.name)
    const why = `${gap.why.charAt(0).toLowerCase()}${gap.why.slice(1)}`
    return `Change a picture is not offered: ${name} ${why}${why.endsWith('.') ? '' : '.'}`
  }, [cat, editStyle])

  // An intent nothing installed can serve is corrected out loud.
  useEffect(() => {
    if (!cat || !cat.styles.length) return
    if (c.mode === 'edit' && !editStyle) store.patch({ mode: 't2i' })
  }, [cat, c.mode, editStyle])

  /**
   * A record reused from the Archive, taken up as a pin and a set of overrides.
   *
   * "Use these settings" writes a whole composition into this desk's draft:
   * the model, the steps, the CFG, the sampler, the seed, the size. The recipe
   * decides all of that now, so without this the values would be written and
   * then silently ignored, which is the worst of the three possible behaviours.
   *
   * `touched` is exactly the set reuseIntoDesk marks, so it is read once,
   * turned into a pinned model and the matching overrides, cleared, and said
   * out loud. Nothing here happens quietly: the notice names the model that was
   * pinned and the panel prints how many values were set by hand.
   */
  const reuseTaken = useRef(false)
  useEffect(() => {
    if (!c.touched.length) {
      reuseTaken.current = false
      return
    }
    if (reuseTaken.current) return
    reuseTaken.current = true

    // Read off the store, not the render-scoped `c`: this runs once when
    // `touched` arrives, and the dependency list says exactly that.
    const snap = store.get()
    const next: Overrides = {}
    for (const f of snap.touched) {
      if (f === 'steps') next.steps = snap.steps
      else if (f === 'cfg') next.cfg = snap.cfg
      else if (f === 'sampler') next.sampler = snap.sampler
      else if (f === 'scheduler') next.scheduler = snap.scheduler
      else if (f === 'negative' && snap.negative != null) next.negative = snap.negative
      else if (f === 'width') next.width = snap.width
      else if (f === 'height') next.height = snap.height
      else if (f === 'seed') {
        next.seed = snap.seed
        next.seedLocked = snap.seedLocked
      } else if (f === 'denoise' && snap.denoise != null) next.denoise = snap.denoise
      else if (f === 'megapixels' && snap.megapixels != null) next.megapixels = snap.megapixels
      else if (f === 'shift' && snap.shift != null) next.shift = snap.shift
      else if (f === 'clipSkip' && snap.clipSkip != null) next.clipSkip = snap.clipSkip
    }

    ov.set(next)
    if (snap.model) setPinned(snap.model)
    store.patch({ touched: [] })
    setCorrection(
      `Settings loaded from a finished picture. ${
        snap.model ? `${PLAIN_NAMES[snap.model] ?? titleFromFilename(snap.model)} is pinned and ` : ''
      }${Object.keys(next).length} values are set by hand. Open More to see them, or put them back there.`,
    )
    // Read once, on the composition that arrived. Re-running this on every
    // keystroke would fight the reader.
  }, [c.touched, ov])

  // The prompt proposes an anatomy level until the reader states one. It never
  // proposes `emphasised`: that stack measured below base, so it is a choice
  // somebody makes, not one made for them.
  useEffect(() => {
    if (anatomySaid) return
    const want = suggestAnatomy(c.prompt)
    setAnatomy((was) => (was === want ? was : want))
  }, [c.prompt, anatomySaid])

  // --- the recipe ---------------------------------------------------------
  const sourceName = c.source?.name ?? ''
  const usingSource = needsSource(c.mode) && !!sourceName

  /**
   * True while a picture is in the well and the pinned model cannot work from
   * one.
   *
   * The pin survives a visit to another room, and "Use as source" there puts
   * the desk on image to image. A pin that can only draw from words then left
   * decide() nothing to choose from, and the button said nothing installed
   * could work from a picture, while six installed families could. The pin is
   * set aside, not dropped: back on "From words" it is used again.
   */
  const pinSetAside = useMemo(() => {
    if (!pinned || !usingSource || c.mode !== 'i2i') return false
    const def = familyOwning(pinned)
    return !def || !(IMG2IMG[def.id] ?? deriveImg2Img(def))
  }, [pinned, usingSource, c.mode])
  /** Said for as long as the pin is set aside, and gone the moment it is not. */
  const pinNote =
    pinSetAside && pinned
      ? `${PLAIN_NAMES[pinned] ?? titleFromFilename(pinned)} is pinned, and it cannot work from a picture, so the desk chooses another model while a picture is in the well. Choose From words and the pin is used again.`
      : null

  const recipe: Recipe = useMemo(() => {
    if (!cat) {
      return waitingRecipe(c.prompt, look as Look, anatomy, 'Reading the model list from ComfyUI.')
    }
    if (c.mode === 'edit') {
      if (!editStyle) {
        return waitingRecipe(
          c.prompt,
          look as Look,
          anatomy,
          'No instruction editing model is installed, so a picture cannot be changed by describing the change.',
        )
      }
      return editRecipe({
        style: editStyle,
        prompt: c.prompt,
        look: look as Look,
        anatomy,
        source: sourceName || null,
        seed: seed0,
        cat,
        hardware,
        lib,
        addOns: { accepted: c.addOnsAccepted, declined: c.addOnsDeclined },
      })
    }
    return decide({
      prompt: c.prompt,
      look: look as Look,
      anatomy,
      sourceImage: usingSource ? sourceName : undefined,
      hardware,
      sizes: cat.sizes,
      installed: pinned && !pinSetAside ? [pinned] : cat.installed,
      loras: lib,
      seed: seed0,
      // The reader's own decisions, threaded in so they survive this recompute.
      // decide() runs on every keystroke; a decision held anywhere but here
      // would be silently overwritten by the next suggestion.
      addOns: { accepted: c.addOnsAccepted, declined: c.addOnsDeclined },
      passBlocks: cat.passBlocks,
    })
  }, [cat, c.mode, c.prompt, look, anatomy, usingSource, sourceName, pinned, pinSetAside, lib, seed0, editStyle,
      c.addOnsAccepted, c.addOnsDeclined, hardware])

  const plan = recipe.ok ? recipe : null

  /** The recipe with whatever the reader took back by hand folded onto it. */
  const settled = useMemo(() => (plan ? settle(plan, ov.value, lib) : null), [plan, ov.value, lib])

  /**
   * The recipe as it is printed under the button, with its memory warning
   * priced on the graph the button will queue.
   *
   * decide() prices the graph it built. An add-on stack edited behind More
   * builds another one, with other files to load, and the warning described
   * the recipe's graph rather than the one that runs. Only the memory warning
   * changes; everything else the recipe says still holds.
   */
  const shown: Recipe = useMemo(() => {
    if (!plan || !settled?.rebuilt || !hardware || !cat) return recipe
    const verdict = feasibility(plan.base, cat.sizes, hardware, buildGraph(settled))
    const before = plan.verdict && plan.verdict.level !== 'ok' ? plan.verdict.reason : null
    const after = verdict.level !== 'ok' ? verdict.reason : null
    if (before === after) return recipe
    const warnings = plan.warnings.filter((w) => w !== before)
    if (after) warnings.push(after)
    return { ...plan, verdict, warnings }
  }, [recipe, plan, settled, hardware, cat])

  /** The catalogue row for the chosen file, for the plate's dagger and the bench. */
  const style = useMemo(
    () => (plan ? (styles.find((s) => s.def.id === plan.familyId && s.model === plan.model) ?? null) : null),
    [styles, plan],
  )

  const houseNegative = style ? (defaultsFor(style.def, style.model).negative ?? '') : ''
  const timing = useMemo(() => timingNote(records, plan?.model ?? ''), [records, plan])

  const canI2I = useMemo(
    () => pictureStyles.some((s) => IMG2IMG[s.def.id] ?? deriveImg2Img(s.def)),
    [pictureStyles],
  )

  // --- mode ---------------------------------------------------------------
  const setMode = useCallback(
    (mode: Mode) => {
      if (mode === store.get().mode) return
      if (mode === 'edit' && !editStyle) return
      store.patch({ mode })
    },
    [editStyle],
  )

  // --- the reader's add-on decisions --------------------------------------
  /**
   * Take an added add-on off again.
   *
   * An accepted add-on is carried on the reader's say-so even after the
   * wording drifts away from it, and it leaves the offers as it joins the
   * chain, so while it was on the main screen had no way to take it off.
   * Removing it in More only set a stack override, which "Go back to the
   * picks" or a walk to another room undid while the decision stayed filed.
   * This withdraws the decision itself. It is not a "no": the add-on can be
   * offered again when the wording asks for it.
   */
  const dropAddOn = useCallback((file: string) => {
    const now = store.get()
    store.patch({ addOnsAccepted: now.addOnsAccepted.filter((f) => f !== file) })
  }, [])

  /**
   * The prompt, written from the field.
   *
   * An emptied prompt is the start of a new composition, so the add-on
   * decisions made for the old one go with it. They used to outlive every
   * prompt and every reload, and one accepted add-on then rode along on every
   * picture that followed.
   */
  const writePrompt = useCallback((prompt: string) => {
    store.patch(prompt.trim() ? { prompt } : { prompt, addOnsAccepted: [], addOnsDeclined: [] })
  }, [])

  // --- the source picture -------------------------------------------------
  const clearSource = useCallback(() => {
    const now = store.get()
    if (now.source?.previewUrl?.startsWith('blob:')) URL.revokeObjectURL(now.source.previewUrl)
    store.patch({ source: null, mode: now.mode === 'i2i' ? 't2i' : now.mode })
    setSourceError(null)
  }, [])

  const takeFile = useCallback(async (file: File) => {
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
      store.patch({ source, mode: now.mode === 't2i' ? 'i2i' : now.mode })
    } catch (err) {
      URL.revokeObjectURL(previewUrl)
      const status = err instanceof Error ? err.message : String(err)
      setSourceError(`ComfyUI refused the file (${status}). Try a PNG or JPEG under 50 MB.`)
    } finally {
      setUploading(false)
    }
  }, [])

  const takeRecord = useCallback((entry: HistoryEntry) => {
    const now = store.get()
    const previous = now.source
    if (previous?.previewUrl?.startsWith('blob:')) URL.revokeObjectURL(previous.previewUrl)
    // A picture taken on "From words" switches the desk to working from it,
    // the way a dropped or pasted file does. Left on t2i, the well showed the
    // picture and the button said "Make the change" while the run drew from
    // the words alone and the record filed no source.
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
      mode: now.mode === 't2i' ? 'i2i' : now.mode,
    })
  }, [])

  // A picture adopted from the archive lives in the OUTPUT folder; LoadImage
  // reads the input folder, so it is copied across once, here.
  useEffect(() => {
    const s = c.source
    if (!s || s.name || !s.ref) return
    // LoadImage reads a still, and a draft saved by an older build can hold a
    // clip here. Copied across, the whole clip was uploaded for a run that
    // could only fail on it. The mode is left alone so the well stays on
    // screen to say why it is empty.
    if (VIDEO_EXT.test(s.ref.filename)) {
      store.patch({ source: null })
      setSourceError('It is a clip, and this desk starts only from a still. Choose a picture instead.')
      return
    }
    // A picture chosen after a refusal is a fresh start; the old notice goes.
    setSourceError(null)
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

  /**
   * Queue the recipe, as settled.
   *
   * Everything is read off `settled`, never off the plan directly: an override
   * the reader set behind More has to reach the server or the panel is a lie.
   */
  const start = useCallback(() => {
    if (!plan || !settled || busy(press)) return
    const base = store.get()
    // The button's own refusals, checked again here, because Ctrl+Enter reaches
    // this without the button: the prompt field lets the key through when the
    // desk is blocked, and the page-wide handler then called this regardless,
    // rendering from the words alone with no picture attached, or queueing the
    // edit graph against its placeholder file.
    if (!base.prompt.trim()) return
    if (needsSource(base.mode) && (!base.source?.name || uploading)) return
    const first = settled.seedLocked ? settled.params.seed : randomSeed()
    const plans: RunPlan[] = []

    for (let i = 0; i < settled.runs; i += 1) {
      const seed = first + i
      const params: Params = { ...settled.params, seed }
      const graph = buildGraph({ ...settled, params })
      const composition: Composition = {
        ...base,
        desk: DESK,
        mode: c.mode,
        familyId: plan.familyId,
        model: plan.model,
        prompt: plan.prompt,
        // The prefix is filed when the prompt sent really opened with it, so
        // the record says what ran. Nothing that rebuilds a picture from its
        // record prepends it blindly: rerun() rebuilds the whole prompt with
        // positiveFor(), which knows the words may already carry it.
        positivePrefix: prefixFiled(style?.positivePrefix ?? null, params.positive, plan.prompt),
        // The prompt as the graph carries it, add-on words and any hand edit
        // included, so the record can send the same words again.
        positive: params.positive,
        negative: params.negative,
        source: needsSource(c.mode) ? base.source : null,
        width: params.width,
        height: params.height,
        megapixels: params.megapixels ?? null,
        denoise: params.denoise ?? null,
        seed,
        seedLocked: settled.seedLocked,
        steps: params.steps,
        cfg: params.cfg,
        sampler: params.sampler,
        scheduler: params.scheduler,
        shift: params.shift ?? null,
        clipSkip: params.clipSkip ?? null,
        split: params.split ?? null,
        length: null,
        fps: null,
        // The LoRA chain is filed in `loras`, not as a variant flag: `noLora`
        // means the reader stripped a family's own Lightning LoRAs, which is a
        // different thing and not something the recipe does.
        noLora: false,
        runs: settled.runs as Composition['runs'],
      }
      plans.push({
        graph,
        composition,
        seed,
        familyLabel: plan.familyLabel,
        modelLabel: plan.label,
        variant: c.mode === 'i2i' ? 'img2img' : null,
        label: plan.label,
        passes: settled.passes,
        loras: settled.specs,
      })
    }

    setSeed0(first)
    startRuns(plans)
  }, [plan, settled, c.mode, uploading, style])

  // Ctrl/⌘+Enter runs, and is the one shortcut that works inside the prompt.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // A key somebody nearer the event already claimed is not ours.
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
        setPicking(true)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [start])

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

  /**
   * A figure taken off a finished picture becomes an override, not a draft
   * field. The recipe decides these now, so writing one into the composition
   * would change nothing and the link would quietly do nothing at all.
   */
  const adopt = useCallback(
    (field: Parameters<typeof adoptValue>[1], value: number | string, label: string) => {
      if (!ADOPTABLE.has(field)) return
      ov.one(field as never, value as never)
      if (field === 'seed') ov.one('seedLocked', true)
      flash(label)
    },
    [flash, ov],
  )

  // --- the passes the finished picture offers -----------------------------

  /**
   * The graph a finished picture was actually made with.
   *
   * The result surface asks this graph what it can carry, so handing it the
   * bare family would offer a picture made through image to image the wrong
   * passes. Image to image first, then the passes the picture already had,
   * then the add-on chain, in the order settle() builds them, so what is
   * offered is what can really be derived and a pass queued from it redraws
   * the same picture. Without the passes, "Fix the hands" on a picture made
   * with "Render it bigger" came back smaller and single pass.
   */
  const defOf = useCallback((entry: HistoryEntry): FamilyDef | DerivedDef | null => {
    const base = BY_ID[entry.familyId]
    if (!base) return null
    let def: FamilyDef | DerivedDef = base
    if (entry.mode === 'i2i') def = IMG2IMG[base.id] ?? deriveImg2Img(base) ?? base
    if (entry.passes?.hires) def = deriveHiresFix(def) ?? def
    if (entry.passes?.face) def = deriveAutoDetail(def, 'face') ?? def
    if (entry.passes?.hand) def = deriveAutoDetail(def, 'hand') ?? def
    if (entry.loras?.length) {
      def = withLoras(def, entry.loras.map((l) => ({ name: l.name, strength: l.strength }))) ?? def
    }
    return def
  }, [])

  const resultDef = useMemo(() => (current ? defOf(current) : null), [current, defOf])

  /** Detector facts per finished picture, from a reading the reader asked for. */
  const [facts, setFacts] = useState<ReadonlyMap<string, ImageFacts | null>>(() => new Map())
  const noteReading = useCallback((id: string, report: VisionReport) => {
    setFacts((prev) => new Map(prev).set(id, report.facts))
    // The tags outlive the session: they go on the record, and the archive
    // can search them.
    if (report.tags) {
      updateRecord(id, {
        tags: report.tags.general.slice(0, 40).map((t) => t.tag),
        rating: report.rating ?? undefined,
      })
    }
  }, [])

  /** The attached picture, read on request. One element, handed to whichever desk is up. */
  const sourceReading = c.source?.name ? (
    <Reading
      compact
      source={{ kind: 'input', rel: c.source.name }}
      cacheKey={`input:${c.source.name}`}
      arch={style ? archFor(style.def, style.model) : null}
      anatomy={anatomy}
      onAnatomy={(level) => {
        setAnatomySaid(true)
        setAnatomy(level)
      }}
      onAddOn={(file) =>
        store.patch({
          addOnsAccepted: [...new Set([...c.addOnsAccepted, file])],
          addOnsDeclined: c.addOnsDeclined.filter((f) => f !== file),
        })
      }
      onUseWords={(words) =>
        store.patch({ prompt: c.prompt.trim() ? `${c.prompt.trim()}, ${words}` : words })
      }
    />
  ) : null

  /**
   * Run one pass on the picture in front of the reader, or make another like it.
   *
   * Queued against the record's OWN composition, not against whatever is in the
   * compose field now: the reader has usually typed something since. Same
   * model, same prompt, same LoRAs, and for a pass the same seed, so "fix the
   * hands" fixes these hands rather than drawing a different picture with
   * better ones. "Make another" is the one that draws a fresh seed.
   */
  const rerun = useCallback(
    (entry: HistoryEntry, kind: 'face' | 'hand' | 'hires' | null) => {
      if (busy(press)) return
      if (!rebuildable(entry)) return
      const base = BY_ID[entry.familyId]
      if (!base) return
      const fresh = kind === null
      const reuse = compositionFromEntry(entry, {
        installedModels: cat?.installed,
        freshSeed: fresh,
      })
      const composition: Composition = {
        ...reuse.composition,
        seed: fresh ? reuse.composition.seed : entry.seed,
        seedLocked: !fresh,
      }

      let def = defOf(entry)
      if (!def) return
      if (kind) {
        const derived = kind === 'hires' ? deriveHiresFix(def) : deriveAutoDetail(def, kind)
        if (!derived) return
        // The chain is re-applied on top of the derivation, because deriving
        // from an already-chained graph is what refine.ts expects and the pass
        // must sample on the same patched weights the picture did.
        def = derived
      }

      const params = toParams(composition, {
        negative: defaultsFor(base, composition.model).negative ?? '',
      })
      // The prompt as the desk sent it, not the bare words the record files.
      // Queued from the words alone, the pass drew a different picture at the
      // same seed: no trained prefix, and add-ons loaded without the words
      // they answer to. A record that kept the prompt as sent is taken at its
      // word, which also carries a hand edit; an older one is rebuilt.
      params.positive =
        entry.positive ??
        positiveFor({
          def: base,
          model: composition.model,
          prompt: composition.prompt,
          loras: entry.loras ?? [],
          lib,
        })
      const graph = instantiate(def, params)
      if ('derived' in def) {
        writeExtras(graph, def, { hiresSteps: hiresStepsFor(params.steps) })
      }

      // The passes it already had are in the graph again (see defOf), so the
      // record says so as well as naming the one just asked for.
      const passes: Passes = {
        face: !!entry.passes?.face || kind === 'face',
        hand: !!entry.passes?.hand || kind === 'hand',
        hires: !!entry.passes?.hires || kind === 'hires',
      }
      startRuns([
        {
          graph,
          composition: { ...composition, positive: params.positive },
          seed: composition.seed,
          familyLabel: entry.familyLabel,
          modelLabel: entry.modelLabel,
          variant: entry.variant ?? null,
          label: kind
            ? `${entry.modelLabel}, ${kind === 'hires' ? 'larger render' : `${kind} pass`}`
            : entry.modelLabel,
          passes,
          loras: entry.loras ?? [],
        },
      ])
    },
    [cat, defOf, lib],
  )

  // --- region refine ------------------------------------------------------

  /** Every model that can actually draw a region, not just the first one found. */
  const refineOptions = useMemo(
    () => pictureStyles.filter((s) => deriveRefine(s.def) !== null),
    [pictureStyles],
  )

  /** The reader's own pick, as `familyId::model`. Null means "the suggested one". */
  const [refinePick, setRefinePick] = useState<string | null>(null)

  const deskRefines = !!style && deriveRefine(style.def) !== null

  /** The model that made the picture on the bench, when it is installed and can draw a region. */
  const refineMaker = useMemo(() => {
    const m = refineSource?.maker
    if (!m) return null
    return refineOptions.find((s) => s.def.id === m.familyId && s.model === m.model) ?? null
  }, [refineSource, refineOptions])

  /**
   * Which style draws the region when the reader has not picked one.
   *
   * The model that made the picture comes first: it drew everything around the
   * region, so its patch matches. When it cannot redraw a region, or is not
   * installed, the stand-in is the capable model best at the kind of picture
   * that model makes, read off the same style table the ranking uses. This
   * used to be the desk's current model, else whichever capable model sorted
   * first by name, which put an anime model on photoreal pictures while the
   * note blamed the picture's maker. With no maker known (an upload), the
   * desk's model draws when it can, else the one best at the desk's look.
   */
  /** The kind of picture a stand-in should be good at: what the maker does best, else the desk's look. */
  const refineAim: Intent = useMemo(() => {
    const m = refineSource?.maker
    const makerDef = m ? BY_ID[m.familyId] : undefined
    return m && makerDef ? strongestLook(m.model, makerDef) : look
  }, [refineSource, look])

  const refineDefault = useMemo(() => {
    if (refineMaker) return refineMaker
    const m = refineSource?.maker
    const makerDef = m ? BY_ID[m.familyId] : undefined
    if (!makerDef && deskRefines) return style
    if (!refineOptions.length) return null
    const makerArch = m && makerDef ? archFor(makerDef, m.model) : null
    let best = refineOptions[0]
    let bestScore = -1
    for (const o of refineOptions) {
      // The look decides; the same kind of weights only breaks a tie.
      const score =
        lookScore(o.model, o.def, refineAim) * 2 + (makerArch && archFor(o.def, o.model) === makerArch ? 1 : 0)
      if (score > bestScore) {
        best = o
        bestScore = score
      }
    }
    return best
  }, [refineMaker, refineSource, refineOptions, deskRefines, style, refineAim])

  const refineStyle = useMemo(() => {
    // A pick the reader made outranks the suggestion, and it is checked
    // against the current list so a stale pick cannot strand the bench on a
    // model that is no longer installed.
    if (refinePick) {
      const chosen = refineOptions.find((s) => `${s.def.id}::${s.model}` === refinePick)
      if (chosen) return chosen
    }
    return refineDefault
  }, [refineOptions, refinePick, refineDefault])

  /**
   * Why no region can be drawn on this ComfyUI, naming the file or node pack
   * the region graph loads and ComfyUI does not have. The upscaler is not in
   * every install, and the region links used to open the bench anyway, which
   * queued a job ComfyUI refused over a file the page never mentioned.
   */
  const regionBlock = cat?.passBlocks.refine ?? null

  const canRefine = refineOptions.length > 0 && !regionBlock

  /**
   * True when the region is drawn by the model the desk is set to. Only then
   * do the desk's add-ons go with it: they were resolved against that model,
   * and an add-on for one kind of model loads on another without error and
   * changes nothing.
   */
  const drawsWithDesk = sameStyle(refineStyle, style)

  /** The desk's add-ons that ride along on the region pass. */
  const deskSpecs: LoraSpec[] = useMemo(
    () => (drawsWithDesk ? (settled?.specs ?? []) : []),
    [drawsWithDesk, settled],
  )

  /**
   * Region add-ons that fit the model drawing the region, offered on the
   * bench, unticked. They were all stacked onto every region pass unasked:
   * every installed one, sliders at full strength included, twenty six files
   * on a face touch up, none of them named on screen or in the record.
   */
  const regionOffers = useMemo(
    () =>
      refineStyle
        ? regionAddOns(lib, refineStyle.def, refineStyle.model, deskSpecs.map((s) => s.name))
        : [],
    [lib, refineStyle, deskSpecs],
  )
  const regionPicked = useMemo(
    () => regionOffers.filter((o) => refinePicked.includes(o.file)),
    [regionOffers, refinePicked],
  )

  /**
   * The refine graph, add-ons and all, with the chain it actually carries.
   * Null means: do not offer the pass.
   */
  const refineChain: { def: DerivedDef; specs: LoraSpec[] } | null = useMemo(() => {
    if (!refineStyle) return null
    const derived = deriveRefine(refineStyle.def)
    if (!derived) return null
    const specs: LoraSpec[] = [
      ...deskSpecs,
      ...regionPicked.map((l) => ({ name: l.file, strength: l.strength })),
    ]
    if (!specs.length) return { def: derived, specs: [] }
    const chained = withLoras(derived, specs)
    // A graph that cannot take add-ons runs without them, and the record and
    // the bench say none were sent.
    return chained ? { def: chained, specs } : { def: derived, specs: [] }
  }, [refineStyle, deskSpecs, regionPicked])
  const refineDef = refineChain?.def ?? null

  /** The reader-facing name of an add-on file. */
  const addOnLabel = useCallback((file: string) => lib.byFile.get(file)?.label ?? file, [lib])

  /**
   * Why this model is drawing the region, said about the right model.
   *
   * "<maker> cannot redraw a region" is printed only when the maker is known
   * and its family really cannot, which the bench used to say about any
   * picture whenever the DESK's model could not, including pictures made by a
   * model that could.
   */
  const refineWhy: string | null = useMemo(() => {
    if (!refineStyle) return null
    const madeBy = refineSource?.madeBy ?? null
    const m = refineSource?.maker
    const makerDef = m ? BY_ID[m.familyId] : undefined
    const picked = !!refinePick && !sameStyle(refineStyle, refineDefault)
    if (picked) return `You chose ${refineStyle.label} to draw the area.`
    if (sameStyle(refineStyle, refineMaker)) {
      return `The area is redrawn by ${refineStyle.label}, the model that made this picture.`
    }
    if (m && makerDef) {
      const who = madeBy ?? 'The model that made this picture'
      const stand = `the area is drawn by ${refineStyle.label}, rated best for ${refineAim} work of the models here that can`
      return deriveRefine(makerDef)
        ? `${who} is not available here, so ${stand}.`
        : `${who} cannot redraw a region, so ${stand}.`
    }
    if (drawsWithDesk) return `The area is drawn by ${refineStyle.label}, the model the desk is set to.`
    if (style && !deskRefines) {
      return `${style.label}, the model the desk is set to, cannot redraw a region, so the area is drawn by ${refineStyle.label}, rated best for ${refineAim} work of the models here that can.`
    }
    return `The area is drawn by ${refineStyle.label}.`
  }, [refineStyle, refineSource, refinePick, refineDefault, refineMaker, drawsWithDesk, style, deskRefines, refineAim])

  /** What goes with the region pass, said by name. */
  const refineCarries: string = useMemo(() => {
    const sent = refineChain?.specs ?? []
    const fromDesk = sent.filter((sp) => deskSpecs.some((d) => d.name === sp.name))
    const parts: string[] = []
    if (fromDesk.length) {
      parts.push(`Your add-ons from the desk go with it: ${fromDesk.map((sp) => addOnLabel(sp.name)).join(', ')}.`)
    } else if (!drawsWithDesk && settled?.specs.length && style) {
      parts.push(`Your add-ons on the desk are left out, because they were chosen for ${style.label}.`)
    }
    if (regionPicked.length && sent.length) {
      parts.push(`Also on for this area: ${regionPicked.map((l) => l.label).join(', ')}.`)
    }
    if (!sent.length && (deskSpecs.length || regionPicked.length)) {
      parts.push(`${refineStyle?.label ?? 'This model'} cannot take add-ons, so none are sent.`)
    }
    if (!parts.length) parts.push('No add-ons go with it.')
    return parts.join(' ')
  }, [refineChain, deskSpecs, drawsWithDesk, settled, style, regionPicked, refineStyle, addOnLabel])

  /**
   * Open the refine surface on a finished picture.
   *
   * The size is always measured off the file, never read out of the record. The
   * record files the size the composer ASKED for, and a two pass render
   * upscales the latent on the way to disk with nothing in the record to say
   * so. Trusting it put the crop, the mask and the composite in the pre upscale
   * space and pasted a correct patch into the wrong part of the frame.
   */
  const openRefine = useCallback(async (entry: RegionPicture, words?: string) => {
    const token = (refineToken.current += 1)
    awaitingRefine.current = false
    setRefinePicked([])
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
        prompt: words ?? entry.prompt,
        madeBy: entry.modelLabel || null,
        maker: entry.model ? { familyId: entry.familyId, model: entry.model } : null,
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

  /**
   * Open the bench on the picture currently attached to the desk.
   *
   * This is the upload route. A record from the archive goes through
   * openRefine, which fetches it out of the OUTPUT folder and copies it into
   * the input folder; an attached source is already in the input folder by the
   * time the well shows it (see the copy effect above), so `source.name` is all
   * LoadImage needs and there is nothing to upload again.
   */
  const openRefineFromSource = useCallback(async (src: SourceRef, words?: string) => {
    if (!src.name) return
    const token = (refineToken.current += 1)
    awaitingRefine.current = false
    setRefinePicked([])
    setRefining(true)
    setRefineSource(null)
    setRefineResult(null)
    setRefineFault(null)
    setOpeningRefine(true)
    // Paint on the INPUT-FOLDER copy, not on the well's preview.
    //
    // src.name is already the LoadImage filename, so this URL serves the exact
    // bytes the graph will read, and it is a stable server URL. The well's
    // previewUrl is a blob: object URL owned by the desk store, which Clear,
    // paste and drop all revoke - out from under the bench, which does not own
    // it. It is also absent entirely for an upload restored from localStorage,
    // which would offer the link and then fault on it.
    const cut = src.name.lastIndexOf('/')
    const url = fileUrl(
      cut === -1
        ? { filename: src.name, subfolder: '', type: 'input' }
        : { filename: src.name.slice(cut + 1), subfolder: src.name.slice(0, cut), type: 'input' },
    )
    try {
      // MEASURE THE FILE, never trust the declared size. A record files the size
      // the composer ASKED for, and a second pass upscales on the way to disk, so
      // a hires picture would open the bench at two thirds of its real resolution
      // and paste a correctly drawn patch into the wrong part of the frame.
      // openRefine's own doc comment says this; this used to do the opposite.
      const measured =
        (await measure(url)) ??
        (src.width && src.height ? { width: src.width, height: src.height } : null)
      if (!measured) throw new Error('the size could not be read')
      if (refineToken.current !== token) return
      // A picture taken from the archive still has its record, and so a maker.
      const made = src.fromEntryId ? allRecords().find((r) => r.id === src.fromEntryId) : undefined
      setRefineSource({
        name: src.name,
        url,
        width: measured.width,
        height: measured.height,
        // Carried when the picture came from the archive; empty for a plain
        // upload, which genuinely has no record behind it.
        entryId: src.fromEntryId ?? '',
        prompt: words ?? store.get().prompt,
        madeBy: made?.modelLabel || null,
        maker: made?.model ? { familyId: made.familyId, model: made.model } : null,
      })
    } catch (err) {
      if (refineToken.current !== token) return
      const why = err instanceof Error ? err.message : String(err)
      setRefineFault(`That picture could not be opened for region editing (${why}).`)
    } finally {
      if (refineToken.current === token) setOpeningRefine(false)
    }
  }, [])

  /**
   * "Make another like this" on a region pass.
   *
   * A region pass cannot be run again from its record: the record files the
   * region's words against the whole picture they were drawn on and keeps no
   * mask, so rebuilt it redrew the whole of that picture. This opens the
   * bench on that picture instead, with the region's words ready, and the
   * reader paints the area again. Where the picture has gone, the row that
   * leads here is not offered and ResultActions says why.
   */
  const openRegionAgain = useCallback(
    (entry: HistoryEntry) => {
      const origin = regionOrigin(entry, allRecords())
      if (!origin || origin.kind === 'gone') return
      if (origin.kind === 'record') return void openRefine(origin.entry, entry.prompt)
      if (origin.kind === 'output') {
        // Only the file is known, not what made it, so no maker is claimed.
        return void openRefine(
          { file: origin.ref, width: null, height: null, id: origin.fromEntryId ?? '', prompt: entry.prompt, modelLabel: '', model: '', familyId: '' },
          entry.prompt,
        )
      }
      void openRefineFromSource({ name: origin.name, fromEntryId: origin.fromEntryId }, entry.prompt)
    },
    [openRefine, openRefineFromSource],
  )

  // A record handed over from the Archive, which is a different route and so
  // cannot call openRefine directly. Runs once per request: takeRegionRequest()
  // clears the slot as it reads it.
  useEffect(() => {
    const handed = takeRegionRequest()
    if (handed) void openRefine(handed)
  }, [openRefine])

  const closeRefine = useCallback(() => {
    refineToken.current += 1
    awaitingRefine.current = false
    setRefining(false)
    setRefineSource(null)
    setRefineResult(null)
    setRefineFault(null)
    setOpeningRefine(false)
    // The "Drawn by" choice belongs to the picture it was made for. Left set,
    // it silently outranked the recipe for every region pass that followed.
    // The ticked region add-ons are the same kind of choice.
    setRefinePick(null)
    setRefinePicked([])
  }, [])

  /**
   * Queue one refine pass.
   *
   * The mask arrives as a source-sized PNG, white on black and opaque, and is
   * uploaded exactly as handed over: re-encoding it with an alpha channel would
   * make LoadImageMask read it as empty and the pass would change nothing.
   */
  const runRefine = useCallback(
    async (req: RefineRequest) => {
      if (!refineChain || !refineSource || !refineStyle || busy(press)) return
      const token = refineToken.current
      setRefineFault(null)
      try {
        const mask = await uploadImage(req.mask, req.maskName)
        const base = store.get()
        const rd = defaultsFor(refineStyle.def, refineStyle.model)
        const seed = req.seed ?? settled?.params.seed ?? seed0
        // The region prompt as it is sent: the drawing model's trained prefix,
        // the region's words, and the word each add-on in the chain answers
        // to. The graph used to be handed the bare region words over the top
        // of all that, so the record claimed a prefix the pass never saw and
        // the add-ons ran without their words.
        const sent = refineChain.specs
        const fromDesk = sent.filter((sp) => deskSpecs.some((d) => d.name === sp.name))
        const target = targetFor(refineStyle.def, refineStyle.model)
        const regionWords = triggersFor(
          regionPicked
            .filter((l) => sent.some((sp) => sp.name === l.file))
            .map((l) => ({ file: l.file, strength: l.strength, enabled: true })),
          lib,
          target,
        )
        const positive = positiveFor({
          def: refineStyle.def,
          model: refineStyle.model,
          prompt: req.prompt,
          loras: fromDesk,
          lib,
          extra: regionWords,
        })
        const composition: Composition = {
          ...base,
          // Filed as image-to-image, which is what it is: a partial denoise of
          // an existing picture. That files no width or height either, which is
          // right: the output is the source's size, not the composer's. The
          // variant `refine` below says it was one region, not the whole frame.
          mode: 'i2i',
          familyId: refineStyle.def.id,
          model: refineStyle.model,
          steps: rd.steps,
          cfg: rd.cfg,
          sampler: rd.sampler,
          scheduler: rd.scheduler,
          negative: rd.negative ?? null,
          positivePrefix: prefixFiled(refineStyle.positivePrefix, positive, req.prompt),
          positive,
          shift: refineStyle.shift,
          clipSkip: refineStyle.clipSkip,
          prompt: req.prompt,
          seed,
          source: {
            name: refineSource.name,
            previewUrl: refineSource.url,
            label: `region of ${refineSource.entryId || refineSource.name}`,
            width: refineSource.width,
            height: refineSource.height,
            fromEntryId: refineSource.entryId,
          },
          denoise: req.denoise,
          megapixels: null,
          length: null,
          fps: null,
        }
        const params = { ...toParams(composition, { negative: rd.negative ?? houseNegative }), positive }
        const graph = instantiateRefine(refineChain.def, params, {
          image: refineSource.name,
          mask,
          crop: req.plan.crop,
          target: req.plan.target,
          denoise: req.denoise,
          grow: req.grow,
          feather: req.feather,
          prompt: positive,
          seed,
        })
        const before = press.job?.id ?? null
        startRuns([
          {
            graph,
            composition,
            seed,
            familyLabel: refineStyle.group,
            modelLabel: refineStyle.label,
            // What rebuildable() reads to keep the whole-picture passes off it.
            variant: 'refine',
            label: `${refineStyle.label}, region refine`,
            // A refine pass carries no detail passes of its own; it IS the
            // detail pass.
            passes: NO_PASSES,
            // Exactly the chain the graph carries, so the record names every
            // add-on that touched the region.
            loras: sent,
          },
        ])
        // drive() announces the new job before its first await, so the live
        // singleton already holds it. startRuns refuses while another run is
        // going (one may have started during the mask upload), and then the
        // job there is not ours and nothing is followed. Nor is a pass whose
        // bench was closed or moved to another picture during the upload.
        const queued = press.job
        if (queued && queued.id !== before && refineToken.current === token) {
          refineJob.current = queued.id
          awaitingRefine.current = true
        }
      } catch (err) {
        setRefineFault(err instanceof Error ? err.message : 'The refine pass could not be queued.')
      }
    },
    [refineChain, refineSource, refineStyle, deskSpecs, regionPicked, lib, settled, seed0, houseNegative],
  )

  // The result is the picture the queued pass produced, and only that: the
  // composer still works while the bench is open, so an ordinary picture made
  // in the meantime must not be presented as the refined one. A pass that is
  // rejected, stopped or lost stops the wait, or the next ordinary picture
  // would land here in its place.
  useEffect(() => {
    if (!awaitingRefine.current) return
    const job = state.job
    if (!job || job.id !== refineJob.current) return
    if (job.status === 'error' || job.status === 'cancelled') awaitingRefine.current = false
    const cur = state.current
    if (!cur || job.status !== 'done' || !awaitingRefine.current) return
    awaitingRefine.current = false
    setRefineResult(cur)
  }, [state])

  /**
   * The graph a region of the picture on the plate would be drawn with, for
   * its "Sharpen a region" row: its own model when that can redraw a region,
   * else the first that can. Never the picture's own graph, which carries any
   * larger render it had, and a graph with a larger render in it cannot take a
   * region, so the row vanished from every such picture although the bench
   * draws with a plain graph and worked on it. Null when nothing installed
   * can redraw a region.
   */
  const resultRegion = useMemo(() => {
    if (!current || !refineOptions.length) return null
    const own = refineOptions.find((o) => o.def.id === current.familyId && o.model === current.model)
    return (own ?? refineOptions[0]).def
  }, [current, refineOptions])

  /** Set when the picture on the plate is a region pass: what it was drawn on, by name. */
  const resultRegionPass = useMemo(() => {
    if (!current) return null
    const origin = regionOrigin(current, records)
    if (!origin) return null
    if (origin.kind === 'gone') return { from: null }
    const no = origin.kind === 'record' ? origin.entry.no : 0
    return { from: no > 0 ? `No. ${no.toLocaleString('en-GB')}` : 'the picture it came from' }
  }, [current, records])

  const refineBlocked =
    refineFault ??
    // The catalogue comes first. The archive can now open this bench straight
    // from another route, which happens before `cat` has loaded, and claiming
    // "no installed style can do this" while we have not yet read the model
    // list is a confident falsehood rather than a delay.
    (!cat
      ? 'Reading the list of installed models from ComfyUI.'
      : !refineDef
        ? 'None of your installed models can redraw part of a picture. They all sample through a fixed schedule with no way to redraw just an area.'
        : (regionBlock ?? (openingRefine ? 'Getting the picture ready.' : null)))

  // --- the advanced panel, mounted only while More is open ----------------
  const advanced = (
    <div className="space-y-7">
      <div>
        <Choice legend="The look, in full" options={INTENTS} value={look} onChange={setLook} />
        <p className="mt-1.5 max-w-[62ch] text-caption text-grey-500">
          The simple screen offers three of these. Cartoon is the fourth and it ranks differently:
          western toon and comic styling rather than Japanese cel shading. The one line under the
          button has three names for these four routes, so a cartoon brief reads there as
          illustration. The ranking above is the one that actually ran.
        </p>
      </div>
      <AdvancedPanel
        recipe={recipe}
        overrides={ov.value}
        onOverrides={ov.set}
        lib={lib}
        onLibraryReload={() => void loadLoraLibrary().then(setLib, () => {})}
        samplers={cat?.samplers ?? []}
        schedulers={cat?.schedulers ?? []}
        pinnedModel={pinned}
        onPinModel={setPinned}
        onCatalogueChange={() => load(true)}
        onDropAddOn={dropAddOn}
        faultNode={state.fault?.node ?? null}
        onClose={() => settings.patch({ expert: false })}
      />
    </div>
  )

  // --- render -------------------------------------------------------------
  if (catError) return <ServerDown onRetry={() => load(true)} retryIn={retryIn} detail={catError} />

  return (
    <main
      className={`relative grid grid-cols-1 items-start xl:h-full ${
        expert ? 'lg:grid-cols-[minmax(0,44rem)_minmax(0,1fr)]' : 'lg:grid-cols-[minmax(0,34rem)_minmax(0,1fr)]'
      }`}
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

      {/* ---- the composer: three answers and a button ---- */}
      <div className="order-1 border-grey-300 px-6 py-6 lg:border-r xl:h-full xl:overflow-y-auto">
        <h2 className="mb-1 border-b-2 border-burgundy-900 pb-1.5 text-overline font-semibold uppercase tracking-[0.18em] text-burgundy-900">
          The Pictures Desk
        </h2>
        <p className="mb-6 text-caption italic text-grey-500">
          {timing ??
            'Nothing timed for this model yet. After three pictures this line says how long one takes here.'}
        </p>

        {offline && (
          <div className="mb-4">
            <Notice tone="correction" title="Correction">
              We have lost the connection to ComfyUI. If ComfyUI is still running, a job already
              under way is picked up when it answers again. If it restarted, that job is gone.
            </Notice>
          </div>
        )}

        {correction && (
          <div className="mb-4">
            <Notice tone="correction" title="Correction">
              {correction} <Link onClick={() => setCorrection(null)}>Dismiss</Link>
            </Notice>
          </div>
        )}

        {pinNote && (
          <div className="mb-4">
            <Notice tone="correction" title="Correction">
              {pinNote}
            </Notice>
          </div>
        )}

        {/*
          What you are doing is the first question, not a parameter. It sits
          here, above the desk, rather than inside one: `c.mode === 'edit'`
          swaps EditDesk for ComposeDesk wholesale, so a control mounted inside
          either desk would unmount itself the moment it was used to cross that
          boundary, taking keyboard focus with it.
        */}
        <div className="mb-6">
          <Choice
            legend="What you want to do"
            hint="Dropping or pasting a picture on the page switches this for you."
            options={MODE_CHOICES.filter((m) => m.id !== 'edit' || !!editStyle)}
            value={c.mode}
            onChange={setMode}
          />
          {editGap ? <p className="mt-1.5 text-caption italic text-grey-500">{editGap}</p> : null}
        </div>

        {c.mode === 'edit' ? (
          <EditDesk
            prompt={c.prompt}
            onPrompt={writePrompt}
            promptRef={promptRef}
            recipe={shown}
            source={c.source}
            sourceBusy={uploading}
            sourceError={sourceError}
            onPickSource={() => setPicking(true)}
            onClearSource={clearSource}
            onRun={start}
            onStop={() => void stopRun()}
            running={running}
            job={state.job}
            queuedAhead={ahead}
            lastMs={state.lastMs}
            reducedMotion={reduced}
            advanced={advanced}
            moreOpen={expert}
            onMoreOpenChange={(open) => settings.patch({ expert: open })}
            onAcceptAddOn={(file) =>
              store.patch({
                addOnsAccepted: [...new Set([...c.addOnsAccepted, file])],
                addOnsDeclined: c.addOnsDeclined.filter((f) => f !== file),
              })
            }
            onDeclineAddOn={(file) =>
              store.patch({
                addOnsDeclined: [...new Set([...c.addOnsDeclined, file])],
                addOnsAccepted: c.addOnsAccepted.filter((f) => f !== file),
              })
            }
            onRemoveAddOn={dropAddOn}
            onEditRegion={
              c.source && c.source.name && canRefine
                ? () => void openRefineFromSource(c.source!)
                : undefined
            }
            sourceReading={sourceReading}
          />
        ) : (
          <ComposeDesk
            prompt={c.prompt}
            look={look as Look}
            anatomy={anatomy}
            onPrompt={writePrompt}
            onLook={setLook}
            onAnatomy={(next) => {
              setAnatomySaid(true)
              setAnatomy(next)
            }}
            onAcceptAddOn={(file) =>
              store.patch({
                addOnsAccepted: [...new Set([...c.addOnsAccepted, file])],
                addOnsDeclined: c.addOnsDeclined.filter((f) => f !== file),
              })
            }
            onDeclineAddOn={(file) =>
              store.patch({
                addOnsDeclined: [...new Set([...c.addOnsDeclined, file])],
                addOnsAccepted: c.addOnsAccepted.filter((f) => f !== file),
              })
            }
            onRemoveAddOn={dropAddOn}
            recipe={shown}
            // A picture left in the store after choosing "From words" is not
            // used by the run, so it is not shown either: the well, the "The
            // change" label and the "Make the change" button all follow what
            // the run will actually do. Switching back brings it back.
            source={needsSource(c.mode) ? c.source : null}
            needsSource={c.mode === 'i2i'}
            // Reachable now that the mode control is on the page rather than buried
            // in More: a reader can ask to work from a picture before handing one
            // over. Refuse in plain words instead of quietly rendering from the
            // prompt alone and throwing the intent away.
            disabled={c.mode === 'i2i' && !c.source}
            disabledWhy="Add a picture to work from, or choose From words."
            sourceBusy={uploading}
            sourceError={sourceError}
            onPickSource={canI2I ? () => setPicking(true) : undefined}
            onEditRegion={
              c.source && c.source.name && canRefine
                ? () => void openRefineFromSource(c.source!)
                : undefined
            }
            sourceReading={sourceReading}
            onClearSource={clearSource}
            onRun={start}
            onStop={() => void stopRun()}
            running={running}
            job={state.job}
            queuedAhead={ahead}
            lastMs={state.lastMs}
            promptRef={promptRef}
            reducedMotion={reduced}
            advanced={advanced}
            moreOpen={expert}
            onMoreOpenChange={(open) => settings.patch({ expert: open })}
          />
        )}

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
      <section className="order-2 px-6 py-6 xl:h-full xl:overflow-y-auto">
        {refining ? (
          <div className="flex h-full flex-col">
            <div className="mb-4 flex items-baseline justify-between gap-4 border-b-2 border-burgundy-900 pb-1.5">
              <Kicker className="text-burgundy-900">The Refine Bench</Kicker>
              <Link onClick={closeRefine}>Back to the plate</Link>
            </div>
            {refineStyle && (
              <p className="mb-3 text-caption leading-snug text-grey-700">
                <Kicker className="block">Drawn by {refineStyle.label}</Kicker>
                {refineWhy} It works at its own settings and the rest of the picture is untouched.{' '}
                {refineCarries}
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
              model={{
                // The first row hands the choice back to the desk, so the
                // reader can undo a pick without reloading. Only drawn when
                // there is a real choice; a one-model machine sees no picker.
                options: [
                  ...(refineOptions.length > 1 && refineDefault
                    ? [{ id: '', label: `Suggested: ${refineDefault.label}`, group: '' }]
                    : []),
                  ...refineOptions.map((o) => ({
                    id: `${o.def.id}::${o.model}`,
                    label: o.label,
                    group: o.group,
                  })),
                ],
                value:
                  refinePick && refineOptions.some((o) => `${o.def.id}::${o.model}` === refinePick)
                    ? refinePick
                    : '',
                onChange: (id: string) => setRefinePick(id || null),
                // Said under the picker only when the picture's own model is
                // not the one drawing, which is the case the reader asks about.
                note:
                  refineSource?.maker && !refinePick && !sameStyle(refineStyle, refineMaker)
                    ? refineWhy
                    : null,
              }}
              addOns={{
                options: regionOffers.map((o) => ({
                  id: o.file,
                  label: o.label,
                  strength: o.strength,
                  why: o.why,
                })),
                picked: refinePicked,
                onToggle: (id: string) =>
                  setRefinePicked((was) => (was.includes(id) ? was.filter((f) => f !== id) : [...was, id])),
              }}
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
          <>
            <Plate
              state={state}
              entry={current}
              c={c}
              style={style}
              reduced={reduced}
              adopted={adopted}
              examples={c.prompt.trim() ? [] : EXAMPLES}
              onExample={(ex) => {
                // A new composition, so the add-on decisions for the old one go.
                store.patch({ prompt: ex.prompt, addOnsAccepted: [], addOnsDeclined: [] })
                setLook(ex.look)
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
              canI2I={canI2I}
              canEdit={!!editStyle}
              canRefine={canRefine}
              onRefine={(entry) => void openRefine(entry)}
            />

            {/*
              What the picture can be told to do next. This is requirement three
              of the simplification: the quality passes are offered HERE, where
              the reader can see whether the hands came out wrong, instead of
              being checkboxes to guess at before anything exists.
            */}
            {current && (
              <ResultActions
                picture={{
                  url: fileUrl(current.file),
                  width: current.width ?? undefined,
                  height: current.height ?? undefined,
                }}
                def={resultDef}
                rebuild={rebuildable(current)}
                region={resultRegion}
                blocks={cat?.passBlocks ?? NO_PASS_BLOCKS}
                regionPass={resultRegionPass}
                canSource={canI2I}
                facts={facts.get(current.id) ?? null}
                busy={running}
                blocked={offline ? 'ComfyUI is not answering, so nothing can be queued.' : null}
                onAction={(id) => {
                  if (id === 'refine') return void openRefine(current)
                  if (id === 'again') {
                    return rebuildable(current) ? rerun(current, null) : openRegionAgain(current)
                  }
                  if (id === 'source') {
                    takeRecord(current)
                    setMode('i2i')
                    return
                  }
                  if (id === 'face' || id === 'hand' || id === 'hires') rerun(current, id)
                }}
              />
            )}

            {/* What the machine sees in it, on request. The hand and face rows above read the result. */}
            {current && !current.missing ? (
              <div className="mt-6 border-t border-grey-300 pt-4">
                <Reading
                  source={{ kind: 'output', rel: relPath(current.file) }}
                  cacheKey={current.id}
                  arch={archFor(familyOwning(current.model), current.model)}
                  known={current.tags ? { tags: current.tags, rating: current.rating ?? null } : null}
                  onRead={(report) => noteReading(current.id, report)}
                  onAddOn={(file) =>
                    store.patch({
                      addOnsAccepted: [...new Set([...c.addOnsAccepted, file])],
                      addOnsDeclined: c.addOnsDeclined.filter((f) => f !== file),
                    })
                  }
                  onUseWords={(words) =>
                    store.patch({ prompt: c.prompt.trim() ? `${c.prompt.trim()}, ${words}` : words })
                  }
                />
              </div>
            ) : null}
          </>
        )}
      </section>

      {picking && (
        <ArchivePicker
          records={records}
          onClose={() => setPicking(false)}
          onFile={() => {
            setPicking(false)
            fileInput.current?.click()
          }}
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

/**
 * The edit desk: the same three-answer page, minus the two answers that mean
 * nothing here.
 *
 * Changing a picture has one model and no look to choose, so printing a look
 * picker and an anatomy picker that change nothing would be exactly the kind of
 * dead control this redesign exists to remove. The pieces are the compose
 * rail's own, used directly.
 *
 * Four controls: the instruction, the picture (a plate and a Remove link), and
 * the button. Plus the More footnote, which is the fifth.
 */
function EditDesk({
  prompt,
  onPrompt,
  promptRef,
  recipe,
  source,
  sourceBusy,
  sourceError,
  onPickSource,
  onClearSource,
  onRun,
  onStop,
  running,
  job,
  queuedAhead,
  lastMs,
  reducedMotion,
  advanced,
  moreOpen,
  onMoreOpenChange,
  onAcceptAddOn,
  onDeclineAddOn,
  onRemoveAddOn,
  onEditRegion,
  sourceReading,
}: {
  prompt: string
  onPrompt: (v: string) => void
  promptRef: RefObject<HTMLTextAreaElement | null>
  recipe: Recipe
  source: SourceRef | null
  sourceBusy: boolean
  sourceError: string | null
  onPickSource: () => void
  onClearSource: () => void
  /** Add-ons that fit the instruction model. Offered, never applied without a decision. */
  onAcceptAddOn?: (file: string) => void
  onDeclineAddOn?: (file: string) => void
  /** Take an added add-on off again. */
  onRemoveAddOn?: (file: string) => void
  /** Open the region bench on the attached picture. Omit it and nothing is printed. */
  onEditRegion?: () => void
  /** A reading of the attached picture, printed under the well. */
  sourceReading?: ReactNode
  onRun: () => void
  onStop: () => void
  running: boolean
  job: DeskJob | null
  queuedAhead: number
  lastMs: number | null
  reducedMotion: boolean
  advanced: ReactNode
  moreOpen: boolean
  onMoreOpenChange: (open: boolean) => void
}) {
  const why = !prompt.trim()
    ? 'Say what to change.'
    : !recipe.ok
      ? recipe.reason
      : !source
        ? 'Choose the picture to change.'
        : !source.name || sourceBusy
          ? 'The picture is still copying across.'
          : ''

  return (
    <section className="max-w-[46rem]">
      <header className="mb-8">
        <Kicker>Compose</Kicker>
        <h2 className="mt-1 text-h2 font-semibold leading-tight text-ink">Change a picture</h2>
        <div className="mt-2 mb-4 border-b-2 border-burgundy-900" />
        <p className="max-w-[62ch] text-body leading-relaxed text-grey-700">
          Hand it a picture and say what should be different. Everything else follows the source.
        </p>
      </header>

      <div className="space-y-8">
        <PromptField
          value={prompt}
          onChange={onPrompt}
          onSubmit={why ? undefined : onRun}
          textRef={promptRef}
          label="The change"
          placeholder="Make the jacket red"
          rows={3}
        />

        <div>
          <SourceWell
            source={source}
            busy={sourceBusy}
            error={sourceError}
            onPick={onPickSource}
            onClear={onClearSource}
          />
          {source && onEditRegion ? (
            <p className="mt-1.5 text-caption text-grey-500">
              <Link onClick={onEditRegion}>Change part of this picture</Link> instead, by painting
              over the area you want redrawn.
            </p>
          ) : null}
          {source && sourceReading ? <div className="mt-2">{sourceReading}</div> : null}
        </div>

        {recipe.ok && onAcceptAddOn && onDeclineAddOn && (
          <AddOnOffers
            offers={recipe.offers}
            applied={recipe.loras.filter((l) => !l.measured)}
            onAccept={onAcceptAddOn}
            onDecline={onDeclineAddOn}
            onRemove={onRemoveAddOn}
          />
        )}

        <div>
          <RunButton
            label="Make the change"
            disabled={Boolean(why)}
            why={why}
            running={running}
            queuedAhead={queuedAhead}
            lastMs={lastMs}
            job={job}
            onRun={onRun}
            onStop={onStop}
            reduced={reducedMotion}
          />

          <RecipeProse recipe={recipe} />
        </div>
      </div>

      <MoreFootnote open={moreOpen} onOpenChange={onMoreOpenChange}>
        {advanced}
      </MoreFootnote>
    </section>
  )
}

/**
 * A failed job, in the words every desk uses (lib/faults.ts).
 *
 * This desk used to call every failure "rejected", say ComfyUI "would not
 * accept it", and send the reader to "the highlighted setting". Nothing here
 * highlights a setting, and a job that ComfyUI took and that then broke part
 * way, or whose error was missed on the socket and settled from its history,
 * was not rejected. The Video desk already said "did not finish" for those.
 */
function Fault({ fault, onDismiss }: { fault: DeskFault; onDismiss: () => void }) {
  const title = fault.message === NO_FILE ? 'No picture came back' : faultTitle(fault)
  const tone = fault.cancelled ? 'correction' : fault.lost ? 'warning' : 'error'
  const oom = title === 'The card ran out of memory'
  // The shared wording for memory mentions a shorter clip, which this desk
  // does not make.
  const body = oom
    ? 'This size needs more than the card has free. Try a smaller shape, or close anything else using the GPU.'
    : faultBody(fault)
  const where = faultWhere(fault)
  return (
    <Notice tone={tone} title={title}>
      {body}
      {where ? <span className="block text-caption">{where}</span> : null}{' '}
      <Link onClick={onDismiss}>Dismiss</Link>
    </Notice>
  )
}

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
                    {LOOKS.find((l) => l.id === ex.look)?.label ?? ex.look}
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
              {/* A 64 pixel tile needs a thumbnail, not the render, which is
                  often a megabyte or more and was decoded in full for every
                  picture made. */}
              <img
                src={thumbUrl(r.file, 256)}
                srcSet={thumbSrcSet(r.file)}
                sizes="64px"
                alt=""
                loading="lazy"
                decoding="async"
                className="h-full w-full object-cover"
              />
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

/**
 * How wide a picker tile is drawn, for choosing a thumbnail: under a third of
 * the screen in the three-column grid, and about 160 pixels at most in the
 * four and five column grids, inside the dialog's 48rem and its margins.
 */
const PICKER_TILE = '(min-width: 640px) 160px, 33vw'

function ArchivePicker({
  records,
  onClose,
  onFile,
  onPick,
}: {
  records: readonly HistoryEntry[]
  onClose: () => void
  /** Choose a file from disk instead. The two used to be separate links. */
  onFile: () => void
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
          <Kicker className="text-burgundy-900">Choose a picture</Kicker>
          <p className="text-caption">
            <Link onClick={onFile}>Choose a file</Link>
            <span className="px-2 text-grey-400" aria-hidden>
              ·
            </span>
            <Link onClick={onClose}>Close</Link>
          </p>
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
                    {/* Up to 120 tiles of about 160 pixels at most. Loaded as
                        the original renders, the dialog fetched and decoded
                        every one in full as it opened. */}
                    <img
                      src={thumbUrl(r.file, 256)}
                      srcSet={thumbSrcSet(r.file)}
                      sizes={PICKER_TILE}
                      alt=""
                      loading="lazy"
                      decoding="async"
                      className="h-full w-full object-cover"
                    />
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
