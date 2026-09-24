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
  missingFilesFor,
  missingWhy,
  packNeededFor,
  passBlocks,
  type PassBlocks,
} from '../lib/availability'
import { measureImage as measure } from '../lib/images'
import { clamp } from '../lib/num'
import { ServerDown } from '../components/ServerDown'
import { onPlanLanded } from '../lib/downloads'
import { annotatedRef } from '../lib/continuation'
import { WAITS_ON_SERVER, holdAwake } from '../lib/wakeLock'
import {
  deviceId,
  fallbackLine,
  follow,
  forgetGivenUp,
  givenUpBatches,
  holdCovers,
  outboxPending,
  recordTemplate,
  runnerAvailable,
  runnerStore,
  stopGroup,
  submitGroup,
  waitLine,
  withdraw,
  type FollowEvent,
  type FollowResult,
  type RunnerJob,
  type RunnerSnapshot,
  type SubmitBody,
  type SubmitResult,
} from '../lib/runner'
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
  fetchPastRun,
  fileUrl,
  followPrompt,
  forgetObjectInfo,
  listJobs,
  newPromptId,
  objectInfo,
  run,
  uploadImage,
  watchConnection,
  type ApiWorkflow,
  type FileRef,
  type Followed,
  type OutputFile,
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
  get as getRecord,
  search as searchRecords,
  star as starRecord,
  subscribe as subscribeRecords,
  type HistoryEntry,
  update as updateRecord
} from '../lib/history'
import {
  MODES,
  adoptValue,
  compositionFromEntry,
  deskStore,
  needsSource,
  randomSeed,
  recordOf,
  settings,
  sourceFile,
  tabStore,
  takePlateRequest,
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
  detailSentence,
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
  ANATOMY_LEVELS,
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
import { RegionRefine, forgetBench, type RefineRequest } from '../components/refine'
import {
  ComposeDesk,
  MoreFootnote,
  PromptField,
  RunButton,
  STOPPING,
  SourceWell,
  Choice,
  type RunJob,
} from '../components/compose'
import {
  AdvancedPanel,
  NO_PASSES,
  buildGraph,
  sanitiseOverrides,
  settle,
  useOverrides,
  type Overrides,
  type Passes,
  type Unloadable,
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

/**
 * Why a pinned file cannot be used, as the end of "X is pinned, but ...": the
 * catalogue's own sentence when it has one ("it needs qwen_3_4b.safetensors"),
 * else that the file is not installed here.
 */
function pinBlockedWhy(model: string, unavailable: readonly { name: string; why: string }[]): string {
  const why = unavailable.find((u) => u.name === model)?.why
  return why ? `it cannot load here: it ${why.replace(/\.$/, '')}` : 'it is not installed here'
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
  /**
   * Weight files that cannot be offered, with why. `files` is true when a file
   * the graph loads is missing (an encoder, a VAE, the node pack that reads
   * it), which no reading of free memory can change; false when only the
   * memory verdict refused it, which the ranking judges again on each visit.
   */
  unavailable: { name: string; why: string; files: boolean }[]
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
  const unavailable: Catalogue['unavailable'] = []

  for (const model of installed) {
    const def = familyOwning(model)
    if (!def || (def.mode !== 'image' && def.mode !== 'edit')) continue

    // Every file the graph references must exist, or the run fails with an
    // opaque backend error. Name the missing file instead; then the memory,
    // priced on this file rather than the family's default. Checked for this
    // file alone: one SDXL checkpoint does not need the other three.
    const avail = availabilityOf(def, inv, hardware, sizes, model)
    if (!avail.ok) {
      unavailable.push({ name: model, why: avail.why, files: missingFilesFor(def, inv, model).length > 0 })
      continue
    }
    styles.push(styleOf(def, model, avail.verdict))
  }

  // A weight file ComfyUI cannot list because the node that reads it is not
  // installed never reaches the loop above: it is on disk, and it looked like
  // nothing at all. The instruction editing model is a .gguf, so without the
  // GGUF node pack "Change a picture" simply vanished. It is named here with
  // the pack it needs, and with any other file of its family that is missing
  // too: told of the pack alone, a reader would install it and only then learn
  // that the family still lacks its encoder or its VAE.
  for (const def of FAMILIES) {
    if (def.mode !== 'image' && def.mode !== 'edit') continue
    const others = missingFilesFor(def, inv).filter((f) => !def.models.includes(f))
    for (const model of def.models) {
      if (installed.has(model) || !sizes.has(model)) continue
      if (!packNeededFor(model, inv)) continue
      unavailable.push({ name: model, why: missingWhy([model, ...others], inv, sizes), files: true })
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
  // The model list is one shared download for every desk (comfy.ts keeps it),
  // so a reload asks for it afresh rather than reading the kept copy again.
  if (reload) forgetObjectInfo()
  if (reload || !cataloguePromise) cataloguePromise = readCatalogue()
  return cataloguePromise
}

// A family fetched from the catalogue changes what this machine can run, and
// the fetch can land while the reader is in another room. The desk reads the
// catalogue once and keeps it, so the kept reading goes here, and the next
// visit reads what is installed now.
onPlanLanded(() => {
  forgetObjectInfo()
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
  /**
   * When ComfyUI began running it: the first 'running' event, which is its
   * execution_start. The time a record files is counted from here, not from
   * the press. A picture sent while a clip was sampling used to file the
   * clip's remaining minutes as its own, and the timing line under the
   * heading then stated that as how long a picture takes.
   */
  ranAt: number | null
  /**
   * A picture of a batch the SwitchGen server is sending (see the server's
   * batch below), from this page or another device. The section bar reports
   * it from the server's own list, so App's bridge for this desk leaves it
   * out rather than report it twice.
   */
  runner?: boolean
  /** What the server says about the wait, when it has more to say than the stage. */
  note?: string | null
  /**
   * What of this batch the hold on the server's lane covers: the picture on
   * the press, and how many after it. Null when it covers none of it. See
   * laneHoldOf.
   */
  onHold?: LaneHold | null
}

/** A batch's pictures held on the server until the reader says, as laneHoldOf counts them. */
type LaneHold = { press: boolean; rest: number }

/**
 * A fault as the press shows it. A batch the server would not take was never
 * sent, so it carries its own title and tone rather than the ones a failed
 * picture gets.
 */
type PressFault = DeskFault & { title?: string; tone?: 'correction' | 'warning' | 'error' }

type PressState = {
  job: DeskJob | null
  /** Everything this desk has made since the page loaded, newest first. */
  results: HistoryEntry[]
  current: HistoryEntry | null
  fault: PressFault | null
  /** Duration of the last finished run, for the button's quiet receipt. */
  lastMs: number | null
  /** What the page before this one never sent of its batch, in a line (see batchRestLine). */
  unsent: string | null
  /** Pictures an earlier page in this tab sent and did not hand on, which this page follows only at the reader's word. */
  left: SentPicture[]
  /** Why the last batch was sent from this page although the server keeps a queue, in a line, or null. */
  fellBack: string | null
  /**
   * This desk's batch that the server keeps while its queue is off, or stands
   * back for another server, in a line, or null. Shown and not taken up: the
   * press stays free for the page's own work meanwhile (see parkedLine).
   */
  parked: string | null
}

export type RunPlan = {
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

/** What a record is filed from: the plan, less the graph that ran. */
type Filing = Pick<RunPlan, 'composition' | 'seed' | 'familyLabel' | 'modelLabel' | 'variant' | 'passes' | 'loras'>

let press: PressState = {
  job: null,
  results: [],
  current: null,
  fault: null,
  lastMs: null,
  unsent: null,
  left: [],
  fellBack: null,
  parked: null,
}
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
/** A picture the last page went while sending, which ComfyUI never heard of. Fault gives it its own title too. */
const NOT_SENT =
  'The page before this one went while it was still sending this picture, and ComfyUI has nothing under its number, so it was not sent. Nothing has been sent in its place.'

function busy(state: PressState): boolean {
  const s = state.job?.status
  return s === 'submitting' || s === 'queued' || s === 'running'
}

// ---------------------------------------------------------------------------
// A picture sent to ComfyUI outlives the page
// ---------------------------------------------------------------------------

/**
 * The picture on the press, kept for this tab from the moment before it is
 * sent until it settles.
 *
 * The press lived only in memory, and a phone reloads a tab it put in the
 * background as a matter of course. The picture then went on rendering in
 * ComfyUI with nothing following it: no progress, no Stop, the section bar
 * calling it a job started outside SwitchGen, and no record until some later
 * page load happened to look for unfiled files, by which time a restarted
 * ComfyUI had forgotten everything but the file.
 *
 * Kept in the tab's session storage, and taken up by the next page only when
 * the page that wrote it said it was going (pagehide) or the browser discarded
 * the tab, as the Video desk's lane is: a tab copied from this one while it is
 * still following the picture gets the same entry, and must not follow and
 * file it a second time. Anything else (a copied tab, a page that crashed, an
 * iPhone that closed the tab in the background, which says nothing to tell it
 * apart from a copy) is put to the reader, never followed on a guess and never
 * dropped without a word.
 *
 * It is written under the prompt's number before the prompt goes, marked as
 * still being sent. The send can reach ComfyUI and the page go before the
 * answer comes back, as when the phone throws away a tab the reader left just
 * after pressing Make. Kept only once ComfyUI had answered, that picture
 * rendered with nothing following it, the desk bare, and a second press made
 * it twice.
 */
const SENT_KEY = 'switchgen.picturesent.v1'
/** Sent pictures an earlier page left that this page would not follow by itself. */
const LEFT_SENT_KEY = 'switchgen.picturesent.v1.left'
/** This page, so a page back from the browser's cache can tell whether a later one took its picture. */
const PAGE = globalThis.crypto?.randomUUID?.() ?? `page_${Date.now()}_${Math.random().toString(36).slice(2)}`

/** A picture sent to ComfyUI, with what its record will say. */
export type SentPicture = Filing & {
  id: string
  promptId: string
  startedAt: number
  label: string
  /** Still on its way: ComfyUI had not answered when this was written, so it may never have got there. */
  sending?: boolean
  /** Where it stood in its batch, so a later page can say what of the batch was never sent. */
  index?: number
  total?: number
}

/** This page's pictures in ComfyUI's hands. One at a time, but a list costs nothing. */
let sent: SentPicture[] = []
/** False when the tab would not keep the last write. */
let sentKept = true

const isObj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v)

function isSent(v: unknown): v is SentPicture {
  return (
    isObj(v) &&
    typeof v.id === 'string' &&
    typeof v.promptId === 'string' &&
    !!v.promptId &&
    typeof v.startedAt === 'number' &&
    typeof v.label === 'string' &&
    isObj(v.composition) &&
    typeof v.seed === 'number' &&
    typeof v.familyLabel === 'string' &&
    typeof v.modelLabel === 'string' &&
    (v.variant === null || v.variant === undefined || typeof v.variant === 'string') &&
    isObj(v.passes) &&
    Array.isArray(v.loras) &&
    (v.sending === undefined || typeof v.sending === 'boolean') &&
    (v.index === undefined || typeof v.index === 'number') &&
    (v.total === undefined || typeof v.total === 'number')
  )
}

/** The saved pictures as written, or null when there are none or they cannot be read. */
export function readSent(raw: string | null): { writer: string; released: boolean; jobs: SentPicture[] } | null {
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
    jobs: v.jobs.filter(isSent),
  }
}

/** Write this page's pictures, or clear them when none are out. `released` says the page is going. */
function saveSent(released = false): void {
  if (sent.length) {
    sentKept = tabStore.set(SENT_KEY, JSON.stringify({ writer: PAGE, released, jobs: sent }))
  } else {
    tabStore.remove(SENT_KEY)
    sentKept = true
  }
}

function saveLeftSent(): void {
  if (press.left.length) tabStore.set(LEFT_SENT_KEY, JSON.stringify({ jobs: press.left }))
  else tabStore.remove(LEFT_SENT_KEY)
}

function noteSent(job: SentPicture): void {
  sent = [...sent.filter((j) => j.id !== job.id), job]
  saveSent()
}

/**
 * Keep the picture on the press for the tab under `promptId`, marked
 * `sending` until ComfyUI has said it has it.
 */
function keepSent(plan: RunPlan, promptId: string, sending: boolean): void {
  const job = press.job
  if (!job) return
  noteSent({
    id: job.id,
    promptId,
    startedAt: job.startedAt,
    label: plan.label,
    ...(sending ? { sending: true } : {}),
    index: job.index,
    total: job.total,
    composition: plan.composition,
    seed: plan.seed,
    familyLabel: plan.familyLabel,
    modelLabel: plan.modelLabel,
    variant: plan.variant,
    passes: plan.passes,
    loras: plan.loras,
  })
}

/**
 * What the page before this one never sent of the batch a picked-up picture
 * belonged to, in a line, or null when it was the last of its batch or its
 * place is not known. The rest of a batch lives only in the page sending it,
 * so a reload loses it. The next page showed the picture it picked up as 1 of
 * 1 and said nothing more, and the reader took the whole batch to have run.
 */
export function batchRestLine(index: number | undefined, total: number | undefined): string | null {
  if (index === undefined || total === undefined) return null
  if (!Number.isInteger(index) || !Number.isInteger(total) || index < 1 || total <= index) return null
  const first = index + 1
  const one = first === total
  const which = one ? `Picture ${total} was` : `Pictures ${first} ${total === first + 1 ? 'and' : 'to'} ${total} were`
  const it = one ? 'it' : 'them'
  return `The picture picked up from the page before this one was number ${index} of a batch of ${total}. ${which} never sent: the page reloaded before it got to ${it}, and nothing has sent ${it} since. Make ${it} again from the desk if you want ${it}.`
}

/** Done, failed, stopped or lost: nothing is left for a later page to follow. */
function settleSent(id: string): void {
  if (!sent.some((j) => j.id === id)) return
  sent = sent.filter((j) => j.id !== id)
  saveSent()
}

/** The same file, as ComfyUI serves it. */
function sameFile(a: FileRef, b: FileRef): boolean {
  return a.filename === b.filename && (a.subfolder ?? '') === (b.subfolder ?? '') && (a.type || 'output') === (b.type || 'output')
}

/**
 * The record that already names this file, when ComfyUI answered from its
 * cache. A graph identical to one it has run (a reused seed and nothing
 * changed) comes back in the time it takes to look it up, with the file it
 * wrote last time under a new prompt id. Filed again, it made a second card
 * and edition number for one file, and its lookup time went into the timing
 * line as a picture that took a tenth of a second. Only a file ComfyUI says
 * was cached is matched: a file name reused after the old file was deleted is
 * a new picture, and is filed as one.
 */
export function cachedRecordFor(file: OutputFile, records: readonly HistoryEntry[]): HistoryEntry | null {
  if (file.cached !== true) return null
  return records.find((r) => sameFile(r.file, file)) ?? null
}

/**
 * How long ComfyUI took, from when it began running the picture to when the
 * answer came back. Zero, which files as not timed, when the start was never
 * seen: counted from the press instead, it would include every job that was
 * in front of this one.
 */
export function ranFor(ranAt: number | null, finishedAt: number): number {
  return ranAt != null && finishedAt >= ranAt ? finishedAt - ranAt : 0
}

/** ComfyUI's own start and end for a prompt, from its history, or 0 when it has none to give. */
async function timedByComfy(promptId: string): Promise<number> {
  if (!promptId) return 0
  try {
    const past = await fetchPastRun(promptId)
    if (past?.startedAt != null && past.finishedAt != null && past.finishedAt >= past.startedAt) {
      return past.finishedAt - past.startedAt
    }
  } catch {
    /* no answer is no timing, not a failed picture */
  }
  return 0
}

/**
 * Put a finished picture on the plate: the record ComfyUI's cache points back
 * to, or a new record filed from the plan. False when the job wrote no file.
 */
function landPicture(files: OutputFile[], filing: Filing, promptId: string, durationMs: number, jobId: string): boolean {
  const picture = files.find((f) => f.kind === 'image') ?? files[0] ?? null
  if (!picture) {
    emit({
      fault: { message: NO_FILE, cancelled: false, lost: false, node: null, nodeType: null, detail: null },
    })
    patchJob({ status: 'error', finishedAt: Date.now() })
    return false
  }
  const known = cachedRecordFor(picture, allRecords())
  if (known) {
    // The picture already on file, with its own real time. No receipt: a
    // lookup is not a run.
    emit({ results: [known, ...press.results.filter((r) => r.id !== known.id)], current: known })
    patchJob({ status: 'done', pct: 1, stage: 'Done', finishedAt: Date.now() })
    return true
  }
  const record = recordOf(filing.composition, {
    file: picture,
    files: files.length > 1 ? files : undefined,
    kind: picture.kind,
    promptId,
    durationMs,
    seed: filing.seed,
    familyLabel: filing.familyLabel,
    modelLabel: filing.modelLabel,
    variant: filing.variant,
    passes: filing.passes,
    loras: filing.loras,
  })
  // A full or unwritable archive must never present as a failed picture:
  // the file is on disk either way, so show it and carry on.
  let entry: HistoryEntry
  try {
    entry = fileRecord(record)
  } catch {
    entry = { ...record, id: `unfiled-${jobId}`, no: 0, at: record.at ?? Date.now() }
  }
  emit({
    results: [entry, ...press.results],
    current: entry,
    // A picture whose run was not timed leaves the last receipt alone.
    lastMs: durationMs > 0 ? durationMs : press.lastMs,
  })
  patchJob({ status: 'done', pct: 1, stage: 'Done', finishedAt: Date.now() })
  return true
}

/**
 * Put a batch on the press. Exported for the tests: the desk's own buttons are
 * the only callers in the app.
 *
 * It resolves with the id the press shows the batch's first picture under, so
 * a caller that needs to know which job is its own (the region bench) is told
 * rather than reading whatever the press holds by then; null when the press
 * was busy and nothing was put on it. It never rejects.
 *
 * Every batch is offered to the server's queue first (see viaServer), which
 * asks whether the queue runs now rather than going by what the page last
 * heard. A queue that was off when the page loaded, with nothing in it, sends
 * the page no word when it comes back, and the page used to send every batch
 * itself from then on, until the tab was hidden and shown again. Where the
 * queue does not take it, the batch is this page's to send, one picture at a
 * time, as it was before the server had a queue.
 */
export function startRuns(plans: RunPlan[]): Promise<string | null> {
  if (driving || handing || viewing || !plans.length) return Promise.resolve(null)
  // A new press answers what the last page never sent, so the line goes, and
  // with it the word on why the last batch was sent from here.
  if (press.unsent || press.fellBack) emit({ unsent: null, fellBack: null })
  return viaServer(plans)
}

/** The batch as this page sends it itself. */
function runInPage(plans: RunPlan[]): void {
  queue = [...plans]
  stopped = false
  void drive()
}

/**
 * The hold on the screen while the rest of a batch waits in the page. It is
 * let go the moment ComfyUI has the last picture (see onProgress), not when
 * that picture lands: from then on nothing waits here, a picture ComfyUI has
 * goes on while the phone sleeps and is kept for the tab (SENT_KEY), and
 * a phone left on the table kept its screen on through the whole of the last
 * render for nothing.
 */
let awake: (() => void) | null = null

function letScreenSleep(): void {
  awake?.()
  awake = null
}

async function drive() {
  driving = true
  let index = 0
  const total = queue.length
  // The rest of a batch is sent from this page, one picture at a time, and a
  // phone that locks its screen sends nothing more until it wakes. Where the
  // browser allows it, the screen is kept on until the last picture is sent;
  // the button says the rest in words either way (see RunButton's waitingLine).
  awake = total > 1 ? holdAwake('pictures batch') : null

  try {
    while (queue.length && !stopped) {
      const plan = queue.shift()!
      index += 1
      const startedAt = Date.now()
      const jobId = `${startedAt.toString(36)}-${index}`
      // The prompt's number, made here so the picture can be kept for the tab
      // under it before it goes (see SENT_KEY), and looked up by a later page
      // if this one goes before ComfyUI answers.
      const sendAs = newPromptId()
      emit({
        fault: null,
        job: {
          id: jobId,
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
          ranAt: null,
        },
      })
      keepSent(plan, sendAs, true)

      // run() also rejects when ComfyUI forgets the prompt, as after a
      // restart mid job, so this await always returns and the desk is freed.
      try {
        const files = await run(plan.graph, (ev) => onProgress(ev, plan), { promptId: sendAs })
        const finishedAt = Date.now()
        const promptId = press.job?.promptId ?? ''
        const durationMs = ranFor(press.job?.ranAt ?? null, finishedAt) || (await timedByComfy(promptId))
        if (!landPicture(files, plan, promptId, durationMs, jobId)) continue
      } catch (err) {
        const fault = faultOf(err)
        emit({ fault })
        patchJob({ status: fault.cancelled ? 'cancelled' : 'error', finishedAt: Date.now() })
        // A rejected queue or a stopped job ends the whole batch: three more of
        // the same mistake helps nobody.
        break
      } finally {
        settleSent(jobId)
      }
    }
  } finally {
    // Every exit releases the desk. Releasing only on the happy path is what
    // used to wedge the run button for the rest of the page's life.
    queue = []
    driving = false
    stopped = false
    letScreenSleep()
    lookAgain()
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
      // The cancel below ends it, so nothing is left for a later page to
      // follow: kept, it would be picked up and read as lost to a restart.
      settleSent(press.job.id)
      patchJob({ promptId: ev.promptId, stage: STOPPING })
      void cancelJob(ev.promptId).catch(() => undefined)
      return
    }
    patchJob({ promptId: ev.promptId, status: 'queued', stage: 'Queued' })
    // From here the picture is ComfyUI's, and outlives this page.
    keepSent(plan, ev.promptId, false)
    // With the last picture in ComfyUI's hands nothing waits in the page.
    if (!queue.length) letScreenSleep()
    return
  }
  if (ev.phase === 'preview') {
    patchJob({ previewUrl: ev.url })
    return
  }
  if (ev.phase !== 'running') return

  const cls = ev.node ? plan.graph[ev.node]?.class_type : null
  showRunning(cls, ev.value, ev.max, stopped)
}

/**
 * A step of the picture on the press, from this page's own socket or from the
 * server's word on a picture it sent.
 */
function showRunning(classType: string | null | undefined, value: number, max: number, stopping: boolean): void {
  if (!press.job) return
  // A job asked to stop may still report a step or two before the stop
  // lands. It says it is stopping until it has.
  const stage = stopping ? STOPPING : stageFor(classType, value, max)
  const sampling = max > 1
  const pct = sampling ? clamp(value / max, press.job.pct, 0.97) : Math.max(press.job.pct, 0.02)
  patchJob({ status: 'running', stage, value, max, pct, ranAt: press.job.ranAt ?? Date.now() })
}

/**
 * Follow a picture an earlier page in this tab sent, to its end, and file it.
 *
 * The socket that would have reported it belonged to that page, so it is
 * followed through ComfyUI's job list instead (followPrompt). It is the desk's
 * own job meanwhile: the press shows it, the section bar counts it as this
 * app's, and Stop cancels it.
 *
 * One that page was still sending is looked for under its number, and counts
 * as queued only once ComfyUI says it has it. If ComfyUI has nothing under
 * that number it was never sent, and it is not sent again from here: the
 * reader decides whether to make it.
 */
async function followSent(job: SentPicture): Promise<void> {
  stopped = false
  const rest = batchRestLine(job.index, job.total)
  /** ComfyUI has said it has the picture, so gone later is gone, not unsent. */
  let arrived = !job.sending
  emit({
    fault: null,
    ...(rest ? { unsent: rest } : {}),
    job: {
      id: job.id,
      promptId: job.promptId,
      status: job.sending ? 'submitting' : 'queued',
      stage: job.sending ? 'Asking ComfyUI whether it got there' : 'Picked up from the page before this one',
      value: 0,
      max: 0,
      pct: 0,
      previewUrl: null,
      label: job.label,
      // One of one, whatever its place in its batch: the rest of that batch
      // went with the page (the unsent line says so), and counted here the
      // button would say they wait to be sent from this one.
      index: 1,
      total: 1,
      startedAt: job.startedAt,
      finishedAt: null,
      ranAt: null,
    },
  })
  let outcome: Followed
  try {
    outcome = await followPrompt(job.promptId, {
      onState: (s) => {
        // The first word from ComfyUI settles the doubt the mark carried: drop
        // it from the saved entry too, so a page after this one follows a
        // picture that got there, not one that may never have.
        if (!arrived && job.sending) {
          const { sending: _sending, ...landed } = job
          noteSent(landed)
        }
        arrived = true
        const cur = press.job
        if (!cur || cur.id !== job.id) return
        patchJob({
          status: s,
          stage: stopped ? STOPPING : s === 'queued' ? 'Queued' : 'Drawing',
          pct: s === 'running' ? Math.max(cur.pct, 0.02) : cur.pct,
        })
      },
    })
  } catch (err) {
    outcome = { status: 'error', message: err instanceof Error ? err.message : String(err), node: null, nodeType: null }
  }
  try {
    if (outcome.status === 'done') {
      // Timed by ComfyUI itself: this page never saw it begin.
      landPicture(outcome.files, job, job.promptId, await timedByComfy(job.promptId), job.id)
    } else if (outcome.status === 'cancelled' || (outcome.status === 'lost' && stopped)) {
      // Stop takes a waiting job out of the queue with no record left behind,
      // so gone after a stop is the stop landing.
      emit({ fault: faultOf({ message: 'Job stopped. Nothing was saved.', cancelled: true }) })
      patchJob({ status: 'cancelled', finishedAt: Date.now() })
    } else if (outcome.status === 'lost') {
      emit({
        fault: faultOf({
          // Still being sent when that page went, and never seen in ComfyUI:
          // not a restart, and no file to look for.
          message: arrived
            ? 'ComfyUI no longer has any record of the picture the page before this one sent, which usually means it restarted. If it finished first, its file is on disk: open the Archive and press “Look for files with no record”.'
            : NOT_SENT,
          lost: true,
        }),
      })
      patchJob({ status: 'error', finishedAt: Date.now() })
    } else {
      emit({
        fault: { message: outcome.message, cancelled: false, lost: false, node: outcome.node, nodeType: outcome.nodeType, detail: null },
      })
      patchJob({ status: 'error', finishedAt: Date.now() })
    }
  } finally {
    settleSent(job.id)
    stopped = false
  }
}

/**
 * Take pictures up as this page's own, and follow them one after another. The
 * press is held from the first to the last, so nothing of the server's is
 * taken up in between (see lookAgain).
 */
function takeUp(jobs: SentPicture[]): void {
  sent = [...sent.filter((j) => !jobs.some((t) => t.id === j.id)), ...jobs]
  saveSent()
  driving = true
  void (async () => {
    try {
      for (const job of jobs) await followSent(job)
    } finally {
      driving = false
      lookAgain()
    }
  })()
}

/**
 * Take up what the last page in this tab left on the press, once, when the
 * module loads. The saved entry is cleared either way: taken up, it is written
 * again as this page's own; left alone, it moves to the desk's question (see
 * LEFT_SENT_KEY), so a copy another tab is following is never taken up later
 * by accident. It used to be dropped without a word, on the reasoning that it
 * was a copied tab's; but an iPhone closes a tab in the background without
 * saying it is going, and so does a crashed page, and that picture then went
 * on with no Stop and reached the Archive only through its look for files
 * with no record, bare if ComfyUI had restarted meanwhile.
 */
function restoreSent(): void {
  const saved = readSent(tabStore.get(SENT_KEY))
  const left = readSent(tabStore.get(LEFT_SENT_KEY))
  tabStore.remove(SENT_KEY)
  let kept = left?.jobs ?? []
  let take: SentPicture[] = []
  if (saved?.jobs.length) {
    const discarded =
      typeof document !== 'undefined' && (document as Document & { wasDiscarded?: boolean }).wasDiscarded === true
    if (saved.released || discarded) take = saved.jobs
    else kept = [...kept, ...saved.jobs.filter((s) => !kept.some((l) => l.id === s.id))]
  }
  emit({ left: kept })
  saveLeftSent()
  if (take.length) takeUp(take)
}

/**
 * Follow what an earlier page sent, from this page, at the reader's word.
 * The press takes one picture at a time, so only while it is free.
 */
export function followLeftSent(): void {
  if (driving || handing || viewing || !press.left.length) return
  const jobs = press.left
  emit({ left: [] })
  saveLeftSent()
  takeUp(jobs)
}

/** Stop offering them. ComfyUI goes on making them. */
export function forgetLeftSent(): void {
  if (!press.left.length) return
  emit({ left: [] })
  saveLeftSent()
}

async function stopRun() {
  // A batch on the server is stopped there, never by cancelling its prompt
  // from here: the server is what sends the rest of it.
  if (handing || viewing || press.job?.runner) return stopOnServer()
  stopped = true
  queue = []
  const id = press.job?.promptId
  if (!id) {
    // Still being sent, so there is nothing to cancel yet: the "queued" event
    // cancels it the moment ComfyUI names it (see onProgress). The job stays
    // busy until then. Marked cancelled here, the Run button came back while
    // the send was still in flight, did nothing when pressed, and the job
    // went on to render.
    patchJob({ stage: STOPPING })
    return
  }
  patchJob({ stage: STOPPING })
  try {
    await cancelJob(id)
  } catch {
    /* the watcher reports the outcome; a failed cancel is not a second error */
  }
}

/**
 * The desk's own Stop, for the section bar and the running slug.
 *
 * A bare cancel of the running prompt ends the batch only when it lands. One
 * that arrives just after the picture saved finds nothing to cancel, and
 * drive() goes on to make the rest of the batch after the reader pressed Stop.
 * This stops the batch as the desk's own button does, whatever the picture in
 * hand is doing. With nothing running it does nothing, so a press that arrives
 * as the last picture finishes leaves that job reading Done, not Stopping.
 */
export function stopPress(): void {
  if (!busy(press)) return
  void stopRun()
}

/** Show a finished picture on the plate without re-running anything. */
function showResult(entry: HistoryEntry) {
  // The reader's pick stands over a record of the server's still on its way.
  wantCurrent = null
  emit({ current: entry })
}

/**
 * What the plate shows when nothing has been made or chosen since the page
 * loaded: the newest picture this desk filed that is still on disk. The plate
 * used to come up empty after every reload, and the face, hand and larger
 * render passes, which only the plate offers, could not be reached for any
 * picture made before it, or on another device.
 */
export function plateFallback(records: readonly HistoryEntry[]): HistoryEntry | null {
  return records.find((r) => r.desk === DESK && r.kind === 'image' && !r.missing) ?? null
}

/**
 * How often the desk asks how many jobs are ahead in ComfyUI's queue, and the
 * asking itself. The next ask is scheduled only once the last one answered: on
 * a fixed interval, a ComfyUI too slow to answer (a machine swapping just
 * before earlyoom acts) stacked a new request every five seconds until they
 * held every connection the browser allows the page, and the archive, the
 * thumbnails and the plate all waited behind them.
 */
export function watchAhead(onAhead: (n: number) => void, everyMs = 5000): () => void {
  let alive = true
  let timer: ReturnType<typeof setTimeout> | null = null
  const poll = async () => {
    try {
      const page = await listJobs({ status: ['pending', 'in_progress'], limit: 20 })
      if (!alive) return
      const mine = press.job?.promptId
      onAhead(page.jobs.filter((j) => j.id !== mine).length)
    } catch {
      if (alive) onAhead(0)
    } finally {
      if (alive) timer = setTimeout(() => void poll(), everyMs)
    }
  }
  void poll()
  return () => {
    alive = false
    if (timer) clearTimeout(timer)
  }
}

if (typeof window !== 'undefined') {
  // A reload or a close says it is going, so the next page in the tab takes
  // the picture up at once.
  window.addEventListener('pagehide', () => {
    if (sent.length) saveSent(true)
  })
  // A phone hides the page (and says pagehide) without always ending it. Back
  // in view, the picture is still this page's to follow, unless a later page
  // in the tab took it over meanwhile, which the pageshow below deals with.
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState !== 'visible' || !sent.length) return
    if (readSent(tabStore.get(SENT_KEY))?.writer === PAGE) saveSent()
  })
  // Back from the browser's cache. If another page ran in this tab meanwhile,
  // it took this picture up and has followed or filed it, so this page's copy
  // must not file it again: the page starts over, as a reload does.
  window.addEventListener('pageshow', (e) => {
    if (!e.persisted || !sent.length) return
    if (sentKept && readSent(tabStore.get(SENT_KEY))?.writer !== PAGE) {
      sent = []
      window.location.reload()
      return
    }
    saveSent()
  })
}

// ---------------------------------------------------------------------------
// A batch the SwitchGen server sends
// ---------------------------------------------------------------------------

/**
 * When the SwitchGen server keeps a queue for this desk (lib/runner.ts), a
 * batch is handed to it whole, in one request, and the server sends each
 * picture to ComfyUI in turn, files its record and goes on to the next while
 * this page is closed or the phone is locked. The rest of a batch used to
 * wait in the page, and a phone that locked its screen sent nothing more
 * until it woke.
 *
 * The page then only watches. It shows the picture on the press, its
 * progress and its ending, whichever device sent the batch, and its Stop asks
 * the server to stop the batch. Nothing here sends to ComfyUI, files a record
 * or keeps the screen on for such a batch, and nothing is kept for the tab
 * under SENT_KEY: the server has it, and a page that took it up from there
 * would send and file it a second time.
 *
 * Everything above (drive, SENT_KEY, followSent, the hold on the screen) is
 * the path for a server without the queue, and runs exactly as it did.
 *
 * While the queue is off, or stands back for another server, the server
 * still lists the work it keeps, as it was last saved, and nothing moves it
 * on: a Stop answers that the queue is not running. Such a batch is never
 * taken up then, and one on the press is let go (see letGoWhileOff). It is
 * said in a line under the desk (parkedLine), without a Stop, and the press
 * is free for the page's own work. Once the queue is back it is taken up as
 * any other.
 */

/** Said on the press while the server has not yet answered for the batch. */
const HANDING = 'Handing it to the server'
/** And under it, once the first asking has had no answer. */
const HANDING_LINE =
  'The SwitchGen server has not answered yet. The batch is handed to it as soon as it does, and is never sent from this page as well, since the server may already have it.'
/** The server's answer when it is already making a batch: one at a time, for every device. */
export const BATCH_BUSY = 'A batch of pictures is already being made, from this page or another.'
/**
 * The batch this tab handed to the server, until a page has shown how it
 * ended. The server goes on while the page is away, and a phone throws away
 * a tab it put in the background as a matter of course: the page that loads
 * next shows how that batch went, its pictures and any fault, rather than a
 * plate that says nothing of it.
 */
const RUNNER_KEY = 'switchgen.pictures.runner.v1'
/**
 * Batches the reader stopped before the server had answered for them. The
 * tab's outbox (lib/runner.ts) sends such a batch no more (see dropHanding),
 * but a request already sent may reach the server all the same, or have
 * reached it with only the answer lost, so one that shows up there, in this
 * page or the next one, is stopped the moment it does.
 */
const DROPPED_KEY = 'switchgen.pictures.dropped.v1'

/** A batch on its way to the server, until the server has answered for it. */
type Handing = {
  groupId: string
  body: SubmitBody
  /** Each picture's plan, by job id, in batch order. */
  own: Map<string, RunPlan>
  /** The server has not answered; the tab's outbox asks again when it is next heard from. */
  pending: boolean
}

/** The server's batch this page shows on the press, and where it stands in it. */
type Viewing = {
  groupId: string
  /** The job on the press. */
  jobId: string | null
  /** Each picture's plan, by job id, when this page sent the batch; null for one taken up. */
  own: Map<string, RunPlan> | null
  /** This page asked the server to stop the batch. */
  stopping: boolean
  /** Ends the follow of the picture on the press when the batch is let go (see letGoWhileOff). */
  quit: AbortController
  /** Let go because the queue went off: nothing of it is shown to its end, and it is taken up again once the queue is back. */
  letGo: boolean
}

let handing: Handing | null = null
let viewing: Viewing | null = null
/** Batches this page has shown to their end, so one is never taken up twice. */
const walked = new Set<string>()
/** Records the server filed that this page has not pulled yet, by id. */
const unpulled = new Set<string>()
/** A record to put on the plate the moment it is pulled, unless the reader picks another first. */
let wantCurrent: string | null = null
/**
 * The plans of a batch this page sent and then let go while the queue was
 * off, by group id, so the batch is shown with them once it is taken up again.
 */
const parkedPlans = new Map<string, Map<string, RunPlan>>()

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/

function readDropped(): Set<string> {
  try {
    const list: unknown = JSON.parse(tabStore.get(DROPPED_KEY) ?? '[]')
    return new Set(Array.isArray(list) ? list.filter((id): id is string => typeof id === 'string' && UUID.test(id)) : [])
  } catch {
    return new Set()
  }
}

let dropped = readDropped()
/** Stops asked for dropped batches and not yet answered. */
const stopsOut = new Set<string>()

/** Ask the server to stop a dropped batch, unless that is already being asked. */
function askStop(groupId: string): void {
  if (stopsOut.has(groupId)) return
  stopsOut.add(groupId)
  void stopGroup(groupId).finally(() => stopsOut.delete(groupId))
}

function keepDropped(): void {
  // A handful at most: the reader stopped each while the server was silent.
  if (dropped.size > 20) dropped = new Set([...dropped].slice(-20))
  if (dropped.size) tabStore.set(DROPPED_KEY, JSON.stringify([...dropped]))
  else tabStore.remove(DROPPED_KEY)
}

/** The same string, no longer than the server takes. */
const upTo = (text: string, n: number) => (text.length > n ? text.slice(0, n) : text)

/**
 * What of a batch the server never sent, in a line, once the batch ended at
 * picture `endedAt`: the server ends a batch at the first picture that fails
 * (one that wrote no file aside) and sends none of the rest. Null when
 * nothing was left.
 */
export function serverRestLine(first: number, last: number, endedAt: number): string | null {
  if (![first, last, endedAt].every(Number.isInteger) || first < 1 || last < first) return null
  const one = first === last
  const which = one ? `Picture ${first} was` : `Pictures ${first} ${last === first + 1 ? 'and' : 'to'} ${last} were`
  const it = one ? 'it' : 'them'
  return `${which} never sent, because picture ${endedAt} failed and a batch ends at the first picture that fails. Make ${it} again from the desk if you want ${it}.`
}

/**
 * The line under a batch the server is sending, or null when nothing of it
 * waits. The page's own line (RunButton's waitingLine) says the rest waits in
 * this page and goes nowhere while the phone is locked, which for such a
 * batch is the opposite of the truth, so the button is handed the job without
 * its place in the batch (see buttonJob) and this is said instead.
 *
 * While the hold on the server's lane covers any of the batch (onHold), the
 * line says which of it is held, and that the notice about held work, which
 * the shell shows on every page whenever the lane is held, sends it or calls
 * it off; it does not say the batch is sent in turn, which it is not until
 * the reader says.
 *
 * Nor does it say so while the queue is not running (`queueOn` false): work
 * that waited while it was off is held when it comes back, not carried on,
 * and the press says the queue is off in its own note. (A batch on the press
 * is let go when the queue goes off, see letGoWhileOff, so this is for the
 * moment in between.)
 */
export function serverWaitingLine(
  job: Pick<DeskJob, 'status' | 'stage' | 'index' | 'total' | 'runner' | 'note' | 'onHold'> | null,
  queueOn: boolean = runnerStore.snapshot().available,
): string | null {
  if (!job?.runner || job.stage === STOPPING) return null
  const waiting = job.status === 'submitting'
  if (job.stage === HANDING) return waiting ? HANDING_LINE : null
  if (!waiting && job.status !== 'queued' && job.status !== 'running') return null
  // Only a picture still waiting is held; the one on the press says so in its own note.
  const pressHeld = waiting && job.onHold?.press === true
  const restHeld = job.onHold?.rest ?? 0
  const left = Math.max(job.total - job.index, restHeld)
  const rest =
    restHeld > 0
      ? heldRestLine(left, restHeld, pressHeld)
      : left === 1
        ? 'One more picture waits on the SwitchGen server and is sent when this one is done.'
        : left > 1
          ? `${left} more pictures wait on the SwitchGen server and are sent one at a time.`
          : null
  if (!rest && !waiting) return null
  const held = (pressHeld ? 1 : 0) + restHeld
  const after =
    held === 0
      ? queueOn
        ? WAITS_ON_SERVER
        : null
      : held === 1
        ? 'The notice about held work sends it, or calls it off, with any other work held.'
        : 'The notice about held work sends them, or calls them off, with any other work held.'
  return [waiting ? job.note : null, rest, after].filter(Boolean).join(' ') || null
}

/**
 * The line under the desk for a batch of this desk's that the server keeps
 * while its queue is off, or stands back for another server: `left` of its
 * pictures are not finished, and `reason` is the server's own sentence for
 * why its queue is not running. A stop asked of it then is answered that the
 * queue is not running, so none is offered; `stopAsked` says this tab asked
 * for one all the same, before the queue went off, and asks again once it is
 * back (see onStore).
 */
function serverParkedLine(label: string, left: number, reason: string | null, stopAsked = false): string {
  const name = label.trim() ? `“${label.trim()}”` : 'A batch of pictures'
  const it = left === 1 ? 'it' : 'them'
  let why = (reason ?? '').trim() || 'The queue on the server is not running.'
  if (!/[.!?]$/.test(why)) why += '.'
  const then = stopAsked
    ? `You asked to stop ${it}, and this tab asks the server again once its queue is back.`
    : `This page does not send ${it}, and cannot stop ${it} until the queue is back. Pictures you make meanwhile are sent from this page.`
  return `${name}: ${left === 1 ? 'one picture waits' : `${left} pictures wait`} on the SwitchGen server, whose queue is off. ${why} ${then}`
}

/** The pictures after the one on the press, `n` of which (of `left`) the hold on the server's lane covers. */
function heldRestLine(left: number, n: number, pressHeld: boolean): string {
  if (n < left) {
    return `${left} more pictures wait on the SwitchGen server, and ${n === 1 ? 'one of them is' : `${n} of them are`} held until you say.`
  }
  const which = left === 1 ? 'One more picture waits' : `${left} more pictures wait`
  if (pressHeld) return `${which} on the SwitchGen server and ${left === 1 ? 'is' : 'are'} held with it.`
  return `${which} on the SwitchGen server, held until you say.`
}

/** The job as the Run button is handed it: its place in the batch goes into the stage. */
function buttonJob(job: DeskJob | null): RunJob | null {
  if (!job?.runner) return job
  return {
    stage: job.total > 1 ? `Picture ${job.index} of ${job.total} · ${job.stage}` : job.stage,
    pct: job.pct,
    status: job.status,
  }
}

const findJob = (id: string, snap: RunnerSnapshot = runnerStore.snapshot()): RunnerJob | undefined =>
  snap.jobs.find((j) => j.id === id)

/**
 * What of the batch of `job`, the picture on the press, the hold on the
 * server's lane covers, counted as the server counts it and as the notice
 * about held work does (holdCovers): a picture still waiting, with the hold
 * on all work or the picture a heavy one, whatever its own wait says. A
 * picture of the batch waiting besides the one on the press comes after it,
 * since each waits for the one before. Null when the lane is not held, or the
 * hold covers none of the batch.
 */
function laneHoldOf(snap: RunnerSnapshot, job: RunnerJob): LaneHold | null {
  const h = snap.lane.held
  if (!h) return null
  let press = false
  let rest = 0
  for (const j of snap.jobs) {
    if (j.groupId !== job.groupId || j.status !== 'waiting' || !holdCovers(h, j)) continue
    if (j.id === job.id) press = true
    else rest += 1
  }
  return press || rest ? { press, rest } : null
}

const sameHold = (a: LaneHold | null, b: LaneHold | null): boolean =>
  a === b || (!!a && !!b && a.press === b.press && a.rest === b.rest)

/**
 * A picture's place in its batch, counted from one, as the desk that sent it
 * wrote it, or null. (The server numbers a group's jobs from one as well, in
 * the order the desk handed them over.)
 */
function placeIn(job: RunnerJob | undefined, key: 'index' | 'total'): number | null {
  const n: unknown = job?.meta?.[key]
  return typeof n === 'number' && Number.isInteger(n) && n > 0 ? n : null
}

/** The group id this tab handed to the server and no page has yet shown to its end, or null. */
function keptGroup(): string | null {
  const id = tabStore.get(RUNNER_KEY)
  return id && UUID.test(id) ? id : null
}

function forgetKept(groupId: string): void {
  if (keptGroup() === groupId) tabStore.remove(RUNNER_KEY)
}

/** Stopped before it was sent, in the words a stop is always given. */
function stoppedBeforeSent(groupId: string): void {
  forgetKept(groupId)
  emit({ fault: faultOf({ message: 'Stopped before it was sent.', cancelled: true }) })
  patchJob({ status: 'cancelled', finishedAt: Date.now() })
}

/** The server would not take the batch: nothing of it was kept, and nothing was sent. */
function refusalOf(answer: { error: string; busy?: 'images' | 'reel' }): PressFault {
  const busyHere = answer.busy === 'images'
  return {
    message: busyHere ? BATCH_BUSY : answer.error,
    cancelled: false,
    lost: false,
    node: null,
    nodeType: null,
    detail: null,
    title: 'Not sent',
    tone: busyHere ? 'correction' : 'error',
  }
}

/**
 * Hand a batch to the server. The press shows the first picture at once, so
 * the button turns to Stop and a second press does nothing while the server
 * is asked. The asking (runnerAvailable) does not go by the page's last word
 * alone: where the server said its queue was off, and nothing streams its
 * word here, the queue is read again, so one that came back since is found.
 * A queue that does not take the batch has it sent from this page instead,
 * as it always was. Resolves with the id the press shows the batch under (see
 * startRuns).
 */
async function viaServer(plans: RunPlan[]): Promise<string | null> {
  const own = new Map<string, RunPlan>()
  let body: SubmitBody
  try {
    const total = plans.length
    body = {
      v: 1,
      group: {
        id: newPromptId(),
        desk: DESK,
        kind: 'batch',
        label: upTo(total > 1 ? `${plans[0].label}, ${total} pictures` : plans[0].label, 200),
        device: deviceId(),
      },
      jobs: plans.map((plan, i) => {
        const id = newPromptId()
        own.set(id, plan)
        return {
          id,
          label: upTo(plan.label, 200),
          prompt: upTo(plan.composition.prompt ?? '', 4000),
          kind: 'image' as const,
          primary: 'image' as const,
          orFirst: true,
          // Said and passed over, and the batch goes on, as on this page (see
          // landPicture).
          noFile: 'fail' as const,
          heavy: false,
          graph: plan.graph,
          record: recordTemplate(plan.composition, {
            seed: plan.seed,
            familyLabel: plan.familyLabel,
            modelLabel: plan.modelLabel,
            variant: plan.variant,
            passes: plan.passes,
            loras: plan.loras,
          }),
          meta: { index: i + 1, total },
        }
      }),
    }
  } catch {
    // A plan the server's record cannot be made from goes from here, where
    // it always could.
    return inPage(plans)
  }
  const h: Handing = { groupId: body.group.id, body, own, pending: false }
  const shownAs = body.jobs[0].id
  handing = h
  // Kept from before it goes: the server may take it with the page already gone.
  tabStore.set(RUNNER_KEY, h.groupId)
  emit({
    fault: null,
    job: {
      id: shownAs,
      promptId: null,
      status: 'submitting',
      stage: 'Sending the job',
      value: 0,
      max: 0,
      pct: 0,
      previewUrl: null,
      label: plans[0].label,
      index: 1,
      total: plans.length,
      startedAt: Date.now(),
      finishedAt: null,
      ranAt: null,
      runner: true,
    },
  })
  let can = { ok: false }
  try {
    can = await runnerAvailable(DESK)
  } catch {
    /* no answer is no queue: the page sends it */
  }
  // Stopped meanwhile (see dropHanding): nothing has been sent.
  if (handing !== h) return shownAs
  if (!can.ok) {
    handing = null
    forgetKept(h.groupId)
    return inPage(plans)
  }
  let answer: SubmitResult
  try {
    answer = await submitGroup(h.body)
  } catch {
    // submitGroup answers every failure in words. A throw all the same is
    // taken as no answer: the outbox has the batch, and it is never sent from
    // here as well, since the server may have it.
    answer = { ok: false, fallback: false, pending: true }
  }
  // Stopped meanwhile: a batch the server took after all is stopped the
  // moment it shows up there (see onStore).
  if (handing !== h) return shownAs
  if (answer.ok) {
    handing = null
    takeUpServerBatch(h.groupId, h.body.jobs.map((j) => j.id), h.own)
    return shownAs
  }
  if (answer.fallback) {
    handing = null
    forgetKept(h.groupId)
    emit({ fellBack: fallbackLine(answer.reason) })
    return inPage(plans)
  }
  if ('pending' in answer) {
    // No answer: the batch waits in the tab's outbox, which sends it again
    // when the server is next heard from, and the press waits with it (see
    // onStore). It is never sent from this page: the server may have it.
    h.pending = true
    patchJob({ stage: HANDING })
    return shownAs
  }
  handing = null
  forgetKept(h.groupId)
  emit({ fault: refusalOf(answer) })
  patchJob({ status: 'error', finishedAt: Date.now() })
  // Refused as busy, the batch in the way is shown on the press, under the
  // refusal (see driveRunner), rather than at the server's next word.
  lookAgain()
  return shownAs
}

/** Sent from this page after all, under the id drive() gives its first picture. */
function inPage(plans: RunPlan[]): string | null {
  runInPage(plans)
  return press.job?.id ?? null
}

/**
 * Let go of a batch the server has not answered for, at the reader's Stop.
 * The desk is free at once: the answer can take a minute to come, and a stop
 * that waited for it held the button for as long.
 *
 * The tab's outbox is told to send it no more (withdraw), from this page or
 * the next. Sent again later, as it used to be, its first picture reached
 * ComfyUI before the stop could, and the reader's next batch was refused as
 * busy. A
 * request already sent may reach the server all the same, or have reached it
 * with only the answer lost, so the server is told now as well, and the batch
 * is stopped the moment it shows up there (see onStore).
 */
function dropHanding(h: Handing): void {
  if (handing === h) handing = null
  withdraw(h.groupId)
  dropped.add(h.groupId)
  keepDropped()
  // A queue that is not running answers a stop that it is not; the stop is
  // asked once it is back (see onStore).
  if (runnerStore.snapshot().available) askStop(h.groupId)
  stoppedBeforeSent(h.groupId)
  lookAgain()
}

/** The desk's Stop, for a batch on the server or on its way there. */
async function stopOnServer(): Promise<void> {
  if (handing) return dropHanding(handing)
  const v = viewing
  if (!v) {
    // A picture of the server's with nothing following it: its batch is
    // stopped all the same, and never by cancelling its prompt from here.
    const job = press.job ? findJob(press.job.id) : undefined
    if (job) await stopGroup(job.groupId)
    return
  }
  v.stopping = true
  patchJob({ stage: STOPPING })
  if ((await stopGroup(v.groupId)) || viewing !== v) return
  // The server did not take the stop (it may be restarting). The press says
  // what the picture is really doing again, at once rather than at its next
  // step, and Stop can be asked again.
  v.stopping = false
  syncFromStore()
  // ComfyUI's number can reach the press before the server's list says the
  // picture is queued, and that wait is not syncFromStore's to show.
  const cur = press.job
  if (cur?.status === 'queued' && cur.stage === STOPPING && !findJob(cur.id)?.stopRequested) patchJob({ stage: 'Queued' })
}

/**
 * Show a batch the server is sending, one picture after another, to its end.
 * `own` holds the plans when this page sent it; a batch taken up from another
 * device, or from the page before this one, has none, and its records are
 * shown once the archive has them. `freedJust`: it is taken up in the look
 * made the moment the press came free (see lookAgain).
 */
async function driveRunner(
  groupId: string,
  jobIds: readonly string[],
  sentWith: Map<string, RunPlan> | null,
  freedJust = false,
): Promise<void> {
  // The plans of a batch this page sent and let go while the queue was off.
  const own = sentWith ?? parkedPlans.get(groupId) ?? null
  parkedPlans.delete(groupId)
  const v: Viewing = { groupId, jobId: null, own, stopping: false, quit: new AbortController(), letGo: false }
  viewing = v
  if (own) tabStore.set(RUNNER_KEY, groupId)
  try {
    for (let i = 0; i < jobIds.length; i += 1) {
      const id = jobIds[i]
      const known = findJob(id)
      if (known?.status === 'skipped') break
      const plan = own?.get(id) ?? null
      const index = placeIn(known, 'index') ?? i + 1
      v.jobId = id
      // Each picture clears the last one's fault, as on this page. A batch
      // taken up rather than sent from here leaves two kinds standing. One is
      // the server's answer that it was already making a batch: cleared, the
      // reader's press seemed to have started someone else's. The other is
      // the fault the press ended on just now: the batch is taken up the
      // moment the press is free (see lookAgain), and would wipe the fault of
      // the reader's own picture before it was read. An older fault goes, or
      // it would read as the new batch's.
      const standing = i === 0 && !own && (freedJust || press.fault?.message === BATCH_BUSY) ? press.fault : null
      emit({
        fault: standing,
        job: {
          id,
          promptId: known?.promptId ?? null,
          status: 'submitting',
          stage: v.stopping ? STOPPING : known ? waitLine(known, DESK).stage : 'Sending the job',
          value: 0,
          max: 0,
          pct: 0,
          previewUrl: null,
          label: known?.label ?? plan?.label ?? '',
          index,
          total: placeIn(known, 'total') ?? jobIds.length,
          startedAt: known?.createdAt ?? Date.now(),
          finishedAt: null,
          ranAt: null,
          runner: true,
        },
      })
      syncFromStore()
      try {
        const out = await follow(id, (ev) => onServerEvent(v, id, ev), { words: { 'no-file': NO_FILE }, signal: v.quit.signal })
        // Let go meanwhile: the press may hold the page's own picture by now.
        if (v.letGo) return
        landFromServer(out, plan)
      } catch (err) {
        if (v.letGo) return
        const fault = faultOf(err)
        const ended = findJob(id)
        const finishedAt = ended?.endedAt ?? Date.now()
        if (ended?.error?.code === 'no-file' || fault.message === NO_FILE) {
          // As on this page, a picture that wrote no file is said and the
          // batch goes on to the next; the server does the same.
          emit({ fault: { message: NO_FILE, cancelled: false, lost: false, node: null, nodeType: null, detail: null } })
          patchJob({ status: 'error', finishedAt })
          continue
        }
        // Anything else ends the batch, as it does on this page, and the
        // server sends none of the rest. A stop needs no line saying so.
        const rest = fault.cancelled ? null : restLine(jobIds.slice(i + 1), index)
        emit({ fault, ...(rest ? { unsent: rest } : {}) })
        patchJob({ status: fault.cancelled ? 'cancelled' : 'error', finishedAt })
        break
      }
    }
  } finally {
    // One let go is not shown to its end: it is taken up again once the queue is back.
    if (!v.letGo) {
      walked.add(groupId)
      forgetKept(groupId)
    }
    if (viewing === v) {
      viewing = null
      lookAgain()
    }
  }
}

/** The pictures after the one that ended a batch, which the server never sent, in a line, or null. */
function restLine(later: readonly string[], endedAt: number): string | null {
  const snap = runnerStore.snapshot()
  const places: number[] = []
  later.forEach((id, k) => {
    const job = findJob(id, snap)
    // Still waiting is skipped a moment later: the server ends the batch and
    // skips the rest in one change, which may reach this page a job at a time.
    if (job && job.status !== 'skipped' && job.status !== 'waiting') return
    places.push(placeIn(job, 'index') ?? endedAt + 1 + k)
  })
  if (!places.length) return null
  return serverRestLine(Math.min(...places), Math.max(...places), endedAt)
}

/** A picture the server finished, on the plate. */
function landFromServer(out: FollowResult, plan: RunPlan | null): void {
  if (out.repeatOf) {
    // ComfyUI answered from its cache with a file already on record: that
    // record, with its own real time. No receipt: a lookup is not a run.
    showFiled(out.repeatOf, null)
  } else if (out.entryId && out.primary) {
    const primary = out.primary
    // Until the archive's pull brings the record the server filed, the same
    // record made here from the plan, under the server's id and number.
    const provisional: HistoryEntry | null = plan
      ? {
          ...recordOf(plan.composition, {
            file: primary,
            files: out.files.length > 1 ? out.files : undefined,
            kind: primary.kind,
            promptId: press.job?.promptId ?? '',
            durationMs: out.durationMs,
            seed: plan.seed,
            familyLabel: plan.familyLabel,
            modelLabel: plan.modelLabel,
            variant: plan.variant,
            passes: plan.passes,
            loras: plan.loras,
          }),
          id: out.entryId,
          no: out.entryNo ?? 0,
          at: out.finishedAt,
        }
      : null
    showFiled(out.entryId, provisional)
    // A picture whose run was not timed leaves the last receipt alone.
    if (out.durationMs > 0) emit({ lastMs: out.durationMs })
  }
  // With neither, it is done with nothing filed: the reader removed its
  // record while it was being filed, and a removed record is not brought back.
  patchJob({ status: 'done', pct: 1, stage: 'Done', finishedAt: out.finishedAt })
}

/** Put the record `id` on the plate: as filed when this page has it, else `provisional` until it is pulled. */
function showFiled(id: string, provisional: HistoryEntry | null): void {
  const filed = getRecord(id)
  const entry = filed ?? provisional
  if (!filed) unpulled.add(id)
  wantCurrent = entry ? null : id
  if (entry) emit({ results: [entry, ...press.results.filter((r) => r.id !== id)], current: entry })
}

/** The server's word on the picture on the press, as follow() passes it on. */
function onServerEvent(v: Viewing, jobId: string, ev: FollowEvent): void {
  const cur = press.job
  if (viewing !== v || !cur || cur.id !== jobId) return
  const stopping = v.stopping || findJob(jobId)?.stopRequested === true
  if (ev.phase === 'queued') {
    if (cur.status === 'submitting') patchJob({ promptId: ev.promptId, status: 'queued', stage: stopping ? STOPPING : 'Queued' })
    return
  }
  if (ev.phase === 'preview') {
    patchJob({ previewUrl: ev.url })
    return
  }
  const node = ev.node ? v.own?.get(jobId)?.graph[ev.node]?.class_type : null
  showRunning(ev.classType ?? node ?? null, ev.value, ev.max, stopping)
}

/**
 * The press brought into line with the server's list, for what follow()
 * does not report: the wait before a picture is sent (its turn, ComfyUI's own
 * queue, a held lane, a server with no room), a stop asked from another page,
 * and filing. It never moves a picture back: one ComfyUI has is not waiting
 * again, and the steps of one running are follow()'s to report, though a
 * stop the server did not take is taken back here at once rather than at the
 * next step, which on a long one can be minutes away.
 */
function syncFromStore(snap: RunnerSnapshot = runnerStore.snapshot()): void {
  const v = viewing
  const cur = press.job
  if (!v || !cur?.runner || cur.id !== v.jobId) return
  const job = findJob(cur.id, snap)
  if (!job) return
  const onHold = laneHoldOf(snap, job)
  if (!sameHold(cur.onHold ?? null, onHold)) patchJob({ onHold })
  const stopping = v.stopping || job.stopRequested
  switch (job.status) {
    case 'waiting':
    case 'releasing':
    case 'sending': {
      if (cur.status !== 'submitting') return
      const line = waitLine(job, DESK)
      const stage = stopping ? STOPPING : line.stage
      if (cur.stage !== stage || (cur.note ?? null) !== line.note) patchJob({ stage, note: line.note })
      return
    }
    case 'queued':
    case 'running': {
      const p = snap.progress[job.id]
      if (cur.status === 'running') {
        // follow() has reported a step, whichever of the two the list says.
        if (stopping) {
          if (cur.stage !== STOPPING) patchJob({ stage: STOPPING })
        } else if (cur.stage === STOPPING) {
          const node = p?.node ? v.own?.get(job.id)?.graph[p.node]?.class_type : null
          patchJob({ stage: stageFor(p?.classType ?? node ?? null, p?.value ?? cur.value, p?.max ?? cur.max) })
        }
        return
      }
      if (job.status === 'queued') {
        if (cur.status === 'submitting') {
          patchJob({ status: 'queued', promptId: job.promptId ?? cur.promptId, stage: stopping ? STOPPING : 'Queued' })
        } else if (cur.status === 'queued') {
          const stage = stopping ? STOPPING : 'Queued'
          if (cur.stage !== stage) patchJob({ stage })
        }
        return
      }
      if (job.promptId && cur.promptId !== job.promptId) patchJob({ promptId: job.promptId })
      const node = p?.node ? v.own?.get(job.id)?.graph[p.node]?.class_type : null
      showRunning(p?.classType ?? node ?? null, p?.value ?? 0, p?.max ?? 0, stopping)
      return
    }
    case 'filing': {
      const stage = waitLine(job, DESK).stage
      if (cur.status !== 'running' || cur.stage !== stage) patchJob({ status: 'running', stage })
      return
    }
    default:
      // Endings are follow()'s to report.
      return
  }
}

/** A job the server has not finished with. */
const UNFINISHED: ReadonlySet<RunnerJob['status']> = new Set(['waiting', 'releasing', 'sending', 'queued', 'running', 'filing'])

/**
 * This desk's batches the server keeps while its queue is not running, in the
 * line under the desk (serverParkedLine), or null. Not the one still being
 * handed over, which the press shows, nor one this page has shown to its end.
 */
function parkedLine(snap: RunnerSnapshot): string | null {
  if (snap.available) return null
  const lines: string[] = []
  for (const g of snap.groups) {
    if (g.desk !== DESK || g.state !== 'active' || walked.has(g.id) || g.id === handing?.groupId) continue
    const left = snap.jobs.filter((j) => j.groupId === g.id && UNFINISHED.has(j.status)).length
    if (left) lines.push(serverParkedLine(g.label, left, snap.reason, dropped.has(g.id)))
  }
  return lines.length ? lines.join(' ') : null
}

function showParked(snap: RunnerSnapshot = runnerStore.snapshot()): void {
  const parked = parkedLine(snap)
  if (parked !== press.parked) emit({ parked })
}

/** The press, free again, when it shows a picture of the server's that is not finished. */
function freePress(): void {
  if (press.job?.runner && busy(press)) emit({ job: null })
}

/**
 * The queue went off, or stood back for another server, with a batch on the
 * press. The server lists it as it was last saved and nothing there moves it
 * on, and a Stop is answered that the queue is not running, so a press that
 * followed it would hold every Make and every rerun until the queue came
 * back. It is let go: the press is free for the page's own work, and the
 * batch is said in the line under the desk (parkedLine), without a Stop.
 * A stop this page asked for that the server had not taken is asked again
 * once the queue is back (see dropped), and the batch is taken up again then,
 * with its plans when this page sent it.
 */
function letGoWhileOff(): void {
  const v = viewing
  if (!v) return
  viewing = null
  v.letGo = true
  v.quit.abort()
  if (v.own) parkedPlans.set(v.groupId, v.own)
  if (v.stopping) {
    dropped.add(v.groupId)
    keepDropped()
  }
  freePress()
}

/**
 * A batch the server has, as this page answers for it: on the press to its
 * end while the queue runs; while it does not, in the line under the desk,
 * with the press free (see letGoWhileOff).
 */
function takeUpServerBatch(groupId: string, jobIds: readonly string[], own: Map<string, RunPlan> | null): void {
  if (runnerStore.snapshot().available) {
    void driveRunner(groupId, jobIds, own)
    return
  }
  if (own) parkedPlans.set(groupId, own)
  freePress()
  showParked()
}

/**
 * Whenever the server's list changes: bring the press into line, settle a
 * batch the server had not answered for, and take up a batch the server is
 * sending when the press is free. There is one such batch at a time for the
 * whole server, and every device's desk shows it, with Stop.
 *
 * While the queue is not running, none is taken up and one on the press is
 * let go (letGoWhileOff): the batch is said in the line under the desk
 * instead, and the press is free.
 */
function onStore(): void {
  const snap = runnerStore.snapshot()
  // Only the first look after the press came free is that moment; a later one is any word of the server's.
  const freedJust = pressFreed
  pressFreed = false
  if (!snap.available) letGoWhileOff()
  syncFromStore(snap)
  showParked(snap)

  // Hand-overs from this tab the outbox gave up on: too old when a page came
  // back, or refused when it was made again. Each is said once.
  const given = givenUpBatches().filter((g) => g.desk === DESK)
  if (given.length) {
    const mine = handing?.pending ? given.find((g) => g.groupId === handing?.groupId) : undefined
    if (mine) {
      handing = null
      patchJob({ status: 'error', finishedAt: Date.now() })
    }
    for (const g of given) forgetKept(g.groupId)
    emit({ unsent: given.map((g) => `${g.label}: ${g.line}`).join(' ') })
    // After this change has been told to every subscriber, not in the middle of it.
    queueMicrotask(() => given.forEach((g) => forgetGivenUp(g.groupId)))
  }

  for (const g of snap.groups) {
    if (!dropped.has(g.id)) continue
    if (g.state !== 'active') {
      dropped.delete(g.id)
      keepDropped()
    } else if (snap.available) {
      // Kept until the server says the batch has ended, so a stop it did not
      // take is asked again at its next word. A queue that is not running
      // answers that it is not, so the stop waits for it to be back.
      askStop(g.id)
    }
  }

  const h = handing
  if (h) {
    // It reached the server after all, sent again by the outbox, or with only
    // the answer lost.
    if (h.pending && snap.groups.some((g) => g.id === h.groupId)) {
      handing = null
      takeUpServerBatch(h.groupId, h.body.jobs.map((j) => j.id), h.own)
    }
    return
  }

  // Nothing is taken up from a queue that is not running (see parkedLine),
  // nor is its list taken at its word about what the server has not got: a
  // server that cannot read its saved list lists nothing.
  if (!snap.available) return
  if (driving || viewing) return
  const kept = keptGroup()
  const take =
    snap.groups.find((g) => g.desk === DESK && g.state === 'active' && !walked.has(g.id) && !dropped.has(g.id)) ??
    (kept ? snap.groups.find((g) => g.id === kept && !walked.has(g.id)) : undefined)
  if (take) {
    void driveRunner(take.id, take.jobIds, null, freedJust)
    return
  }
  // The server has answered and has not got it, nor is the outbox still
  // handing it over: there is nothing to show.
  if (
    kept &&
    (snap.connected || snap.boot) &&
    !snap.groups.some((g) => g.id === kept) &&
    !outboxPending().some((e) => e.groupId === kept)
  ) {
    tabStore.remove(RUNNER_KEY)
  }
  // The plans of a batch let go while the queue was off, once the server has it going no longer.
  for (const id of [...parkedPlans.keys()]) {
    if (!snap.groups.some((g) => g.id === id && g.state === 'active')) parkedPlans.delete(id)
  }
}

/**
 * The press is free again, so the server's list is looked at now rather than
 * at its next word. A batch the server lists while the press is busy is not
 * taken up (see onStore), and one held on its lane, or waiting on a queue that
 * came back while the page made its own pictures, may bring no further word
 * for hours: the desk said nothing of it meanwhile, and the next Make was
 * refused as busy. It runs once the caller is done, so the caller's last
 * change to the press (a picture marked done, stopped or refused) lands on
 * its own picture, not on the batch taken up after it.
 */
function lookAgain(): void {
  pressFreed = true
  queueMicrotask(onStoreSafely)
}

/** Set by lookAgain until the next look at the server's list (see driveRunner's fault). */
let pressFreed = false

/** A change that lands while one is being taken is taken after it, never inside it. */
let inStore = false
let storeAgain = false

function onStoreSafely(): void {
  if (inStore) {
    storeAgain = true
    return
  }
  inStore = true
  try {
    do {
      storeAgain = false
      try {
        onStore()
      } catch {
        /* the desk's own path is untouched either way */
      }
    } while (storeAgain)
  } finally {
    inStore = false
  }
}

// A record the server filed reaches this page with the archive's pull. It
// replaces the one made here in the meantime, and goes on the plate if the
// press was waiting for it.
subscribeRecords(() => {
  if (!unpulled.size) return
  let { results, current } = press
  let changed = false
  for (const id of [...unpulled]) {
    const record = getRecord(id)
    if (!record) continue
    unpulled.delete(id)
    changed = true
    results = results.some((r) => r.id === id) ? results.map((r) => (r.id === id ? record : r)) : [record, ...results]
    if (current?.id === id || wantCurrent === id) current = record
    if (wantCurrent === id) wantCurrent = null
  }
  if (changed) emit({ results, current })
})

// Last in the engine, so everything it calls is defined.
restoreSent()

// A batch the server is sending is taken up now, and whenever the server's
// word changes. App starts the store; until it has answered there is nothing
// to take up.
runnerStore.subscribe(onStoreSafely)
onStoreSafely()

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
 * Kept for this tab, in its session storage, never in localStorage. An
 * override is a statement about this session's picture (see useOverrides), so
 * another tab and a later session start clean. A reload of this tab does not:
 * a phone reloads a tab it put in the background, and "Use these settings"
 * then kept the record's prompt in the field while the pin, the values set by
 * hand, the seed and the notice saying so all went, so the next press made a
 * different picture from a desk that still looked loaded.
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

const HELD_KEY = 'switchgen.pictures.held.v1'

const INTENT_IDS = new Set<string>(INTENTS.map((i) => i.id))
const ANATOMY_IDS = new Set<string>(ANATOMY_LEVELS.map((a) => a.id))

/**
 * What the desk was set to, read back from the tab's storage. Each field is
 * checked, since whatever wrote it may be an older build: a look, a detail
 * setting, a mode or a seed that cannot be read discards the whole of it, and
 * the desk starts as it would have; a bad pin, notice or override is dropped
 * on its own.
 */
export function parseHeld(raw: string | null): Held | null {
  let v: unknown
  try {
    v = JSON.parse(raw ?? 'null')
  } catch {
    return null
  }
  if (!v || typeof v !== 'object' || Array.isArray(v)) return null
  const h = v as Record<string, unknown>
  if (typeof h.look !== 'string' || !INTENT_IDS.has(h.look)) return null
  if (typeof h.anatomy !== 'string' || !ANATOMY_IDS.has(h.anatomy)) return null
  if (typeof h.mode !== 'string' || !MODES.images.includes(h.mode as Mode)) return null
  if (typeof h.seed0 !== 'number' || !Number.isFinite(h.seed0) || h.seed0 < 0) return null
  return {
    look: h.look as Intent,
    anatomy: h.anatomy as AnatomyLevel,
    anatomySaid: h.anatomySaid === true,
    pinned: typeof h.pinned === 'string' && h.pinned ? h.pinned : null,
    overrides: sanitiseOverrides(h.overrides),
    seed0: Math.floor(h.seed0),
    correction: typeof h.correction === 'string' && h.correction ? h.correction : null,
    mode: h.mode as Mode,
  }
}

let held: Held | null = parseHeld(tabStore.get(HELD_KEY))

/**
 * The region bench as the reader left it, for the next visit to the desk.
 *
 * Memory only: the bench is the picture in front of the reader, and a mask is
 * too large and too particular to keep past the page. App mounts one room at
 * a time, and a look at the Archive or at a clip on the Video desk closed the
 * bench and threw away the model choice, the region add-ons and the pass on
 * its way, so a pass that landed meanwhile came back as an ordinary picture
 * with no comparison. The strokes and the region words are kept by the bench
 * itself (components/refine) and forgotten with this when the bench closes.
 */
type Bench = {
  source: RefineSource
  result: HistoryEntry | null
  pick: string | null
  picked: string[]
  /** The press job the region pass was queued as, and whether its result is still awaited. */
  job: string | null
  awaiting: boolean
}

let bench: Bench | null = null

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
 * A finished picture, named for LoadImage and measured, ready for a refine
 * pass. Both halves are needed before anything can be queued: the graph needs
 * the file, and the crop arithmetic needs real pixels.
 */
type RefineSource = {
  /**
   * The file as LoadImage takes it: an output by its annotated path, read
   * where it lies, or an upload by its name in the input folder.
   */
  name: string
  /** The output file, when the picture is one of ours, so its record can point back at it. */
  ref?: FileRef
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
        face: detailSentence('face'),
        hand: `${detailSentence('hand')} It is given more freedom than a face. Not measured here.`,
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

  // Kept for the next visit, and for a reload of this tab. See `held`.
  useEffect(() => {
    held = { look, anatomy, anatomySaid, pinned, overrides: ov.value, seed0, correction, mode: c.mode }
    tabStore.set(HELD_KEY, JSON.stringify(held))
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

  // The bench comes back as it was left on the last visit (see `bench`).
  const [refining, setRefining] = useState(() => bench !== null)
  const [refineSource, setRefineSource] = useState<RefineSource | null>(() => bench?.source ?? null)
  const [refineResult, setRefineResult] = useState<HistoryEntry | null>(() => bench?.result ?? null)
  const [refineFault, setRefineFault] = useState<string | null>(null)
  const [openingRefine, setOpeningRefine] = useState(false)
  const refineToken = useRef(0)
  /**
   * True between queueing a refine and its result landing on the plate. Back
   * from another room, it is still awaited only while the press holds the job
   * it was queued as; the landing effect below then settles it either way.
   */
  const awaitingRefine = useRef(
    !!bench?.awaiting && !!bench.job && pressSnapshot().job?.id === bench.job,
  )
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
  const refineJob = useRef<string | null>(bench?.job ?? null)
  /** The region add-ons the reader ticked on the bench, by filename. None until they do. */
  const [refinePicked, setRefinePicked] = useState<string[]>(() => bench?.picked ?? [])

  const fileInput = useRef<HTMLInputElement | null>(null)
  const promptRef = useRef<HTMLTextAreaElement | null>(null)
  const dragDepth = useRef(0)

  const running = busy(state)
  const serverLine = serverWaitingLine(state.job)

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
  useEffect(() => watchAhead(setAhead), [])

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

  /**
   * The weight files the ranking may choose from: every file ComfyUI lists,
   * less those of a picture family that cannot load here for a missing file.
   *
   * decide() was handed every listed file, and ranks on memory alone, so a
   * family whose text encoder or VAE ComfyUI does not list was picked, counted
   * as runnable and queued, and ComfyUI refused the job over the missing file,
   * while the catalogue had already worked out the plain "needs" sentence and
   * shown it nowhere. That sentence is now in More instead (see unloadable). A
   * file held back only for memory stays in: the ranking judges memory again
   * on each visit, against a fresher reading than the catalogue's.
   */
  const cannotLoad = useMemo(
    () => new Set((cat?.unavailable ?? []).filter((u) => u.files).map((u) => u.name)),
    [cat],
  )
  const rankable = useMemo(
    () => (cat ? cat.installed.filter((m) => !cannotLoad.has(m)) : []),
    [cat, cannotLoad],
  )
  /** The pinned file cannot load here, or is not installed at all: said, and set aside. */
  const pinUnloadable = !!pinned && !!cat && !rankable.includes(pinned)

  /** Said for as long as the pin is set aside, and gone the moment it is not. */
  const pinNote = pinUnloadable && pinned
    ? `${PLAIN_NAMES[pinned] ?? titleFromFilename(pinned)} is pinned, but ${
        pinBlockedWhy(pinned, (cat?.unavailable ?? []).filter((u) => u.files))
      }, so the desk chooses another model until it can load.`
    : pinSetAside && pinned
      ? `${PLAIN_NAMES[pinned] ?? titleFromFilename(pinned)} is pinned, and it cannot work from a picture, so the desk chooses another model while a picture is in the well. Choose From words and the pin is used again.`
      : null

  /** The files More lists as unable to load here, for the mode the desk is on. */
  const unloadable: Unloadable[] = useMemo(() => {
    const want = c.mode === 'edit' ? 'edit' : 'image'
    return (cat?.unavailable ?? [])
      .filter((u) => u.files && familyOwning(u.name)?.mode === want)
      .map((u) => ({ model: u.name, label: PLAIN_NAMES[u.name] ?? titleFromFilename(u.name), why: u.why }))
  }, [cat, c.mode])

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
      installed: pinned && !pinSetAside && !pinUnloadable ? [pinned] : rankable,
      loras: lib,
      seed: seed0,
      // The reader's own decisions, threaded in so they survive this recompute.
      // decide() runs on every keystroke; a decision held anywhere but here
      // would be silently overwritten by the next suggestion.
      addOns: { accepted: c.addOnsAccepted, declined: c.addOnsDeclined },
      passBlocks: cat.passBlocks,
    })
  }, [cat, c.mode, c.prompt, look, anatomy, usingSource, sourceName, pinned, pinSetAside, pinUnloadable, rankable,
      lib, seed0, editStyle, c.addOnsAccepted, c.addOnsDeclined, hardware])

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

  // A picture adopted from the archive lives in the OUTPUT folder, and
  // LoadImage reads it there by its annotated path ("x.png [output]"), as the
  // reel's key frames already do. It used to be fetched to the browser and
  // uploaded back into the input folder: over a phone's uplink that was
  // seconds each way for every pick, with the button waiting, and a copy left
  // behind in the input folder each time.
  useEffect(() => {
    const s = c.source
    if (!s || s.name || !s.ref) return
    // LoadImage reads a still, and a draft saved by an older build can hold a
    // clip here. Sent on, the run could only fail on it. The mode is left
    // alone so the well stays on screen to say why it is empty.
    if (VIDEO_EXT.test(s.ref.filename)) {
      store.patch({ source: null })
      setSourceError('It is a clip, and this desk starts only from a still. Choose a picture instead.')
      return
    }
    // A picture chosen after a refusal is a fresh start; the old notice goes.
    setSourceError(null)
    store.patch({ source: { ...s, name: annotatedRef(s.ref), previewUrl: s.previewUrl ?? fileUrl(s.ref) } })
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
    void startRuns(plans)
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
  // With nothing made or chosen since the page loaded, it shows the newest
  // picture on file, with its offers (see plateFallback).
  const current = useMemo(() => {
    const cur = state.current
    if (!cur) return plateFallback(records)
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

  /**
   * The attached picture, read on request. One element, handed to whichever
   * desk is up. Read where the file is: an upload in the input folder, one of
   * our outputs where it lies.
   */
  const sourceAt = c.source?.name ? sourceFile(c.source) : null
  const sourceKind = sourceAt?.type === 'output' ? 'output' : 'input'
  const sourceReading = sourceAt ? (
    <Reading
      compact
      source={{ kind: sourceKind, rel: relPath(sourceAt) }}
      cacheKey={`${sourceKind}:${relPath(sourceAt)}`}
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
      void startRuns([
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
  const [refinePick, setRefinePick] = useState<string | null>(() => bench?.pick ?? null)

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
    // The "Drawn by" choice and the ticked region add-ons were made for the
    // picture the bench was on, as closeRefine says. The bench now comes back
    // after a look at another room with them in place, so a picture handed
    // over from the Archive was drawn by the model picked for the last one.
    // A pass carried on from its own result loses nothing: that result files
    // the model that drew it, which is then the suggested one.
    setRefinePick(null)
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
      if (refineToken.current !== token) return
      setRefineSource({
        // Read where it lies, as a picture taken from the archive is (see the
        // source effect above). It used to be downloaded and uploaded back on
        // every opening of the bench.
        name: annotatedRef(entry.file),
        ref: entry.file,
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
      setRefineFault(`That picture could not be opened for region editing (${why}).`)
    } finally {
      if (refineToken.current === token) setOpeningRefine(false)
    }
  }, [])

  /**
   * Open the bench on the picture currently attached to the desk.
   *
   * An attached source is already named for LoadImage by the time the well
   * shows it, an upload by its input-folder name and an archive picture by its
   * annotated path, so `source.name` is all the graph needs and there is
   * nothing to upload again.
   */
  const openRefineFromSource = useCallback(async (src: SourceRef, words?: string) => {
    if (!src.name) return
    const token = (refineToken.current += 1)
    awaitingRefine.current = false
    // A new picture starts from the suggested model and no add-ons (see openRefine).
    setRefinePick(null)
    setRefinePicked([])
    setRefining(true)
    setRefineSource(null)
    setRefineResult(null)
    setRefineFault(null)
    setOpeningRefine(true)
    // Paint on the file the graph will read, not on the well's preview.
    //
    // sourceFile resolves the LoadImage name to that file, so this URL serves
    // the exact bytes the graph will read, and it is a stable server URL. The
    // well's previewUrl can be a blob: object URL owned by the desk store,
    // which Clear, paste and drop all revoke - out from under the bench, which
    // does not own it.
    const url = fileUrl(sourceFile(src) ?? { filename: src.name, subfolder: '', type: 'input' })
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
        ref: src.ref,
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
    // The ticked region add-ons are the same kind of choice, and so are the
    // strokes and words the bench kept for the next visit.
    setRefinePick(null)
    setRefinePicked([])
    forgetBench()
    bench = null
  }, [])

  // A record the Archive asked to see on the plate, with the passes the plate
  // offers. The same one-shot handover; the bench, if it was left open, gives
  // way to the plate the reader asked for.
  useEffect(() => {
    const handed = takePlateRequest()
    if (!handed) return
    closeRefine()
    showResult(handed)
  }, [closeRefine])

  // Kept for the next visit, whatever changed: the refs move with the press
  // job, so the press state is a dependency too. See `bench`.
  useEffect(() => {
    bench =
      refining && refineSource
        ? {
            source: refineSource,
            result: refineResult,
            pick: refinePick,
            picked: refinePicked,
            job: refineJob.current,
            awaiting: awaitingRefine.current,
          }
        : null
  }, [refining, refineSource, refineResult, refinePick, refinePicked, state])

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
            ref: refineSource.ref,
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
        const mine = await startRuns([
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
        // startRuns names the job the pass is shown under: at once when this
        // page sends it, and once the server has answered when the server
        // does, by which time the press may hold something else. It names
        // none while another run is going (one may have started during the
        // mask upload), and then nothing is followed. Nor is a pass whose
        // bench was closed or moved to another picture meanwhile.
        if (mine && refineToken.current === token) {
          refineJob.current = mine
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
        unloadable={unloadable}
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

        {/* Pictures an earlier page sent, which this page will not follow on a guess */}
        {state.left.length ? <LeftSent left={state.left} running={running} /> : null}

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
            job={buttonJob(state.job)}
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
            job={buttonJob(state.job)}
            queuedAhead={ahead}
            lastMs={state.lastMs}
            promptRef={promptRef}
            reducedMotion={reduced}
            advanced={advanced}
            moreOpen={expert}
            onMoreOpenChange={(open) => settings.patch({ expert: open })}
          />
        )}

        {serverLine && <p className="mt-5 text-caption leading-snug text-grey-700">{serverLine}</p>}

        {/* A batch the server keeps while its queue is off: said, never followed, and with no Stop (see parkedLine). */}
        {state.parked && <p className="mt-5 text-caption leading-snug text-grey-700">{state.parked}</p>}

        {state.fellBack && <p className="mt-5 text-caption italic leading-snug text-grey-500">{state.fellBack}</p>}

        {state.unsent && (
          <div className="mt-5">
            <Notice tone="warning" title="Not sent">
              {state.unsent} <Link onClick={() => emit({ unsent: null })}>Dismiss</Link>
            </Notice>
          </div>
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
  job: RunJob | null
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
 * Pictures the page before this one sent and went without handing on, put to
 * the reader. Following one a tab copied from this one is still following
 * would file it twice, and only the reader knows whether that tab is open.
 */
function LeftSent({ left, running }: { left: SentPicture[]; running: boolean }) {
  const one = left.length === 1
  const it = one ? 'it' : 'them'
  const button = 'inline-flex items-center underline [@media(pointer:coarse)]:min-h-11'
  return (
    <div className="mb-4">
      <Notice tone="warning" title={one ? 'A picture sent before this page' : 'Pictures sent before this page'}>
        {one ? 'One picture was' : `${left.length} pictures were`} sent to ComfyUI from this tab by the page before
        this one, which went away without handing {it} on, as happens when the phone closes a tab in the background, a
        page crashes or a tab is copied.{' '}
        {left.some((j) => j.sending)
          ? `That page was still sending ${one ? 'it' : 'one of them'} when it went, so ComfyUI may never have had ${one ? 'it' : 'that one'}. `
          : ''}
        If this tab was copied from one that is still open, that one is following {it} already, and following from here
        as well could file {it} twice.
        <ul className="my-1 list-none p-0">
          {left.map((j) => (
            <li key={j.id} className="truncate italic">
              {j.composition.prompt || 'No words'} · {j.modelLabel || j.familyLabel}
            </li>
          ))}
        </ul>
        {running ? (
          <>Once the picture on the press is done, {it} can be followed from here. </>
        ) : (
          <>
            <button className={button} onClick={followLeftSent}>
              Follow {it} from here
            </button>{' '}
            ·{' '}
          </>
        )}
        <button className={button} onClick={forgetLeftSent}>
          Forget {it}
        </button>{' '}
        Forgetting does not stop ComfyUI making {it}; the Archive’s “Look for files with no record” files{' '}
        {one ? 'it once it has' : 'them once they have'} landed.
      </Notice>
    </div>
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
function Fault({ fault, onDismiss }: { fault: PressFault; onDismiss: () => void }) {
  const title =
    fault.title ??
    (fault.message === NO_FILE ? 'No picture came back' : fault.message === NOT_SENT ? 'The picture was not sent' : faultTitle(fault))
  const tone = fault.tone ?? (fault.cancelled ? 'correction' : fault.lost ? 'warning' : 'error')
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
          Nothing yet. Press Make the picture.
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
        Type a line and press Make the picture. Everything you make is filed in the archive with
        the settings that made it, so you can find it again and change one word.
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
        Made by {entry.modelLabel} · {DATE.format(entry.at)}
        {/* No time is printed for a run nobody timed: 0.0 s would read as one. */}
        {entry.durationMs > 0 ? (
          <>
            {' '}
            · <span className="tabular-nums">{seconds(entry.durationMs)}</span>
          </>
        ) : null}
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
