/**
 * Session state: what each desk is composing, what the reader has chosen, and
 * how an archive record becomes the seed of a new generation.
 *
 * Three things live here.
 *
 *   1. {@link Composition} — everything a desk holds while you write. The
 *      Pictures desk and the Video desk each own one, independently and
 *      persistently, so switching sections mid-thought never costs you a
 *      half-written prompt.
 *   2. {@link settings} — the expert flag and the small display preferences,
 *      global across desks and persisted.
 *   3. {@link reuseIntoDesk} — the bridge from a history record back to a
 *      composition, with an undo and a plain-English list of anything that
 *      could not be carried across.
 *
 * Nothing here imports the registry or the workflow builder. Desks pass a
 * family's defaults in; this module only holds and persists the state. That
 * keeps the store testable and stops a registry change rippling into storage.
 */

import type { FileRef } from './comfy'
import type { HistoryEntry, NewEntry } from './history'

export type { FileRef }

// ---------------------------------------------------------------------------
// Storage that never throws
// ---------------------------------------------------------------------------

/**
 * localStorage access that degrades instead of failing.
 *
 * `localStorage` throws on the *getter itself* in a locked-down or private
 * window, so every access — read, write and the availability probe — is
 * wrapped. When it is unavailable the app keeps working, in memory, for the
 * lifetime of the tab.
 */
const memory = new Map<string, string>()
/**
 * Keys whose last write could only be held in memory. Without this a write that
 * localStorage refused for a reason other than quota would leave `get` reading
 * the older value straight back out of localStorage, so the tab would show a
 * setting the reader had just changed away from.
 */
const memoryOnly = new Set<string>()
let persistent: boolean | null = null

/** True when writes survive a reload. False means we are running in memory. */
export function storageWorks(): boolean {
  if (persistent !== null) return persistent
  try {
    const probe = '__switchgen_probe__'
    localStorage.setItem(probe, '1')
    localStorage.removeItem(probe)
    persistent = true
  } catch {
    persistent = false
  }
  return persistent
}

export const store = {
  get(key: string): string | null {
    if (memoryOnly.has(key) || !storageWorks()) return memory.get(key) ?? null
    try {
      return localStorage.getItem(key) ?? memory.get(key) ?? null
    } catch {
      return memory.get(key) ?? null
    }
  },

  /** @returns false when the value could only be held in memory. */
  set(key: string, value: string): boolean {
    memory.set(key, value)
    if (!storageWorks()) {
      memoryOnly.add(key)
      return false
    }
    try {
      localStorage.setItem(key, value)
      memoryOnly.delete(key)
      return true
    } catch (err) {
      // The newest value is in memory either way, and memory is now the only
      // place it is right, so reads must come from there until a write lands.
      memoryOnly.add(key)
      // Quota is the caller's problem to handle: it knows what it can shed.
      if (isQuotaError(err)) throw err
      return false
    }
  },

  remove(key: string): void {
    memory.delete(key)
    memoryOnly.delete(key)
    if (!storageWorks()) return
    try {
      localStorage.removeItem(key)
    } catch {
      /* nothing useful to do */
    }
  },
}

/** True for the several spellings browsers use for a full quota. */
export function isQuotaError(err: unknown): boolean {
  if (!err || typeof err !== 'object') return false
  const e = err as { name?: string; code?: number }
  return (
    e.name === 'QuotaExceededError' ||
    e.name === 'NS_ERROR_DOM_QUOTA_REACHED' ||
    e.code === 22 ||
    e.code === 1014
  )
}

/** Watch one key for writes made by another tab. Returns an unsubscribe. */
export function onStorage(key: string, fn: (value: string | null) => void): () => void {
  // Both stores below subscribe at module scope, so importing this file from a
  // script or a test must not depend on there being a window.
  if (typeof window === 'undefined') return () => {}
  const handler = (e: StorageEvent) => {
    if (e.key === null || e.key === key) fn(e.newValue)
  }
  window.addEventListener('storage', handler)
  return () => window.removeEventListener('storage', handler)
}

/**
 * Read one stored object. Everything stored here is a plain object, so an array
 * or a bare number is as unusable as unparseable text and takes the fallback.
 * The result is still untrusted: callers check it field by field.
 */
function readJson<T>(key: string, fallback: T): T {
  const raw = store.get(key)
  if (!raw) return fallback
  try {
    const parsed = JSON.parse(raw)
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? (parsed as T) : fallback
  } catch {
    return fallback
  }
}

const isFinite_ = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)
/**
 * A number, or null when the value cannot be read as one. A numeric string
 * counts: a draft written by an older build should restore the size the reader
 * chose, not silently fall back to the blank desk's.
 */
const asNumberOrNull = (v: unknown): number | null => {
  if (typeof v === 'number') return Number.isFinite(v) ? v : null
  if (typeof v === 'string' && v.trim() !== '') {
    const n = Number(v)
    return Number.isFinite(n) ? n : null
  }
  return null
}
const asNumber = (v: unknown, fallback: number): number => asNumberOrNull(v) ?? fallback
const asString = (v: unknown, fallback: string): string => (typeof v === 'string' ? v : fallback)
const asStringOrNull = (v: unknown): string | null => (typeof v === 'string' ? v : null)
/** Filenames off disk, so anything that is not a string is dropped rather than trusted. */
const asStringList = (v: unknown): string[] =>
  Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string') : []

const asBoolean = (v: unknown, fallback: boolean): boolean =>
  typeof v === 'boolean' ? v : fallback

// ---------------------------------------------------------------------------
// Vocabulary
// ---------------------------------------------------------------------------

/**
 * The two desks. The Pictures desk's route is `#/pictures`; its stored id is
 * `images`, which is also what a history record carries.
 */
export type DeskId = 'images' | 'video'

/** How a generation was made. The desk owns the modes; the mode owns the UI. */
export type Mode = 't2i' | 'i2i' | 'edit' | 't2v' | 'i2v'

export const MODES: Record<DeskId, readonly Mode[]> = {
  images: ['t2i', 'i2i', 'edit'],
  video: ['t2v', 'i2v'],
}

/** Which desk a mode belongs to. A video record can never land in Pictures. */
export function deskOf(mode: Mode): DeskId {
  return mode === 't2v' || mode === 'i2v' ? 'video' : 'images'
}

/** True when the mode needs a source picture before it can run. */
export function needsSource(mode: Mode): boolean {
  return mode === 'i2i' || mode === 'edit' || mode === 'i2v'
}

/** Plain labels, for the source tabs and for the archive's facets. */
export const MODE_LABEL: Record<Mode, string> = {
  t2i: 'From words',
  i2i: 'From a picture',
  edit: 'Change a picture',
  t2v: 'From words',
  i2v: 'From a picture',
}

/**
 * A picture standing by as the source of the next generation.
 *
 * `name` is what goes into the graph's LoadImage — the file as ComfyUI's input
 * folder knows it, returned by `uploadImage()`. `ref` is set instead when the
 * picture is one of our own outputs being reused, in which case nothing needs
 * uploading again.
 */
export type SourceRef = {
  /** ComfyUI input filename for LoadImage. Empty until the upload lands. */
  name: string
  /** Set when the source is an existing ComfyUI output rather than an upload. */
  ref?: FileRef
  /** Where to show it in the well. An object URL or a /comfy/view URL. */
  previewUrl?: string
  /** For the well's caption: `street.png · 1216 × 832 · 1.2 MB`. */
  label?: string
  width?: number
  height?: number
  bytes?: number
  /** Set when the source came from the archive, for the lineage link. */
  fromEntryId?: string
  /** Set when the source is a frame lifted out of a clip. */
  fromFrame?: number
}

// ---------------------------------------------------------------------------
// Composition
// ---------------------------------------------------------------------------

/**
 * Everything one desk holds. Simple mode shows five of these fields; expert
 * mode shows the rest, already carrying the values simple mode was using.
 *
 * `null` means "follow the family default", and is not the same as a value the
 * user chose. That distinction is what lets the style picker re-apply a new
 * model's recipe without trampling a deliberate override.
 */
export type Composition = {
  desk: DeskId
  mode: Mode

  /** Base family id — never a derived `__img2img` / `__i2v` suffix. */
  familyId: string
  /** The weight file. Empty on a dual-model family, which carries its own. */
  model: string

  prompt: string
  /** null => use the family's house negative. */
  negative: string | null
  /** Prepended to the prompt at submit time, from the model's author card. */
  positivePrefix: string | null

  source: SourceRef | null

  width: number
  height: number
  /** Output budget for image-to-image, where the source dictates the shape. */
  megapixels: number | null
  /** Image-to-image strength. 0.25 touch up … 0.85 start over. */
  denoise: number | null

  seed: number
  /** True once the reader pins a seed; false means a fresh one every run. */
  seedLocked: boolean
  steps: number
  cfg: number
  sampler: string
  scheduler: string

  /** Video only. Frames, constrained to 4n+1, and the clip's frame rate. */
  length: number | null
  fps: number | null

  /**
   * Dual-model pass boundary. null => round(steps / 2), which is what
   * `instantiate` writes and what every registry preset hardcodes.
   */
  split: number | null
  /** ModelSampling* shift. */
  shift: number | null
  /** CLIPSetLastLayer, e.g. -2 on Illustrious. */
  clipSkip: number | null
  /** Strip the Lightning LoRAs and run the full-step recipe. */
  noLora: boolean

  /** How many sequential runs the button queues, with successive seeds. */
  runs: 1 | 2 | 4

  /**
   * Add-ons the reader has decided about, by filename. The recipe matches
   * add-ons to the wording and OFFERS them; nothing is applied until it appears
   * in `addOnsAccepted`, and anything in `addOnsDeclined` is never offered
   * again. Kept here rather than inside the recipe because decide() is pure and
   * runs again on every keystroke: a decision held anywhere else would be
   * recomputed away, which is the one thing a suggestion must never do.
   */
  addOnsAccepted: string[]
  addOnsDeclined: string[]

  /**
   * Fields the reader has set by hand. Everything else follows the family's
   * defaults and is re-derived when the style changes.
   */
  touched: TunableField[]
}

/**
 * Fields a reader can set by hand, and which `applyDefaults` will respect.
 *
 * The runtime list is the definition and the type is derived from it, so the
 * loader can check a stored `touched` array against the same thing the type
 * system checks against, and the two can never drift apart.
 */
export const TUNABLE_FIELDS = [
  'negative',
  'positivePrefix',
  'width',
  'height',
  'megapixels',
  'denoise',
  'seed',
  'steps',
  'cfg',
  'sampler',
  'scheduler',
  'length',
  'fps',
  'split',
  'shift',
  'clipSkip',
] as const

export type TunableField = (typeof TUNABLE_FIELDS)[number]

const IS_TUNABLE = new Set<string>(TUNABLE_FIELDS)

/** A family's effective recipe, as the desk reads it from the registry. */
export type FamilyDefaults = {
  familyId: string
  model: string
  steps: number
  cfg: number
  width: number
  height: number
  sampler: string
  scheduler: string
  length?: number
  fps?: number
  negative?: string
  clipSkip?: number
  positivePrefix?: string
}

/** A seed ComfyUI will accept: a non-negative integer well inside 2^53. */
export function randomSeed(): number {
  return Math.floor(Math.random() * 2 ** 48)
}

/** A blank desk. Values are placeholders until `applyDefaults` runs. */
export function newComposition(desk: DeskId, patch: Partial<Composition> = {}): Composition {
  const base: Composition = {
    desk,
    mode: desk === 'video' ? 't2v' : 't2i',
    familyId: '',
    model: '',
    prompt: '',
    negative: null,
    positivePrefix: null,
    source: null,
    width: desk === 'video' ? 832 : 1024,
    height: desk === 'video' ? 480 : 1024,
    megapixels: null,
    denoise: null,
    seed: randomSeed(),
    seedLocked: false,
    steps: 20,
    cfg: 4,
    sampler: 'euler',
    scheduler: 'simple',
    length: desk === 'video' ? 81 : null,
    fps: desk === 'video' ? 16 : null,
    split: null,
    shift: null,
    clipSkip: null,
    noLora: false,
    runs: 1,
    addOnsAccepted: [],
    addOnsDeclined: [],
    touched: [],
  }
  return { ...base, ...patch, desk }
}

/**
 * Apply a family's defaults over a composition, leaving every field the reader
 * has touched exactly as they left it.
 *
 * Call this whenever the style changes. It is what makes simple mode run the
 * registry's verified recipe while still honouring a deliberate override.
 */
export function applyDefaults(c: Composition, d: FamilyDefaults): Composition {
  const keep = new Set(c.touched)
  const take = <T>(field: TunableField, value: T, current: T): T =>
    keep.has(field) ? current : value

  return {
    ...c,
    familyId: d.familyId,
    model: d.model,
    steps: take('steps', d.steps, c.steps),
    cfg: take('cfg', d.cfg, c.cfg),
    width: take('width', d.width, c.width),
    height: take('height', d.height, c.height),
    sampler: take('sampler', d.sampler, c.sampler),
    scheduler: take('scheduler', d.scheduler, c.scheduler),
    length: take('length', d.length ?? c.length, c.length),
    fps: take('fps', d.fps ?? c.fps, c.fps),
    negative: take('negative', null, c.negative),
    clipSkip: take('clipSkip', d.clipSkip ?? null, c.clipSkip),
    positivePrefix: take('positivePrefix', d.positivePrefix ?? null, c.positivePrefix),
  }
}

/** Record that the reader set a field by hand. Idempotent. */
export function markTouched(c: Composition, ...fields: TunableField[]): Composition {
  const next = new Set(c.touched)
  for (const f of fields) next.add(f)
  return next.size === c.touched.length ? c : { ...c, touched: [...next] }
}

/** Forget a hand-set field, so the family default takes over again. */
export function clearTouched(c: Composition, ...fields: TunableField[]): Composition {
  const drop = new Set<string>(fields)
  const touched = c.touched.filter((f) => !drop.has(f))
  return touched.length === c.touched.length ? c : { ...c, touched }
}

/**
 * Parameters for `instantiate()`.
 *
 * Structurally the `Params` type in `src/lib/workflows.ts`, restated so this
 * module does not depend on the workflow builder.
 *
 * `split`, `shift` and `clipSkip` have no binding in the generated registry, so
 * `instantiate` writes them by finding the node that owns each one. They are
 * carried here for the same reason every other field is: what the composer
 * collects has to reach the graph, or the archive files a setting that never
 * ran.
 */
export type CompositionParams = {
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
  image?: string
  denoise?: number
  megapixels?: number
  split?: number
  shift?: number
  clipSkip?: number
}

/**
 * Turn a composition into the parameters a graph is instantiated with.
 *
 * @param c         the desk's state.
 * @param fallback  the family's house negative, used when the reader has not
 *                  written one. Passing it here keeps the registry out of this
 *                  module.
 */
export function toParams(
  c: Composition,
  fallback: { negative?: string; split?: number } = {},
): CompositionParams {
  const prefix = c.positivePrefix ?? ''
  const p: CompositionParams = {
    model: c.model,
    positive: prefix ? `${prefix}${c.prompt}` : c.prompt,
    negative: c.negative ?? fallback.negative ?? '',
    seed: Math.floor(c.seed),
    steps: Math.floor(c.steps),
    cfg: c.cfg,
    width: Math.floor(c.width),
    height: Math.floor(c.height),
    sampler: c.sampler,
    scheduler: c.scheduler,
  }
  if (c.length != null) p.length = Math.floor(c.length)
  if (c.fps != null) p.fps = c.fps
  if (c.source?.name) p.image = c.source.name
  if (c.denoise != null) p.denoise = c.denoise
  if (c.megapixels != null) p.megapixels = c.megapixels
  if (c.split != null) p.split = Math.floor(c.split)
  else if (fallback.split != null) p.split = Math.floor(fallback.split)
  if (c.shift != null) p.shift = c.shift
  if (c.clipSkip != null) p.clipSkip = Math.floor(c.clipSkip)
  return p
}

/**
 * Build the history record for a finished run, from the composition that made
 * it. The job engine supplies what only it knows: the file, the prompt id and
 * how long it took.
 */
export function recordOf(
  c: Composition,
  result: {
    file: FileRef
    files?: FileRef[]
    kind: 'image' | 'video'
    promptId: string
    durationMs: number
    /** The seed actually used, which differs from `c.seed` on an unlocked run. */
    seed?: number
    familyLabel: string
    modelLabel: string
    variant?: HistoryEntry['variant']
    at?: number
    /**
     * Quality passes that ran. They live on the desk rather than on the
     * composition, so the engine has to hand them over or the record says a
     * plain single-pass render made a two-pass picture.
     */
    passes?: HistoryEntry['passes']
    /** The LoRA chain that was inserted, resolved to files and strengths. */
    loras?: HistoryEntry['loras']
  },
): NewEntry {
  const sized = c.mode === 'i2i' || c.mode === 'edit'
  const passes = result.passes
  const ranAPass = !!passes && (!!passes.face || !!passes.hand || !!passes.hires)
  return {
    at: result.at,
    desk: c.desk,
    kind: result.kind,
    mode: c.mode,
    file: result.file,
    files: result.files,
    familyId: c.familyId,
    familyLabel: result.familyLabel,
    variant: result.variant ?? variantOf(c.mode, c.noLora),
    model: c.model,
    modelLabel: result.modelLabel,
    prompt: c.prompt,
    negative: c.negative,
    seed: result.seed ?? c.seed,
    steps: c.steps,
    cfg: c.cfg,
    sampler: c.sampler,
    scheduler: c.scheduler,
    width: sized ? null : c.width,
    height: sized ? null : c.height,
    megapixels: c.megapixels ?? undefined,
    denoise: c.denoise ?? undefined,
    split: c.split ?? undefined,
    shift: c.shift ?? undefined,
    clipSkip: c.clipSkip ?? undefined,
    length: c.length ?? undefined,
    fps: c.fps ?? undefined,
    positivePrefix: c.positivePrefix ?? undefined,
    passes: ranAPass ? { ...passes } : undefined,
    loras: result.loras?.length ? result.loras.map((l) => ({ ...l })) : undefined,
    source: c.source
      ? {
          name: c.source.name,
          ref: c.source.ref,
          fromEntryId: c.source.fromEntryId,
          fromFrame: c.source.fromFrame,
        }
      : undefined,
    promptId: result.promptId,
    durationMs: result.durationMs,
  }
}

function variantOf(mode: Mode, noLora: boolean): HistoryEntry['variant'] {
  if (noLora) return 'nolora'
  if (mode === 'i2i') return 'img2img'
  if (mode === 'i2v') return 'i2v'
  return null
}

// ---------------------------------------------------------------------------
// The desk stores
// ---------------------------------------------------------------------------

const DESK_KEY: Record<DeskId, string> = {
  images: 'switchgen.desk.images.v1',
  video: 'switchgen.desk.video.v1',
}

/** Writes are debounced so typing does not hit localStorage on every key. */
const SAVE_DEBOUNCE_MS = 400

export type DeskStore = {
  /** useSyncExternalStore(store.subscribe, store.get) */
  subscribe: (fn: () => void) => () => void
  get: () => Composition
  set: (next: Composition) => void
  /** Shallow patch. Marks nothing as touched; use `edit` for reader input. */
  patch: (patch: Partial<Composition>) => void
  /**
   * Apply a reader's edit: patches, and records the named fields as touched so
   * a later style change leaves them alone.
   */
  edit: (patch: Partial<Composition>, ...touched: TunableField[]) => void
  /** Back to a blank desk. */
  reset: () => void
  /** Write the pending draft now, rather than when the debounce lands. */
  flush: () => void
}

/**
 * Bring a stored draft onto the current shape.
 *
 * The desk key carries no version number, so a draft can arrive from an older
 * build, from a hand edit, or half-written from a tab that died mid-save. Every
 * reader downstream trusts the declared type — `reuseIntoDesk` opens with
 * `previous.prompt.trim()`, `toParams` opens with `Math.floor(c.width)` — so a
 * null prompt or a string width is not a cosmetic problem: it is a blank screen
 * on load, or a NaN posted to ComfyUI.
 *
 * Each field is therefore checked and falls back to the blank desk's value. A
 * payload that is not an object at all is discarded whole, which is the one
 * case where starting clean is better than guessing.
 */
function sanitiseDraft(desk: DeskId, raw: unknown): Composition {
  const base = newComposition(desk)
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return base
  const d = raw as Record<string, unknown>

  const video = desk === 'video'
  // A draft written by an older build can carry a mode from the other desk.
  const mode = MODES[desk].includes(d.mode as Mode) ? (d.mode as Mode) : MODES[desk][0]
  const touched = Array.isArray(d.touched)
    ? (d.touched.filter((f): f is TunableField => typeof f === 'string' && IS_TUNABLE.has(f)))
    : []

  return {
    ...base,
    mode,
    familyId: asString(d.familyId, base.familyId),
    model: asString(d.model, base.model),
    prompt: asString(d.prompt, ''),
    negative: asStringOrNull(d.negative),
    positivePrefix: asStringOrNull(d.positivePrefix),
    source: sanitiseSource(d.source),
    width: asNumber(d.width, base.width),
    height: asNumber(d.height, base.height),
    megapixels: asNumberOrNull(d.megapixels),
    denoise: asNumberOrNull(d.denoise),
    seed: asNumber(d.seed, base.seed),
    seedLocked: asBoolean(d.seedLocked, base.seedLocked),
    steps: asNumber(d.steps, base.steps),
    cfg: asNumber(d.cfg, base.cfg),
    sampler: asString(d.sampler, base.sampler),
    scheduler: asString(d.scheduler, base.scheduler),
    // Frames and frame rate belong to the video desk. A stale length left on a
    // picture draft would be sent to a graph that has nowhere to put it.
    length: video ? (asNumberOrNull(d.length) ?? base.length) : null,
    fps: video ? (asNumberOrNull(d.fps) ?? base.fps) : null,
    split: asNumberOrNull(d.split),
    shift: asNumberOrNull(d.shift),
    clipSkip: asNumberOrNull(d.clipSkip),
    noLora: asBoolean(d.noLora, base.noLora),
    runs: d.runs === 2 || d.runs === 4 ? d.runs : 1,
    addOnsAccepted: asStringList(d.addOnsAccepted),
    addOnsDeclined: asStringList(d.addOnsDeclined),
    touched,
  }
}

function sanitiseSource(raw: unknown): SourceRef | null {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null
  const s = raw as Record<string, unknown>
  const ref = s.ref
  const file =
    ref && typeof ref === 'object' && typeof (ref as FileRef).filename === 'string'
      ? {
          filename: String((ref as FileRef).filename),
          subfolder: String((ref as FileRef).subfolder ?? ''),
          type: String((ref as FileRef).type ?? 'output'),
        }
      : undefined
  const name = asString(s.name, '')
  if (!name && !file) return null
  // An object URL belongs to the page that made it. Restoring one from a
  // previous load points the well at a blob the browser has already released.
  const preview = asStringOrNull(s.previewUrl)
  return {
    name,
    ref: file,
    previewUrl: preview && !preview.startsWith('blob:') ? preview : undefined,
    label: asStringOrNull(s.label) ?? file?.filename ?? name,
    width: asNumberOrNull(s.width) ?? undefined,
    height: asNumberOrNull(s.height) ?? undefined,
    bytes: asNumberOrNull(s.bytes) ?? undefined,
    fromEntryId: asStringOrNull(s.fromEntryId) ?? undefined,
    fromFrame: asNumberOrNull(s.fromFrame) ?? undefined,
  }
}

function loadComposition(desk: DeskId): Composition {
  return sanitiseDraft(desk, readJson<unknown>(DESK_KEY[desk], null))
}

/** Field-by-field equality, so a patch that changes nothing announces nothing. */
function sameComposition(a: Composition, b: Composition): boolean {
  if (a === b) return true
  for (const key of Object.keys(a) as (keyof Composition)[]) {
    if (key === 'touched' || key === 'source') continue
    if (a[key] !== b[key]) return false
  }
  if (a.touched.length !== b.touched.length) return false
  for (let i = 0; i < a.touched.length; i += 1) if (a.touched[i] !== b.touched[i]) return false
  return sameSource(a.source, b.source)
}

function sameSource(a: SourceRef | null, b: SourceRef | null): boolean {
  if (a === b) return true
  if (!a || !b) return false
  return (
    a.name === b.name &&
    a.previewUrl === b.previewUrl &&
    a.label === b.label &&
    a.width === b.width &&
    a.height === b.height &&
    a.bytes === b.bytes &&
    a.fromEntryId === b.fromEntryId &&
    a.fromFrame === b.fromFrame &&
    a.ref?.filename === b.ref?.filename &&
    a.ref?.subfolder === b.ref?.subfolder &&
    a.ref?.type === b.ref?.type
  )
}

function makeDeskStore(desk: DeskId): DeskStore {
  let current = loadComposition(desk)
  const listeners = new Set<() => void>()
  let timer: ReturnType<typeof setTimeout> | null = null

  const flush = () => {
    timer = null
    try {
      store.set(DESK_KEY[desk], JSON.stringify(current))
    } catch {
      // A full quota must never cost the reader their draft on screen; the
      // archive's own eviction will free room on the next write.
    }
  }

  const announce = () => {
    for (const fn of [...listeners]) {
      try {
        fn()
      } catch {
        /* a broken subscriber must not stop the others */
      }
    }
    if (timer) clearTimeout(timer)
    timer = setTimeout(flush, SAVE_DEBOUNCE_MS)
  }

  const set = (next: Composition) => {
    // `patch` and `edit` spread, so they hand back a fresh object even when
    // nothing in it moved. Announcing that would re-render every subscriber,
    // and a component that patches while rendering would never settle, because
    // `useSyncExternalStore` compares the snapshot by reference.
    if (next === current || sameComposition(current, next)) return
    current = { ...next, desk }
    announce()
  }

  const flushNow = () => {
    if (!timer) return
    clearTimeout(timer)
    flush()
  }

  // A draft lost because the tab closed inside the debounce window is a draft
  // the reader was still writing. `pagehide` fires on close, on navigation and
  // on mobile backgrounding; nothing else does all three.
  if (typeof window !== 'undefined') window.addEventListener('pagehide', flushNow)

  return {
    subscribe(fn) {
      listeners.add(fn)
      return () => {
        listeners.delete(fn)
      }
    },
    get: () => current,
    set,
    patch: (patch) => set({ ...current, ...patch }),
    edit: (patch, ...touched) => set(markTouched({ ...current, ...patch }, ...touched)),
    reset: () => set(newComposition(desk)),
    flush: flushNow,
  }
}

const stores: Partial<Record<DeskId, DeskStore>> = {}

/**
 * The store for one desk. A module singleton, so a desk keeps its draft while
 * you are on the other one and while a job of its own is running.
 */
export function deskStore(desk: DeskId): DeskStore {
  const existing = stores[desk]
  if (existing) return existing
  const made = makeDeskStore(desk)
  stores[desk] = made
  return made
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

export const SETTINGS_KEY = 'switchgen.settings.v1'

export type Settings = {
  /** All controls, rather than the five that matter. */
  expert: boolean
  /** Player: loop by default. These are five-second clips. */
  loop: boolean
  /** Player speed: 0.25 | 0.5 | 1 | 2. */
  speed: number
  /** Archive: grid or the dense parameter table. */
  view: 'grid' | 'list'
  sort: 'newest' | 'oldest'
}

const SETTINGS_DEFAULT: Settings = {
  expert: false,
  loop: true,
  speed: 1,
  view: 'grid',
  sort: 'newest',
}

/** The speeds the player offers. A stored value outside them is not playable. */
const SPEEDS: readonly number[] = [0.25, 0.5, 1, 2]

/**
 * Settings arrive from localStorage and from other tabs, neither of which this
 * build wrote. A `speed` of "fast" or a `view` of "banana" would be handed
 * straight to the player and to the archive, so each field is checked against
 * what it is allowed to be rather than merged over the defaults on trust.
 */
function sanitiseSettings(raw: unknown): Settings {
  const d = (raw && typeof raw === 'object' && !Array.isArray(raw) ? raw : {}) as Record<
    string,
    unknown
  >
  return {
    expert: asBoolean(d.expert, SETTINGS_DEFAULT.expert),
    loop: asBoolean(d.loop, SETTINGS_DEFAULT.loop),
    speed: isFinite_(d.speed) && SPEEDS.includes(d.speed) ? d.speed : SETTINGS_DEFAULT.speed,
    view: d.view === 'grid' || d.view === 'list' ? d.view : SETTINGS_DEFAULT.view,
    sort: d.sort === 'newest' || d.sort === 'oldest' ? d.sort : SETTINGS_DEFAULT.sort,
  }
}

export type SettingsStore = {
  subscribe: (fn: () => void) => () => void
  get: () => Settings
  patch: (patch: Partial<Settings>) => void
  toggleExpert: () => void
}

function makeSettingsStore(): SettingsStore {
  let current: Settings = sanitiseSettings(readJson<unknown>(SETTINGS_KEY, null))
  const listeners = new Set<() => void>()

  const announce = () => {
    for (const fn of [...listeners]) {
      try {
        fn()
      } catch {
        /* as above */
      }
    }
  }

  const write = (next: Settings) => {
    current = next
    try {
      store.set(SETTINGS_KEY, JSON.stringify(current))
    } catch {
      /* settings are small; a quota failure here is not worth a notice */
    }
    announce()
  }

  // A second tab turning expert mode on should not leave this one lying.
  onStorage(SETTINGS_KEY, (value) => {
    if (value === null) return
    try {
      current = sanitiseSettings(JSON.parse(value))
      announce()
    } catch {
      /* ignore an unreadable write from elsewhere */
    }
  })

  return {
    subscribe(fn) {
      listeners.add(fn)
      return () => {
        listeners.delete(fn)
      }
    },
    get: () => current,
    patch: (patch) => write({ ...current, ...patch }),
    toggleExpert: () => write({ ...current, expert: !current.expert }),
  }
}

/** Global display settings. `useSyncExternalStore(settings.subscribe, settings.get)`. */
export const settings: SettingsStore = makeSettingsStore()

// ---------------------------------------------------------------------------
// Reuse — a record becomes the next generation
// ---------------------------------------------------------------------------

/** Something the record carried that the desk could not take. */
export type ReuseNote = { field: string; reason: string }

export type Reuse = {
  desk: DeskId
  composition: Composition
  notes: ReuseNote[]
}

export type ReuseOptions = {
  /** Draw a new seed rather than reusing the record's. "Make another" does. */
  freshSeed?: boolean
  /** Models installed right now. A record naming a missing one is reported. */
  installedModels?: readonly string[]
  /** Used when the record's model is gone: what to load instead. */
  substitute?: { familyId: string; model: string; modelLabel?: string }
  /** Samplers the substitute offers, so an impossible one is dropped, not sent. */
  availableSamplers?: readonly string[]
  availableSchedulers?: readonly string[]
}

/**
 * Map a history record back onto a composition.
 *
 * Restores every field the record carries: prompt, negative, model, sampler,
 * scheduler, steps, CFG, size, length, denoise, split, shift, clip skip, the
 * source picture by its existing ComfyUI filename (no re-upload) and the seed
 * with the lock on.
 *
 * Two things it cannot restore, because they live on the desk rather than in
 * the composition: the quality passes and the LoRA rack. A record that ran with
 * either says so in `notes`, by name and strength, rather than leaving the
 * reader to discover it from a picture that came out different at the same
 * seed. Restoring them is the desk's job; naming them is this function's.
 *
 * Nothing runs. The reader sees precisely what they are about to make.
 */
export function compositionFromEntry(entry: HistoryEntry, opts: ReuseOptions = {}): Reuse {
  const desk = deskOf(entry.mode)
  const notes: ReuseNote[] = []

  let familyId = entry.familyId
  let model = entry.model
  const installed = opts.installedModels
  const missingModel = !!installed && !!model && !installed.includes(model)
  if (missingModel && opts.substitute) {
    notes.push({
      field: 'model',
      reason: `${entry.modelLabel} is no longer installed. We loaded ${opts.substitute.modelLabel ?? opts.substitute.model} instead.`,
    })
    familyId = opts.substitute.familyId
    model = opts.substitute.model
  } else if (missingModel) {
    notes.push({
      field: 'model',
      reason: `${entry.modelLabel} is no longer installed. Choose another style before you run this.`,
    })
  }

  const sampler = pick(entry.sampler, opts.availableSamplers)
  if (sampler === null) {
    notes.push({
      field: 'sampler',
      reason: `Not carried over: the sampler, because this model does not offer ${entry.sampler}.`,
    })
  }
  const scheduler = pick(entry.scheduler, opts.availableSchedulers)
  if (scheduler === null) {
    notes.push({
      field: 'scheduler',
      reason: `Not carried over: the scheduler, because this model does not offer ${entry.scheduler}.`,
    })
  }

  const current = deskStore(desk).get()
  const composition: Composition = {
    ...newComposition(desk),
    mode: entry.mode,
    familyId,
    model,
    prompt: entry.prompt,
    negative: entry.negative,
    positivePrefix: entry.positivePrefix ?? null,
    source: sourceFromEntry(entry),
    width: entry.width ?? current.width,
    height: entry.height ?? current.height,
    megapixels: entry.megapixels ?? null,
    denoise: entry.denoise ?? null,
    seed: opts.freshSeed ? randomSeed() : entry.seed,
    seedLocked: !opts.freshSeed,
    steps: entry.steps,
    cfg: entry.cfg,
    sampler: sampler ?? current.sampler,
    scheduler: scheduler ?? current.scheduler,
    length: entry.length ?? null,
    fps: entry.fps ?? null,
    split: entry.split ?? null,
    shift: entry.shift ?? null,
    clipSkip: entry.clipSkip ?? null,
    noLora: entry.variant === 'nolora',
    runs: 1,
    // Every value here was chosen once already, so a style change must not
    // silently rewrite it underneath the reader.
    touched: [
      'steps',
      'cfg',
      'sampler',
      'scheduler',
      'negative',
      'seed',
      ...(entry.width != null ? (['width', 'height'] as TunableField[]) : []),
      ...(entry.length != null ? (['length'] as TunableField[]) : []),
      ...(entry.fps != null ? (['fps'] as TunableField[]) : []),
      ...(entry.denoise != null ? (['denoise'] as TunableField[]) : []),
      ...(entry.megapixels != null ? (['megapixels'] as TunableField[]) : []),
      ...(entry.split != null ? (['split'] as TunableField[]) : []),
      ...(entry.shift != null ? (['shift'] as TunableField[]) : []),
      ...(entry.clipSkip != null ? (['clipSkip'] as TunableField[]) : []),
    ],
  }

  if (entry.missing && entry.source) {
    notes.push({
      field: 'source',
      reason: 'The source picture may no longer be on disk. Check the well before you run this.',
    })
  }

  const ran: string[] = []
  if (entry.passes?.hires) ran.push('the second pass')
  if (entry.passes?.face) ran.push('the face detail pass')
  if (entry.passes?.hand) ran.push('the hand detail pass')
  if (ran.length) {
    notes.push({
      field: 'passes',
      reason: `Not carried over: ${sentenceList(ran)}. Switch ${ran.length > 1 ? 'them' : 'it'} back on before you run this.`,
    })
  }

  if (entry.loras?.length) {
    const named = entry.loras.map((l) => `${l.name} at ${l.strength}`)
    notes.push({
      field: 'loras',
      reason: `Not carried over: ${entry.loras.length === 1 ? 'the LoRA' : `the ${entry.loras.length} LoRAs`} this used. Set the rack to ${sentenceList(named)} before you run this.`,
    })
  }

  return { desk, composition, notes }
}

/** `a`, `a and b`, `a, b and c`. */
function sentenceList(items: readonly string[]): string {
  if (items.length <= 1) return items[0] ?? ''
  return `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`
}

function pick(value: string, available: readonly string[] | undefined): string | null {
  if (!value) return null
  if (!available || available.length === 0) return value
  return available.includes(value) ? value : null
}

function sourceFromEntry(entry: HistoryEntry): SourceRef | null {
  if (!entry.source) return null
  return {
    name: entry.source.name,
    ref: entry.source.ref,
    label: entry.source.ref?.filename ?? entry.source.name,
    fromEntryId: entry.source.fromEntryId,
    fromFrame: entry.source.fromFrame,
  }
}

export type AppliedReuse = Reuse & {
  /**
   * True when the desk already held a different, non-empty prompt. The desk
   * shows the loaded settings behind a notice offering both ways out.
   */
  clobbered: boolean
  /** Put the desk back exactly as it was. */
  undo: () => void
}

/**
 * Load a record into its own desk and return an undo.
 *
 * The desk is decided by the record, never by where the reader happens to be:
 * a clip always restores into the Video desk, with the video controls.
 */
export function reuseIntoDesk(entry: HistoryEntry, opts: ReuseOptions = {}): AppliedReuse {
  const reuse = compositionFromEntry(entry, opts)
  const target = deskStore(reuse.desk)
  const previous = target.get()
  const clobbered =
    previous.prompt.trim().length > 0 && previous.prompt.trim() !== entry.prompt.trim()

  target.set(reuse.composition)
  return {
    ...reuse,
    clobbered,
    undo: () => target.set(previous),
  }
}

/**
 * Cross-route handoff for the region bench.
 *
 * The bench lives on the Pictures desk, but a reader can ask for it from the
 * Archive, which is a different route. Rather than thread a prop through the
 * router, the Archive leaves the record here and navigates; Pictures takes it
 * on its next render and opens the bench.
 *
 * Deliberately NOT part of Composition: this is a one-shot request, not desk
 * state. Persisting it would reopen the bench on every reload, which is exactly
 * the kind of sticky surprise the desk is trying to get away from.
 */
let pendingRegion: HistoryEntry | null = null

/** Ask the Pictures desk to open the region bench on this record. */
export function requestRegionEdit(entry: HistoryEntry): void {
  pendingRegion = entry
}

/** Take the pending request, if any. Reading it clears it, so it fires once. */
export function takeRegionRequest(): HistoryEntry | null {
  const held = pendingRegion
  pendingRegion = null
  return held
}

/**
 * Send only the picture to a desk, leaving the prompt and settings alone.
 * The archive's "Use as source" verb.
 */
export function adoptSource(
  entry: HistoryEntry,
  desk: DeskId,
  opts: { name?: string; frame?: number } = {},
): () => void {
  const target = deskStore(desk)
  const previous = target.get()
  const source: SourceRef = {
    // An output of ours is already inside ComfyUI, but LoadImage reads the
    // *input* folder, so the desk uploads it and fills `name` in. Until then
    // the ref is enough to show the well.
    name: opts.name ?? '',
    ref: entry.file,
    previewUrl: undefined,
    label: entry.file.filename,
    fromEntryId: entry.id,
    fromFrame: opts.frame,
  }
  const mode: Mode = desk === 'video' ? 'i2v' : previous.mode === 'edit' ? 'edit' : 'i2i'
  target.set({ ...previous, mode, source })
  return () => target.set(previous)
}

/**
 * Adopt one value from a record into a desk — the caption's numbers are links,
 * and clicking `seed 1839204718` should take that seed and nothing else.
 */
export function adoptValue<K extends TunableField>(
  desk: DeskId,
  field: K,
  value: Composition[K],
): () => void {
  const target = deskStore(desk)
  const previous = target.get()
  target.edit({ [field]: value } as Partial<Composition>, field)
  if (field === 'seed') target.patch({ seedLocked: true })
  return () => target.set(previous)
}
