/**
 * The archive: every picture and clip this machine has made, with the settings
 * that made it.
 *
 * The record and the file are deliberately separate things. The record is
 * small, searchable and lives in localStorage; the file is large and lives in
 * ComfyUI's output folder. That separation is what lets you remove a record
 * without losing a file, keep a record whose file you deleted by hand, and
 * remake something you lost — the settings outlive the picture.
 *
 * Storage rules:
 *   - Every localStorage access is wrapped. In a private window the getter
 *     itself throws, so the store degrades to memory for the tab's lifetime
 *     rather than taking the app down with it.
 *   - The payload is versioned. An envelope this build does not understand is
 *     kept under {@link BROKEN_KEY} and a fresh archive is started. An index is
 *     never silently discarded.
 *   - A generation never fails because the archive is full. On a quota error
 *     the oldest unstarred records are shed and the write retried once. With
 *     no server behind it they are gone. With one, they stay on screen for the
 *     session and only this browser's saved copy is shorter: the server still
 *     holds them. A record the server does not yet have is never shed.
 *
 * React: `useSyncExternalStore(subscribe, all)`. `all()` returns the same array
 * reference until something actually changes.
 */

import { fileUrl, relPath, type FileRef } from './comfy'
import { isQuotaError, onStorage, store, storageWorks, type DeskId, type Mode } from './session'

export type { FileRef, DeskId, Mode }

// ---------------------------------------------------------------------------
// The record
// ---------------------------------------------------------------------------

const HISTORY_KEY = 'switchgen.archive.v2'
const BROKEN_KEY = 'switchgen.archive.v2.broken'
const HISTORY_VERSION = 2

/**
 * How many records this browser keeps. A record with its tags serialises to
 * about 1 KB (119 records on the live archive averaged 1,018 characters), so
 * 5,000 of them come close to the roughly five million characters a browser
 * allows one origin, which the desk drafts share. Beyond the cap the oldest
 * unstarred records fall off the end. With the server behind it, that only
 * narrows what this browser shows: the server keeps them all.
 */
const MAX_ENTRIES = 5000

/** How many unstarred records are shed when the browser reports a full quota. */
const EVICT_ON_QUOTA = 200

export type HistoryEntry = {
  id: string
  /** Monotonic edition number. Shown as "No. 1,284"; never reused. */
  no: number
  /** Epoch ms, when the job finished. */
  at: number

  desk: DeskId
  /** From `collectFiles`, which reads the animated flag, not the container. */
  kind: 'image' | 'video'
  mode: Mode

  /** The file this record stands for. */
  file: FileRef
  /** Every file the run produced, when it made more than one. */
  files?: FileRef[]

  /** Base family id — never a derived `__img2img` / `__i2v` suffix. */
  familyId: string
  familyLabel: string
  /** Which derived shape of the family ran, if any. */
  variant: null | 'img2img' | 'i2v' | 'nolora'
  /** The weight file, e.g. `moodyCutieMixKrea2_v50_int8.safetensors`. */
  model: string
  modelLabel: string

  prompt: string
  /** null means the family's house negative was used, unchanged. */
  negative: string | null
  /** The author-card prefix that was prepended, when one was. */
  positivePrefix?: string

  seed: number
  steps: number
  cfg: number
  sampler: string
  scheduler: string
  /** null for image-to-image and edit, where the source decided the size. */
  width: number | null
  height: number | null
  megapixels?: number
  denoise?: number
  split?: number
  shift?: number
  clipSkip?: number
  /** Frames. */
  length?: number
  fps?: number

  /**
   * The quality passes that ran, when any did.
   *
   * These are desk state, not composition state, so nothing else in this record
   * implies them: two records can hold identical scalars and stand for visibly
   * different pictures unless this says otherwise. Structurally the Pictures
   * desk's `Passes`, restated so the archive does not import a route.
   */
  passes?: { face?: boolean; hand?: boolean; hires?: boolean }
  /**
   * The LoRA chain that was inserted, resolved to files and strengths.
   * Structurally `LoraSpec` from `src/lib/refine.ts`.
   */
  loras?: { name: string; strength: number; clipStrength?: number }[]

  /** The picture this was made from, when it was made from one. */
  source?: {
    /** The ComfyUI input filename the graph loaded. */
    name: string
    /** The output it came from, when it came from the archive. */
    ref?: FileRef
    fromEntryId?: string
    fromFrame?: number
  }

  promptId: string
  durationMs: number

  starred?: boolean
  /** Reader's marginalia. */
  note?: string
  /** Set by the file audit. Never blocks anything. */
  missing?: boolean
  /**
   * Changed in this browser since the server last stamped it. Saved with the
   * record, so a change made while the server was away is still sent after the
   * tab closes, and until it is sent the server's older copy does not replace
   * it. Only ever set on a record that has a `rev`; one without is unsent
   * already. Never sent to the server.
   */
  pending?: boolean

  /**
   * The server's revision stamp. Absent on a record the server has never
   * seen, which is exactly the set the sync pushes on start.
   */
  rev?: number
  /** Filed from the outputs folder after the fact, not by the desk that made it. */
  recovered?: boolean

  /** What the tagger saw, booru spelling, strongest first. Absent until a reading. */
  tags?: string[]
  /** WD14's own four-way rating, when a reading was made. */
  rating?: 'general' | 'sensitive' | 'questionable' | 'explicit'
}

/**
 * Alias for readers of the build specification, which calls the same record an
 * ArchiveEntry. One type, two names; no conversion anywhere.
 */
export type ArchiveEntry = HistoryEntry

/** What a caller hands to {@link add}. Identity and numbering are ours. */
export type NewEntry = Omit<HistoryEntry, 'id' | 'no' | 'at'> & { at?: number }

type Envelope = {
  v: number
  nextNo: number
  entries: HistoryEntry[]
  /** Ids removed here whose removal the server has not yet acknowledged. */
  gone?: string[]
  /**
   * How far into the server's log these entries have been brought, and which
   * log. Saved with the entries it describes, so the two can never disagree.
   */
  sync?: { rev: number; epoch: string | null }
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

let entries: HistoryEntry[] = []
let nextNo = 1
/** Removals the server has not yet acknowledged. See {@link gone}. */
let goneIds = new Set<string>()
/** The server revision these entries are current to, and the epoch of its log. */
let syncRev = 0
let syncEpoch: string | null = null
/**
 * Set once this browser's archive mirrors a server's: the first pull, or a
 * saved cursor from an earlier one. It decides what a full quota means, and
 * whether a removal of a never-stamped record still has to be sent.
 */
let serverBacked = false
/**
 * Records sent in by the reader that the local cap dropped at once (an old
 * restore, a large recovery). They never reach the screen, but the server
 * keeps everything, so they wait here to be pushed. Memory only.
 */
const outbox = new Map<string, HistoryEntry>()

/** A record that exists only here until it is pushed: never stamped, or changed since. */
function unpushed(e: HistoryEntry): boolean {
  return e.rev === undefined || e.pending === true
}

let loadIssueMessage: string | null = null
/** Set when the stored envelope was older than this build and was brought forward. */
let migratedOnLoad = false

const listeners = new Set<() => void>()
let saveTimer: ReturnType<typeof setTimeout> | null = null

/**
 * The least a stored record must have to be worth keeping: a file we can name
 * and a time we can order it by. Everything else {@link normalise} can supply,
 * including a missing id, so a record is never thrown away over a field that
 * can be rebuilt.
 */
function sane(e: any): e is HistoryEntry {
  return (
    !!e &&
    typeof e === 'object' &&
    !Array.isArray(e) &&
    numOrNull(e.at) !== null &&
    !!e.file &&
    typeof e.file.filename === 'string' &&
    e.file.filename.length > 0
  )
}

/**
 * A number, or null when the value cannot be read as one.
 *
 * A numeric string is accepted. An archive exists so that what was lost can be
 * made again, and a seed stored as `"1839204718"` by an older build is a seed
 * that still reproduces the picture. `"wide"` is not a width and takes the
 * fallback.
 */
const numOrNull = (v: unknown): number | null => {
  if (typeof v === 'number') return Number.isFinite(v) ? v : null
  if (typeof v === 'string' && v.trim() !== '') {
    const n = Number(v)
    return Number.isFinite(n) ? n : null
  }
  return null
}
const num = (v: unknown, fallback: number): number => numOrNull(v) ?? fallback
const str = (v: unknown, fallback: string): string => (typeof v === 'string' ? v : fallback)

function fileRef(v: any, fallbackName = ''): FileRef {
  return {
    filename: str(v?.filename, fallbackName),
    subfolder: str(v?.subfolder, ''),
    type: str(v?.type, 'output'),
  }
}

/**
 * Bring one record onto the declared shape.
 *
 * {@link sane} only guarantees an id, a timestamp and a filename. Everything
 * else in a record read back from storage is whatever was there: a string where
 * a number belongs survives `JSON.parse` and then reaches `toParams`, which
 * floors it, and the archive's numeric search, which compares it. So every
 * field is coerced here, once, at the boundary.
 */
function normalise(e: any, fallbackNo: number): HistoryEntry {
  return {
    ...e,
    id: str(e.id, '') || cryptoId(),
    no: num(e.no, fallbackNo),
    at: num(e.at, Date.now()),
    desk: e.desk === 'video' ? 'video' : 'images',
    kind: e.kind === 'video' ? 'video' : 'image',
    file: fileRef(e.file),
    files: Array.isArray(e.files)
      ? e.files.filter((f: any) => typeof f?.filename === 'string').map((f: any) => fileRef(f))
      : undefined,
    rev: numOrNull(e.rev) ?? undefined,
    pending: e.pending === true ? true : undefined,
    recovered: e.recovered === true ? true : undefined,
    tags: Array.isArray(e.tags) ? e.tags.filter((t: unknown): t is string => typeof t === 'string') : undefined,
    rating: ['general', 'sensitive', 'questionable', 'explicit'].includes(e.rating) ? e.rating : undefined,
    prompt: str(e.prompt, ''),
    negative: typeof e.negative === 'string' ? e.negative : null,
    familyId: str(e.familyId, ''),
    familyLabel: str(e.familyLabel, str(e.familyId, '')),
    model: str(e.model, ''),
    modelLabel: str(e.modelLabel, str(e.model, '')),
    variant: e.variant ?? null,
    seed: num(e.seed, 0),
    steps: num(e.steps, 0),
    cfg: num(e.cfg, 0),
    sampler: str(e.sampler, ''),
    scheduler: str(e.scheduler, ''),
    width: numOrNull(e.width),
    height: numOrNull(e.height),
    promptId: str(e.promptId, ''),
    durationMs: num(e.durationMs, 0),
    source:
      e.source && typeof e.source === 'object'
        ? {
            name: str(e.source.name, ''),
            ref: e.source.ref?.filename ? fileRef(e.source.ref) : undefined,
            fromEntryId: typeof e.source.fromEntryId === 'string' ? e.source.fromEntryId : undefined,
            fromFrame: numOrNull(e.source.fromFrame) ?? undefined,
          }
        : undefined,
    loras: Array.isArray(e.loras)
      ? e.loras
          .filter((l: any) => typeof l?.name === 'string')
          .map((l: any) => ({
            name: l.name,
            strength: num(l.strength, 1),
            clipStrength: numOrNull(l.clipStrength) ?? undefined,
          }))
      : undefined,
  }
}

function load(): void {
  const raw = store.get(HISTORY_KEY)
  if (!raw) return

  let parsed: any
  try {
    parsed = JSON.parse(raw)
  } catch {
    keepBroken(raw, 'We could not read your archive and have started a new one.')
    return
  }

  if (!parsed || typeof parsed !== 'object' || !Array.isArray(parsed.entries)) {
    keepBroken(raw, 'Your archive was in a shape this version does not recognise.')
    return
  }

  const version = typeof parsed.v === 'number' ? parsed.v : 0
  if (version > HISTORY_VERSION) {
    // Forward migration is guesswork, and guessing would rewrite the newer
    // archive under the key the newer build reads. Stand aside instead.
    keepBroken(raw, 'Your archive was written by a newer version of SwitchGen.')
    return
  }
  if (version !== HISTORY_VERSION) {
    const migrated = migrate(parsed)
    if (!migrated) {
      keepBroken(raw, 'Your archive was written by a version we cannot read.')
      return
    }
    parsed = migrated
    migratedOnLoad = true
  }

  const kept: HistoryEntry[] = []
  let n = 0
  for (const e of parsed.entries) {
    if (sane(e)) kept.push(normalise(e, ++n))
  }
  kept.sort((a, b) => b.at - a.at)
  entries = kept
  nextNo =
    typeof parsed.nextNo === 'number' && parsed.nextNo > 0
      ? parsed.nextNo
      : kept.reduce((m, e) => Math.max(m, e.no), 0) + 1
  readSyncState(parsed)
}

/** The removal list and log position an envelope carries. Both are optional; an older envelope has neither. */
function readSyncState(parsed: any, merge = false): void {
  const ids: string[] = Array.isArray(parsed?.gone) ? parsed.gone.filter((x: unknown) => typeof x === 'string') : []
  goneIds = merge ? new Set([...goneIds, ...ids]) : new Set(ids)
  const sync = parsed?.sync
  syncRev = numOrNull(sync?.rev) ?? 0
  syncEpoch = typeof sync?.epoch === 'string' ? sync.epoch : null
  if (syncRev > 0) serverBacked = true
}

const MODES: readonly Mode[] = ['t2i', 'i2i', 'edit', 't2v', 'i2v']

/**
 * Bring an older envelope forward. Returns null when the shape is unknown, in
 * which case the raw text is kept rather than thrown away.
 *
 * v1 held bare output files with no settings, and a versionless envelope is the
 * same problem seen from further back, so both are read the same way: keep
 * every field that is already right, fill the rest, and drop only the entries
 * that carry no filename at all. One unreadable record must not cost the reader
 * the archive, so each is built inside its own try.
 */
function migrate(parsed: any): Envelope | null {
  const version = typeof parsed.v === 'number' ? parsed.v : 0
  if (version > 1 || !Array.isArray(parsed.entries)) return null

  const out: HistoryEntry[] = []
  const total = parsed.entries.length
  parsed.entries.forEach((e: any, i: number) => {
    try {
      const made = fromEarly(e, i, total)
      if (made) out.push(made)
    } catch {
      /* one record we cannot read is one record we leave behind */
    }
  })
  return { v: HISTORY_VERSION, nextNo: out.length + 1, entries: out }
}

/** One pre-v2 record, or null when there is not enough of it to keep. */
function fromEarly(e: any, i: number, total: number): HistoryEntry | null {
  if (!e || typeof e !== 'object') return null
  const filename =
    typeof e.file?.filename === 'string'
      ? e.file.filename
      : typeof e.filename === 'string'
        ? e.filename
        : null
  // A record whose file we cannot name stands for nothing and reproduces
  // nothing. Everything else is recoverable.
  if (!filename) return null

  const kind: 'image' | 'video' = e.kind === 'video' ? 'video' : 'image'
  const file: FileRef = e.file?.filename
    ? fileRef(e.file)
    : { filename, subfolder: str(e.subfolder, ''), type: str(e.type, 'output') }

  return {
    // Anything a build between v1 and v2 already wrote correctly is kept;
    // `normalise` runs over the result and coerces what is left.
    ...e,
    id: str(e.id, '') || cryptoId(),
    no: num(e.no, i + 1),
    at: num(e.at, Date.now() - (total - i) * 1000),
    desk: e.desk === 'video' || kind === 'video' ? 'video' : 'images',
    kind,
    mode: MODES.includes(e.mode) ? e.mode : kind === 'video' ? 't2v' : 't2i',
    file,
    familyId: str(e.familyId, 'unknown'),
    familyLabel: str(e.familyLabel, 'Unknown'),
    variant: e.variant ?? null,
    model: str(e.model, ''),
    modelLabel: str(e.modelLabel, ''),
    prompt: str(e.prompt, ''),
    negative: typeof e.negative === 'string' ? e.negative : null,
    seed: num(e.seed, 0),
    steps: num(e.steps, 0),
    cfg: num(e.cfg, 0),
    sampler: str(e.sampler, ''),
    scheduler: str(e.scheduler, ''),
    width: numOrNull(e.width),
    height: numOrNull(e.height),
    promptId: str(e.promptId, ''),
    durationMs: num(e.durationMs, 0),
  }
}

function keepBroken(raw: string, message: string) {
  try {
    store.set(BROKEN_KEY, raw)
  } catch {
    /* if even this fails there is nothing left to try */
  }
  entries = []
  nextNo = 1
  loadIssueMessage = `${message} The old data is saved under ${BROKEN_KEY} in case you want it back. Your files are untouched.`
}

function cryptoId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `id_${Date.now().toString(36)}_${Math.random().toString(36).slice(2)}`
}

/**
 * Set when the stored archive could not be read on load, with the exact
 * correction copy to show. Null on a clean load.
 */
export function loadIssue(): string | null {
  return loadIssueMessage
}

/** Dismiss the load correction once the reader has seen it. */
export function clearLoadIssue(): void {
  loadIssueMessage = null
}

/** False when records only live for this tab — a private window, say. */
export function isPersistent(): boolean {
  return storageWorks()
}

load()

// ---------------------------------------------------------------------------
// Save
// ---------------------------------------------------------------------------

let quotaMessage: string | null = null
/**
 * With a server behind it, how many records the last full save found room for.
 * Later saves keep to it rather than failing on the whole list every time.
 */
let roomFor: number | null = null

function envelope(list: HistoryEntry[] = entries): Envelope {
  return {
    v: HISTORY_VERSION,
    nextNo,
    entries: list,
    gone: goneIds.size ? [...goneIds] : undefined,
    sync: syncRev > 0 ? { rev: syncRev, epoch: syncEpoch } : undefined,
  }
}

/** False only when the browser said it is full. Any other failure is memory-only storage, already handled by `store`. */
function tryWrite(list: HistoryEntry[]): boolean {
  try {
    store.set(HISTORY_KEY, JSON.stringify(envelope(list)))
    return true
  } catch (err) {
    return !isQuotaError(err)
  }
}

function writeNow(): void {
  saveTimer = null
  const kept = roomFor === null ? entries : trimTo(entries, roomFor, unpushed)
  if (tryWrite(kept)) return

  // Full. Shed the oldest unstarred records and try once more. A generation
  // must never fail because the archive filled up.
  const shed = trimTo(kept, Math.max(0, kept.length - EVICT_ON_QUOTA), serverBacked ? unpushed : undefined)
  const fits = tryWrite(shed)
  if (serverBacked) {
    // Only records the server already holds were shed, so this browser's copy
    // is a window onto the archive and nothing has been lost. The records stay
    // on screen for this session; only the saved copy is shorter. Telling the
    // reader to clear records out here would have them remove real records
    // for every device.
    roomFor = shed.length
    const shown = shed.length.toLocaleString('en-GB')
    quotaMessage = fits
      ? `This browser has run out of room, so after a reload it will show only the newest ${shown} records. Nothing was removed: the older ones are still in the archive on the server.`
      : 'This browser has run out of room and could not save its copy of the archive. Nothing was removed from the archive on the server, and changes made here are still sent to it while this page is open.'
  } else {
    entries = shed
    quotaMessage = fits
      ? `Your archive was full, so we removed the ${kept.length - shed.length} oldest records. The files are still on disk.`
      : 'Your archive is full and we could not save it. Save a copy, then clear out some records.'
  }
  announce()
}

/** Set when records had to be dropped to fit. Show it as a correction. */
export function quotaIssue(): string | null {
  return quotaMessage
}

export function clearQuotaIssue(): void {
  quotaMessage = null
}

const SAVE_DEBOUNCE_MS = 400

function save(): void {
  if (saveTimer) clearTimeout(saveTimer)
  saveTimer = setTimeout(writeNow, SAVE_DEBOUNCE_MS)
}

/**
 * Write any pending change immediately.
 *
 * Writes are debounced so a burst of edits costs one serialisation, but a tab
 * closed inside that window would lose the record of a generation that really
 * happened. `pagehide` is the one event that fires reliably on close, on
 * navigation and on mobile backgrounding, so the flush hangs off it.
 */
export function flush(): void {
  if (!saveTimer) return
  clearTimeout(saveTimer)
  writeNow()
}

if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  window.addEventListener('pagehide', flush)
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') flush()
  })
}

// A migrated archive is written back in the current shape. Without this the old
// envelope is re-read and re-migrated on every load, and `nextNo` is recomputed
// from the entry count each time, so edition numbers would restart until the
// reader happened to make something.
if (migratedOnLoad) save()

function announce(): void {
  for (const fn of [...listeners]) {
    try {
      fn()
    } catch {
      /* one broken subscriber must not stop the rest */
    }
  }
}

/**
 * What one commit changed, for the sync. `remote` marks a change that came
 * from the server, which must not be echoed back to it.
 */
export type CommitDelta = { upserted: HistoryEntry[]; removed: string[]; remote: boolean }

const commitListeners = new Set<(delta: CommitDelta) => void>()

/** Hear every change as a delta. The sync is the one subscriber. */
export function onCommit(fn: (delta: CommitDelta) => void): () => void {
  commitListeners.add(fn)
  return () => {
    commitListeners.delete(fn)
  }
}

/**
 * Make `next` the archive.
 *
 * `untrimmed` is the list as it stood before a trim to the local cap, when
 * there was one. A trim is this browser running out of room, not the reader
 * removing anything, so what it dropped is never reported as a removal. With
 * a server behind it, anything the trim dropped that the server does not yet
 * have still goes there, because the server keeps everything; what the reader
 * brought in and the trim dropped at once is reported as a change.
 */
function commit(next: HistoryEntry[], opts: { remote?: boolean; untrimmed?: readonly HistoryEntry[] } = {}): void {
  const prev = entries
  const remote = opts.remote === true
  entries = next
  indexCache = new WeakMap()
  if (remote && !commitListeners.size) {
    save()
    announce()
    return
  }
  // Records are replaced, never mutated, so identity says what changed.
  const prevById = new Map(prev.map((e) => [e.id, e]))
  const nextIds = new Set(next.map((e) => e.id))
  const beforeTrim = opts.untrimmed ? new Set(opts.untrimmed.map((e) => e.id)) : nextIds
  const upserted = next.filter((e) => prevById.get(e.id) !== e)
  const removedEntries = prev.filter((e) => !beforeTrim.has(e.id))
  if (!remote) {
    // A removal is kept until the server says it has it, so closing the tab
    // first does not bring the record back on the next pull.
    for (const e of removedEntries) if (e.rev !== undefined || serverBacked) goneIds.add(e.id)
    for (const e of upserted) goneIds.delete(e.id)
    if (serverBacked && opts.untrimmed) {
      for (const e of opts.untrimmed) {
        if (nextIds.has(e.id)) continue
        const changedNow = prevById.get(e.id) !== e
        if (!changedNow && !unpushed(e)) continue
        outbox.set(e.id, e)
        if (changedNow) upserted.push(e)
      }
    }
  }
  save()
  announce()
  if (!commitListeners.size) return
  const removed = removedEntries.map((e) => e.id)
  if (!upserted.length && !removed.length) return
  const delta: CommitDelta = { upserted, removed, remote }
  for (const fn of [...commitListeners]) {
    try {
      fn(delta)
    } catch {
      /* one broken subscriber must not stop the rest */
    }
  }
}

// A second tab writing the archive must not leave this one showing a stale one.
onStorage(HISTORY_KEY, (value) => {
  if (value === null) return
  try {
    const parsed = JSON.parse(value)
    if (!parsed || !Array.isArray(parsed.entries) || parsed.v !== HISTORY_VERSION) return
    const kept: HistoryEntry[] = []
    let n = 0
    for (const e of parsed.entries) if (sane(e)) kept.push(normalise(e, ++n))
    kept.sort((a, b) => b.at - a.at)
    entries = kept
    nextNo = typeof parsed.nextNo === 'number' ? parsed.nextNo : nextNo
    // The log position describes the entries it was saved with, so it is
    // taken with them. Removals are merged rather than replaced: one this tab
    // has not yet sent is still this tab's to send.
    readSyncState(parsed, true)
    indexCache = new WeakMap()
    announce()
  } catch {
    /* a write we cannot read is a write we ignore */
  }
})

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/** Subscribe to every change. `useSyncExternalStore(subscribe, all)`. */
export function subscribe(fn: () => void): () => void {
  listeners.add(fn)
  return () => {
    listeners.delete(fn)
  }
}

/** Every record, newest first. Stable reference between changes. */
export function all(): readonly HistoryEntry[] {
  return entries
}

export function count(): number {
  return entries.length
}

export function get(id: string): HistoryEntry | undefined {
  return entries.find((e) => e.id === id)
}

// The archive filters through `runQuery` in `components/archive/query.tsx`,
// which composes `search()` below with its date and numeric tokens. A `byKind`
// or `byDesk` helper here would be a second, narrower way to ask the same
// question, and each call would hand back a fresh array, which is exactly the
// snapshot a `useSyncExternalStore` caller must not be given.

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/** File a finished generation. Returns the stored record. */
export function add(input: NewEntry): HistoryEntry {
  const entry: HistoryEntry = {
    ...input,
    id: cryptoId(),
    no: nextNo++,
    at: input.at ?? Date.now(),
  }
  const next = [entry, ...entries]
  commit(trim(next), { untrimmed: next })
  return entry
}

/**
 * File several records as one change: one sort, one trim, one commit. The
 * recovery pass can file thousands of outputs at once, and one {@link add}
 * each copied the whole archive every time and held the page for seconds.
 * Numbers are handed out in the order given, so pass them oldest first.
 */
export function addMany(inputs: readonly NewEntry[]): HistoryEntry[] {
  if (!inputs.length) return []
  const now = Date.now()
  const made: HistoryEntry[] = inputs.map((input) => ({
    ...input,
    id: cryptoId(),
    no: nextNo++,
    at: input.at ?? now,
  }))
  const next = [...made, ...entries].sort((a, b) => b.at - a.at)
  commit(trim(next), { untrimmed: next })
  return made
}

// ---------------------------------------------------------------------------
// The server's copy
// ---------------------------------------------------------------------------

/**
 * Merge what the server sent. The server wins by id, with two exceptions that
 * both mean this browser holds something the server has not yet taken: a
 * record changed here or never stamped, and a record removed here. Those wait
 * for the push, which is what settles them; replacing them first would wipe
 * the change off the screen while the older copy was still on its way. A
 * removal is honoured only for a record the server had stamped: a local record
 * it has never seen cannot have been removed there, and stays to be pushed.
 *
 * `cursor` is the log position the answer brings these entries to.
 */
export function mergeFromServer(
  records: readonly unknown[],
  removed: readonly string[],
  serverNextNo: number,
  cursor?: { rev: number; epoch: string | null },
): number {
  const map = new Map(entries.map((e) => [e.id, e]))
  let n = 0
  let changed = 0
  for (const raw of records) {
    if (!sane(raw)) continue
    n++
    const id = typeof (raw as { id?: unknown }).id === 'string' ? (raw as { id: string }).id : ''
    const local = id ? map.get(id) : undefined
    if (local) {
      // Unchanged since this browser last saw it, which is most of a full
      // pull: settled before paying for `normalise`.
      if (local.rev !== undefined && local.rev === numOrNull((raw as { rev?: unknown }).rev)) continue
      if (unpushed(local)) continue
    } else if (goneIds.has(id)) continue
    const e = normalise(raw, n)
    e.pending = undefined
    map.set(e.id, e)
    changed++
  }
  for (const id of removed) {
    // The server has it removed; nothing is left for this browser to send.
    goneIds.delete(id)
    outbox.delete(id)
    const local = map.get(id)
    if (local && local.rev !== undefined) {
      map.delete(id)
      changed++
    }
  }
  nextNo = Math.max(nextNo, serverNextNo)
  const moved = cursor !== undefined && (cursor.rev !== syncRev || cursor.epoch !== syncEpoch)
  if (cursor) {
    syncRev = cursor.rev
    syncEpoch = cursor.epoch
    serverBacked = true
  }
  if (!changed) {
    if (moved) save()
    return 0
  }
  // The server keeps every record; this browser keeps the newest of them, and
  // never one that exists only here.
  const merged = [...map.values()].sort((a, b) => b.at - a.at)
  commit(trimTo(merged, MAX_ENTRIES, unpushed), { remote: true, untrimmed: merged })
  return changed
}

/**
 * Stamp records the server just accepted with its revision and, when it
 * reassigned one, its edition number.
 *
 * `sent` is exactly what went, so the stamp lands on the content the server
 * now holds at that revision and on nothing else. A record changed here again
 * while the push was in flight keeps its mark and goes next; it takes only the
 * number, so it is not renumbered a second time. A record that was replaced
 * some other way meanwhile (another tab's save) becomes what was sent, stamped,
 * so this browser and the server never hold different content at one revision.
 */
export function applyServerMeta(
  assigned: readonly { id: string; no: number; rev: number }[],
  sent: readonly HistoryEntry[] = [],
): void {
  if (!assigned.length) return
  const meta = new Map(assigned.map((a) => [a.id, a]))
  const sentById = new Map(sent.map((e) => [e.id, e]))
  for (const e of sent) if (meta.has(e.id) && outbox.get(e.id) === e) outbox.delete(e.id)
  let changed = false
  const next = entries.map((e) => {
    const m = meta.get(e.id)
    if (!m) return e
    const went = sentById.get(e.id)
    if (!went || went === e) {
      if (e.rev === m.rev && e.no === m.no && !e.pending) return e
      changed = true
      return { ...e, rev: m.rev, no: m.no, pending: undefined }
    }
    if (unpushed(e)) {
      if (e.no === m.no) return e
      changed = true
      return { ...e, no: m.no }
    }
    changed = true
    return { ...went, rev: m.rev, no: m.no, pending: undefined }
  })
  if (!changed) return
  nextNo = Math.max(nextNo, next.reduce((mx, e) => Math.max(mx, e.no), 0) + 1)
  commit(next, { remote: true })
}

/**
 * Everything the server does not yet have from this browser: records it has
 * never stamped, records changed here since, and records the local cap
 * dropped before they could be sent.
 */
export function unsynced(): HistoryEntry[] {
  const out = entries.filter(unpushed)
  for (const e of outbox.values()) out.push(e)
  return out
}

/** Ids removed here that the server has not yet acknowledged removing. */
export function gone(): string[] {
  return [...goneIds]
}

/** The server has these removals. */
export function acknowledgeRemoved(ids: readonly string[]): void {
  let changed = false
  for (const id of ids) changed = goneIds.delete(id) || changed
  if (changed) save()
}

/** How many changes are waiting for the server. */
export function pendingCount(): number {
  let n = goneIds.size + outbox.size
  for (const e of entries) if (unpushed(e)) n++
  return n
}

/**
 * Drop records the server refused: a stale copy of one removed on another
 * device, or a second record for a file it already has. It is the server's
 * word, so it is not sent back.
 */
export function forget(ids: readonly string[]): void {
  if (!ids.length) return
  const drop = new Set(ids)
  for (const id of ids) outbox.delete(id)
  if (!entries.some((e) => drop.has(e.id))) return
  commit(
    entries.filter((e) => !drop.has(e.id)),
    { remote: true },
  )
}

/** Where this browser's copy stands in the server's log. */
export function syncCursor(): { rev: number; epoch: string | null } {
  return { rev: syncRev, epoch: syncEpoch }
}

/** Say whether a server holds the archive behind this browser. The sync is the one caller. */
export function setServerBacked(on: boolean): void {
  serverBacked = on
  if (!on) {
    roomFor = null
    outbox.clear()
  }
}

/** Drop the oldest unstarred records down to the cap. Starred are exempt. */
function trim(list: HistoryEntry[]): HistoryEntry[] {
  return trimTo(list, MAX_ENTRIES)
}

/**
 * Drop the oldest records down to `max`, never a starred one or one `keep`
 * covers. Oldest means furthest down the list, which is kept newest first.
 * Returns the same array when nothing needs to go.
 */
function trimTo(list: HistoryEntry[], max: number, keep?: (e: HistoryEntry) => boolean): HistoryEntry[] {
  let over = list.length - max
  if (over <= 0) return list
  const drop = new Set<HistoryEntry>()
  for (let i = list.length - 1; i >= 0 && over > 0; i--) {
    const e = list[i]
    if (e.starred || keep?.(e)) continue
    drop.add(e)
    over--
  }
  return drop.size ? list.filter((e) => !drop.has(e)) : list
}

/** Change one record — star it, note it, mark it missing. */
export function update(id: string, patch: Partial<HistoryEntry>): HistoryEntry | null {
  const i = entries.findIndex((e) => e.id === id)
  if (i < 0) return null
  const was = entries[i]
  const updated = {
    ...was,
    ...patch,
    id: was.id,
    no: was.no,
    // A stamped record changed here is marked until the server takes it.
    pending: was.rev !== undefined ? true : undefined,
  }
  const next = [...entries]
  next[i] = updated
  commit(next)
  return updated
}

export function star(id: string, starred = true): HistoryEntry | null {
  return update(id, { starred })
}

/**
 * Remove the record. The file stays on disk.
 *
 * This is the cheap, undoable verb behind "Remove from the archive", and it is
 * what ninety per cent of "delete" means here. Pass the returned record to
 * {@link restore} to put it back.
 */
export function remove(id: string): HistoryEntry | null {
  const entry = entries.find((e) => e.id === id)
  if (!entry) return null
  commit(entries.filter((e) => e.id !== id))
  return entry
}

export function removeMany(ids: readonly string[]): HistoryEntry[] {
  const set = new Set(ids)
  const removed = entries.filter((e) => set.has(e.id))
  if (!removed.length) return []
  commit(entries.filter((e) => !set.has(e.id)))
  return removed
}

/**
 * Put removed records back, in their original places. The undo.
 *
 * They come back unstamped. By the time the reader undoes, the removal may
 * already be on the server, and a stamped copy arriving after its removal is
 * refused there as stale. An unstamped one is a record arriving, which is what
 * an undo is.
 */
export function restore(...records: HistoryEntry[]): void {
  if (!records.length) return
  const known = new Set(entries.map((e) => e.id))
  const back = records
    .filter((r) => !known.has(r.id))
    .map((r) => ({ ...r, rev: undefined, pending: undefined }))
  if (!back.length) return
  commit([...back, ...entries].sort((a, b) => b.at - a.at))
}

/** Empty the archive. Files are untouched. */
export function clear(): void {
  commit([])
}

// ---------------------------------------------------------------------------
// Files on disk
// ---------------------------------------------------------------------------

export type DeleteResult =
  | { ok: true; freed: number }
  /** `reason` is written for a person to read. */
  | { ok: false; reason: string; unsupported?: boolean }

/**
 * Remove the record *and* the file from ComfyUI's output folder.
 *
 * Needs the local API (`POST /api/delete`), which only exists when the app is
 * served by the SwitchGen dev or preview server. On a bare static host the
 * call 404s and the result carries `unsupported: true`, so the UI can hide the
 * action rather than offer something that cannot work.
 *
 * The record is dropped only when the file actually went, so a failed delete
 * never loses you the settings.
 */
async function deleteFile(entry: HistoryEntry): Promise<DeleteResult> {
  let res: Response
  try {
    res = await fetch('/api/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kind: 'output', rel: relPath(entry.file) }),
    })
  } catch {
    return { ok: false, reason: 'The server did not answer.' }
  }

  if (res.status === 404 || res.status === 405) {
    const body = await res.json().catch(() => null)
    // The endpoint itself answers 404 for a file that is already gone; the
    // router answers 404 with a different body when there is no endpoint.
    if (body?.error === 'not found') {
      remove(entry.id)
      return { ok: true, freed: 0 }
    }
    return {
      ok: false,
      unsupported: true,
      reason:
        'Deleting files needs the SwitchGen dev or preview server. Right now we can only remove records; your files stay on disk.',
    }
  }

  if (!res.ok) {
    const body = await res.json().catch(() => null)
    return { ok: false, reason: String(body?.error ?? `The server refused (HTTP ${res.status}).`) }
  }

  const body = await res.json().catch(() => null)
  remove(entry.id)
  return { ok: true, freed: Number(body?.freed ?? 0) }
}

/** Delete several files. Each result is reported separately. */
export async function deleteFiles(
  records: readonly HistoryEntry[],
): Promise<{ entry: HistoryEntry; result: DeleteResult }[]> {
  const out: { entry: HistoryEntry; result: DeleteResult }[] = []
  for (const entry of records) out.push({ entry, result: await deleteFile(entry) })
  return out
}

/** What one audit found. `unanswered` records were left exactly as they were. */
export type AuditResult = { checked: number; missing: number; unanswered: number }

/**
 * Ask ComfyUI whether one file is there. True when it serves it, false only
 * when it answers 404, and null for everything else: a 502 from the proxy
 * while ComfyUI is down, a 5xx, the network gone. Only a 404 says the file is
 * not there; the rest say the question was not answered.
 */
async function fileIsThere(f: FileRef): Promise<boolean | null> {
  try {
    const r = await fetch(fileUrl(f), { method: 'HEAD' })
    if (r.ok) return true
    return r.status === 404 ? false : null
  } catch {
    return null
  }
}

/**
 * Check which records' files are still on disk, marking those that are not.
 *
 * A missing file is never an error: the record keeps its settings, so what was
 * lost can be made again. Runs in small batches so it can be done on idle.
 *
 * A record is marked only on a definite answer. The mark is shared with every
 * device, so a guess would put "moved or deleted" on files that are intact.
 * A batch with no answer at all ends the pass: ComfyUI is down or the network
 * is, and nothing after it would fare better.
 */
export async function checkMissing(
  records: readonly HistoryEntry[] = entries,
  batch = 20,
): Promise<AuditResult> {
  let checked = 0
  let missing = 0
  for (let i = 0; i < records.length; i += batch) {
    const slice = records.slice(i, i + batch)
    const found = await Promise.all(slice.map((e) => fileIsThere(e.file)))
    if (found.every((f) => f === null)) break
    slice.forEach((e, n) => {
      const there = found[n]
      if (there === null) return
      checked++
      if (!there) missing++
      if (Boolean(e.missing) !== !there) update(e.id, { missing: !there })
    })
  }
  return { checked, missing, unanswered: records.length - checked }
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

let indexCache = new WeakMap<HistoryEntry, string>()

/**
 * The lowercased haystack one record is matched against: the positive prompt as
 * it was actually submitted (the author-card prefix included), the negative, the
 * family and model in both machine and human spellings, the mode, the kind, the
 * desk, every filename the run produced, the source picture's filename and any
 * note.
 *
 * The prefix matters because it is prepended at submit time: a reader who saw
 * "masterpiece, best quality" on the record and searched for it was, until this
 * was fixed, told there were no such records. The same for the second and third
 * file of a multi-picture run, which the archive holds but did not index.
 */
function searchIndex(e: HistoryEntry): string {
  const cached = indexCache.get(e)
  if (cached !== undefined) return cached
  const built = [
    e.positivePrefix ?? '',
    e.prompt,
    e.negative ?? '',
    e.familyId,
    e.familyLabel,
    e.model,
    e.modelLabel,
    e.mode,
    e.kind,
    e.desk,
    e.file.filename,
    ...(e.files ?? []).map((f) => f.filename),
    e.source?.name ?? '',
    e.source?.ref?.filename ?? '',
    e.note ?? '',
    ...(e.tags ?? []).map((t) => t.replace(/_/g, ' ')),
  ]
    .join(' \u0000 ')
    .toLowerCase()
  indexCache.set(e, built)
  return built
}

type Term =
  | { kind: 'text'; value: string; negated: boolean }
  | { kind: 'field'; field: string; value: string; negated: boolean }

const TOKENS = /"([^"]*)"|(\S+)/g

function parse(query: string): Term[] {
  const terms: Term[] = []
  let m: RegExpExecArray | null
  TOKENS.lastIndex = 0
  while ((m = TOKENS.exec(query)) !== null) {
    let raw = m[1] !== undefined ? m[1] : m[2]
    if (!raw) continue
    let negated = false
    if (m[1] === undefined && raw.startsWith('-') && raw.length > 1) {
      negated = true
      raw = raw.slice(1)
    }
    const colon = m[1] === undefined ? raw.indexOf(':') : -1
    if (colon > 0 && colon < raw.length - 1) {
      terms.push({
        kind: 'field',
        field: raw.slice(0, colon).toLowerCase(),
        value: raw.slice(colon + 1).toLowerCase(),
        negated,
      })
    } else {
      terms.push({ kind: 'text', value: raw.toLowerCase(), negated })
    }
  }
  return terms
}

function matchField(e: HistoryEntry, field: string, value: string): boolean {
  switch (field) {
    case 'is':
      switch (value) {
        case 'image':
        case 'video':
          return e.kind === value
        case 'starred':
          return !!e.starred
        case 'missing':
          return !!e.missing
        case 'source':
          return !!e.source
        case 'tagged':
          return !!e.tags?.length
        default:
          return false
      }
    case 'kind':
      return e.kind === value
    case 'desk':
      return e.desk === value
    case 'mode':
      return e.mode === value
    case 'model':
      return `${e.model} ${e.modelLabel}`.toLowerCase().includes(value)
    case 'family':
      return `${e.familyId} ${e.familyLabel}`.toLowerCase().includes(value)
    case 'tag': {
      // Booru spelling has underscores; a person types spaces. Match either.
      const want = value.replace(/_/g, ' ')
      return (e.tags ?? []).some((t) => t.replace(/_/g, ' ').includes(want))
    }
    case 'seed':
      return String(e.seed) === value
    case 'no':
      return String(e.no) === value.replace(/[.,]/g, '')
    default:
      // An unknown prefix is just text — `neon:lit` should still find itself.
      return searchIndex(e).includes(`${field}:${value}`)
  }
}

/** True when one record satisfies the whole query. */
export function matches(e: HistoryEntry, query: string): boolean {
  const terms = parse(query)
  for (const t of terms) {
    const hit =
      t.kind === 'text' ? searchIndex(e).includes(t.value) : matchField(e, t.field, t.value)
    if (hit === t.negated) return false
  }
  return true
}

/**
 * Search the archive, newest first.
 *
 * All terms must match. `"a phrase"` matches exactly, `-word` excludes, and
 * these prefixes narrow: `is:image` `is:video` `is:starred` `is:missing`
 * `is:source`, `kind:` `desk:` `mode:` `model:` `family:` `seed:` `no:`.
 * Everything else is matched as text against {@link searchIndex}: the submitted
 * prompt and its prefix, the negative, the family and model names, the mode,
 * every filename the run produced, the source picture and the note.
 *
 * A plain `Array.filter` over a cached index string. At five thousand records
 * that is well under a frame, so there is no library and no worker.
 */
export function search(query: string, within: readonly HistoryEntry[] = entries): HistoryEntry[] {
  const q = query.trim()
  if (!q) return [...within]
  const terms = parse(q)
  if (!terms.length) return [...within]
  return within.filter((e) => {
    for (const t of terms) {
      const hit =
        t.kind === 'text' ? searchIndex(e).includes(t.value) : matchField(e, t.field, t.value)
      if (hit === t.negated) return false
    }
    return true
  })
}

// ---------------------------------------------------------------------------
// Export and import
// ---------------------------------------------------------------------------

/**
 * This browser's archive as JSON, what "Save a copy" downloads when there is
 * no server to ask for the whole of it. The sync bookkeeping stays behind: it
 * describes this browser, not the records.
 */
export function exportJson(): string {
  return JSON.stringify({ v: HISTORY_VERSION, nextNo, entries }, null, 2)
}

/**
 * Merge a saved copy back in. Records are matched by id; the newer `at` wins.
 *
 * A record this browser does not hold comes in unstamped, like an undo:
 * restoring from a file is a deliberate act, and the server takes an unstamped
 * record even where it had been removed. Past the local cap the oldest records
 * do not stay in this browser; with a server behind it they are still sent
 * there and counted, and without one they are not counted, because they are
 * not kept anywhere.
 * @returns how many records were added or updated.
 */
export function importJson(text: string): number {
  let parsed: any
  try {
    parsed = JSON.parse(text)
  } catch {
    throw new Error('That file is not a SwitchGen archive.')
  }
  const incoming: unknown[] = Array.isArray(parsed) ? parsed : (parsed?.entries ?? [])
  if (!Array.isArray(incoming)) throw new Error('That file is not a SwitchGen archive.')

  const byId = new Map(entries.map((e) => [e.id, e]))
  const touched: string[] = []
  let n = 0
  for (const raw of incoming) {
    if (!sane(raw)) continue
    const e = normalise(raw, ++n)
    const existing = byId.get(e.id)
    if (!existing || e.at > existing.at) {
      byId.set(
        e.id,
        existing
          ? // The server's stamp on this browser's copy is the one that counts, not the file's.
            { ...existing, ...e, rev: existing.rev, pending: existing.rev !== undefined ? true : undefined }
          : { ...e, no: e.no || nextNo++, rev: undefined, pending: undefined },
      )
      touched.push(e.id)
    }
  }
  if (!touched.length) return 0
  const merged = [...byId.values()].sort((a, b) => b.at - a.at)
  nextNo = Math.max(nextNo, merged.reduce((m, e) => Math.max(m, e.no), 0) + 1)
  const kept = trim(merged)
  commit(kept, { untrimmed: merged })
  if (serverBacked || kept === merged) return touched.length
  const keptIds = new Set(kept.map((e) => e.id))
  return touched.filter((id) => keptIds.has(id)).length
}

/**
 * The store as one object, for code that would rather pass it around than
 * import a dozen names. Same functions, no copies.
 */
export const history = {
  subscribe,
  all,
  count,
  get,
  add,
  addMany,
  update,
  star,
  remove,
  removeMany,
  restore,
  clear,
  mergeFromServer,
  applyServerMeta,
  unsynced,
  gone,
  acknowledgeRemoved,
  pendingCount,
  forget,
  syncCursor,
  setServerBacked,
  deleteFile,
  deleteFiles,
  checkMissing,
  search,
  matches,
  searchIndex,
  exportJson,
  importJson,
  flush,
  loadIssue,
  clearLoadIssue,
  quotaIssue,
  clearQuotaIssue,
  isPersistent,
}
