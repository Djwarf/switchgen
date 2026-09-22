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
 *     the oldest unstarred records are evicted and the write retried once.
 *
 * React: `useSyncExternalStore(subscribe, all)`. `all()` returns the same array
 * reference until something actually changes.
 */

import { headFile, relPath, type FileRef } from './comfy'
import { isQuotaError, onStorage, store, storageWorks, type DeskId, type Mode } from './session'

export type { FileRef, DeskId, Mode }

// ---------------------------------------------------------------------------
// The record
// ---------------------------------------------------------------------------

const HISTORY_KEY = 'switchgen.archive.v2'
const BROKEN_KEY = 'switchgen.archive.v2.broken'
const HISTORY_VERSION = 2

/**
 * How many records are kept. Roughly 800 bytes for a picture and 1.1 KB for a
 * clip, so 5,000 records is about 4.5 MB — inside a 5 MB localStorage budget,
 * and about two years of heavy use. Beyond the cap the oldest unstarred
 * records fall off the end.
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

type Envelope = { v: number; nextNo: number; entries: HistoryEntry[] }

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

let entries: HistoryEntry[] = []
let nextNo = 1
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

function envelope(): Envelope {
  return { v: HISTORY_VERSION, nextNo, entries }
}

function writeNow(): void {
  saveTimer = null
  const body = JSON.stringify(envelope())
  try {
    store.set(HISTORY_KEY, body)
    return
  } catch (err) {
    if (!isQuotaError(err)) return
  }

  // Full. Shed the oldest unstarred records and try once more. A generation
  // must never fail because the archive filled up.
  const before = entries.length
  const starred = entries.filter((e) => e.starred)
  const rest = entries.filter((e) => !e.starred)
  const evicted = Math.min(EVICT_ON_QUOTA, rest.length)
  entries = [...starred, ...rest.slice(0, Math.max(0, rest.length - evicted))].sort(
    (a, b) => b.at - a.at,
  )
  try {
    store.set(HISTORY_KEY, JSON.stringify(envelope()))
    quotaMessage = `Your archive was full, so we removed the ${before - entries.length} oldest records. The files are still on disk.`
  } catch {
    quotaMessage =
      'Your archive is full and we could not save it. Save a copy, then clear out some records.'
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

function commit(next: HistoryEntry[], opts: { remote?: boolean; trimmed?: boolean } = {}): void {
  const prev = entries
  entries = next
  indexCache = new WeakMap()
  save()
  announce()
  if (!commitListeners.size) return
  // Records are replaced, never mutated, so identity says what changed.
  const prevById = new Map(prev.map((e) => [e.id, e]))
  const nextIds = new Set(next.map((e) => e.id))
  const upserted = next.filter((e) => prevById.get(e.id) !== e)
  // A trim to the local cap is this browser running out of room, not the
  // reader removing anything: it is never reported as a removal.
  const removed = opts.trimmed ? [] : prev.filter((e) => !nextIds.has(e.id)).map((e) => e.id)
  if (!upserted.length && !removed.length) return
  const delta: CommitDelta = { upserted, removed, remote: opts.remote === true }
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
  commit(next.length > MAX_ENTRIES ? trim(next) : next, { trimmed: next.length > MAX_ENTRIES })
  return entry
}

// ---------------------------------------------------------------------------
// The server's copy
// ---------------------------------------------------------------------------

/**
 * Merge what the server sent. The server wins by id. A removal is honoured
 * only for a record the server had stamped: a local record it has never seen
 * cannot have been removed there, and stays to be pushed.
 */
export function mergeFromServer(records: readonly unknown[], removed: readonly string[], serverNextNo: number): number {
  const map = new Map(entries.map((e) => [e.id, e]))
  let n = 0
  let changed = 0
  for (const raw of records) {
    if (!sane(raw)) continue
    const e = normalise(raw, ++n)
    const local = map.get(e.id)
    if (local && local.rev === e.rev && local.rev !== undefined) continue
    map.set(e.id, e)
    changed++
  }
  for (const id of removed) {
    const local = map.get(id)
    if (local && local.rev !== undefined) {
      map.delete(id)
      changed++
    }
  }
  nextNo = Math.max(nextNo, serverNextNo)
  if (!changed) return 0
  commit([...map.values()].sort((a, b) => b.at - a.at), { remote: true })
  return changed
}

/** Stamp records the server just accepted with its revision and, when it reassigned one, its edition number. */
export function applyServerMeta(assigned: readonly { id: string; no: number; rev: number }[]): void {
  if (!assigned.length) return
  const meta = new Map(assigned.map((a) => [a.id, a]))
  let changed = false
  const next = entries.map((e) => {
    const m = meta.get(e.id)
    if (!m || (e.rev === m.rev && e.no === m.no)) return e
    changed = true
    return { ...e, rev: m.rev, no: m.no }
  })
  if (!changed) return
  nextNo = Math.max(nextNo, next.reduce((mx, e) => Math.max(mx, e.no), 0) + 1)
  commit(next, { remote: true })
}

/** Records the server has never stamped. */
export function unsynced(): HistoryEntry[] {
  return entries.filter((e) => e.rev === undefined)
}

/** Drop the oldest unstarred records down to the cap. Starred are exempt. */
function trim(list: HistoryEntry[]): HistoryEntry[] {
  const over = list.length - MAX_ENTRIES
  if (over <= 0) return list
  const out = [...list]
  for (let i = out.length - 1; i >= 0 && out.length > MAX_ENTRIES; i--) {
    if (!out[i].starred) out.splice(i, 1)
  }
  return out
}

/** Change one record — star it, note it, mark it missing. */
export function update(id: string, patch: Partial<HistoryEntry>): HistoryEntry | null {
  const i = entries.findIndex((e) => e.id === id)
  if (i < 0) return null
  const updated = { ...entries[i], ...patch, id: entries[i].id, no: entries[i].no }
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

/** Put removed records back, in their original places. The undo. */
export function restore(...records: HistoryEntry[]): void {
  if (!records.length) return
  const known = new Set(entries.map((e) => e.id))
  const back = records.filter((r) => !known.has(r.id))
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

/**
 * Check which records' files are still on disk, marking those that are not.
 *
 * A missing file is never an error: the record keeps its settings, so what was
 * lost can be made again. Runs in small batches so it can be done on idle.
 */
export async function checkMissing(
  records: readonly HistoryEntry[] = entries,
  batch = 20,
): Promise<number> {
  let missingCount = 0
  for (let i = 0; i < records.length; i += batch) {
    const slice = records.slice(i, i + batch)
    const found = await Promise.all(slice.map((e) => headFile(e.file)))
    slice.forEach((e, n) => {
      const missing = !found[n]
      if (missing) missingCount++
      if (Boolean(e.missing) !== missing) update(e.id, { missing })
    })
  }
  return missingCount
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

/** The whole archive as JSON — what "Save a copy" downloads. */
export function exportJson(): string {
  return JSON.stringify(envelope(), null, 2)
}

/**
 * Merge a saved copy back in. Records are matched by id; the newer `at` wins.
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
  let changed = 0
  let n = 0
  for (const raw of incoming) {
    if (!sane(raw)) continue
    const e = normalise(raw, ++n)
    const existing = byId.get(e.id)
    if (!existing || e.at > existing.at) {
      byId.set(e.id, existing ? { ...existing, ...e } : { ...e, no: e.no || nextNo++ })
      changed++
    }
  }
  if (!changed) return 0
  const merged = [...byId.values()].sort((a, b) => b.at - a.at)
  nextNo = Math.max(nextNo, merged.reduce((m, e) => Math.max(m, e.no), 0) + 1)
  commit(merged.length > MAX_ENTRIES ? trim(merged) : merged)
  return changed
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
  update,
  star,
  remove,
  removeMany,
  restore,
  clear,
  mergeFromServer,
  applyServerMeta,
  unsynced,
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
