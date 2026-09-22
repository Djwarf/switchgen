/**
 * The reel engine.
 *
 * A reel of ten shots is ten separate generations of several minutes each. They
 * are run one at a time, in order, because shot N+1 literally cannot be built
 * until shot N has written its last frame to disk, and because the measured
 * peak that fits in this machine was measured one job at a time.
 *
 * A module singleton, like the video desk's engine and for the same reason: an
 * hour of rendering must survive the reader walking to another section, and the
 * shell unmounting the route. What it has rendered also survives a reload: the
 * finished clips are saved, and a shot that was on the press when the page went
 * away is picked up again from ComfyUI's own queue rather than rendered twice.
 *
 * WHAT THE ENGINE OWNS
 *   the queue, the per shot state, and the handoff frames.
 * WHAT IT DOES NOT OWN
 *   the plan. Jobs arrive already built by `shotPlan`, so reordering, pinning
 *   and family choice stay in one place and the engine never second guesses
 *   them.
 *
 * STALENESS, SAID OUT LOUD
 * Re-render shot 4 and shots 5 onward still open on the frame shot 4 used to
 * end on. They are not wrong, they are out of date, and the difference matters:
 * the reader may be perfectly happy with them. So the strip says so, rather
 * than silently wiping or silently re-queueing them.
 *
 * Out of date is worked out, never remembered. Each clip records what it was
 * made from (`made`), and `currencyOf` compares that with the strip as it is
 * now. A flag set at render time could only ever notice a re-render; comparing
 * notices a cut, a move, a pasted reel, an edited line and a changed bench
 * just the same, because they all change what the clip would be made from.
 */
import { useSyncExternalStore } from 'react'

import { releaseComfyMemory, waitForIdleComfy, type ClipMemory } from '../../lib/clipMemory'
import {
  cancelJob,
  fetchPastRun,
  getJob,
  relPath,
  run,
  type OutputFile,
  type PastRun,
  type ProgressEvent,
  type ServerJob,
} from '../../lib/comfy'
import { faultBody, faultOf, faultWhere } from '../../lib/faults'
import {
  annotatedRef,
  chainFrameOf,
  clipFrames,
  instantiateShot,
  jobSignature,
  type ShotJob,
} from '../../lib/continuation'
import { history } from '../../lib/history'
import { onStorage, recordOf, store as kv, type Composition } from '../../lib/session'

export type ShotStatus = 'waiting' | 'queued' | 'running' | 'done' | 'error' | 'stopped'

/** What a clip on disk was made from. Recorded when it lands. */
export type Made = {
  /** `jobSignature` of the job that made the clip. */
  signature: string
  /** The seed it was rendered with. */
  seed: number
  /** The handoff frame a continued shot opened on, as an annotated ref. Null for any other start. */
  openedOn: string | null
  /** Frames in the clip as written, and the rate and size it was written at. */
  frames: number
  fps: number
  width: number
  height: number
}

export type ShotState = {
  shotId: string
  status: ShotStatus
  promptId: string | null
  /** Sampler steps. 0 and 0 until the sampler reports. */
  value: number
  max: number
  stage: string
  previewUrl: string | null
  startedAt: number | null
  finishedAt: number | null
  durationMs: number
  error: string | null
  /** "The trouble is in LoadImage (node 7)." when ComfyUI named the node. */
  detail: string | null
  files: OutputFile[]
  /** The clip itself. */
  clip: OutputFile | null
  /** The last frame, which the next shot opens on. */
  frame: OutputFile | null
  entryId: string | null
  /** What the clip on disk was made from. Null until a render lands. */
  made: Made | null
  /** Frames this pass asks for, so the band can weight progress by real work. */
  frames: number
}

export type RunStatus = 'idle' | 'running' | 'done' | 'stopped' | 'error'

export type RunState = {
  id: string
  status: RunStatus
  startedAt: number | null
  finishedAt: number | null
  /** Shot ids in reel order when the pass started. */
  order: string[]
  states: Record<string, ShotState>
  currentShotId: string | null
  /**
   * Shot ids this pass set out to render, in queue order. The band counts and
   * times these, not the whole reel: one shot rendered alone is a pass of one.
   */
  queue: string[]
  stopRequested: boolean
  /** One line about why the queue stopped where it did. */
  note: string | null
}

/** Everything the engine needs from the desk to file a finished clip. */
export type RunContext = {
  familyLabel: string
  modelLabel: string
  /** The archive record's settings, per job. The desk knows the family. */
  compositionFor: (job: ShotJob) => Composition
  /**
   * The memory verdict for one clip (lib/clipMemory). A refused clip is never
   * sent, and a clip whose verdict says release waits for ComfyUI's queue to
   * empty, has its cached models released, and is queued straight after.
   */
  memory?: (job: ShotJob) => ClipMemory
}

function blankShot(shotId: string, frames: number): ShotState {
  return {
    shotId,
    status: 'waiting',
    promptId: null,
    value: 0,
    max: 0,
    stage: '',
    previewUrl: null,
    startedAt: null,
    finishedAt: null,
    durationMs: 0,
    error: null,
    detail: null,
    files: [],
    clip: null,
    frame: null,
    entryId: null,
    made: null,
    frames,
  }
}

const IDLE: RunState = {
  id: 'idle',
  status: 'idle',
  startedAt: null,
  finishedAt: null,
  order: [],
  states: {},
  currentShotId: null,
  queue: [],
  stopRequested: false,
  note: null,
}

let state: RunState = IDLE
const listeners = new Set<() => void>()
/** One walk at a time. A second press while the queue moves is ignored. */
let walking = false
/** The wait for ComfyUI's queue to empty before a shot that needs memory released, so Stop can end it. */
let waiting: AbortController | null = null

function emit(): void {
  for (const fn of [...listeners]) {
    try {
      fn()
    } catch {
      /* one bad subscriber must not strand the rest */
    }
  }
}

function setRun(patch: Partial<RunState>): void {
  state = { ...state, ...patch }
  emit()
}

function setShot(shotId: string, patch: Partial<ShotState>): void {
  const current = state.states[shotId]
  if (!current) return
  state = { ...state, states: { ...state.states, [shotId]: { ...current, ...patch } } }
  emit()
}

function shotOf(shotId: string | undefined): ShotState | null {
  if (!shotId) return null
  return state.states[shotId] ?? null
}

const delay = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms))

// ---------------------------------------------------------------------------
// Is a clip still the clip the strip describes?
// ---------------------------------------------------------------------------

/**
 *   current  nothing that decides the clip has moved since it was made
 *   changed  its line, length, pins, the bench, or a fixed seed has moved
 *   stale    it continues the shot before it, and that shot's last frame is no
 *            longer the frame this clip opened on: the shot before was
 *            rendered again, cut, moved, or replaced
 */
export type Currency = 'current' | 'changed' | 'stale'

/**
 * Whether the rendered shot at `index` still matches the strip. Null for a
 * shot with no finished clip.
 *
 * A seed counts only when it is fixed. On a Random reel every render draws a
 * new one, so the seed a clip was made with is not something the reader asked
 * for, and a different one is not an edit.
 */
export function currencyOf(
  index: number,
  order: readonly string[],
  jobs: readonly ShotJob[],
  states: Readonly<Record<string, ShotState>>,
): Currency | null {
  const shotId = order[index]
  const shot = shotId ? states[shotId] : undefined
  const job = jobs[index]
  if (!shot || !job || shot.status !== 'done' || !shot.clip) return null
  if (!shot.made || shot.made.signature !== jobSignature(job)) return 'changed'
  if (job.seedFixed && shot.made.seed !== job.params.seed) return 'changed'
  if (job.start.from === 'previous') {
    const upstream = states[order[index - 1] ?? '']?.frame
    if (!upstream || annotatedRef(upstream) !== shot.made.openedOn) return 'stale'
  }
  return 'current'
}

/**
 * The shots a whole-reel pass would render, as indices into `order`.
 *
 * A shot needs rendering when it has no current clip of its own. Then the need
 * spreads forward: every shot that opens on the one before it has to follow,
 * because the frame it was going to open on is about to change. `force`
 * renders every shot.
 */
export function shotsToRender(
  order: readonly string[],
  jobs: readonly ShotJob[],
  states: Readonly<Record<string, ShotState>>,
  force = false,
): number[] {
  const need = new Set<number>()
  order.forEach((shotId, i) => {
    const current = !force && !!states[shotId]?.frame && currencyOf(i, order, jobs, states) === 'current'
    if (!current) need.add(i)
  })
  for (let i = 1; i < order.length; i++) {
    if (need.has(i - 1) && jobs[i]?.start.from === 'previous') need.add(i)
  }
  return [...need].sort((a, b) => a - b)
}

// ---------------------------------------------------------------------------
// Keeping the run across a reload
//
// Only the draft used to be saved, so a reload forgot every rendered shot: the
// strip read "unwritten", the cutting room emptied, and the next press queued
// the whole reel again behind the job still running in ComfyUI. Now every clip
// on disk is remembered with what it was made from, and so is the one shot on
// the press, by prompt id, so the next page load can follow it to the end.
// ---------------------------------------------------------------------------

const RUN_KEY = 'switchgen.reelrun.v1'

/**
 * Every open tab loads this engine, and they share one saved run. So the shot
 * on the press is saved with the tab that is following it, and that tab says
 * it is still there every few seconds. Another tab leaves a shot alone while
 * its follower is alive: two tabs following one prompt could file its clip
 * twice, and whichever finished second wrote back a strip that was out of
 * date, without the next shot on the press.
 *
 * The allowance is long because browsers slow the timers of a tab nobody is
 * looking at (Chrome to once a minute, after five minutes hidden), and that
 * is usually the tab the reel is rendering in.
 */
const TAB = newRunId()
const BEAT_MS = 5_000
const STALE_MS = 150_000
/** How often a tab that is not following the shot checks whether its follower has gone. */
const WATCH_MS = 15_000

/** The shot on the press, with everything needed to file it if the page goes away. */
type Pending = {
  shotId: string
  promptId: string
  /** "Shot 7", for the notes. */
  label: string
  order: string[]
  startedAt: number
  made: Made
  frames: number
  composition: Composition
  familyLabel: string
  modelLabel: string
}

/** Who is following a saved shot on the press. Stamped when it is written. */
type Stamp = {
  /** The tab following it. */
  owner: string
  /** When that tab last wrote, in epoch ms. */
  beat: number
  /** True when that page went away (a reload, a close) and left it for the next one. */
  released: boolean
}

type StoredPending = Pending & Stamp

type SavedShot = Pick<
  ShotState,
  'shotId' | 'clip' | 'frame' | 'files' | 'entryId' | 'durationMs' | 'finishedAt' | 'made'
>

/** The shot this tab is following, if any. */
let pending: Pending | null = null
let beatTimer: ReturnType<typeof setInterval> | null = null
let watchTimer: ReturnType<typeof setTimeout> | null = null
/** The saved shots this tab last wrote or took in, as written. */
let mirrored = ''

/** Set or clear the shot this tab follows, keep its heartbeat going while there is one, and save. */
function setPending(p: Pending | null): void {
  pending = p
  if (p && !beatTimer) beatTimer = setInterval(() => persist(), BEAT_MS)
  if (!p && beatTimer) {
    clearInterval(beatTimer)
    beatTimer = null
  }
  persist()
}

/**
 * Write every clip on disk, whatever its status says: a shot waiting to be
 * rendered again, or one whose new render failed, still has its earlier clip.
 * Written immediately, not debounced, because it runs at shot boundaries and
 * on the heartbeat, and the moment it matters is the moment before the page
 * goes.
 *
 * A shot another tab is following is written back as it was found. This tab's
 * view of the run can be behind that tab's, and leaving the entry out would
 * forget the shot if that tab then closed.
 */
function persist(opts: { released?: boolean } = {}): void {
  let entry: StoredPending | null = null
  if (pending) entry = { ...pending, owner: TAB, beat: Date.now(), released: opts.released ?? false }
  else {
    const theirs = readStored(kv.get(RUN_KEY))?.pending ?? null
    if (theirs && theirs.owner !== TAB) entry = theirs
  }

  const shots: SavedShot[] = []
  for (const s of Object.values(state.states)) {
    if (!s.clip || !s.made) continue
    shots.push({
      shotId: s.shotId,
      clip: s.clip,
      frame: s.frame,
      files: s.files,
      entryId: s.entryId,
      durationMs: s.durationMs,
      finishedAt: s.finishedAt,
      made: s.made,
    })
  }
  mirrored = JSON.stringify(shots)
  try {
    if (!shots.length && !entry) kv.remove(RUN_KEY)
    else kv.set(RUN_KEY, JSON.stringify({ shots, pending: entry }))
  } catch {
    // A full quota costs the saved copy, never the run on screen.
  }
}

const isObject = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object'
const isString = (v: unknown): v is string => typeof v === 'string'
const isNumber = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)

function isFile(v: unknown): v is OutputFile {
  return isObject(v) && isString(v.filename) && isString(v.subfolder) && isString(v.type)
}

function isMade(v: unknown): v is Made {
  return (
    isObject(v) &&
    isString(v.signature) &&
    isNumber(v.seed) &&
    (v.openedOn === null || isString(v.openedOn)) &&
    isNumber(v.frames) &&
    isNumber(v.fps) &&
    isNumber(v.width) &&
    isNumber(v.height)
  )
}

function readSaved(v: unknown): ShotState | null {
  if (!isObject(v) || !isString(v.shotId) || !isFile(v.clip) || !isMade(v.made)) return null
  return {
    ...blankShot(v.shotId, v.made.frames),
    status: 'done',
    stage: 'Done',
    clip: v.clip,
    frame: isFile(v.frame) ? v.frame : null,
    files: Array.isArray(v.files) ? v.files.filter(isFile) : [],
    entryId: isString(v.entryId) ? v.entryId : null,
    durationMs: isNumber(v.durationMs) ? v.durationMs : 0,
    finishedAt: isNumber(v.finishedAt) ? v.finishedAt : null,
    made: v.made,
  }
}

function readPending(v: unknown): StoredPending | null {
  if (
    !isObject(v) ||
    !isString(v.shotId) ||
    !isString(v.promptId) ||
    !isString(v.label) ||
    !Array.isArray(v.order) ||
    !v.order.every(isString) ||
    !isNumber(v.startedAt) ||
    !isMade(v.made) ||
    !isNumber(v.frames) ||
    !isObject(v.composition) ||
    !isString(v.familyLabel) ||
    !isString(v.modelLabel)
  ) {
    return null
  }
  // An entry saved before entries were stamped reads as unowned and long
  // quiet, so the next page picks it up as it always did.
  return {
    ...(v as unknown as Pending),
    owner: isString(v.owner) ? v.owner : '',
    beat: isNumber(v.beat) ? v.beat : 0,
    released: v.released === true,
  }
}

type Saved = {
  states: Record<string, ShotState>
  pending: StoredPending | null
  /** The saved shots as written, to tell a write that changed them from a heartbeat. */
  shotsKey: string
}

/** The saved run, or null when there is none or it cannot be read. */
function readStored(raw: string | null): Saved | null {
  let v: unknown
  try {
    v = JSON.parse(raw ?? 'null')
  } catch {
    return null
  }
  if (!isObject(v)) return null
  const list: unknown[] = Array.isArray(v.shots) ? v.shots : []
  const states: Record<string, ShotState> = {}
  for (const s of list) {
    const shot = readSaved(s)
    if (shot) states[shot.shotId] = shot
  }
  return { states, pending: readPending(v.pending), shotsKey: JSON.stringify(list) }
}

/** File a finished clip, unless the archive already holds that exact file. */
function fileClip(
  clip: OutputFile,
  files: OutputFile[],
  record: { compose: () => Composition; familyLabel: string; modelLabel: string },
  promptId: string,
  seed: number,
  startedAt: number,
  finishedAt: number,
): { entryId: string | null; durationMs: number; unchanged: boolean } {
  // ComfyUI answers a graph it has already run from its cache, and the answer
  // is the clip it wrote last time, returned in the time it takes to look it
  // up. Filing that again would add a second record for one file, with a
  // duration that measures nothing and would drag every finish time the band
  // prints towards zero. So the existing record stands, with its real time.
  const known = history.all().find((e) => relPath(e.file) === relPath(clip))
  if (known) return { entryId: known.id, durationMs: known.durationMs, unchanged: true }
  try {
    const entry = history.add(
      recordOf(record.compose(), {
        file: clip,
        files: files.length > 1 ? files : undefined,
        kind: 'video',
        promptId,
        durationMs: finishedAt - startedAt,
        seed,
        familyLabel: record.familyLabel,
        modelLabel: record.modelLabel,
        at: finishedAt,
      }),
    )
    return { entryId: entry.id, durationMs: finishedAt - startedAt, unchanged: false }
  } catch {
    // A full archive must never cost the reader the clip itself.
    return { entryId: null, durationMs: finishedAt - startedAt, unchanged: false }
  }
}

// ---------------------------------------------------------------------------
// Following a shot that outlived its page
// ---------------------------------------------------------------------------

/** How often the queue is asked about a shot picked up after a reload. */
const POLL_MS = 4000
/** Unanswered asks in a row before the desk stops waiting (about a minute). */
const POLL_GIVE_UP = 15

/**
 * The page that queued this shot went away while it was on the press. The
 * socket that would have reported it belonged to that page, so the shot is
 * followed through ComfyUI's job list instead, and filed when it lands.
 * `kept` is the clip the shot had before, which comes back if this one does
 * not arrive.
 */
async function resume(p: Pending, kept: ShotState | null): Promise<void> {
  let verdict: RunStatus = 'error'
  let note: string | null = null
  try {
    ;[verdict, note] = await follow(p)
  } catch {
    note = `${p.label} was left on the press by the page before this one, and following it failed. If it finishes, the Archive's recover action files it.`
  }

  if (verdict !== 'done') {
    if (kept) state = { ...state, states: { ...state.states, [p.shotId]: kept } }
    else {
      setShot(p.shotId, {
        status: verdict === 'stopped' ? 'stopped' : 'error',
        stage: verdict === 'stopped' ? 'Stopped' : 'Failed',
        error: verdict === 'stopped' ? null : note,
        finishedAt: Date.now(),
      })
    }
  }

  walking = false
  holdUnload(false)
  setRun({ status: verdict, finishedAt: Date.now(), currentShotId: null, stopRequested: false, note })
  setPending(null)
}

/** Poll one job to its end, and file its clip if it lands. */
async function follow(p: Pending): Promise<[RunStatus, string]> {
  const left = `${p.label} was left on the press by the page before this one`
  let misses = 0
  let historyMisses = 0
  for (;;) {
    let job: ServerJob | null | undefined
    try {
      job = await getJob(p.promptId)
    } catch {
      job = undefined
    }
    if (job === undefined) {
      if (++misses < POLL_GIVE_UP) {
        await delay(POLL_MS)
        continue
      }
      return ['error', `${left}, and ComfyUI is not answering, so the desk stopped waiting for it. If it finishes, the Archive's recover action files it.`]
    }
    misses = 0
    if (job && (job.status === 'pending' || job.status === 'in_progress')) {
      const waiting = job.status === 'pending'
      setShot(p.shotId, {
        status: waiting ? 'queued' : 'running',
        stage: state.stopRequested ? 'Stopping' : waiting ? 'Queued' : 'Drawing',
      })
      await delay(POLL_MS)
      continue
    }
    // Stop takes a waiting job out of ComfyUI's queue, and a job taken out
    // that way leaves no record at all. So gone after a stop is the stop
    // landing, not a job that was lost.
    if (!job) {
      return state.stopRequested
        ? ['stopped', `${p.label} was stopped.`]
        : ['error', `${left}, and ComfyUI no longer knows the job, so it did not finish.`]
    }
    if (job.status === 'cancelled') return ['stopped', `${p.label} was stopped.`]
    if (job.status === 'failed') return ['error', `${left}, and it failed in ComfyUI.`]

    // This one prompt's record, asked for by id. A page of recent history
    // could already have scrolled past a shot that finished a while ago.
    let past: PastRun | null
    try {
      past = await fetchPastRun(p.promptId)
    } catch {
      // Could not ask, which is not the same as no record.
      if (++historyMisses < POLL_GIVE_UP) {
        await delay(POLL_MS)
        continue
      }
      return ['error', `${left}. It finished, but ComfyUI did not answer when asked for its clip, so the desk stopped waiting. The Archive's recover action can still file it.`]
    }
    const files = past?.files ?? []
    const clip = files.find((f) => f.kind === 'video') ?? null
    if (!clip) {
      return ['error', `${left}. It finished, but ComfyUI's history does not list its clip. The Archive's recover action can still file it.`]
    }
    const finishedAt = past?.finishedAt ?? Date.now()
    const filed = fileClip(
      clip,
      files,
      { compose: () => p.composition, familyLabel: p.familyLabel, modelLabel: p.modelLabel },
      p.promptId,
      p.made.seed,
      p.startedAt,
      finishedAt,
    )
    setShot(p.shotId, {
      status: 'done',
      stage: 'Done',
      files,
      clip,
      frame: chainFrameOf(files),
      entryId: filed.entryId,
      finishedAt,
      durationMs: filed.durationMs,
      previewUrl: null,
      error: null,
      detail: null,
      made: p.made,
    })
    return ['done', `${left}. It has finished and is filed with the rest.`]
  }
}

/** Read what the last page left behind, once, when the module loads. */
function restore(): void {
  const saved = readStored(kv.get(RUN_KEY))
  if (!saved) return
  mirrored = saved.shotsKey
  if (Object.keys(saved.states).length) state = { ...IDLE, states: saved.states }
  pickUp(saved.pending, true)
}

/**
 * Follow a shot a page left on the press, unless a tab that is still open is
 * following it. A page that went away released it, and the next page to load
 * takes it at once: that is usually the same tab, reloaded. A tab that was
 * already open waits for the entry to go quiet instead, so a reload gets its
 * own shot back, and a tab that closed or crashed is still covered.
 */
function pickUp(p: StoredPending | null, atLoad: boolean): void {
  if (!p || p.owner === TAB) return
  const free = Date.now() - p.beat >= STALE_MS || (atLoad && p.released)
  if (walking || !free) {
    watchPress()
    return
  }

  const kept = state.states[p.shotId] ?? null
  state = {
    ...IDLE,
    id: newRunId(),
    status: 'running',
    startedAt: p.startedAt,
    order: p.order,
    queue: [p.shotId],
    states: {
      ...state.states,
      [p.shotId]: {
        ...(kept ?? blankShot(p.shotId, p.frames)),
        status: 'running',
        stage: 'Still on the press from before the reload',
        promptId: p.promptId,
        startedAt: p.startedAt,
        finishedAt: null,
        error: null,
        detail: null,
        frames: p.frames,
      },
    },
    currentShotId: p.shotId,
  }
  walking = true
  holdUnload(true)
  // Claimed before anything else runs, so another tab reading now sees it taken.
  setPending(p)
  emit()
  void resume(p, kept)
}

/** Look again shortly at a shot another tab is following, and pick it up if that tab has gone. */
function watchPress(): void {
  if (watchTimer) return
  watchTimer = setTimeout(() => {
    watchTimer = null
    const saved = readStored(kv.get(RUN_KEY))
    if (!walking) mirror(saved)
    pickUp(saved?.pending ?? null, false)
  }, WATCH_MS)
}

/**
 * Take in the run as another tab saved it, while this tab is rendering
 * nothing, so its strip is not left behind and a later save from here does
 * not write an old strip over a newer one. A heartbeat that changed no shot
 * changes nothing here.
 */
function mirror(saved: Saved | null): void {
  const key = saved?.shotsKey ?? ''
  if (key === mirrored) return
  mirrored = key
  state = saved && Object.keys(saved.states).length ? { ...state, states: saved.states } : IDLE
  emit()
}

/**
 * While the queue walks, closing the page stops every shot after the one on
 * the press, because nothing else will queue them. So the browser asks first.
 * The shot on the press itself is picked up again on the next load.
 */
function holdPage(e: BeforeUnloadEvent): void {
  e.preventDefault()
}

function holdUnload(on: boolean): void {
  if (typeof window === 'undefined') return
  if (on) window.addEventListener('beforeunload', holdPage)
  else window.removeEventListener('beforeunload', holdPage)
}

// ---------------------------------------------------------------------------
// Preparing a pass
// ---------------------------------------------------------------------------

/**
 * Seed the run with one state per shot, keeping what earlier passes produced.
 *
 * A re-render of shot 4 must not wipe shots 1 to 3: their clips are on disk and
 * their handoff frames are how shot 4 gets built at all.
 */
function adopt(order: string[], jobs: readonly ShotJob[]): Record<string, ShotState> {
  const next: Record<string, ShotState> = {}
  order.forEach((shotId, i) => {
    const frames = jobs[i]?.params.length ?? 0
    const kept = state.states[shotId]
    next[shotId] = kept ? { ...kept, frames } : blankShot(shotId, frames)
  })
  return next
}

// ---------------------------------------------------------------------------
// One shot
// ---------------------------------------------------------------------------

type ShotRun = {
  shotId: string
  job: ShotJob
  previous: OutputFile | null
  ctx: RunContext
  made: Made
  /** Release ComfyUI's cached models before queueing, per the memory verdict. */
  release: boolean
  label: string
  order: string[]
}

async function renderShot(r: ShotRun): Promise<'done' | 'unchanged' | 'error' | 'stopped'> {
  const { shotId, job, previous, ctx } = r
  let workflow
  try {
    workflow = instantiateShot(job, previous)
  } catch (err) {
    setShot(shotId, {
      status: 'error',
      stage: 'Not built',
      error: (err as Error).message,
      finishedAt: Date.now(),
    })
    return 'error'
  }

  const startedAt = Date.now()
  setShot(shotId, {
    status: 'queued',
    stage: 'Sending it to the press',
    promptId: null,
    value: 0,
    max: 0,
    error: null,
    detail: null,
    startedAt,
    finishedAt: null,
    previewUrl: null,
  })

  if (r.release) {
    // The two-model families are killed for memory at the final decode when
    // models from earlier runs are still resident. ComfyUI spends a release
    // on whichever prompt it takes next, so one sent while other work waits
    // in its queue frees memory for that work, and this shot then starts with
    // whatever that work loaded. So the shot waits until ComfyUI has nothing
    // queued or running, releases, and is queued straight after.
    const stopped = () => {
      setShot(shotId, { status: 'stopped', stage: 'Stopped', finishedAt: Date.now() })
      return 'stopped' as const
    }
    const wait = new AbortController()
    waiting = wait
    const idle = await waitForIdleComfy(wait.signal, (ahead) =>
      setShot(shotId, {
        stage: `Waiting for ComfyUI to finish ${ahead === 1 ? 'one other job' : `${ahead} other jobs`}`,
      }),
    )
    waiting = null
    if (!idle || state.stopRequested) return stopped()
    setShot(shotId, { stage: 'Freeing memory first' })
    await releaseComfyMemory()
    if (state.stopRequested) return stopped()
    setShot(shotId, { stage: 'Sending it to the press' })
  }

  const onEvent = (e: ProgressEvent) => {
    if (e.phase === 'queued') {
      setShot(shotId, { promptId: e.promptId, stage: 'Queued' })
      let entry: Pending | null = null
      try {
        entry = {
          shotId,
          promptId: e.promptId,
          label: r.label,
          order: r.order,
          startedAt,
          made: r.made,
          frames: job.params.length ?? 0,
          composition: ctx.compositionFor(job),
          familyLabel: ctx.familyLabel,
          modelLabel: ctx.modelLabel,
        }
      } catch {
        entry = null
      }
      setPending(entry)
      if (state.stopRequested) void cancelJob(e.promptId).catch(() => undefined)
    } else if (e.phase === 'running') {
      setShot(shotId, { status: 'running', value: e.value, max: e.max, stage: 'Drawing' })
    } else if (e.phase === 'preview') {
      setShot(shotId, { previewUrl: e.url })
    }
  }

  try {
    const files = await run(workflow, onEvent)
    const finishedAt = Date.now()
    const clip = files.find((f) => f.kind === 'video') ?? null
    const frame = chainFrameOf(files)

    let filed = { entryId: null as string | null, durationMs: finishedAt - startedAt, unchanged: false }
    if (clip) {
      filed = fileClip(
        clip,
        files,
        { compose: () => ctx.compositionFor(job), familyLabel: ctx.familyLabel, modelLabel: ctx.modelLabel },
        shotOf(shotId)?.promptId ?? '',
        job.params.seed,
        startedAt,
        finishedAt,
      )
    }

    setShot(shotId, {
      status: 'done',
      stage: 'Done',
      files,
      clip,
      frame,
      entryId: filed.entryId,
      finishedAt,
      durationMs: filed.durationMs,
      previewUrl: null,
      made: r.made,
    })
    setPending(null)
    return filed.unchanged ? 'unchanged' : 'done'
  } catch (err) {
    const f = faultOf(err)
    const finishedAt = Date.now()
    setShot(shotId, {
      status: f.cancelled ? 'stopped' : 'error',
      stage: f.cancelled ? 'Stopped' : 'Failed',
      error: f.cancelled ? null : faultBody(f),
      detail: f.cancelled ? null : faultWhere(f),
      finishedAt,
      durationMs: finishedAt - startedAt,
      previewUrl: null,
    })
    setPending(null)
    return f.cancelled ? 'stopped' : 'error'
  }
}

// ---------------------------------------------------------------------------
// The walk
// ---------------------------------------------------------------------------

type Pass = {
  order: string[]
  jobs: readonly ShotJob[]
  ctx: RunContext
  /** Indices into `order` this pass renders, in queue order. */
  indices: number[]
  /** Each queued shot as it stood before the pass marked it waiting. */
  before: Record<string, ShotState>
}

/** "Shot 3" or "Shots 2 and 5", for a note. */
function shotsWord(numbers: readonly number[]): string {
  if (numbers.length === 1) return `Shot ${numbers[0]}`
  return `Shots ${numbers.slice(0, -1).join(', ')} and ${numbers[numbers.length - 1]}`
}

async function walk(pass: Pass): Promise<void> {
  walking = true
  holdUnload(true)
  const { order, jobs, ctx, indices, before } = pass
  let verdict: RunStatus = 'done'
  let note: string | null = null
  /** Shots the pass got as far as, and shots it stopped before a new clip landed. */
  const reached = new Set<string>()
  const stopped = new Set<string>()
  const unchanged: number[] = []

  // The notes speak for this pass only. It can be one shot rendered alone, or
  // the few shots a whole-reel pass found missing, so nothing is said about
  // shots it never touched, and later shots are said to depend on a failed one
  // only when the next one really opens on its last frame.
  /** Shot numbers this pass finished, cached answers included. */
  const finished: number[] = []
  const finishedLine = () =>
    finished.length
      ? ` ${shotsWord(finished)} finished in this pass and ${finished.length === 1 ? 'is' : 'are'} on disk and in the archive.`
      : ''
  /** How a note ends when a shot at `position` in the queue stopped the pass. */
  const restLine = (position: number) => (position + 1 < indices.length ? ', so the rest of this pass was not sent.' : '.')

  for (const [position, index] of indices.entries()) {
    if (state.stopRequested) {
      verdict = 'stopped'
      note = `Stopped before shot ${index + 1}.${finishedLine()}`
      break
    }
    const shotId = order[index]
    const job = jobs[index]
    if (!shotId || !job) continue
    reached.add(shotId)

    setRun({ currentShotId: shotId })

    // The desk refuses these before the press, and this is the last place
    // before the server, so the same two checks stand here too.
    const memory = ctx.memory?.(job) ?? null
    const refusal = job.blocked ?? (memory?.level === 'refuse' ? memory.reason : null)
    if (refusal) {
      setShot(shotId, { status: 'error', stage: 'Not sent', error: refusal, detail: null, finishedAt: Date.now() })
      verdict = 'error'
      note = `Shot ${index + 1} was not sent to ComfyUI${restLine(position)}`
      break
    }

    let previous: OutputFile | null = null
    if (job.start.from === 'previous') {
      const upstreamId = order[index - 1]
      previous = shotOf(upstreamId)?.frame ?? null
      if (!previous) {
        setShot(shotId, {
          status: 'error',
          stage: 'Not built',
          error: `This shot opens on shot ${index}'s last frame, and shot ${index} has not produced one yet. Render the shot before it, or pin an opening frame here.`,
          finishedAt: Date.now(),
        })
        verdict = 'error'
        note = `Shot ${index + 1} had nothing to continue from${restLine(position)}`
        break
      }
    }

    const result = await renderShot({
      shotId,
      job,
      previous,
      ctx,
      made: {
        signature: jobSignature(job),
        seed: job.params.seed,
        openedOn: previous ? annotatedRef(previous) : null,
        frames: clipFrames(job),
        fps: job.params.fps ?? 0,
        width: job.params.width,
        height: job.params.height,
      },
      release: memory?.release ?? false,
      label: `Shot ${index + 1}`,
      order,
    })

    if (result === 'done' || result === 'unchanged') {
      finished.push(index + 1)
      if (result === 'unchanged') unchanged.push(index + 1)
      continue
    }
    if (result === 'stopped') {
      stopped.add(shotId)
      verdict = 'stopped'
      const kept = before[shotId]?.status === 'done' ? ' Its earlier clip is kept.' : ''
      note = `Stopped during shot ${index + 1}.${kept}${finishedLine()}`
      break
    }
    verdict = 'error'
    const next = indices[position + 1]
    note =
      next === index + 1 && jobs[next]?.start.from === 'previous'
        ? `Shot ${index + 1} failed, and shot ${index + 2} was going to open on its last frame, so the rest of this pass was not sent.`
        : `Shot ${index + 1} failed${restLine(position)}`
    break
  }

  // A shot the pass never reached, or stopped before its new clip landed, goes
  // back to exactly what it was: its earlier clip and frame were never touched,
  // so leaving it on "waiting" would drop a finished clip from the cut and
  // re-queue it. Whether it still matches the strip is worked out as always.
  // Anything else left on "queued" goes back to waiting rather than read as
  // work in progress.
  const states = { ...state.states }
  for (const shotId of order) {
    const shot = states[shotId]
    if (!shot) continue
    const prior = before[shotId]
    if (prior?.status === 'done' && (!reached.has(shotId) || stopped.has(shotId))) {
      states[shotId] = prior
    } else if (shot.status === 'queued' || shot.status === 'running') {
      states[shotId] = { ...shot, status: 'waiting', stage: '', previewUrl: null }
    }
  }
  state = { ...state, states }

  if (verdict === 'done' && unchanged.length) {
    const one = unchanged.length === 1
    note = `${shotsWord(unchanged)} came back as the ${one ? 'clip' : 'clips'} already on disk. Nothing that decides ${one ? 'it' : 'them'} had changed, the seed included, so ComfyUI handed back what it had already made. Change the shot, or set the seed to Random, for a different take.`
  }

  walking = false
  holdUnload(false)
  setRun({
    status: verdict,
    finishedAt: Date.now(),
    currentShotId: null,
    stopRequested: false,
    note,
  })
  persist()
}

// ---------------------------------------------------------------------------
// The public engine
// ---------------------------------------------------------------------------

function newRunId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `run_${Date.now().toString(36)}`
}

function startPass(
  order: string[],
  jobs: readonly ShotJob[],
  ctx: RunContext,
  indices: number[],
  states: Record<string, ShotState>,
): void {
  const before: Record<string, ShotState> = {}
  for (const i of indices) {
    const shotId = order[i]
    const shot = shotId ? states[shotId] : undefined
    if (!shotId || !shot) continue
    before[shotId] = shot
    // Marked up front so the strip shows the whole queue from the first
    // moment. The clip and frame stay, and walk puts back any it never reaches.
    states[shotId] = { ...shot, status: 'waiting', stage: '', error: null, detail: null }
  }

  state = {
    id: newRunId(),
    status: 'running',
    startedAt: Date.now(),
    finishedAt: null,
    order,
    states,
    currentShotId: null,
    queue: indices.map((i) => order[i]).filter((id): id is string => id !== undefined),
    stopRequested: false,
    note: null,
  }
  emit()
  void walk({ order, jobs, ctx, indices, before })
}

export const reelRun = {
  subscribe(fn: () => void): () => void {
    listeners.add(fn)
    return () => {
      listeners.delete(fn)
    }
  },
  snapshot: (): RunState => state,

  /** True while the queue is walking. A second press does nothing. */
  busy: (): boolean => walking,

  /**
   * Render the whole reel, skipping shots that are already done and still
   * current. Pass `force` to render every shot again from the top.
   */
  renderAll(order: string[], jobs: readonly ShotJob[], ctx: RunContext, opts: { force?: boolean } = {}): void {
    if (walking) return
    const states = adopt(order, jobs)
    const queue = shotsToRender(order, jobs, states, opts.force)
    if (!queue.length) return
    startPass(order, jobs, ctx, queue, states)
  },

  /** Render one shot, leaving every other shot exactly as it stands. */
  renderOne(index: number, order: string[], jobs: readonly ShotJob[], ctx: RunContext): void {
    if (walking || !order[index]) return
    startPass(order, jobs, ctx, [index], adopt(order, jobs))
  },

  /** Stop the running shot and abandon the rest of the queue. */
  stop(): void {
    if (state.status !== 'running') return
    setRun({ stopRequested: true })
    // A shot still waiting for ComfyUI's queue to empty has nothing to cancel yet.
    waiting?.abort()
    const shot = shotOf(state.currentShotId ?? undefined)
    if (shot?.promptId) {
      setShot(shot.shotId, { stage: 'Stopping' })
      void cancelJob(shot.promptId).catch(() => undefined)
    }
  },

  /** Clear a finished pass's note without touching any rendered shot. */
  dismissNote(): void {
    if (state.note) setRun({ note: null })
  },

  /** Forget every rendered shot. The files stay on disk and in the archive. */
  clear(): void {
    if (walking) return
    state = IDLE
    setPending(null)
    emit()
  },
}

export function useReelRun(): RunState {
  return useSyncExternalStore(reelRun.subscribe, reelRun.snapshot, reelRun.snapshot)
}

if (typeof window !== 'undefined') {
  // A reload or a close hands the shot on the press to the next page at once,
  // rather than after the quiet time another tab waits.
  window.addEventListener('pagehide', () => {
    if (pending) persist({ released: true })
  })
  window.addEventListener('pageshow', (e) => {
    if (e.persisted && pending) persist()
  })
}

// Another tab saved the run. While this one renders nothing it takes in that
// tab's strip, and keeps an eye on that tab's shot on the press.
onStorage(RUN_KEY, (value) => {
  if (walking) return
  const saved = readStored(value)
  mirror(saved)
  if (saved?.pending && saved.pending.owner !== TAB) watchPress()
})

// Last, so everything it calls is defined.
restore()
