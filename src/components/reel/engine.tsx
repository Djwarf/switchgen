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

import { releaseComfyMemory, releaseIfOthersAhead, waitForIdleComfy, type ClipMemory } from '../../lib/clipMemory'
import {
  cancelJob,
  fetchPastRun,
  getJob,
  newPromptId,
  relPath,
  run,
  type ApiWorkflow,
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
  samplerPass,
  type SamplerPass,
  type ShotJob,
} from '../../lib/continuation'
import { history } from '../../lib/history'
import { onStorage, recordOf, store as kv, tabStore, type Composition } from '../../lib/session'
import { holdAwake } from '../../lib/wakeLock'

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
  /**
   * Which sampling pass the steps belong to, for a family that samples in
   * two (the Wan 2.2 14B pairs), each counted from one. Null for one pass.
   */
  pass: SamplerPass | null
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

/** What another open tab has on the press. This tab sends nothing while it does. */
export type Elsewhere = {
  /** The shot on the press there, or null between two of its shots. */
  shotId: string | null
  /**
   * True when the page that had the shot went away (closed, reloaded or
   * crashed) and left it for another page to follow, which this tab does
   * once that page has been quiet long enough.
   */
  left: boolean
}

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
  /**
   * Set while another tab renders this reel. This tab only mirrors that run,
   * so it must not start a pass of its own: it would queue the shot that tab
   * has on the press a second time, and the two tabs' saves would overwrite
   * each other's shots and the one saved shot on the press.
   */
  elsewhere: Elsewhere | null
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
    pass: null,
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
  elsewhere: null,
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

/**
 * How much of a running shot's sampling is done, from 0 to 1. A shot that
 * samples in two passes counts each as half (see samplerPass), so step 1 of
 * the second pass reads past the middle rather than back at the start.
 */
export function drawnFraction(shot: Pick<ShotState, 'status' | 'value' | 'max'> & { pass?: SamplerPass | null }): number {
  if (shot.status !== 'running' || shot.max <= 1) return 0
  const steps = Math.min(1, Math.max(0, shot.value / shot.max))
  const pass = shot.pass ?? null
  if (!pass || pass.count < 2) return steps
  return Math.min(1, (Math.min(Math.max(pass.index, 1), pass.count) - 1 + steps) / pass.count)
}

/**
 * Shots of this tab's pass that wait in the page to be sent: the ones the
 * pass has not reached, and the one on the press until ComfyUI has taken it.
 * Only the page sends them, so none goes while it is closed, hidden or the
 * phone is locked, and the desk says so while there are any. A shot ComfyUI
 * has taken carries on whatever the page does.
 */
export function waitingInPage(run: Pick<RunState, 'status' | 'queue' | 'states' | 'elsewhere'>): number {
  if (run.status !== 'running' || run.elsewhere) return 0
  return run.queue.filter((id) => {
    const s = run.states[id]
    return s?.status === 'waiting' || (s?.status === 'queued' && !s.promptId)
  }).length
}

/**
 * Shot numbers after `shotId` in `order` that have no clip. A shot picked up
 * after a reload was the one on the press when the page went; whatever that
 * page had still to send went with it, and nothing sends it now.
 */
export function unsentAfter(
  order: readonly string[],
  shotId: string,
  states: Readonly<Record<string, ShotState>>,
): number[] {
  const at = order.indexOf(shotId)
  if (at < 0) return []
  const out: number[] = []
  for (let i = at + 1; i < order.length; i++) if (!states[order[i] ?? '']?.clip) out.push(i + 1)
  return out
}

/**
 * The line that closes a picked-up shot's note when shots after it have no
 * clip, naming the press that sends them. It reads "Render what is missing"
 * once the reel has a clip, and "Render the reel" before.
 */
export function unsentLine(numbers: readonly number[], anyClip: boolean): string {
  if (!numbers.length) return ''
  const one = numbers.length === 1
  return `${shotsWord(numbers)} after it ${one ? 'has' : 'have'} no clip. Anything that page was still to send went with it, so ${anyClip ? 'Render what is missing' : 'Render the reel'} sends ${one ? 'it' : 'them'} from here.`
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
 * on the press, and the pass it belongs to, are saved with the tab that is
 * following them, and that tab says it is still there every few seconds.
 * Another tab leaves them alone while their follower is alive, and sends
 * nothing of its own meanwhile (RunState.elsewhere): two tabs following one
 * prompt could file its clip twice, and whichever finished second wrote back
 * a strip that was out of date, without the next shot on the press.
 *
 * The allowance is long because browsers slow the timers of a tab nobody is
 * looking at (Chrome to once a minute, after five minutes hidden), and that
 * is usually the tab the reel is rendering in.
 */
const TAB = newRunId()
/**
 * The id the page before this one in the same tab ran under, and this page's
 * id kept for the next. A phone that throws a hidden tab away to free memory
 * fires no pagehide, so that page's shot and pass are saved as held and
 * freshly beaten, and without this the reloaded tab read them as another
 * tab's: it said another tab had the reel, offered no Stop, and sent nothing
 * for minutes. sessionStorage belongs to the tab and outlives a reload.
 */
const TAB_KEY = 'switchgen.reeltab.v1'
const PREVIOUS: string | null = tabStore.get(TAB_KEY)
tabStore.set(TAB_KEY, TAB)
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
  /**
   * True from just before the shot is sent until ComfyUI answers. The prompt
   * id is made here and saved first, so a page lost while the send is on its
   * way leaves the next page a number to ask ComfyUI about. Without it the
   * shot rendered with nothing following it, and read as never made, so the
   * next press sent it a second time.
   */
  sending?: boolean
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

/**
 * The pass a tab is walking, saved beside the shot on the press. The shot
 * alone left gaps: between two shots, and while a shot waits for ComfyUI's
 * queue to empty before its memory release, no shot is saved, and another
 * tab read the reel as free and could start a pass over the same shots.
 */
type StoredPress = Stamp & {
  /** The shot the pass is on. Null before it reaches one. */
  shotId: string | null
  /**
   * The reel's shots in order, the ones the pass set out to render, and when
   * it started, so the page after one that went mid-pass can say which shots
   * it never sent. Absent from saves made before they were kept.
   */
  order?: string[]
  queue?: string[]
  startedAt?: number | null
}

type SavedShot = Pick<
  ShotState,
  'shotId' | 'clip' | 'frame' | 'files' | 'entryId' | 'durationMs' | 'finishedAt' | 'made'
>

/** The shot this tab is following, if any. */
let pending: Pending | null = null
/**
 * This tab's hold on what it is rendering: the pass it walks, or the shot it
 * picked up. Aborted when the page comes back from the browser's back and
 * forward cache to find that another page took over while it was away (see
 * pageshow), which ends the pass here without sending or filing anything more.
 */
let claim: AbortController | null = null
let beatTimer: ReturnType<typeof setInterval> | null = null
let watchTimer: ReturnType<typeof setTimeout> | null = null
/** The saved shots this tab last wrote or took in, as written. */
let mirrored = ''
/**
 * The saved run exactly as this tab last wrote it while holding the press,
 * or null before its first write of a hold. While a tab holds the press no
 * other page writes the run: they hold their own press while this one's beat
 * is fresh. So a saved run that differs from this means another page acted
 * on the reel while this one was not running: it took the shot over, or
 * started a pass, once this tab went quiet (see holdStands).
 */
let wrote: string | null = null
/** Lets go of the screen wake lock taken while this tab walks (lib/wakeLock). */
let awake: (() => void) | null = null

/** True while this tab holds a pass or a shot that no other page has taken over. */
function holding(): boolean {
  return claim !== null && !claim.signal.aborted
}

/**
 * True while this tab still holds what it is rendering. False, having let go,
 * once another page has written the saved run since this tab last did.
 * `seen` is the saved run: as read now, or as another tab's write carried it.
 *
 * A phone freezes a tab it is not showing and later thaws it, with no
 * pagehide or pageshow either side. Its beat stops meanwhile, and once it
 * has been quiet long enough another tab takes its shot over, follows it and
 * files it, and may start a pass of its own. The thawed tab used to carry on
 * where it left off: it filed the shot the other tab had filed, said ComfyUI
 * had answered it from the cache, sent the next shot a second time, and wrote
 * its own run over the other tab's. So it asks here before every write,
 * before it sends a shot, and before it files one.
 */
function holdStands(seen: string | null = kv.get(RUN_KEY)): boolean {
  if (!holding()) return false
  if (wrote !== null && seen !== wrote) {
    letGo()
    return false
  }
  return true
}

/** Keep the heartbeat going while this tab holds a pass or a shot, and only then. */
function keepBeat(): void {
  const on = holding()
  if (on && !beatTimer) beatTimer = setInterval(() => persist(), BEAT_MS)
  if (!on && beatTimer) {
    clearInterval(beatTimer)
    beatTimer = null
  }
}

/** Set or clear the shot this tab follows, and save. */
function setPending(p: Pending | null): void {
  pending = p
  keepBeat()
  persist()
}

/** Take the press for a pass or a picked-up shot. Returns the signal that says another page took it over. */
function startWalking(): AbortSignal {
  claim = new AbortController()
  walking = true
  // The run saved before this hold may have been written by any page.
  wrote = null
  holdUnload(true)
  keepBeat()
  return claim.signal
}

function stopWalking(): void {
  claim = null
  walking = false
  holdUnload(false)
  keepBeat()
  awake?.()
  awake = null
}

/**
 * Write every clip on disk, whatever its status says: a shot waiting to be
 * rendered again, or one whose new render failed, still has its earlier clip.
 * Written immediately, not debounced, because it runs at shot boundaries and
 * on the heartbeat, and the moment it matters is the moment before the page
 * goes.
 *
 * A shot or a pass another tab is following is written back as it was found.
 * This tab's view of the run can be behind that tab's, and leaving the entry
 * out would forget the shot if that tab then closed.
 */
function persist(opts: { released?: boolean } = {}): void {
  // Another page has taken over since this tab last wrote: writing now would
  // put this tab's view back over that page's shot and pass.
  if (holding() && !holdStands()) return
  const beat = Date.now()
  const released = opts.released ?? false
  const stored = pending && walking ? null : readStored(kv.get(RUN_KEY))
  let entry: StoredPending | null = null
  if (pending) entry = { ...pending, owner: TAB, beat, released }
  else if (stored?.pending && stored.pending.owner !== TAB) entry = stored.pending
  let press: StoredPress | null = null
  if (walking) {
    press = {
      owner: TAB,
      beat,
      released,
      shotId: state.currentShotId,
      order: state.order,
      queue: state.queue,
      startedAt: state.startedAt,
    }
  }
  else if (stored?.press && stored.press.owner !== TAB) press = stored.press

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
  const raw = !shots.length && !entry && !press ? null : JSON.stringify({ shots, pending: entry, press })
  // Noted before the write: a write the quota refuses is still what this tab
  // reads back (the store keeps it in memory), so it is not another page's.
  wrote = holding() ? raw : null
  try {
    if (raw === null) kv.remove(RUN_KEY)
    else kv.set(RUN_KEY, raw)
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
    sending: v.sending === true,
  }
}

function readPress(v: unknown): StoredPress | null {
  if (!isObject(v) || !isString(v.owner) || !isNumber(v.beat)) return null
  const ids = (x: unknown): string[] | undefined => (Array.isArray(x) && x.every(isString) ? x : undefined)
  return {
    owner: v.owner,
    beat: v.beat,
    released: v.released === true,
    shotId: isString(v.shotId) ? v.shotId : null,
    order: ids(v.order),
    queue: ids(v.queue),
    startedAt: isNumber(v.startedAt) ? v.startedAt : null,
  }
}

type Saved = {
  states: Record<string, ShotState>
  pending: StoredPending | null
  press: StoredPress | null
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
  return { states, pending: readPending(v.pending), press: readPress(v.press), shotsKey: JSON.stringify(list) }
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
  //
  // Two records for one file are not always a cached answer. One with this
  // job's prompt id was filed by another page following the same job, and is
  // simply this clip's record. One the recovery pass filed, from the file
  // alone, is not kept: the archive puts this record in its place under the
  // same number, star and notes (history.add), with the frame it opened on
  // and the settings the recovery pass could not know.
  const known = history.all().find((e) => relPath(e.file) === relPath(clip))
  if (known && !known.recovered) {
    const sameJob = promptId !== '' && known.promptId === promptId
    return { entryId: known.id, durationMs: known.durationMs, unchanged: !sameJob }
  }
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

/**
 * The control that files a clip on disk no record describes, as the Archive
 * labels it (components/archive/FacetRail.tsx). The notes used to send the
 * reader to "the Archive's recover action", which no control is called.
 */
const RECOVER = 'open the Archive and press "Look for files with no record"'

/** How often the queue is asked about a shot picked up after a reload. */
const POLL_MS = 4000
/** Unanswered asks in a row before the desk stops waiting (about a minute). */
const POLL_GIVE_UP = 15

/**
 * The page that queued this shot went away while it was on the press. The
 * socket that would have reported it belonged to that page, so the shot is
 * followed through ComfyUI's job list instead, and filed when it lands.
 * `kept` is the clip the shot had before, which comes back if this one does
 * not arrive. `lost` says another page has taken the shot over.
 */
async function resume(p: Pending, kept: ShotState | null, lost: AbortSignal): Promise<void> {
  let verdict: RunStatus = 'error'
  let note: string | null = null
  let unsent = false
  try {
    ;[verdict, note, unsent = false] = await follow(p, lost)
  } catch {
    note = `${p.label} was left on the press by the page before this one, and following it failed. If it finishes, ${RECOVER} to file it.`
  }
  if (lost.aborted) {
    handedOver()
    return
  }
  if (verdict !== 'done') {
    if (kept) state = { ...state, states: { ...state.states, [p.shotId]: kept } }
    else {
      setShot(p.shotId, {
        status: verdict === 'stopped' ? 'stopped' : 'error',
        stage: verdict === 'stopped' ? 'Stopped' : unsent ? 'Not sent' : 'Failed',
        error: verdict === 'stopped' ? null : note,
        finishedAt: Date.now(),
      })
    }
  }

  // The note used to end at the shot it followed, so a reel stopped one shot
  // after a reload read as finished.
  if (verdict !== 'stopped') {
    const rest = unsentAfter(p.order, p.shotId, state.states)
    const anyClip = p.order.some((id) => state.states[id]?.clip && state.states[id]?.status === 'done')
    if (rest.length) note = `${note ?? ''} ${unsentLine(rest, anyClip)}`.trim()
  }

  stopWalking()
  setRun({ status: verdict, finishedAt: Date.now(), currentShotId: null, stopRequested: false, note })
  setPending(null)
}

/**
 * Poll one job to its end, and file its clip if it lands. Once `lost` is
 * aborted another page follows the job, so this one touches nothing more and
 * returns; resume hands over. The third value is true when the shot was still
 * being sent as that page went and ComfyUI never had it.
 */
async function follow(p: Pending, lost: AbortSignal): Promise<[RunStatus, string, boolean?]> {
  const left = `${p.label} was left on the press by the page before this one`
  const gone: [RunStatus, string] = ['stopped', '']
  let misses = 0
  let historyMisses = 0
  /** Still on its way when that page went, and not yet seen in ComfyUI. */
  let sending = p.sending === true
  /** ComfyUI said once that it has no such job. */
  let absent = false
  for (;;) {
    if (lost.aborted || !holdStands()) return gone
    let job: ServerJob | null | undefined
    try {
      job = await getJob(p.promptId)
    } catch {
      job = undefined
    }
    if (lost.aborted) return gone
    if (job === undefined) {
      if (++misses < POLL_GIVE_UP) {
        await delay(POLL_MS)
        continue
      }
      return ['error', `${left}, and ComfyUI is not answering, so the desk stopped waiting for it. If it finishes, ${RECOVER} to file it.`]
    }
    misses = 0
    if (job && sending) {
      // It arrived, so from here it is a shot on the press like any other,
      // and a reload after this one follows it as such. Its number goes on
      // the shot only now: the section bar reads a number as the queue's word
      // that the job is in it, and Stop cancels by it.
      sending = false
      if (pending?.promptId === p.promptId) setPending({ ...pending, sending: false })
      setShot(p.shotId, { promptId: p.promptId })
      if (state.stopRequested) void cancelJob(p.promptId).catch(() => undefined)
    }
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
      // A send cut off by the page going may never have left it. ComfyUI
      // checks a graph before it queues it, so one no-such-job is asked again
      // a moment later before the shot is called not sent, or stopped.
      if (sending && !absent) {
        absent = true
        await delay(POLL_MS)
        continue
      }
      if (state.stopRequested) return ['stopped', `${p.label} was stopped.`]
      if (!sending) return ['error', `${left}, and ComfyUI no longer knows the job, so it did not finish.`]
      // Nothing more happens without the reader: sending it again now would
      // render it twice if the send lands after all.
      return [
        'error',
        `${p.label} was being sent when the page before this one went, and ComfyUI has no job under its number, so nothing is rendering it. It may never have arrived.`,
        true,
      ]
    }
    if (job.status === 'cancelled') return ['stopped', `${p.label} was stopped.`]
    if (job.status === 'failed') return ['error', `${left}, and it failed in ComfyUI.`]

    // This one prompt's record, asked for by id. A page of recent history
    // could already have scrolled past a shot that finished a while ago.
    let past: PastRun | null
    try {
      past = await fetchPastRun(p.promptId)
      if (lost.aborted || !holdStands()) return gone
    } catch {
      // Could not ask, which is not the same as no record.
      if (++historyMisses < POLL_GIVE_UP) {
        await delay(POLL_MS)
        continue
      }
      return ['error', `${left}. It finished, but ComfyUI did not answer when asked for its clip, so the desk stopped waiting. To file it, ${RECOVER}.`]
    }
    const files = past?.files ?? []
    const clip = files.find((f) => f.kind === 'video') ?? null
    if (!clip) {
      return ['error', `${left}. It finished, but ComfyUI's history does not list its clip. If the clip is on disk, ${RECOVER} to file it.`]
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
    return ['done', `${left}. It has finished and is filed.`]
  }
}

/** Read what the last page left behind, once, when the module loads. */
function restore(): void {
  const found = readStored(kv.get(RUN_KEY))
  if (!found) return
  const saved = reclaimDiscarded(found)
  mirrored = saved.shotsKey
  if (Object.keys(saved.states).length) state = { ...IDLE, states: saved.states }
  pickUp(saved.pending, true)
  if (!saved.pending) noteUnsentPass(saved.press)
  lookElsewhere(saved)
}

/** True when the browser threw this tab's last page away (a phone does, to free memory) and this is it loading again. */
const wasDiscarded = (): boolean =>
  typeof document !== 'undefined' && (document as Document & { wasDiscarded?: boolean }).wasDiscarded === true

/** Change the saved run as it stands and write it back, leaving alone what this tab does not read. */
function amendSaved(change: (v: Record<string, unknown>) => void): void {
  try {
    const v: unknown = JSON.parse(kv.get(RUN_KEY) ?? 'null')
    if (!isObject(v)) return
    change(v)
    kv.set(RUN_KEY, JSON.stringify(v))
  } catch {
    // Unreadable or a full quota: this page still acts on what it read.
  }
}

/**
 * After a discard, the shot and pass this tab's own last page held read as
 * let go, as a pagehide would have marked them. Only a discarded page: a
 * duplicated tab starts with the same sessionStorage but was not discarded,
 * so it still waits for the tab it was copied from. Written back, so other
 * tabs stop holding their press for a page that is gone.
 */
function reclaimDiscarded(saved: Saved): Saved {
  if (PREVIOUS === null || !wasDiscarded()) return saved
  const left = (s: Stamp | null): boolean => s !== null && s.owner === PREVIOUS && !s.released
  const shot = left(saved.pending)
  const pass = left(saved.press)
  if (!shot && !pass) return saved
  amendSaved((v) => {
    if (shot && isObject(v.pending)) v.pending.released = true
    if (pass && isObject(v.press)) v.press.released = true
  })
  return {
    ...saved,
    pending: shot && saved.pending ? { ...saved.pending, released: true } : saved.pending,
    press: pass && saved.press ? { ...saved.press, released: true } : saved.press,
  }
}

/**
 * A pass whose page went with no shot on the press, while it waited for
 * ComfyUI's queue to empty or between two shots, left nothing to follow, and
 * used to vanish without a word. This says which of its shots have no clip
 * and were never sent, then drops the entry, so the note is said once.
 *
 * Only for this tab's own last page, which this page has replaced. Another
 * tab's page that let its pass go may be in the browser's back and forward
 * cache, and takes the pass back on its return if the entry is still there.
 */
function noteUnsentPass(press: StoredPress | null): void {
  if (!press?.released || press.owner !== PREVIOUS || !press.order || !press.queue) return
  const { order, queue } = press
  const at = press.shotId ? queue.indexOf(press.shotId) : 0
  const numbers = queue
    .slice(Math.max(at, 0))
    .filter((id) => !state.states[id]?.clip)
    .map((id) => order.indexOf(id) + 1)
    .filter((n) => n > 0)
    .sort((a, b) => a - b)
  amendSaved((v) => {
    if (isObject(v.press) && v.press.owner === press.owner && v.press.beat === press.beat) v.press = null
  })
  if (!numbers.length) return
  const one = numbers.length === 1
  const anyClip = order.some((id) => state.states[id]?.clip)
  state = {
    ...state,
    id: newRunId(),
    status: 'stopped',
    startedAt: press.startedAt ?? null,
    // Its last sign of life. When it went after that is not known.
    finishedAt: press.beat,
    order,
    queue,
    note: `The page before this one went before it sent ${shotsWord(numbers).toLowerCase()}, so ${one ? 'it has' : 'they have'} no clip and nothing is rendering ${one ? 'it' : 'them'}. ${anyClip ? 'Render what is missing' : 'Render the reel'} sends ${one ? 'it' : 'them'} from here.`,
  }
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
        stage: p.sending
          ? 'Asking ComfyUI whether it arrived before the reload'
          : 'Still on the press from before the reload',
        // Not until ComfyUI shows it, for one still being sent (see follow).
        promptId: p.sending ? null : p.promptId,
        startedAt: p.startedAt,
        finishedAt: null,
        error: null,
        detail: null,
        frames: p.frames,
      },
    },
    currentShotId: p.shotId,
  }
  const lost = startWalking()
  // Claimed before anything else runs, so another tab reading now sees it taken.
  setPending(p)
  emit()
  void resume(p, kept, lost)
}

/** Look again shortly at a shot another tab is following, and pick it up if that tab has gone. */
function watchPress(): void {
  if (watchTimer) return
  watchTimer = setTimeout(() => {
    watchTimer = null
    const saved = readStored(kv.get(RUN_KEY))
    if (!walking) mirror(saved)
    pickUp(saved?.pending ?? null, false)
    lookElsewhere(saved)
  }, WATCH_MS)
}

/**
 * What another tab has on the press, as saved: the shot it follows, or the
 * pass it walks between two shots. A shot is held whoever saved it, even one
 * whose page went away, because this tab takes that one over itself (see
 * pickUp). A pass whose page went away is over, and one whose tab has not
 * written for the quiet time is taken to have crashed.
 */
function heldBy(saved: Saved | null): Elsewhere | null {
  const p = saved?.pending
  if (p && p.owner !== TAB) return { shotId: p.shotId, left: p.released || Date.now() - p.beat >= STALE_MS }
  const w = saved?.press
  if (w && w.owner !== TAB && !w.released && Date.now() - w.beat < STALE_MS) return { shotId: w.shotId, left: false }
  return null
}

/**
 * Note what another tab has on the press, so the desk holds its own press, and
 * keep looking until it has nothing there. A tab that is walking has already
 * got the press, and notes nothing.
 */
function lookElsewhere(saved: Saved | null): void {
  const next = walking ? null : heldBy(saved)
  const was = state.elsewhere
  if (next?.shotId !== was?.shotId || next?.left !== was?.left) {
    state = { ...state, elsewhere: next }
    emit()
  }
  if (next) watchPress()
}

/**
 * End this tab's pass after another page took the reel over while this one
 * was away, and show that page's run instead. Nothing is saved from here:
 * this page's view is older than the saved one, and writing it would put back
 * a strip without the other page's shots and drop its shot on the press.
 */
function handedOver(): void {
  pending = null
  stopWalking()
  const saved = readStored(kv.get(RUN_KEY))
  mirrored = saved?.shotsKey ?? ''
  state = {
    ...state,
    status: 'stopped',
    finishedAt: Date.now(),
    currentShotId: null,
    stopRequested: false,
    states: saved?.states ?? {},
    note: 'While this page was away, another tab took the reel over, so this page stopped following it. The clips that tab makes show here as they land.',
  }
  emit()
  lookElsewhere(saved)
}

/**
 * True when nothing saved while this page was away says another page has
 * taken its shot or its pass. The page's own entries are still there, marked
 * released on the way out, unless a page that loaded since took them.
 */
function stillMine(saved: Saved | null): boolean {
  const p = saved?.pending ?? null
  if (pending ? p?.owner !== TAB || p.promptId !== pending.promptId : p !== null && p.owner !== TAB) return false
  return saved?.press?.owner === TAB
}

/**
 * Let go of the pass or shot this page held, after it came back to find
 * another page following it. The walk or the follow sees the abort, sends
 * and files nothing more, and hands over.
 */
function letGo(): void {
  pending = null
  claim?.abort()
  keepBeat()
  waiting?.abort()
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
  /** Aborted when another page has taken this pass over (see letGo). */
  lost: AbortSignal
  /** The pass's last shot: once ComfyUI has it, nothing waits in the page. */
  last: boolean
}

/** Rejects once `lost` aborts, so a shot stops waiting on a job another page now follows. */
function whenLost(lost: AbortSignal): Promise<never> {
  return new Promise<never>((_, reject) => {
    const give = () => reject(new Error('Another page took this shot over.'))
    if (lost.aborted) give()
    else lost.addEventListener('abort', give, { once: true })
  })
}

async function renderShot(r: ShotRun): Promise<'done' | 'unchanged' | 'error' | 'stopped' | 'handed'> {
  const { shotId, job, previous, ctx, lost } = r
  if (lost.aborted) return 'handed'
  let workflow: ApiWorkflow
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
    pass: null,
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
        stage:
          ahead < 0
            ? 'Waiting for ComfyUI, which is not answering and may be restarting'
            : `Waiting for ComfyUI to finish ${ahead === 1 ? 'one other job' : `${ahead} other jobs`}`,
      }),
    )
    waiting = null
    // A page that slept through the wait asks before it frees memory for a
    // shot another page may now be sending.
    if (lost.aborted || !holdStands()) return 'handed'
    if (!idle || state.stopRequested) return stopped()
    setShot(shotId, { stage: 'Freeing memory first' })
    await releaseComfyMemory()
    if (lost.aborted) return 'handed'
    if (state.stopRequested) return stopped()
    setShot(shotId, { stage: 'Sending it to the press' })
  }

  // The prompt id is made here and the shot saved under it, marked as still
  // being sent, before the send: a page lost while the send is on its way
  // leaves the next page a number to ask ComfyUI about (see follow).
  const promptId = newPromptId()
  let entry: Pending | null = null
  try {
    entry = {
      shotId,
      promptId,
      label: r.label,
      order: r.order,
      startedAt,
      made: r.made,
      frames: job.params.length ?? 0,
      composition: ctx.compositionFor(job),
      familyLabel: ctx.familyLabel,
      modelLabel: ctx.modelLabel,
      sending: true,
    }
  } catch {
    entry = null
  }

  const onEvent = (e: ProgressEvent) => {
    // Another page follows the job now, and its events are that page's to report.
    if (lost.aborted) return
    if (e.phase === 'queued') {
      setShot(shotId, { promptId: e.promptId, stage: 'Queued' })
      // The release above counts only if nothing reached ComfyUI between it
      // and this shot, and the video desk, another tab or anything else could
      // have sent work in that moment. If anything else is in the queue now,
      // the release is sent again: ComfyUI applies one that arrives while a
      // job runs after that job, before it takes the next.
      if (r.release) void releaseIfOthersAhead(e.promptId).catch(() => undefined)
      // ComfyUI's answer names the id; one too old to take ours names its own.
      setPending(entry && { ...entry, promptId: e.promptId, sending: false })
      if (state.stopRequested) void cancelJob(e.promptId).catch(() => undefined)
      // Nothing of the pass waits in the page any more, and following the
      // shot needs no screen: a reload picks it up from the saved entry. Kept
      // on, a phone left on the table lit its screen through the whole render.
      if (r.last) {
        awake?.()
        awake = null
      }
    } else if (e.phase === 'running') {
      // A node between two passes keeps the pass it follows.
      const pass = samplerPass(workflow, e.node) ?? shotOf(shotId)?.pass ?? null
      setShot(shotId, { status: 'running', value: e.value, max: e.max, pass, stage: 'Drawing' })
    } else if (e.phase === 'preview') {
      setShot(shotId, { previewUrl: e.url })
    }
  }

  // Last look before the shot goes: a page that slept through the wait above
  // may find another page has taken the reel over meanwhile.
  if (!holdStands()) return 'handed'
  setPending(entry)

  try {
    const files = await Promise.race([run(workflow, onEvent, { promptId }), whenLost(lost)])
    // A thawed page's job can land before anything else runs, so the check is
    // made again before filing: the page that took the shot over files it.
    if (lost.aborted || !holdStands()) return 'handed'
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
    if (lost.aborted || !holdStands()) return 'handed'
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
  const lost = startWalking()
  // Only this page sends the next shot, and a phone that locks its screen
  // suspends the page, so the screen is kept on while shots wait in the page
  // to be sent, where the browser allows it (a secure page; see
  // lib/wakeLock), and let go once ComfyUI has the last one (renderShot). A
  // shot picked up after a reload was sent by the page before, so following
  // it does not ask.
  awake?.()
  awake = holdAwake('reel')
  // Saved at once, so another tab holds its press from the first moment, even
  // while this pass waits for ComfyUI's queue to empty before its first shot.
  persist()
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
    if (lost.aborted) break
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
    // The saved pass names its shot, for another tab to show where it is.
    persist()
    if (lost.aborted) break

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
      lost,
      last: position === indices.length - 1,
    })

    if (result === 'handed') break
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
  if (lost.aborted) {
    handedOver()
    return
  }
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

  stopWalking()
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
    elsewhere: null,
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
   * current. Pass `force` to render every shot again from the top. Nothing
   * starts while another tab has the reel on the press (RunState.elsewhere).
   */
  renderAll(order: string[], jobs: readonly ShotJob[], ctx: RunContext, opts: { force?: boolean } = {}): void {
    if (walking || state.elsewhere) return
    const states = adopt(order, jobs)
    const queue = shotsToRender(order, jobs, states, opts.force)
    if (!queue.length) return
    startPass(order, jobs, ctx, queue, states)
  },

  /** Render one shot, leaving every other shot exactly as it stands. */
  renderOne(index: number, order: string[], jobs: readonly ShotJob[], ctx: RunContext): void {
    if (walking || state.elsewhere || !order[index]) return
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

  /**
   * Forget every rendered shot. The files stay on disk and in the archive.
   * Not while another tab renders: that tab's next save would bring them all
   * back, and this one's would wipe its shots from the saved run meanwhile.
   */
  clear(): void {
    if (walking || state.elsewhere) return
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
  // rather than after the quiet time another tab waits, and frees the press
  // for other tabs: the rest of the pass goes with the page.
  window.addEventListener('pagehide', () => {
    if (holding()) persist({ released: true })
  })
  // A page the browser kept whole comes back with its pass still walking. It
  // takes the press back only if nothing was saved in the meantime by a page
  // that took it over. A page that loaded since takes a released shot at once
  // (pickUp), and two pages following one prompt could file its clip twice.
  window.addEventListener('pageshow', (e) => {
    if (!e.persisted || !holding()) return
    if (stillMine(readStored(kv.get(RUN_KEY)))) persist()
    else letGo()
  })
}
if (typeof document !== 'undefined') {
  // A phone thaws a frozen tab, or shows a hidden one again, with no pageshow.
  // The save asks whether another page took over meanwhile (holdStands), and
  // otherwise freshens the beat that other tabs read, before a slowed timer
  // gets to it.
  const back = () => {
    if (holding()) persist()
  }
  document.addEventListener('resume', back)
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') back()
  })
}

// Another tab saved the run. While this one renders nothing it takes in that
// tab's strip, and keeps an eye on what that tab has on the press. While this
// one holds the press, no other page writes the run unless it has taken over.
onStorage(RUN_KEY, (value) => {
  if (walking) {
    // The value the other page wrote, rather than a read: a read falls back
    // to this tab's own copy when that page removed the run.
    holdStands(value)
    return
  }
  const saved = readStored(value)
  mirror(saved)
  lookElsewhere(saved)
})

// Last, so everything it calls is defined.
restore()
