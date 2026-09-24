/**
 * The queue on the SwitchGen server, seen from the page.
 *
 * A phone that locks its screen suspends the page, and work that waits in the
 * page waits with it: the next clip in the Video lane, the next shot of a
 * reel, the rest of a pictures batch. Where the server runs its own queue
 * (server/runner.mjs), a desk builds every graph and record as it always did,
 * hands the lot over in one POST, and from then on only watches. The server
 * sends each job to ComfyUI in turn, follows it, files it in the archive and
 * says what happened to every page on every device. This module is the page's
 * whole side of that:
 *
 *   1. A store of what the server holds, read once with GET /api/runner and
 *      kept current by one event stream. Every event carries the revision of
 *      the commit that made it; a gap means events were missed, and the whole
 *      state is read again rather than guessed at.
 *   2. The hand-over. What is being handed over is written to the tab before
 *      the POST goes, so a page that is thrown away mid send can send it again
 *      when it comes back. The server takes the same ids twice as one, so
 *      sending again is always safe; running the work in the page as well
 *      would not be, which is why an unclear answer is never taken as a no.
 *   3. The words: a wait in plain English, the error a desk already knows how
 *      to print, and the row the section bar reads.
 *
 * Nothing here runs a job. Without a queue on the server the desks work exactly
 * as they did, in the page, and this module stays quiet.
 */
import { useEffect, useSyncExternalStore } from 'react'
import type { Reported } from '../components/shell/mirror'
import { forgetCapabilities, serverCapabilities, type RunnerDesk } from './capabilities'
import {
  ComfyError,
  ENDED_UNSENT,
  LostJob,
  newPromptId,
  withDeadline,
  type ApiWorkflow,
  type OutputFile,
} from './comfy'
import type { NewEntry } from './history'
import { recordOf, store, tabStore, type Composition } from './session'

export { WAITS_ON_SERVER } from './wakeLock'
export type { RunnerDesk }

// ---------------------------------------------------------------------------
// Shared literals (the server spells them the same in server/runner/comfyRecord.mjs)
// ---------------------------------------------------------------------------

/**
 * What a reel shot's graph holds where the frame of the shot before it goes,
 * while that shot has not been made. The server puts the frame in its place
 * once it has one, so a whole pass can be handed over at once.
 */
export const CHAIN_TOKEN = 'switchgen:chain:previous-frame'

/** The node a continuation graph saves its last frame from. */
export const FRAME_NODE = '__cont_frame'

// ---------------------------------------------------------------------------
// Types (the same shapes as the server's)
// ---------------------------------------------------------------------------

export type RunnerStatus =
  | 'waiting'
  | 'releasing'
  | 'sending'
  | 'queued'
  | 'running'
  | 'filing'
  | 'done'
  | 'failed'
  | 'stopped'
  | 'lost'
  | 'unsent'
  | 'skipped'

export type Wait = {
  for: 'turn' | 'before' | 'heavy' | 'queue' | 'comfy' | 'held' | 'disk'
  ahead?: number
} | null

export type RunnerErrorCode =
  | 'refused'
  | 'failed'
  | 'lost'
  | 'unsent'
  | 'ended-unsent'
  | 'no-file'
  | 'no-frame'
  | 'stopped'
  | 'skipped'
  | 'internal'

export type RunnerError = {
  code: RunnerErrorCode
  message: string | null
  node: string | null
  nodeType: string | null
  nodeErrors: Record<string, unknown> | null
  /** ComfyUI said the job ended without its result, so its file may be on disk. */
  mayExist: boolean
  /** The job reached ComfyUI's queue before it ended. */
  sent: boolean
  /** For a skipped job, the job whose ending ended its group. */
  after: { jobId: string; index: number } | null
}

export type RunnerLane = {
  held: null | {
    /**
     * A heavy job was lost, or may never have reached ComfyUI; the machine
     * restarted while work waited; or the queue was off (turned off, or
     * standing back for another server) while work waited.
     */
    why: 'lost' | 'unsent' | 'restart' | 'paused'
    /** 'heavy' covers the heavy jobs waiting; 'all' covers every job waiting. */
    scope: 'heavy' | 'all'
    jobId: string | null
    since: number
    /**
     * On a hold on all the work: when a heavy job (`jobId`) was lost, or may
     * never have reached ComfyUI, while it stood. From then on it covers every
     * heavy job too, as the hold after a lost clip does; `since` stays as it
     * was, so a word naming it still answers it.
     */
    heavyAfter?: number | null
  }
}

export type RunnerJob = {
  id: string
  groupId: string
  desk: RunnerDesk
  kind: 'image' | 'video'
  /** The server's intake order, across every group. */
  seq: number
  /** Place in its group, from 1, and the group's size. */
  index: number
  total: number
  label: string
  prompt: string
  /** The browser that handed it over (deviceId). */
  device: string
  heavy: boolean
  status: RunnerStatus
  wait: Wait
  stopRequested: boolean
  stopLanded: boolean
  /** ComfyUI's prompt id. Null until the queue has taken the job. */
  promptId: string | null
  attempt: number
  createdAt: number
  sentAt: number | null
  /** When ComfyUI started it, by ComfyUI's clock. */
  ranAt: number | null
  /** When ComfyUI ended it, by ComfyUI's clock. */
  finishedAt: number | null
  endedAt: number | null
  files: OutputFile[]
  primary: OutputFile | null
  frame: OutputFile | null
  /** For a chained reel shot, the frame it opened on, as the graph names it. */
  openedOn: string | null
  entryId: string | null
  entryNo: number | null
  /** A cached answer: the record the same file is already filed under. */
  repeatOf: string | null
  /** ComfyUI's own running time. 0 when it was not measured, and then not shown. */
  durationMs: number
  error: RunnerError | null
  // What the desk handed over with the job, read back by the desk that knows its shape.
  meta: Record<string, any> | null
  dismissed: boolean
}

export type RunnerGroup = {
  id: string
  desk: RunnerDesk
  kind: 'batch' | 'clips' | 'pass'
  label: string
  device: string
  createdAt: number
  state: 'active' | 'ended'
  endedBy: {
    jobId: string
    why: 'failed' | 'refused' | 'stopped' | 'lost' | 'unsent' | 'no-frame'
  } | null
  endedAt: number | null
  jobIds: string[]
  dismissed: boolean
}

export type RunnerProgress = {
  id: string
  value: number
  max: number
  node: string | null
  classType: string | null
  pass: { index: number; count: number } | null
  at: number
  /** How many previews have come; the newest is at previewUrl(id, previewN). */
  previewN: number
}

type Snapshot = {
  v: 1
  available: boolean
  reason: string | null
  /** Random for each start of the server's queue. */
  boot: string
  rev: number
  comfy: { answering: boolean | null; since: number }
  lane: RunnerLane
  groups: RunnerGroup[]
  jobs: RunnerJob[]
  progress: Record<string, RunnerProgress>
}

export type RunnerSnapshot = Snapshot & {
  /** The event stream is open and has said where the server stands. */
  connected: boolean
}

/** A record as the server files it, less what only the finished job can say. */
export type RecordTemplate = Omit<NewEntry, 'file' | 'files' | 'kind' | 'promptId' | 'durationMs' | 'at'>

/** The contract's names for the same shapes, for code written against them. */
export type JobView = RunnerJob
export type GroupView = RunnerGroup
export type Lane = RunnerLane
export type Progress = RunnerProgress
export type { Snapshot }

export type SubmitJob = {
  /** A lowercase version 4 UUID, made by the desk (newPromptId). */
  id: string
  label: string
  prompt: string
  kind: 'image' | 'video'
  /** The kind of file the record is filed under. */
  primary: 'image' | 'video'
  /** With no file of the primary kind, take the first file of any kind. */
  orFirst: boolean
  /** What a run that wrote no file is: a failure, or done with nothing filed. */
  noFile: 'fail' | 'done'
  heavy: boolean
  graph: ApiWorkflow
  record: RecordTemplate
  /** A reel shot opening on the frame of an earlier job in the same pass. */
  chain?: { after: string; at: [string, string][] }
  meta?: Record<string, any>
}

export type SubmitBody = {
  v: 1
  group: { id: string; desk: RunnerDesk; kind: 'batch' | 'clips' | 'pass'; label: string; device: string }
  jobs: SubmitJob[]
}

export type SubmitTaken = { ok: true; replayed: boolean; group: RunnerGroup; jobs: RunnerJob[] }
/** Nothing here takes the work: the desk runs it in the page, as before. */
export type SubmitFallback = { ok: false; fallback: true; reason: string }
/** The server answered no, and kept nothing. */
export type SubmitRefused = { ok: false; fallback: false; status: number; error: string; busy?: 'images' | 'reel' }
/**
 * No clear answer yet. The work may already be on the server, so it must not
 * be run in the page as well; it stays in the tab and is sent again (with the
 * same ids, which the server takes as one) when the server is back.
 */
export type SubmitPending = { ok: false; fallback: false; pending: true }

export type SubmitResult = SubmitTaken | SubmitFallback | SubmitRefused | SubmitPending

/** What follow() resolves with when the job is done. */
export type FollowResult = {
  files: OutputFile[]
  primary: OutputFile | null
  frame: OutputFile | null
  entryId: string | null
  entryNo: number | null
  repeatOf: string | null
  openedOn: string | null
  durationMs: number
  /** When it ended: ComfyUI's time when known, else when the server saw it. */
  finishedAt: number
}

export type FollowEvent =
  | { phase: 'queued'; promptId: string }
  | {
      phase: 'running'
      node: string | null
      value: number
      max: number
      /** The class of the node, for a desk that names the stage from it. */
      classType?: string | null
      /** The sampling pass the steps count within, for a two-pass family. */
      pass?: { index: number; count: number } | null
    }
  | { phase: 'preview'; url: string }

export type FaultWords = Partial<Record<RunnerErrorCode, string>>

// ---------------------------------------------------------------------------
// Words
// ---------------------------------------------------------------------------

/** A job the server sent and ComfyUI then forgot: the same sentence the desks say today. */
export const RUNNER_LOST =
  'We lost track of this job. ComfyUI has no record of it any more, which usually means it restarted. Nothing was saved.'

/** A send that got no clear answer, where ComfyUI then had no record of the job. */
export const RUNNER_UNSENT =
  'This job may never have reached ComfyUI: the send got no clear answer, and ComfyUI has no record of it. The server never sends a job twice, so it was not sent again. Nothing was saved.'

/** A hand-over sent without an answer until it was too old to send. */
export const OUTBOX_GIVEN_UP = 'The server never answered for this batch, so it was not sent.'

/**
 * A hand-over never sent: its page closed while it waited its turn behind an
 * earlier one, and the page loaded after it found it too old to send.
 */
export const OUTBOX_TOO_OLD = 'The page closed before it handed this batch over, and it is now too old to send.'

/**
 * A hand-over the server answered, when it was sent again or sent from the
 * page loaded after its own, with no queue to take it.
 */
export const OUTBOX_NOT_TAKEN = 'The queue on the server had stopped when this batch reached it, so it was not taken.'

/** Why work waits after a heavy clip was lost, as the Video desk says it today. */
export const HELD_AFTER_LOSS = 'Held until you say, because the heavy clip before it was lost.'

/** Why work waits after a heavy clip's send got no clear answer and ComfyUI had no record of it. */
export const HELD_AFTER_UNSENT = 'Held until you say, because the heavy clip before it may never have reached ComfyUI.'

/** Why work waits after the whole machine was restarted. */
export const HELD_AFTER_RESTART = 'The machine restarted while this work waited, so it is held until you say.'

/** Why work waits after the queue on the server was off, or stood back for another server, while it waited. */
export const HELD_AFTER_PAUSE = 'The queue on the server was off while this work waited, so it is held until you say.'

/** Each reason for a hold, as the note under a job it holds. */
const HELD_NOTE: Record<NonNullable<RunnerLane['held']>['why'], string> = {
  lost: HELD_AFTER_LOSS,
  unsent: HELD_AFTER_UNSENT,
  restart: HELD_AFTER_RESTART,
  paused: HELD_AFTER_PAUSE,
}

/**
 * Whether a hold on the lane covers a job, as the queue decides it (covers()
 * in server/runner/engine.mjs). Of the jobs it covers, the ones still waiting
 * are the ones the lane word sends or calls off, whatever their wait says: in
 * a batch or a pass only the first says 'held', and the rest wait for the one
 * before them, held all the same. A count of held work counts those.
 *
 * A hold made because the machine restarted, or the queue was off, while work
 * waited is about that work: whatever was made after it began goes as usual,
 * unless it is heavy and a heavy clip was lost while the hold stood.
 */
export function holdCovers(held: RunnerLane['held'], job: Pick<RunnerJob, 'heavy' | 'createdAt'>): boolean {
  if (!held) return false
  if (held.scope === 'heavy') return job.heavy === true
  if (held.why !== 'restart' && held.why !== 'paused') return true
  return !(job.createdAt > held.since) || (held.heavyAfter != null && job.heavy === true)
}

/**
 * The note under a job a hold covers. A heavy job made after a hold on all
 * the work began is held for the heavy clip lost since, not for the restart
 * or the pause it never waited through.
 */
function heldNote(held: NonNullable<RunnerLane['held']>, job: Pick<RunnerJob, 'createdAt'>): string {
  if (held.scope === 'all' && held.heavyAfter != null && job.createdAt > held.since) {
    return held.jobId && jobsById.get(held.jobId)?.status === 'unsent' ? HELD_AFTER_UNSENT : HELD_AFTER_LOSS
  }
  return HELD_NOTE[held.why] ?? HELD_AFTER_LOSS
}

/** The Pictures desk's sentence for a run that wrote no file. */
const NO_FILE = 'The job finished but wrote no file. Check ComfyUI’s own log for the reason.'
const NO_FRAME = 'The shot before this one left no last frame to open on, so this shot was not sent.'
const INTERNAL = 'The server could not read its own copy of this job, so it was not sent. Nothing was saved.'
const FORGOTTEN = 'The server has no record of this job any more, so there is nothing to follow.'
const STOPPED_SENT = 'Job stopped. Nothing was saved.'
const STOPPED_UNSENT = 'Stopped before it was sent.'

const NO_QUEUE = 'This server has no queue of its own.'
const NOT_RUNNING = 'The queue on the server is not running.'
const NOT_THIS_DESK = 'The queue on the server does not take this desk’s work.'

/**
 * "This page sends the work itself: <reason>." The reasons are the server's own
 * sentences, capitalised and stopped, so the first word is lowered (unless it
 * is a name such as ComfyUI or an all-capitals variable) and the stop dropped.
 */
export function fallbackLine(reason: string): string {
  let r = reason.trim().replace(/[.\s]+$/, '')
  if (/^[A-Z][a-z]+\b/.test(r) && !/^[A-Z][a-z]+[A-Z]/.test(r)) r = r[0].toLowerCase() + r.slice(1)
  return `This page sends the work itself: ${r || 'the server has no queue'}.`
}

// ---------------------------------------------------------------------------
// This browser
// ---------------------------------------------------------------------------

const DEVICE_KEY = 'switchgen.device.v1'
let deviceInMemory: string | null = null

/**
 * A name for this browser, kept for good, so a desk can say "Sent from this
 * browser" or "Sent from another browser" over work every device can see.
 * Where nothing can be kept it lasts as long as the page.
 */
export function deviceId(): string {
  const kept = store.get(DEVICE_KEY)
  if (kept && /^[A-Za-z0-9-]{1,64}$/.test(kept)) return kept
  const id = deviceInMemory ?? newPromptId()
  deviceInMemory = id
  try {
    store.set(DEVICE_KEY, id)
  } catch {
    /* full: the id still lasts as long as the page */
  }
  return id
}

// ---------------------------------------------------------------------------
// Records
// ---------------------------------------------------------------------------

export type TemplateFields = Omit<
  Parameters<typeof recordOf>[1],
  'file' | 'files' | 'kind' | 'promptId' | 'durationMs' | 'at'
>

/** Stands in for the file while the record is built; removed before it is sent. */
const PLACEHOLDER = { filename: '', subfolder: '', type: 'output' }

/**
 * The record a desk would file, less the six fields only the finished job can
 * fill: the file, the other files, their kind, the prompt id, the running time
 * and when it ended. The server fills them from what ComfyUI returns. Built by
 * recordOf itself, so a record filed by the server says exactly what the same
 * job filed from the page would have said.
 */
export function recordTemplate(c: Composition, fields: TemplateFields): RecordTemplate {
  const full = recordOf(c, {
    ...fields,
    file: PLACEHOLDER,
    kind: c.desk === 'video' ? 'video' : 'image',
    promptId: '',
    durationMs: 0,
  })
  const { file: _file, files: _files, kind: _kind, promptId: _promptId, durationMs: _durationMs, at: _at, ...rest } = full
  // recordOf spells an absent field as undefined; the server reads JSON, where
  // it is simply not there, and a template compared in a test should say the same.
  const clean = rest as Record<string, unknown>
  for (const k of Object.keys(clean)) if (clean[k] === undefined) delete clean[k]
  return rest
}

/** Every [node, input] of a graph that holds CHAIN_TOKEN. */
export function tokenSites(graph: ApiWorkflow): [string, string][] {
  const sites: [string, string][] = []
  for (const [id, node] of Object.entries(graph)) {
    for (const [input, value] of Object.entries(node?.inputs ?? {})) {
      if (value === CHAIN_TOKEN) sites.push([id, input])
    }
  }
  return sites
}

// ---------------------------------------------------------------------------
// The wire
// ---------------------------------------------------------------------------

const API = '/api/runner'
/** A small read. */
const READ_MS = 10_000
/** A stop or a lane word. */
const WORD_MS = 15_000
/**
 * One hand-over. A pass of long shots is a few megabytes and a phone on a slow
 * link takes a while to send it; a deadline shorter than the upload would cut
 * off a send that was about to land and send it all again.
 */
const SUBMIT_MS = 60_000

type Answer = {
  status: number
  /** 'broken' is a body that said it was JSON and could not be read as it. */
  type: 'json' | 'html' | 'other' | 'broken'
  json: any
}

async function ask(path: string, init: RequestInit, ms: number): Promise<Answer> {
  return withDeadline(ms, undefined, async (signal) => {
    const r = await fetch(path, { ...init, signal })
    const ct = String(r.headers?.get?.('content-type') ?? '').toLowerCase()
    if (ct.includes('json')) {
      try {
        return { status: r.status, type: 'json', json: await r.json() }
      } catch {
        return { status: r.status, type: 'broken', json: null }
      }
    }
    return { status: r.status, type: ct.includes('html') ? 'html' : 'other', json: null }
  })
}

function posting(body: unknown): RequestInit {
  return {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
      // Who asked, for the server's log. Nothing is decided by it.
      'X-SwitchGen-Device': deviceId(),
    },
    body: JSON.stringify(body),
  }
}

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms))

// ---------------------------------------------------------------------------
// The store
// ---------------------------------------------------------------------------

const TERMINAL: ReadonlySet<RunnerStatus> = new Set(['done', 'failed', 'stopped', 'lost', 'unsent', 'skipped'])
const LIVE: ReadonlySet<RunnerStatus> = new Set(['waiting', 'releasing', 'sending', 'queued', 'running', 'filing'])

/** '' until the server has said which start of its queue this is. */
let boot = ''
/** The highest revision applied, and the one the last whole state was read at. */
let lastRev = 0
let baseRev = 0
/**
 * The revision each job, group and the lane was last changed at. A whole read
 * and the stream can pass each other, and an event older than what is held
 * must not put a finished job back to running.
 */
const revOf = new Map<string, number>()
let jobsById = new Map<string, RunnerJob>()
let groupsById = new Map<string, RunnerGroup>()
let progressById = new Map<string, RunnerProgress>()
let lane: RunnerLane = { held: null }
let available = false
let reason: string | null = null
let comfy: Snapshot['comfy'] = { answering: null, since: 0 }
/**
 * The newest revision an event skipped over. A read already on its way when
 * the gap was seen may have left before that commit, so the state is read
 * again until it has caught up.
 */
let missedUpTo = 0
let connected = false
/** The server answered with something that is not the queue: nothing to watch. */
let absent = false
/** Whole states taken, so a caller can tell a read that got through from one that did not. */
let reads = 0
/**
 * Times the stream has brought a new start of the queue, or the same start
 * going off or coming back on. A read sent before that, answered by the queue
 * as it was, must not put the old state back.
 */
let turnsSeen = 0
/** A read's whole state is being taken in, and whoever hears of it now is hearing that read. */
let takingRead = false

const listeners = new Set<() => void>()
let current: RunnerSnapshot = build()

function build(): RunnerSnapshot {
  return {
    v: 1,
    available,
    reason,
    boot,
    rev: lastRev,
    comfy,
    lane,
    groups: [...groupsById.values()].sort((a, b) => a.createdAt - b.createdAt),
    jobs: [...jobsById.values()].sort((a, b) => a.seq - b.seq),
    progress: Object.fromEntries(progressById),
    connected,
  }
}

function changed(): void {
  current = build()
  for (const fn of [...listeners]) {
    try {
      fn()
    } catch {
      /* one broken subscriber must not stop the rest */
    }
  }
}

function subscribe(fn: () => void): () => void {
  listeners.add(fn)
  return () => {
    listeners.delete(fn)
  }
}

function setConnected(v: boolean): void {
  if (connected === v) return
  connected = v
  changed()
}

const isObj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v)

function readSnapshot(x: unknown): Snapshot | null {
  if (!isObj(x) || typeof x.boot !== 'string' || typeof x.rev !== 'number') return null
  if (!Array.isArray(x.jobs) || !Array.isArray(x.groups)) return null
  return {
    v: 1,
    available: x.available === true,
    reason: typeof x.reason === 'string' ? x.reason : null,
    boot: x.boot,
    rev: x.rev,
    comfy: isObj(x.comfy) ? (x.comfy as Snapshot['comfy']) : { answering: null, since: 0 },
    lane: isObj(x.lane) ? (x.lane as RunnerLane) : { held: null },
    groups: (x.groups as RunnerGroup[]).filter((g) => isObj(g) && typeof g.id === 'string'),
    jobs: (x.jobs as RunnerJob[]).filter((j) => isObj(j) && typeof j.id === 'string'),
    progress: isObj(x.progress) ? (x.progress as Record<string, RunnerProgress>) : {},
  }
}

/**
 * Take a whole state. One from another start of the queue replaces everything,
 * and so does one from this start that has gone off or come back on: a queue
 * that stands back (another server took the archive) or is turned off says so
 * with its revisions started again, or read back from its saved list, and the
 * jobs it lists are no longer being sent by anything. One from this start, as
 * it was, that is behind what is already held (a slow read that the stream
 * overtook) is left alone. Returns false for something that is not a state at
 * all.
 */
function takeSnapshot(x: unknown, readFrom: number | null = null): boolean {
  const s = readSnapshot(x)
  if (!s) return false
  const newBoot = s.boot !== boot
  const turned = !newBoot && s.available !== available
  const stale =
    (!newBoot && !turned && s.rev < lastRev) ||
    // A read that left before the stream brought a new start, or the queue
    // going off or on, answered as things were before it.
    (readFrom !== null && readFrom !== turnsSeen && (newBoot || turned))
  if (stale) {
    reads++
    return true
  }
  if (newBoot || turned) {
    missedUpTo = 0
    if (readFrom === null) turnsSeen++
  }
  boot = s.boot
  lastRev = s.rev
  baseRev = s.rev
  revOf.clear()
  jobsById = new Map(s.jobs.map((j) => [j.id, j]))
  groupsById = new Map(s.groups.map((g) => [g.id, g]))
  progressById = new Map(
    Object.entries(s.progress).filter(([id]) => {
      const status = jobsById.get(id)?.status
      return status !== undefined && LIVE.has(status)
    }),
  )
  lane = s.lane
  available = s.available
  reason = s.reason
  comfy = s.comfy
  absent = false
  reads++
  changed()
  return true
}

const entityRev = (key: string) => revOf.get(key) ?? baseRev

function putJob(job: RunnerJob, rev: number): void {
  const key = `job:${job.id}`
  if (rev < entityRev(key)) return
  revOf.set(key, rev)
  jobsById.set(job.id, job)
  if (TERMINAL.has(job.status)) progressById.delete(job.id)
}

function putGroup(group: RunnerGroup, rev: number): void {
  const key = `group:${group.id}`
  if (rev < entityRev(key)) return
  revOf.set(key, rev)
  groupsById.set(group.id, group)
  // The server has it: a hand-over of this tab whose answer never came (the
  // page slept through it) is done with.
  settleEntry(group.id)
}

/**
 * An event from a commit. The next revision, or one already seen, is applied
 * (each thing it names only if it is not older than what is held); anything
 * further ahead means events were missed, and the whole state is read again.
 */
function onCommit(rev: unknown, apply: (rev: number) => void): void {
  if (typeof rev !== 'number' || !Number.isFinite(rev)) return
  if (rev > lastRev + 1) {
    missedUpTo = Math.max(missedUpTo, rev)
    void refresh()
    return
  }
  apply(rev)
  lastRev = Math.max(lastRev, rev)
  changed()
}

const EVENTS: Record<string, (d: any) => void> = {
  state: (d) => {
    const before = boot
    if (!takeSnapshot(d)) return
    backoff = BACKOFF_FIRST
    setConnected(true)
    // The queue was started again: its state came whole with the stream, and
    // is read once more so nothing said between the two starts is missed.
    if (before && before !== boot) void refresh()
    void replayOutbox()
  },
  job: (d) => {
    if (!isObj(d?.job) || typeof d.job.id !== 'string') return
    onCommit(d.rev, (rev) => putJob(d.job, rev))
  },
  group: (d) => {
    if (!isObj(d?.group) || typeof d.group.id !== 'string') return
    onCommit(d.rev, (rev) => putGroup(d.group, rev))
  },
  lane: (d) => {
    if (!isObj(d?.lane)) return
    onCommit(d.rev, (rev) => {
      if (rev < entityRev('lane')) return
      revOf.set('lane', rev)
      lane = d.lane
    })
  },
  gone: (d) => {
    onCommit(d?.rev, (rev) => {
      for (const id of Array.isArray(d.jobs) ? d.jobs : []) {
        if (rev < entityRev(`job:${id}`)) continue
        revOf.set(`job:${id}`, rev)
        jobsById.delete(id)
        progressById.delete(id)
      }
      for (const id of Array.isArray(d.groups) ? d.groups : []) {
        if (rev < entityRev(`group:${id}`)) continue
        revOf.set(`group:${id}`, rev)
        groupsById.delete(id)
      }
    })
  },
  comfy: (d) => {
    if (!isObj(d)) return
    comfy = { answering: typeof d.answering === 'boolean' ? d.answering : null, since: Number(d.since) || 0 }
    changed()
  },
  progress: (d) => {
    if (!isObj(d) || typeof d.id !== 'string') return
    const job = jobsById.get(d.id)
    if (job && TERMINAL.has(job.status)) return
    progressById.set(d.id, d as RunnerProgress)
    changed()
  },
  preview: (d) => {
    if (!isObj(d) || typeof d.id !== 'string' || typeof d.n !== 'number') return
    const had = progressById.get(d.id)
    progressById.set(
      d.id,
      had
        ? { ...had, previewN: d.n }
        : { id: d.id, value: 0, max: 0, node: null, classType: null, pass: null, at: Date.now(), previewN: d.n },
    )
    changed()
  },
}

// ------------------------------------------------------------ read + stream --

let started = false
let refreshing: Promise<void> | null = null
let stream: EventSource | null = null
let reconnectTimer: ReturnType<typeof setTimeout> | null = null
const BACKOFF_FIRST = 1000
const BACKOFF_LONGEST = 30_000
let backoff = BACKOFF_FIRST

const hidden = () => typeof document !== 'undefined' && document.visibilityState === 'hidden'

/**
 * A stream is worth its connection only while there is something to watch.
 * Over plain http a browser allows six connections to one host across every
 * tab, and a queue that is turned off, with nothing in it, would hold one for
 * nothing; a page shown again reads the state and opens it if that changed.
 */
const worthStreaming = () => !absent && (available || jobsById.size > 0)

/** Nothing serves the queue here: an older server, or a static host. */
function settleAbsent(): void {
  absent = true
  available = false
  reason = NO_QUEUE
  closeStream()
  changed()
  void replayOutbox()
}

/** Reads made in a row to catch up with a gap, so a server that stays behind cannot keep one asking. */
let catchUps = 0

/**
 * Read the whole state. A server that answers with anything but JSON (the
 * page Vite serves for a route it has not got) has no queue, and is left
 * alone; one that does not answer is asked again later.
 *
 * A read already out is shared. With `fresh`, the read is one that leaves
 * after this call, behind the one already out, which may have left before
 * what the caller has just heard of (a word refused because a hold changed).
 */
function refresh(opts: { fresh?: boolean } = {}): Promise<void> {
  if (refreshing && opts.fresh) return refreshing.then(() => refresh(), () => refresh())
  if (refreshing) return refreshing
  let again = false
  const readFrom = turnsSeen
  refreshing = (async () => {
    let a: Answer
    try {
      a = await ask(API, { headers: { Accept: 'application/json' } }, READ_MS)
    } catch {
      unanswered()
      return
    }
    if (a.status === 200 && a.type === 'json') {
      let took: boolean
      takingRead = true
      try {
        took = takeSnapshot(a.json, readFrom)
      } finally {
        takingRead = false
      }
      if (took) {
        if (started && !hidden() && worthStreaming()) openStream()
        void replayOutbox()
        // A read that left before the missed commit is asked once more; the
        // stream's next event, or its next state, covers anything after that.
        again = lastRev < missedUpTo && catchUps++ < 1
        if (!again) catchUps = 0
        // The stream brought a new start while this read was out, and asked
        // for a read of its own, which this one stood in for; make it. So too
        // when the queue went off or came back on meanwhile: this read may
        // tell of the queue as it was, and was then left alone.
        if (readFrom !== turnsSeen) again = true
      } else settleAbsent()
      return
    }
    if (a.status === 404 || (a.status < 500 && a.type !== 'json' && a.type !== 'broken')) {
      settleAbsent()
      return
    }
    unanswered()
  })().finally(() => {
    refreshing = null
    if (again) void refresh()
  })
  return refreshing
}

function unanswered(): void {
  setConnected(false)
  scheduleReconnect()
}

function scheduleReconnect(): void {
  if (!started || absent || reconnectTimer || hidden()) return
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null
    void refresh()
  }, backoff)
  backoff = Math.min(BACKOFF_LONGEST, backoff * 2)
}

function openStream(): void {
  if (stream || absent || typeof EventSource === 'undefined') return
  let source: EventSource
  try {
    source = new EventSource(`${API}/stream`)
  } catch {
    return
  }
  stream = source
  for (const [name, handle] of Object.entries(EVENTS)) {
    source.addEventListener(name, (ev) => {
      if (stream !== source) return
      let data: unknown
      try {
        data = JSON.parse(String((ev as MessageEvent).data))
      } catch {
        return
      }
      handle(data)
    })
  }
  // The browser reconnects a stream by itself after a network error, and the
  // server's first word on every connection is the whole state. It does not
  // after an answer that is not the stream (a proxy's 502 while the server
  // restarts): the stream closes for good, so it is dropped and opened again
  // from a fresh read, later each time.
  source.onerror = () => {
    if (stream !== source) return
    setConnected(false)
    if (source.readyState === 2 /* CLOSED */) {
      stream = null
      scheduleReconnect()
    }
  }
}

function closeStream(): void {
  stream?.close()
  stream = null
  if (connected) {
    connected = false
    changed()
  }
}

/**
 * A hidden tab gives its stream back, as the archive's does (archiveSync.ts):
 * nobody is reading it, and the server goes on without it. Shown again, the
 * tab reads the whole state, since the stream said nothing meanwhile.
 */
function onVisibility(): void {
  if (absent) return
  if (hidden()) {
    closeStream()
    return
  }
  backoff = BACKOFF_FIRST
  void refresh()
}

/** A page restored from the browser's back-forward cache has a dead stream. */
function onPageShow(ev: Event): void {
  if (absent || !(ev as PageTransitionEvent).persisted) return
  closeStream()
  void refresh()
}

function start(): void {
  if (started) return
  started = true
  if (typeof document !== 'undefined' && typeof document.addEventListener === 'function') {
    document.addEventListener('visibilitychange', onVisibility)
  }
  if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
    window.addEventListener('pageshow', onPageShow)
  }
  void refresh()
}

export const runnerStore = {
  subscribe,
  snapshot: (): RunnerSnapshot => current,
  /** Idempotent; App.tsx calls it once. */
  start,
  /** Read the whole state; `{ fresh: true }` for a read that leaves after the call. */
  refresh,
}

/** The server's queue, for a component. */
export function useRunner(): RunnerSnapshot {
  useEffect(() => runnerStore.start(), [])
  return useSyncExternalStore(subscribe, runnerStore.snapshot, runnerStore.snapshot)
}

/**
 * Whether a desk's next Make, Run or Render should go to the server's queue.
 * The server must say it has one and takes this desk's work, and the live
 * state must not say it has stopped since (another server took the archive,
 * or the queue gave up after repeated faults).
 *
 * The server's answer is kept for the page's life, and a queue that was off
 * when it was given may be running now: the server started again with it on,
 * or the other server let the archive go. So where it said no, the live state
 * is asked, read afresh unless the stream keeps it, and where that says the
 * queue runs, the server is asked again.
 *
 * The live state's own no is read afresh too, unless the stream keeps it: a
 * queue that is off with nothing in it keeps no stream open (worthStreaming),
 * so nothing tells the page when it comes back. A desk whose store says the
 * queue is off asks here, not the store alone, for that reason.
 */
export async function runnerAvailable(desk: RunnerDesk): Promise<{ ok: boolean; reason: string | null }> {
  let caps = await serverCapabilities()
  if (!caps.runner) {
    start()
    if (!connected) await refresh()
    if (boot && available) {
      forgetCapabilities()
      caps = await serverCapabilities()
    }
  }
  if (!caps.runner) return { ok: false, reason: caps.runnerReason ?? caps.reason ?? NO_QUEUE }
  if (!caps.runnerDesks.includes(desk)) return { ok: false, reason: NOT_THIS_DESK }
  start()
  if (absent) return { ok: false, reason: NO_QUEUE }
  if (boot && !available && !connected) await refresh()
  if (boot && !available) return { ok: false, reason: reason ?? NOT_RUNNING }
  return { ok: true, reason: null }
}

// ---------------------------------------------------------------------------
// Handing work over
// ---------------------------------------------------------------------------

const OUTBOX_KEY = 'switchgen.runner.outbox.v1'
const WITHDRAWN_KEY = 'switchgen.runner.withdrawn.v1'
/** How old a hand-over may be and still be sent when its page comes back. */
export const OUTBOX_MAX_AGE_MS = 10 * 60_000
/** How long one hand-over keeps asking before it says it is still pending. */
export const SUBMIT_RETRY_MS = 30_000
const RETRY_WAITS = [1000, 2000, 4000, 8000]
/** How many withdrawn ids the tab keeps, the newest. An id matters only while its hand-over could still go. */
const WITHDRAWN_KEPT = 200

/** A hand-over in the tab. `sent`: a POST of it has left, so the server may have it. */
type OutboxEntry = { at: number; body: SubmitBody; sent: boolean }

export type GivenUp = { groupId: string; desk: RunnerDesk; label: string; at: number; line: string }

function readOutbox(): OutboxEntry[] {
  const raw = tabStore.get(OUTBOX_KEY)
  if (!raw) return []
  try {
    const list: unknown = JSON.parse(raw)
    if (!Array.isArray(list)) return []
    return list
      .filter(
        (e): e is OutboxEntry =>
          isObj(e) &&
          typeof e.at === 'number' &&
          isObj(e.body) &&
          isObj(e.body.group) &&
          typeof e.body.group.id === 'string' &&
          Array.isArray(e.body.jobs),
      )
      // One kept before the tab said whether it was sent may have been.
      .map((e) => ({ at: e.at, body: e.body, sent: e.sent !== false }))
  } catch {
    return []
  }
}

function readWithdrawn(): string[] {
  try {
    const list: unknown = JSON.parse(tabStore.get(WITHDRAWN_KEY) ?? '[]')
    return Array.isArray(list) ? list.filter((id): id is string => typeof id === 'string') : []
  } catch {
    return []
  }
}

/** What waits to be handed over, oldest first. Memory is the truth; the tab keeps a copy for a reload. */
let outbox: OutboxEntry[] = readOutbox()
/** Hand-overs a submitGroup call is still asking about, which a replay leaves alone. */
const inFlight = new Set<string>()
const givenUp: GivenUp[] = []
/**
 * Work the reader stopped before the server listed it: a group's id for all
 * of it, a job's for that job. Nothing here hands it over again.
 */
const withdrawn = new Set<string>(readWithdrawn())

function keepOutbox(): void {
  // A tab that will not keep it (full, or private) still has the copy in memory.
  if (outbox.length) tabStore.set(OUTBOX_KEY, JSON.stringify(outbox))
  else tabStore.remove(OUTBOX_KEY)
}

function settleEntry(groupId: string): void {
  const before = outbox.length
  outbox = outbox.filter((e) => e.body.group.id !== groupId)
  if (outbox.length !== before) keepOutbox()
}

function giveUp(entry: OutboxEntry, line: string): void {
  settleEntry(entry.body.group.id)
  givenUp.push({
    groupId: entry.body.group.id,
    desk: entry.body.group.desk,
    label: entry.body.group.label,
    at: entry.at,
    line,
  })
  changed()
}

/** Hand-overs from this tab that were never sent, each with the line that says why. */
export function givenUpBatches(): GivenUp[] {
  return [...givenUp]
}

/** The reader has seen that a hand-over was given up. */
export function forgetGivenUp(groupId: string): void {
  const i = givenUp.findIndex((g) => g.groupId === groupId)
  if (i < 0) return
  givenUp.splice(i, 1)
  changed()
}

/**
 * Hand-overs from this tab still waiting for the server's answer. `sent` says
 * whether a POST of one has left: until it has, the server cannot have it,
 * and it was pressed but never handed over.
 */
export function outboxPending(): { groupId: string; desk: RunnerDesk; label: string; at: number; jobIds: string[]; sent: boolean }[] {
  return outbox.map((e) => ({
    groupId: e.body.group.id,
    desk: e.body.group.desk,
    label: e.body.group.label,
    at: e.at,
    jobIds: e.body.jobs.map((j) => j.id),
    sent: e.sent,
  }))
}

/** A body less the work withdrawn from it; null when none of it is left. */
function lessWithdrawn(body: SubmitBody): SubmitBody | null {
  if (withdrawn.has(body.group.id)) return null
  const jobs = body.jobs.filter((j) => !withdrawn.has(j.id))
  if (jobs.length === body.jobs.length) return body
  return jobs.length ? { ...body, jobs } : null
}

/**
 * The reader stopped work the server has not listed: the jobs named, or the
 * whole group. It comes out of the tab's outbox, and the tab keeps the word,
 * so neither submitGroup nor a replay, in this page or one loaded after it in
 * the tab, hands it over; a group with nothing left goes from the outbox. A
 * POST of it that already left may have got there, so what the server lists
 * of it is still the desk's to stop.
 */
export function withdraw(groupId: string, jobIds?: string[]): void {
  for (const id of jobIds ?? [groupId]) withdrawn.add(id)
  tabStore.set(WITHDRAWN_KEY, JSON.stringify([...withdrawn].slice(-WITHDRAWN_KEPT)))
  outbox = outbox.flatMap((e) => {
    if (e.body.group.id !== groupId) return [e]
    const body = lessWithdrawn(e.body)
    return body === e.body ? [e] : body ? [{ ...e, body }] : []
  })
  keepOutbox()
  if (!outbox.some((e) => e.body.group.id === groupId)) inFlight.delete(groupId)
  armReplay()
}

type Verdict =
  | { kind: 'done'; result: SubmitTaken }
  | { kind: 'fallback'; result: SubmitFallback }
  | { kind: 'refused'; result: SubmitRefused }
  | { kind: 'retry' }

/**
 * One POST, sorted by what its answer proves.
 *
 * Only the queue's own no, or an answer that shows there is no queue at this
 * address, proves nothing was kept. A dropped connection, a deadline, a
 * proxy's 502 or 504, or a body cut short could each come after the server
 * saved the work, so each is asked again with the same ids; running it in the
 * page instead could make it twice.
 */
async function postGroup(body: SubmitBody): Promise<Verdict> {
  let a: Answer
  try {
    a = await ask(`${API}/groups`, posting(body), SUBMIT_MS)
  } catch {
    return { kind: 'retry' }
  }
  const s = a.status
  if (a.type === 'broken') return { kind: 'retry' }
  if (s === 200 && a.type === 'json' && isObj(a.json?.group) && Array.isArray(a.json?.jobs)) {
    takeIntake(a.json)
    return {
      kind: 'done',
      result: { ok: true, replayed: a.json.replayed === true, group: a.json.group, jobs: a.json.jobs },
    }
  }
  if (s === 404 || s === 405 || s === 501 || s === 503 || (s < 500 && a.type === 'html')) {
    const said = a.json?.reason ?? a.json?.error
    return { kind: 'fallback', result: { ok: false, fallback: true, reason: typeof said === 'string' && said ? said : NO_QUEUE } }
  }
  if (s === 409 && a.json?.conflict === 'id') {
    // The server has work under these ids: a send whose answer was lost got
    // there, and this is the same group less what was withdrawn since. Its
    // own list says what it has, and that settles the hand-over.
    void refresh({ fresh: true })
    return { kind: 'retry' }
  }
  if ((s >= 400 && s < 500) || s === 507) {
    const busy = a.json?.busy === 'images' || a.json?.busy === 'reel' ? (a.json.busy as 'images' | 'reel') : undefined
    const error = typeof a.json?.error === 'string' && a.json.error ? a.json.error : `The server would not take this work (HTTP ${s}).`
    return { kind: 'refused', result: { ok: false, fallback: false, status: s, error, ...(busy ? { busy } : {}) } }
  }
  return { kind: 'retry' }
}

/** Post a hand-over of the outbox, noting first that a POST of it has left. */
function send(entry: OutboxEntry): Promise<Verdict> {
  if (!entry.sent) {
    entry.sent = true
    keepOutbox()
  }
  return postGroup(entry.body)
}

/** The server's answer for a group it lists already, from the store. */
function listedAs(groupId: string): SubmitTaken | null {
  const group = groupsById.get(groupId)
  if (!group) return null
  const jobs = group.jobIds.map((id) => jobsById.get(id)).filter((j): j is RunnerJob => j !== undefined)
  return { ok: true, replayed: true, group, jobs }
}

/** The intake's own answer, taken into the store at its revision. */
function takeIntake(answer: any): void {
  const rev = Number(answer.rev)
  if (!Number.isFinite(rev)) return
  putGroup(answer.group, rev)
  for (const j of answer.jobs as RunnerJob[]) if (isObj(j) && typeof j.id === 'string') putJob(j, rev)
  // A state read from before the intake must not take these jobs away again.
  lastRev = Math.max(lastRev, rev)
  changed()
  if (started && !hidden()) openStream()
}

/**
 * Keep a hand-over in the tab before anything is sent: the group as it will
 * go, with this browser named where the desk left the device blank, less
 * anything withdrawn. A desk whose press must wait for an earlier hand-over's
 * answer stages its own at once, so the work is never only in the page's
 * memory: a tab the browser throws away meanwhile sends it, in the order it
 * was pressed, when the page loads again within ten minutes. Until then a
 * replay leaves it alone, since submitGroup(body) sends it in its turn.
 *
 * Staged again under the same group id, it keeps its place, the time it was
 * first asked for and whether it was sent, and takes the new body. Returns
 * the body as kept, or null where all of it was withdrawn.
 */
export function stage(body: SubmitBody): SubmitBody | null {
  const left = lessWithdrawn(body)
  if (!left) return null
  const kept: SubmitBody = { ...left, group: { ...left.group, device: left.group.device || deviceId() } }
  const groupId = kept.group.id
  outbox = outbox.some((e) => e.body.group.id === groupId)
    ? outbox.map((e) => (e.body.group.id === groupId ? { ...e, body: kept } : e))
    : [...outbox, { at: Date.now(), body: kept, sent: false }]
  keepOutbox()
  inFlight.add(groupId)
  return kept
}

const pending = (): SubmitPending => ({ ok: false, fallback: false, pending: true })

/**
 * Hand a group of jobs to the server's queue.
 *
 * The hand-over is kept in the tab before it goes (see stage), and dropped
 * once the server has answered yes or no. Without an answer it is asked again
 * for about half a minute with the same ids; after that it is left in the tab,
 * reported as pending, and sent again on its own (see armReplay), or when the
 * page is loaded again within ten minutes. Each try sends what is left of it:
 * work withdrawn meanwhile is not sent, and a group withdrawn whole is not
 * sent at all, and reported as pending, since an earlier try may have got
 * there.
 */
export async function submitGroup(body: SubmitBody): Promise<SubmitResult> {
  const groupId = body.group.id
  stage(body)
  try {
    const until = Date.now() + SUBMIT_RETRY_MS
    for (let n = 0; ; n++) {
      // Listed already: an earlier try got there and only its answer was lost.
      const listed = listedAs(groupId)
      if (listed) {
        settleEntry(groupId)
        return listed
      }
      const entry = outbox.find((e) => e.body.group.id === groupId)
      if (!entry) return pending()
      const v = await send(entry)
      // After an attempt with no clear answer, the work may be on the server
      // already, waiting for a queue that has since stood back; running it in
      // the page too could make it twice, so it stays handed over.
      if (v.kind === 'fallback' && n > 0) return pending()
      if (v.kind !== 'retry') {
        settleEntry(groupId)
        return v.result
      }
      const wait = RETRY_WAITS[Math.min(n, RETRY_WAITS.length - 1)]
      if (Date.now() + wait > until) return pending()
      await sleep(wait)
    }
  } finally {
    inFlight.delete(groupId)
    armReplay()
  }
}

let replaying = false
let replayTimer: ReturnType<typeof setTimeout> | null = null
let replayWait = BACKOFF_FIRST

/**
 * Send again what this tab handed over without an answer. A hand-over the
 * server already lists got there: the answer was lost, not the work (a phone
 * that locked its screen just after the POST went, say), and it is let go.
 * Anything else older than ten minutes is dropped with a line instead: the
 * reader has likely moved on, and work arriving long after it was asked for
 * would be a surprise. A replay the server refuses is dropped too, with the
 * server's reason; one it does not answer stays, and goes again on its own.
 *
 * Nothing is judged before the server's state has been read, since only that
 * says what it already has; every whole state taken calls this again.
 */
async function replayOutbox(): Promise<void> {
  if (replaying || !outbox.length) return
  // Where nothing serves the queue there is no state to read, and nothing
  // there could have taken the work.
  if (!boot && !absent) return
  replaying = true
  try {
    for (const entry of [...outbox]) {
      if (inFlight.has(entry.body.group.id) || !outbox.includes(entry)) continue
      if (groupsById.has(entry.body.group.id) || entry.body.jobs.some((j) => jobsById.has(j.id))) {
        settleEntry(entry.body.group.id)
        continue
      }
      if (Date.now() - entry.at > OUTBOX_MAX_AGE_MS) {
        giveUp(entry, entry.sent ? OUTBOX_GIVEN_UP : OUTBOX_TOO_OLD)
        continue
      }
      const v = await send(entry)
      // Nothing answered: nor would the next, so they wait for the server together.
      if (v.kind === 'retry') break
      const groupId = entry.body.group.id
      if (v.kind === 'done') settleEntry(groupId)
      // Gone from the outbox while the POST was out: the reader stopped all of
      // it, or another send settled it, and the desk has said so already.
      else if (!outbox.some((e) => e.body.group.id === groupId)) continue
      else if (v.kind === 'fallback') giveUp(entry, OUTBOX_NOT_TAKEN)
      else giveUp(entry, v.result.error)
    }
  } finally {
    replaying = false
    armReplay()
  }
}

/**
 * While the outbox holds a hand-over that nothing is sending, send it again
 * on a timer, a second at first and longer each time, up to half a minute.
 * The stream can stay up the whole while and bring no whole state to send it
 * on. With the stream down the state is read first, since only that says
 * whether the server has it already, and the read sends it.
 */
function armReplay(): void {
  if (!outbox.some((e) => !inFlight.has(e.body.group.id))) {
    if (replayTimer) clearTimeout(replayTimer)
    replayTimer = null
    replayWait = BACKOFF_FIRST
    return
  }
  if (replayTimer) return
  replayTimer = setTimeout(() => {
    replayTimer = null
    void (connected ? replayOutbox() : refresh()).then(armReplay, armReplay)
  }, replayWait)
  replayWait = Math.min(BACKOFF_LONGEST, replayWait * 2)
}

// ---------------------------------------------------------------------------
// Following a job
// ---------------------------------------------------------------------------

/** The error a desk's catch already knows, for a job that did not end done. */
export function runnerFault(job: RunnerJob, words: FaultWords = {}): ComfyError | LostJob {
  const e = job.error
  const code: RunnerErrorCode = e?.code ?? codeOf(job.status)
  const promptId = job.promptId
  switch (code) {
    case 'refused':
    case 'failed':
      return new ComfyError(e?.message || words[code] || '', {
        node: e?.node ?? null,
        nodeType: e?.nodeType ?? null,
        nodeErrors: e?.nodeErrors ?? null,
        promptId,
      })
    case 'stopped':
    case 'skipped': {
      const sent = e ? e.sent : promptId !== null
      return new ComfyError(words[code] ?? (sent ? STOPPED_SENT : STOPPED_UNSENT), { cancelled: true, promptId })
    }
    case 'lost':
      return new LostJob(words.lost ?? RUNNER_LOST, promptId ?? '')
    case 'unsent':
      return new LostJob(words.unsent ?? RUNNER_UNSENT, promptId ?? '')
    case 'ended-unsent':
      return new LostJob(words['ended-unsent'] ?? ENDED_UNSENT, promptId ?? '', { mayExist: true })
    case 'no-file':
      return new ComfyError(words['no-file'] ?? NO_FILE, { promptId })
    case 'no-frame':
      return new ComfyError(words['no-frame'] ?? NO_FRAME, { promptId })
    case 'internal':
      return new ComfyError(words.internal ?? (e?.message || INTERNAL), { promptId })
  }
}

function codeOf(status: RunnerStatus): RunnerErrorCode {
  switch (status) {
    case 'failed':
    case 'stopped':
    case 'lost':
    case 'unsent':
    case 'skipped':
      return status
    default:
      return 'internal'
  }
}

function abortReason(signal: AbortSignal | undefined): unknown {
  if (signal?.reason !== undefined) return signal.reason
  const err = new Error('Stopped following the job.')
  err.name = 'AbortError'
  return err
}

/**
 * Follow one of the server's jobs to its end, in the shape run() gives a desk:
 * the same events while it goes, the files when it is done, and the same
 * ComfyError or LostJob when it is not, so a desk's words for an ending stay
 * as they are. `words` replaces the sentence for an ending only the desk can
 * put well (which shot a reel shot opens on, say).
 *
 * The server follows the job whether or not anything here does; the signal
 * only stops this page listening.
 */
export function follow(
  jobId: string,
  on: (e: FollowEvent) => void,
  opts: { signal?: AbortSignal; words?: FaultWords } = {},
): Promise<FollowResult> {
  start()
  return new Promise<FollowResult>((resolve, reject) => {
    let settled = false
    let told = false
    let lastRun = ''
    let lastPreview = 0
    /** The count of whole reads when this job was found missing, before asking again. */
    let readsAtAsk: number | null = null
    /** The read this follow waits on to end before it judges the job again. */
    let awaited: Promise<void> | null = null
    let unsubscribe = () => {}

    const tell = (e: FollowEvent) => {
      try {
        on(e)
      } catch {
        /* the desk's own trouble; the job goes on */
      }
    }
    const finish = (then: () => void) => {
      if (settled) return
      settled = true
      unsubscribe()
      opts.signal?.removeEventListener('abort', onAbort)
      then()
    }
    const onAbort = () => finish(() => reject(abortReason(opts.signal)))

    const check = () => {
      if (settled) return
      const job = jobsById.get(jobId)
      if (!job && absent) {
        finish(() => reject(new LostJob(FORGOTTEN, '')))
        return
      }
      if (!job) {
        // Not heard of yet: the state has not been read, or the hand-over is
        // still on its way. Otherwise it is asked about once more before the
        // job is taken to be gone.
        if (!boot || outbox.some((e) => e.body.jobs.some((j) => j.id === jobId))) return
        if (readsAtAsk === null) {
          readsAtAsk = reads
          // The read being taken in now is the one that found it missing, so
          // the question goes to a new read once that one is done.
          const asked = takingRead && refreshing ? refreshing.then(() => refresh()) : refresh()
          void asked.then(check)
          return
        }
        // A read is on its way: the job is judged by what it brings, once it
        // has ended, whether or not anything changes after it.
        if (refreshing) {
          if (awaited !== refreshing) {
            awaited = refreshing
            void refreshing.then(check)
          }
          return
        }
        // Only a read that got through says the server does not have it.
        if (reads === readsAtAsk) return
        finish(() => reject(new LostJob(FORGOTTEN, '')))
        return
      }
      if (job.promptId && !told) {
        told = true
        tell({ phase: 'queued', promptId: job.promptId })
      }
      if (job.status === 'done') {
        finish(() => resolve(resultOf(job)))
        return
      }
      if (TERMINAL.has(job.status)) {
        finish(() => reject(runnerFault(job, opts.words)))
        return
      }
      const p = progressById.get(jobId)
      if (job.status === 'running' || (job.status === 'queued' && p)) {
        // Steps when the server has heard them, else 0 of 1, as run() says it.
        const run = {
          phase: 'running' as const,
          node: p?.node ?? null,
          value: p ? p.value : 0,
          max: p ? p.max : 1,
          classType: p?.classType ?? null,
          pass: p?.pass ?? null,
        }
        const key = `${run.node}|${run.value}|${run.max}|${run.classType}|${run.pass?.index ?? ''}/${run.pass?.count ?? ''}`
        if (key !== lastRun) {
          lastRun = key
          tell(run)
        }
      }
      if (p && p.previewN > lastPreview) {
        lastPreview = p.previewN
        tell({ phase: 'preview', url: previewUrl(jobId, p.previewN) })
      }
    }

    unsubscribe = subscribe(check)
    if (opts.signal?.aborted) {
      onAbort()
      return
    }
    opts.signal?.addEventListener('abort', onAbort, { once: true })
    check()
  })
}

function resultOf(job: RunnerJob): FollowResult {
  return {
    files: job.files,
    primary: job.primary,
    frame: job.frame,
    entryId: job.entryId,
    entryNo: job.entryNo,
    repeatOf: job.repeatOf,
    openedOn: job.openedOn,
    durationMs: job.durationMs,
    finishedAt: job.finishedAt ?? job.endedAt ?? Date.now(),
  }
}

/** The newest preview of a job. `n` makes each one a new address. */
export function previewUrl(id: string, n: number): string {
  return `${API}/jobs/${encodeURIComponent(id)}/preview?n=${n}`
}

// ---------------------------------------------------------------------------
// Words to the queue
// ---------------------------------------------------------------------------

/**
 * One word to the queue. What it changed arrives on the stream, at its
 * revision, so the answer's copy is not taken here: an answer that came back
 * after a later event would put a job back as it was. A page without the
 * stream reads the state instead.
 */
async function word(path: string, body: unknown): Promise<boolean> {
  try {
    const a = await ask(path, posting(body), WORD_MS)
    if (a.status !== 200 || a.type !== 'json') {
      // The queue says the page's copy is behind it (a hold another device
      // already answered, or one that has changed since this page showed
      // it): read it, so what the page shows goes with it. A read already
      // out may have left before the change, so one is made after it.
      if (a.status === 409 && a.type === 'json') void refresh({ fresh: true })
      return false
    }
  } catch {
    return false
  }
  if (!connected) void refresh()
  return true
}

/** Stop one job, from any page on any device. */
export function stopJob(id: string): Promise<boolean> {
  return word(`${API}/jobs/${encodeURIComponent(id)}/stop`, {})
}

/** Stop every job of a group that has not ended, and end the group. */
export function stopGroup(id: string): Promise<boolean> {
  return word(`${API}/groups/${encodeURIComponent(id)}/stop`, {})
}

/**
 * Answer a held lane: send what it holds, or stop all of it. `since` is the
 * `since` of the hold the page showed (lane.held.since), so the word answers
 * that hold and no other: where the hold has changed since (answered on
 * another device, and a new one made), the server refuses it, and the page
 * reads the state again and shows the hold as it now stands.
 */
export function laneWord(action: 'send' | 'stop', since: number): Promise<boolean> {
  return word(`${API}/lane`, { action, since })
}

/** Put finished jobs out of sight on every device. */
export function dismiss(ids: string[]): Promise<boolean> {
  return word(`${API}/dismiss`, { jobIds: ids })
}

// ---------------------------------------------------------------------------
// What a desk says
// ---------------------------------------------------------------------------

const NOUN: Record<RunnerDesk, string> = { video: 'clip', images: 'picture', reel: 'shot' }

/**
 * What a job is doing, as a desk's stage line and the note under it. The
 * sentences are the Video desk's own for its lane, so a clip reads the same
 * whichever of the two sends it; a picture or a shot is named as one.
 */
export function waitLine(job: RunnerJob, desk: RunnerDesk = job.desk): { stage: string; note: string | null } {
  const noun = NOUN[desk] ?? 'job'
  // The queue is off, or stands back for another server, and lists the work
  // it keeps as it was last saved: the job is shown as it stands, with the
  // server's own sentence for why nothing here moves it on. A stop asked for
  // before the queue went off cannot land while it is off either, so that
  // sentence goes under Stopping too.
  const off = queueOff() && LIVE.has(job.status) ? (reason ?? NOT_RUNNING) : null
  if (job.stopRequested && LIVE.has(job.status)) return { stage: 'Stopping', note: off }
  if (off) {
    return job.status === 'waiting'
      ? { stage: 'Waiting for the server’s queue', note: off }
      : { stage: lineOf(job, noun, desk).stage, note: off }
  }
  return lineOf(job, noun, desk)
}

/** The server has said where its queue stands, and it is not running. */
function queueOff(): boolean {
  return boot !== '' && !absent && !available
}

/** waitLine for a queue that is running. */
function lineOf(job: RunnerJob, noun: string, desk: RunnerDesk): { stage: string; note: string | null } {
  switch (job.status) {
    case 'releasing':
      return { stage: 'Releasing memory', note: null }
    case 'sending':
      // Sent without a clear answer, and ComfyUI is not answering the question.
      if (job.wait?.for === 'comfy') {
        return {
          stage: 'Asking ComfyUI whether it has it',
          note: `ComfyUI is not answering; it may be restarting. The server asks again until it answers, and never sends this ${noun} twice.`,
        }
      }
      return { stage: 'Sending it to the press', note: null }
    case 'queued':
      return { stage: 'Queued', note: null }
    case 'running':
      return { stage: 'Drawing', note: null }
    case 'filing':
      return {
        stage: 'Filing',
        note: job.wait?.for === 'disk' ? 'The archive could not take the record yet. The server keeps trying.' : null,
      }
    case 'done':
      return { stage: 'Done', note: null }
    case 'failed':
      return { stage: 'Failed', note: null }
    case 'stopped':
      return { stage: 'Stopped', note: null }
    case 'lost':
      return { stage: 'Lost', note: null }
    case 'unsent':
    case 'skipped':
      return { stage: 'Not sent', note: null }
    case 'waiting':
      break
  }

  const w = job.wait
  switch (w?.for) {
    case 'before':
      return { stage: `Waiting for the ${noun} before it`, note: null }
    case 'heavy':
      return {
        stage: 'Waiting for a heavy clip to go first',
        note: 'That clip needs the card’s memory to itself, so this goes once it has finished.',
      }
    case 'queue': {
      const n = w.ahead ?? 0
      return {
        stage: 'Waiting for the press',
        note:
          n > 0
            ? `ComfyUI has ${n} ${n === 1 ? 'job' : 'jobs'} to finish first. This ${noun} waits for them, so the memory they hold can be released before it starts.`
            : `ComfyUI has work to finish first. This ${noun} waits for it, so the memory it holds can be released before it starts.`,
      }
    }
    case 'comfy':
      return {
        stage: 'Waiting for ComfyUI',
        note: job.heavy
          ? `ComfyUI is not answering; it may be restarting. This ${noun} waits until it answers, then releases its memory and goes.`
          : `ComfyUI is not answering; it may be restarting. This ${noun} waits until it answers, then goes.`,
      }
    case 'held':
      // The word that answered the hold clears it at once, and the job's own
      // wait follows on the queue's next round, as a later event: until then
      // it waits its turn, and is not shown held by a hold that is gone.
      if (lane.held && holdCovers(lane.held, job)) return { stage: 'Held', note: heldNote(lane.held, job) }
      return turnLine(job, noun, desk)
    case 'disk':
      return {
        stage: 'Waiting for room on the server',
        note: 'The server’s disk is full, so it cannot save its list of work. This goes on by itself once there is room.',
      }
    case 'turn':
      return turnLine(job, noun, desk)
    default:
      // Taken in, and not yet looked at by the queue's next round.
      return { stage: 'Waiting its turn', note: null }
  }
}

/** A job that waits for the work ahead of it on the press. */
function turnLine(job: RunnerJob, noun: string, desk: RunnerDesk): { stage: string; note: string | null } {
  return {
    stage: 'Waiting its turn',
    note: !job.heavy
      ? null
      : desk === 'video'
        ? 'Waits for the clip before it to finish, so that one’s memory can be released before this starts.'
        : `Waits for the work before it to finish, so that its memory can be released before this ${noun} starts.`,
  }
}

const REPORTED: Record<RunnerStatus, Reported['status']> = {
  waiting: 'submitting',
  releasing: 'submitting',
  sending: 'submitting',
  queued: 'queued',
  running: 'running',
  filing: 'running',
  done: 'done',
  failed: 'error',
  lost: 'error',
  unsent: 'error',
  stopped: 'cancelled',
  skipped: 'cancelled',
}

/**
 * A job as the section bar reads it. The prompt id is there from 'queued' on,
 * so every device counts the job as SwitchGen's own and not as someone else's
 * work in ComfyUI's queue. Without `progress`, the store's is used.
 *
 * Its start is the one the desk's own bridge gives: a clip is timed from when
 * it was made, as the Video desk's stopwatch counts; a picture or a shot, which
 * goes one at a time after the one before it, from when it was sent, or, not
 * sent yet, from now. The section bar reads it once, when it opens the job, and
 * it opens a picture's or a shot's as that one comes up; timed from the press
 * that made the batch or the pass, every picture after the first would show
 * the whole batch's time as its own, and its time left from that.
 */
export function reportedOf(job: RunnerJob, progress?: RunnerProgress | null): Reported {
  const p = progress === undefined ? progressById.get(job.id) ?? null : progress
  const status = REPORTED[job.status] ?? 'submitting'
  return {
    key: job.id,
    status,
    promptId: job.promptId,
    label: job.label,
    prompt: job.prompt,
    value: p?.value ?? 0,
    max: p?.max ?? 0,
    pass: p?.pass ?? null,
    entryId: job.entryId,
    error: status === 'error' || status === 'cancelled' ? runnerFault(job).message || null : null,
    startedAt: job.desk === 'video' ? job.createdAt : (job.sentAt ?? Date.now()),
    stage: waitLine(job).stage,
  }
}

// What a page thrown away mid hand-over left in the tab goes again as it
// loads, once the server's state is read and says whether it already has it.
if (outbox.length) void refresh()
