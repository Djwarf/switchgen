/**
 * The press ledger.
 *
 * A module singleton that lives outside React, because a four-minute video
 * must not care that you walked over to the pictures desk. Every desk reports
 * its work here; the section bar reads it from wherever you happen to be.
 *
 * It also counts ComfyUI's own queue, so work started somewhere else — a
 * second tab, the ComfyUI page itself — is reported honestly rather than
 * pretended away. There is one graphics card and one queue.
 *
 * What happened to a job is its desk's to say, never the ledger's. The desk
 * follows the job to its end and decides, after asking the server about it
 * and reading /history, whether it finished, failed, was stopped or was lost.
 * A ledger that guessed an ending of its own, from a missing row or from a
 * stop it had only asked for, said Stopped or Lost over work that was then
 * filed.
 *
 * Desk integration is two lines:
 *
 *   const id = jobs.start({ desk: 'video', label: 'Wan 2.2 5B', prompt, kind: 'video' })
 *   const files = await run(workflow, jobs.handler(id, workflow))
 *   jobs.succeed(id)            // or jobs.fail(id, message)
 */
import { useSyncExternalStore } from 'react'
import {
  ComfyError,
  cancelJob,
  listJobs,
  type ApiWorkflow,
  type ProgressEvent,
  type ServerJobsPage,
} from '../../lib/comfy'
import type { DeskId } from '../../lib/session'
import { dismissNotice, postNotice } from './Notice'

// ---------------------------------------------------------------------------
// Stage names
// ---------------------------------------------------------------------------

const STAGES: ReadonlyArray<readonly [RegExp, string]> = [
  [/^(UnetLoaderGGUF|UNETLoader|CheckpointLoaderSimple|VAELoader|CLIPVisionLoader)$/, 'Loading the model'],
  [/^(CLIPLoader|DualCLIPLoader|CLIPTextEncode|TextEncodeQwenImage.*|CLIPSetLastLayer)$/, 'Reading the prompt'],
  [/^LoraLoader/, 'Loading add-ons'],
  [/^(VAEEncode|ImageScaleToTotalPixels|LoadImage|FluxKontextImageScale|ImageScale.*)$/, 'Preparing your picture'],
  [/^(Wan22ImageToVideoLatent|WanImageToVideo|Empty.*Latent.*)$/, 'Setting up the frames'],
  [/^(KSampler|KSamplerAdvanced|SamplerCustomAdvanced|CFGGuider)$/, 'Drawing'],
  [/^(VAEDecode|VAEDecodeTiled)$/, 'Developing the frames'],
  [/^SaveWEBM$/, 'Encoding the clip'],
  [/^(SaveImage|PreviewImage|SaveAnimated.*)$/, 'Writing the file'],
]

/** A human name for what the card is doing, from the node's class. */
export function stageFor(classType: string | null | undefined): string {
  if (!classType) return 'Working'
  for (const [re, name] of STAGES) if (re.test(classType)) return name
  return 'Working'
}

// ---------------------------------------------------------------------------
// The record
// ---------------------------------------------------------------------------

export type JobStatus = 'submitting' | 'queued' | 'running' | 'done' | 'error' | 'cancelled'

/**
 * Where a job came from. The two desks, plus the reel, which has no draft
 * store of its own and so is not a `DeskId`, but is its own room: the slug has
 * to send a reader there, not to the video desk, and stopping one of its shots
 * stops the reel.
 */
export type JobDesk = DeskId | 'reel'

export type Job = {
  /** Local id, assigned before anything is submitted. */
  id: string
  /** ComfyUI's prompt id, once the queue has accepted it. */
  promptId: string | null
  desk: JobDesk
  kind: 'image' | 'video'
  /** What made it: "Krea 2", "Wan 2.2 5B". */
  label: string
  /** The prompt, for the slug and the archive. */
  prompt: string
  status: JobStatus
  /** Sampler steps. `max` is 0 until the first progress message. */
  value: number
  max: number
  /** Dual-model families run two passes; the bar must not jump backwards. */
  pass: 1 | 2 | null
  stage: string
  previewUrl: string | null
  startedAt: number
  finishedAt: number | null
  error: string | null
  /**
   * A stop has been asked for and the job has not ended yet. It clears when
   * the desk reports the ending, or when ComfyUI says there was nothing left
   * to stop.
   */
  cancelling: boolean
  /** The archive record it produced, when the desk tells us. */
  entryId: string | null
}

export type JobInit = {
  desk: JobDesk
  kind: 'image' | 'video'
  label: string
  prompt: string
  /** Set when the job was submitted before it was reported here. */
  promptId?: string | null
  /** Total sampler steps, so the rule can start at a sensible width. */
  steps?: number
  /**
   * How to stop it, when stopping means more than cancelling one prompt. The
   * desk that owns the job does the stopping and reports the outcome back
   * through the usual calls; `cancel()` only marks the job as stopping.
   */
  stop?: () => void
}

/** What ComfyUI says about its own queue, including work we did not start. */
export type ServerQueue = {
  running: number
  pending: number
  /** Jobs on the server that are not ours. */
  foreign: number
  /** False until the first successful poll. */
  known: boolean
}

export type JobsSnapshot = {
  /** Newest first. */
  jobs: readonly Job[]
  /** Submitting, queued or running. */
  active: readonly Job[]
  running: Job | null
  queued: readonly Job[]
  /** Finished jobs, newest first. The slug shows one for a few seconds. */
  recent: readonly Job[]
  server: ServerQueue
}

const LIVE: readonly JobStatus[] = ['submitting', 'queued', 'running']
const isLive = (j: Job) => LIVE.includes(j.status)

/** How long a finished job keeps its place in the slug. */
export const RECENT_MS = 8000

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let ledger: Job[] = []
let server: ServerQueue = { running: 0, pending: 0, foreign: 0, known: false }
let snapshot: JobsSnapshot = build()
const listeners = new Set<() => void>()
/** Ledger job id → the owning desk's own stop, from `JobInit.stop`. */
const stoppers = new Map<string, () => void>()

function build(): JobsSnapshot {
  const active = ledger.filter(isLive)
  return {
    jobs: ledger,
    active,
    running: active.find((j) => j.status === 'running') ?? null,
    queued: active.filter((j) => j.status !== 'running'),
    recent: ledger.filter((j) => !isLive(j) && j.finishedAt !== null),
    server,
  }
}

function emit(): void {
  snapshot = build()
  for (const fn of [...listeners]) {
    try {
      fn()
    } catch {
      /* one bad subscriber must not strand the rest */
    }
  }
}

function patch(id: string, fn: (job: Job) => Job): Job | null {
  const i = ledger.findIndex((j) => j.id === id)
  if (i < 0) return null
  const next = fn(ledger[i])
  if (next === ledger[i]) return next
  ledger = [...ledger.slice(0, i), next, ...ledger.slice(i + 1)]
  emit()
  return next
}

function newId(): string {
  try {
    return crypto.randomUUID()
  } catch {
    return `job-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`
  }
}

// A finished job stays in the ledger so the desk can read it back, but the
// ledger is not an archive: the archive is. Keep the last fifty.
const KEEP = 50

// ---------------------------------------------------------------------------
// Server reconciliation
// ---------------------------------------------------------------------------

let timer = 0

const BUSY_MS = 4000
const IDLE_MS = 12000

/** How many jobs a filtered list holds in all, not just on the page it sent. */
const totalOf = (page: ServerJobsPage) => Math.max(page.pagination.total, page.jobs.length)

/**
 * Count ComfyUI's queue. Nothing here decides that a job of ours is lost: its
 * desk does, after a direct lookup and a read of /history, and reports it like
 * any other ending.
 */
async function poll(): Promise<void> {
  try {
    // Totals, not a page of rows. A page comes newest first, so behind a long
    // queue the job that is running is not on it at all, and counting rows
    // said nothing was running.
    const [inProgress, waiting] = await Promise.all([
      listJobs({ status: ['in_progress'], limit: 1 }),
      listJobs({ status: ['pending'], limit: 1 }),
    ])
    const running = totalOf(inProgress)
    const pending = totalOf(waiting)
    // Which of those are ours is the desks' word: every job they report as
    // live has been accepted by the queue and not yet ended.
    const ours = ledger.filter((j) => isLive(j) && j.promptId !== null).length
    const next: ServerQueue = {
      running,
      pending,
      foreign: Math.max(0, running + pending - ours),
      known: true,
    }
    const same =
      server.known &&
      server.running === next.running &&
      server.pending === next.pending &&
      server.foreign === next.foreign
    server = next
    if (!same) emit()
  } catch {
    // The offline banner owns this news; the ledger just stops guessing.
    if (server.known) {
      server = { ...server, known: false }
      emit()
    }
  }
}

function schedule(): void {
  clearTimeout(timer)
  if (listeners.size === 0) return
  const busy = ledger.some(isLive) || server.running > 0 || server.pending > 0
  timer = setTimeout(() => {
    void poll().finally(schedule)
  }, busy ? BUSY_MS : IDLE_MS) as unknown as number
}

// ---------------------------------------------------------------------------
// Public store
// ---------------------------------------------------------------------------

function subscribe(fn: () => void): () => void {
  const first = listeners.size === 0
  listeners.add(fn)
  if (first) {
    void poll().finally(schedule)
  }
  return () => {
    listeners.delete(fn)
    if (listeners.size === 0) clearTimeout(timer)
  }
}

/** Open a job. Returns its local id; nothing has been submitted yet. */
function start(init: JobInit): string {
  const job: Job = {
    id: newId(),
    promptId: init.promptId ?? null,
    desk: init.desk,
    kind: init.kind,
    label: init.label,
    prompt: init.prompt,
    status: init.promptId ? 'queued' : 'submitting',
    value: 0,
    max: init.steps ?? 0,
    pass: null,
    stage: 'Queued',
    previewUrl: null,
    startedAt: Date.now(),
    finishedAt: null,
    error: null,
    cancelling: false,
    entryId: null,
  }
  ledger = [job, ...ledger].slice(0, KEEP)
  if (init.stop) stoppers.set(job.id, init.stop)
  for (const id of stoppers.keys()) if (!ledger.some((j) => j.id === id)) stoppers.delete(id)
  emit()
  schedule()
  return job.id
}

/** The queue accepted it. If a stop was asked for first, send it now. */
function attach(id: string, promptId: string): void {
  const before = ledger.find((j) => j.id === id)
  if (!before) return
  const job = patch(id, (j) => ({
    ...j,
    promptId,
    status: isLive(j) ? (j.status === 'submitting' ? 'queued' : j.status) : j.status,
  }))
  // Only for the first id, and only when the ledger itself holds the stop: a
  // desk with a stop of its own was already told.
  if (job && !before.promptId && job.cancelling && isLive(job) && !stoppers.has(id)) {
    void sendCancel(id, promptId)
  }
  schedule()
}

/** Fold one ComfyUI progress event into the job. */
function apply(id: string, e: ProgressEvent, graph?: ApiWorkflow): void {
  switch (e.phase) {
    case 'queued':
      attach(id, e.promptId)
      patch(id, (j) => (isLive(j) ? { ...j, status: 'queued', stage: 'Queued' } : j))
      return
    case 'running': {
      const node = e.node ? graph?.[e.node] : undefined
      const cls = node?.class_type ?? null
      let pass: 1 | 2 | null = null
      if (cls === 'KSamplerAdvanced' && node) {
        pass = node.inputs.return_with_leftover_noise === 'enable' ? 1 : 2
      }
      const stage = pass ? `Drawing, ${pass === 1 ? 'first' : 'second'} pass` : stageFor(cls)
      // A report that arrives after the ending must not bring the job back:
      // that re-enabled Stop on finished work and froze its clock.
      patch(id, (j) =>
        isLive(j)
          ? {
              ...j,
              status: 'running',
              value: e.value,
              max: e.max || j.max,
              pass: pass ?? j.pass,
              stage,
            }
          : j,
      )
      return
    }
    case 'preview':
      patch(id, (j) => (isLive(j) ? { ...j, previewUrl: e.url } : j))
      return
    case 'done':
      patch(id, (j) =>
        isLive(j)
          ? { ...j, status: 'done', stage: 'Done', finishedAt: Date.now(), value: j.max, cancelling: false }
          : j,
      )
      schedule()
      return
    case 'error':
      patch(id, (j) =>
        isLive(j)
          ? {
              ...j,
              status: e.cancelled ? 'cancelled' : 'error',
              stage: e.cancelled ? 'Stopped' : 'Stopped short',
              finishedAt: Date.now(),
              error: e.cancelled ? null : e.message,
              cancelling: false,
            }
          : j,
      )
      schedule()
      return
  }
}

/**
 * A `ProgressEvent` handler bound to one job, for `run(workflow, handler)`.
 * Pass the instantiated graph and the stage names become specific.
 */
function handler(id: string, graph?: ApiWorkflow): (e: ProgressEvent) => void {
  return (e) => apply(id, e, graph)
}

/**
 * When an ending was reached. The same verdict reported twice keeps its first
 * time; a different one is news, and the slug shows news for a few seconds
 * from this moment.
 */
function endedAt(j: Job, status: JobStatus): number {
  return j.status === status && j.finishedAt !== null ? j.finishedAt : Date.now()
}

/** The desk finished writing the record. */
function succeed(id: string, opts: { entryId?: string } = {}): void {
  patch(id, (j) => ({
    ...j,
    status: 'done',
    stage: 'Done',
    finishedAt: endedAt(j, 'done'),
    value: j.max,
    error: null,
    cancelling: false,
    entryId: opts.entryId ?? j.entryId,
  }))
  schedule()
}

function fail(id: string, message: string, opts: { cancelled?: boolean } = {}): void {
  const status: JobStatus = opts.cancelled ? 'cancelled' : 'error'
  patch(id, (j) => ({
    ...j,
    status,
    stage: opts.cancelled ? 'Stopped' : 'Stopped short',
    finishedAt: endedAt(j, status),
    error: opts.cancelled ? null : message,
    cancelling: false,
  }))
  schedule()
}

/** Give up asking: the stop was refused, or there was nothing left to stop. */
function stopAsking(id: string): void {
  patch(id, (j) => (j.cancelling ? { ...j, cancelling: false } : j))
}

/**
 * Ask ComfyUI to stop one prompt, and leave the ending to the job's desk.
 *
 * Asking is not stopping. A running job carries on until ComfyUI next checks
 * for the interrupt, and one on its last node can finish and save first, so
 * the desk reports whatever really happened: stopped, finished or failed.
 */
async function sendCancel(id: string, promptId: string): Promise<void> {
  const notice = `stop-${id}`
  let stopped: boolean
  try {
    stopped = await cancelJob(promptId)
  } catch (err) {
    stopAsking(id)
    const reason = err instanceof ComfyError ? err.message : 'We could not reach ComfyUI.'
    postNotice({
      key: notice,
      tone: 'error',
      title: 'Could not stop that job',
      body: `${reason} It may still be running. Try again in a moment.`,
    })
    return
  }
  // A refusal from an earlier try is no longer news once one gets through.
  dismissNotice(notice)
  // False means it had already ended, and its desk is about to say how.
  if (!stopped) stopAsking(id)
}

/**
 * Stop one job and only that job.
 *
 * `cancelJob` is prompt-targeted, so a picture stopped behind a four-minute
 * clip cannot take the clip with it. The job is marked as stopping and ends
 * when its desk reports the ending.
 */
async function cancel(id: string): Promise<void> {
  const job = ledger.find((j) => j.id === id)
  if (!job || !isLive(job) || job.cancelling) return
  patch(id, (j) => ({ ...j, cancelling: true }))
  const stop = stoppers.get(id)
  if (stop) {
    try {
      stop()
    } catch {
      stopAsking(id)
    }
    return
  }
  // Nothing was submitted yet; `attach` sends the stop the moment it is.
  if (!job.promptId) return
  await sendCancel(id, job.promptId)
}

/** Take a finished job off the bar without touching the archive. */
function dismiss(id: string): void {
  stoppers.delete(id)
  const before = ledger.length
  ledger = ledger.filter((j) => j.id !== id)
  if (ledger.length !== before) emit()
}

function get(id: string): Job | null {
  return ledger.find((j) => j.id === id) ?? null
}

function clearFinished(): void {
  const before = ledger.length
  ledger = ledger.filter(isLive)
  for (const id of stoppers.keys()) if (!ledger.some((j) => j.id === id)) stoppers.delete(id)
  if (ledger.length !== before) emit()
}

export const jobs = {
  subscribe,
  snapshot: () => snapshot,
  start,
  attach,
  handler,
  apply,
  succeed,
  fail,
  cancel,
  dismiss,
  get,
  clearFinished,
}

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------

/** The whole ledger. Re-renders on every change, from any desk. */
export function useJobs(): JobsSnapshot {
  return useSyncExternalStore(subscribe, jobs.snapshot, jobs.snapshot)
}

/** One job, or null once it has been dismissed. */
export function useJob(id: string | null): Job | null {
  const snap = useJobs()
  if (!id) return null
  return snap.jobs.find((j) => j.id === id) ?? null
}

// ---------------------------------------------------------------------------
// Readouts
// ---------------------------------------------------------------------------

/** Fraction done, 0–1, or null when the job has not said yet. */
export function progressOf(job: Job): number | null {
  if (!job.max) return null
  const within = Math.min(1, job.value / job.max)
  if (job.pass === null) return within
  // Two passes, reported as one rule that only ever moves forward.
  return (job.pass - 1 + within) / 2
}

/** `2:04`, `11.4 s`. Plain, tabular, never a spinner's worth of precision. */
export function elapsedText(ms: number): string {
  if (ms < 1000) return `${ms} ms`
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s`
  const total = Math.round(ms / 1000)
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`
}

/** `about 2 min 10 s`. Only ever from a measurement. */
export function roughText(ms: number): string {
  const total = Math.round(ms / 1000)
  if (total < 60) return `about ${Math.max(5, Math.round(total / 5) * 5)} s`
  const m = Math.floor(total / 60)
  const s = Math.round((total % 60) / 10) * 10
  return s ? `about ${m} min ${s} s` : `about ${m} min`
}

/**
 * What is left, measured from this run's own rate — never invented.
 * Null until there is enough of a sample to mean anything.
 */
export function remainingOf(job: Job, now = Date.now()): number | null {
  if (job.status !== 'running' || !job.max || job.value < 3) return null
  const done = progressOf(job)
  if (done === null || done <= 0.02 || done >= 1) return null
  const spent = now - job.startedAt
  const total = spent / done
  const left = total - spent
  return left > 2000 ? left : null
}

/**
 * The one job the section bar should be talking about: whatever is running,
 * else whatever is waiting, else the last one to finish — but only while it is
 * still news.
 */
export function headline(snap: JobsSnapshot, now = Date.now()): Job | null {
  if (snap.running) return snap.running
  if (snap.queued.length) return snap.queued[0]
  const last = snap.recent[0]
  if (last && last.finishedAt !== null && now - last.finishedAt < RECENT_MS) return last
  return null
}
