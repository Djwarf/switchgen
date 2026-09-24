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
  getJob,
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
 * Which sampling pass a job's steps belong to, each counted from one. The Wan
 * 2.2 14B pairs sample in two KSamplerAdvanced passes, one per model, and
 * ComfyUI counts each pass's steps from one again.
 */
export type SamplingPass = { index: number; count: number }

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
  pass: SamplingPass | null
  /**
   * When this pass was first seen, and at which step, so an estimate uses
   * the current pass's own pace rather than a rate that counts model loading
   * and the other pass. Null while there is no pass.
   */
  passAt: number | null
  passFrom: number
  /**
   * What the card is doing, in words. For a job not yet sent, what its desk
   * says it is doing (see JobInit.stage).
   */
  stage: string
  previewUrl: string | null
  startedAt: number
  finishedAt: number | null
  error: string | null
  /**
   * A stop has been asked for and the job has not ended yet. It clears when
   * the desk reports the ending, when ComfyUI says there was nothing left to
   * stop, or when the job shows the stop did not take (see STOP_WAIT_MS).
   */
  cancelling: boolean
  /**
   * Why the job cannot be stopped from here just now, in the words the slug
   * shows in Stop's place, or null when it can be. The queue on the server
   * takes no stop while it is off, so the slug offers none for its work, as
   * the desks do not, and gives the server's reason.
   */
  noStop: string | null
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
   * When the desk started the job, by the desk's own clock. A job taken up
   * again after a reload has been going since before this page opened, and a
   * clock started when the ledger first heard of it said a clip ten minutes
   * in had just begun. Now, when it is left out.
   */
  startedAt?: number
  /**
   * What the desk says a job it has not sent yet is doing: held behind a lost
   * clip, waiting for ComfyUI to come back, waiting its turn. Left out, the
   * job is said to be on its way over.
   */
  stage?: string
  /**
   * How to stop it, when stopping means more than cancelling one prompt. The
   * desk that owns the job does the stopping and reports the outcome back
   * through the usual calls; `cancel()` only marks the job as stopping.
   *
   * A stop that has to travel (a word to the queue on the server) may say
   * whether it got there. False means it did not, so no ending is coming of
   * it: the stopping mark comes off and the reader is told, so Stop can be
   * held again. Nothing else about the ending is read from it.
   */
  stop?: Stopper
}

/** A desk's own stop. See `JobInit.stop`. */
export type Stopper = () => void | Promise<boolean>

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

/** What a job not yet on ComfyUI's queue is doing, when its desk says nothing more. */
export const SENDING = 'Sending it over'

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let ledger: Job[] = []
let server: ServerQueue = { running: 0, pending: 0, foreign: 0, known: false }
let snapshot: JobsSnapshot = build()
const listeners = new Set<() => void>()
/** Ledger job id → the owning desk's own stop, from `JobInit.stop`. */
const stoppers = new Map<string, Stopper>()
/** Ledger job id → when its stop was asked for, and the check that follows it up. */
const asked = new Map<string, Ask>()

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
    passAt: null,
    passFrom: 0,
    stage: init.promptId ? 'Queued' : init.stage || SENDING,
    previewUrl: null,
    startedAt: init.startedAt ?? Date.now(),
    finishedAt: null,
    error: null,
    cancelling: false,
    noStop: null,
    entryId: null,
  }
  ledger = [job, ...ledger].slice(0, KEEP)
  if (init.stop) stoppers.set(job.id, init.stop)
  forgetGone()
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
    // The stop is only now on its way, so its wait starts now.
    askedFor(id)
    void sendCancel(id, promptId)
  }
  schedule()
}

/**
 * The desk's word for what a job it has not sent yet is doing. Only such a
 * job's: once the queue has it, the ledger's own stages take over.
 */
function setStage(id: string, stage: string): void {
  if (!stage) return
  patch(id, (j) => (j.status === 'submitting' && j.stage !== stage ? { ...j, stage } : j))
}

/**
 * The desk's word on whether the job can be stopped from here just now (see
 * Job.noStop). A notice from a stop that did not get through, saying to hold
 * Stop again, goes once there is no Stop to hold: the slug says why instead.
 */
function setNoStop(id: string, why: string | null): void {
  const job = get(id)
  if (!job || job.noStop === why) return
  patch(id, (j) => ({ ...j, noStop: why }))
  if (why) dismissNotice(stopNotice(id))
}

/** The pass a sampler node runs, in a graph that samples in two. Null for any other node. */
function passOf(graph: ApiWorkflow | undefined, node: string | null): SamplingPass | null {
  const n = node ? graph?.[node] : undefined
  if (!n || n.class_type !== 'KSamplerAdvanced' || !graph) return null
  const passes = Object.values(graph).filter((x) => x.class_type === 'KSamplerAdvanced').length
  if (passes < 2) return null
  return { index: n.inputs.return_with_leftover_noise === 'enable' ? 1 : 2, count: 2 }
}

const samePass = (a: SamplingPass | null, b: SamplingPass | null) =>
  a === b || (a !== null && b !== null && a.index === b.index && a.count === b.count)

/**
 * What a desk knows beside the event: the graph, so stages get their node's
 * name, and the sampling pass, when the desk has worked it out itself. A pass
 * left out is kept as it was; null says the steps belong to no pass.
 */
export type ApplyContext = { graph?: ApiWorkflow; pass?: SamplingPass | null }

/** Fold one ComfyUI progress event into the job. */
function apply(id: string, e: ProgressEvent, context: ApplyContext = {}): void {
  const { graph } = context
  switch (e.phase) {
    case 'queued':
      attach(id, e.promptId)
      patch(id, (j) => (isLive(j) ? { ...j, status: 'queued', stage: 'Queued' } : j))
      return
    case 'running': {
      const node = e.node ? graph?.[e.node] : undefined
      const cls = node?.class_type ?? null
      const found = passOf(graph, e.node)
      const told = context.pass !== undefined
      // A report that arrives after the ending must not bring the job back:
      // that re-enabled Stop on finished work and froze its clock.
      patch(id, (j) => {
        if (!isLive(j)) return j
        const pass = told ? (context.pass ?? null) : (found ?? j.pass)
        // A node starting reports 0 of 1; only a sampler's steps are drawing.
        const stepping = e.max > 1
        // The pace is timed from the first step seen in this pass, not from
        // the node starting: the pass's model loads in between.
        const changed = pass !== null && !samePass(pass, j.pass)
        const startsNow = stepping && (changed || j.passAt === null)
        return {
          ...j,
          status: 'running',
          value: e.value,
          max: e.max || j.max,
          pass,
          passAt: pass === null ? null : startsNow ? Date.now() : changed ? null : j.passAt,
          passFrom: pass === null ? 0 : startsNow ? e.value : changed ? 0 : j.passFrom,
          stage:
            pass && stepping && (found !== null || told)
              ? `Drawing, pass ${pass.index} of ${pass.count}`
              : stageFor(cls),
        }
      })
      // ComfyUI checks for a stop before it reports a step, so steps still
      // coming well after one was asked for mean it did not take. A node
      // starting (value 0) proves nothing: ComfyUI announces the next node
      // before it notices the stop there and ends the job.
      if (e.value > 0) stepAfterStop(id)
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
      ended(id)
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
      ended(id)
      schedule()
      return
  }
}

/**
 * A `ProgressEvent` handler bound to one job, for `run(workflow, handler)`.
 * Pass the instantiated graph and the stage names become specific.
 */
function handler(id: string, graph?: ApiWorkflow): (e: ProgressEvent) => void {
  return (e) => apply(id, e, { graph })
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
  ended(id)
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
  ended(id)
  schedule()
}

/** Give up asking: the stop was refused, or there was nothing left to stop. */
function stopAsking(id: string): void {
  forgetAsk(id)
  patch(id, (j) => (j.cancelling ? { ...j, cancelling: false } : j))
}

/**
 * How long a stop is given before the ledger stops vouching for it.
 *
 * A stop can be accepted and still not take: ComfyUI clears its stop flag as
 * each prompt starts, so one that lands in that moment is lost, and a desk's
 * own cancel can fail without a word. Left marked as stopping, such a job kept
 * Stop out of reach for the rest of its run. So once this long has passed, the
 * mark comes off as soon as the job shows the stop did not take: a step
 * reported, or ComfyUI still listing it as waiting, since a stop takes a
 * waiting job off the queue at once. A running job that shows neither is
 * still being stopped: ComfyUI notices a stop at the next step or node, and
 * one long step or a slow decode can take a while to get there.
 */
const STOP_WAIT_MS = 15_000

type Ask = { at: number; timer: number }

/** The stop's own notice, which says the stop failed or did not take. */
const stopNotice = (id: string) => `stop-${id}`

/** Start (or start again) the wait on a stop that has just been asked for. */
function askedFor(id: string): void {
  forgetAsk(id)
  const ask: Ask = { at: Date.now(), timer: 0 }
  ask.timer = setTimeout(() => void stillWaiting(id, ask), STOP_WAIT_MS) as unknown as number
  asked.set(id, ask)
}

function forgetAsk(id: string): void {
  const ask = asked.get(id)
  if (!ask) return
  clearTimeout(ask.timer)
  asked.delete(id)
}

/** The job, while this stop of it is still the one being waited on. */
function stillStopping(id: string, ask: Ask): Job | null {
  const job = ledger.find((j) => j.id === id)
  return asked.get(id) === ask && job && job.cancelling && isLive(job) ? job : null
}

/**
 * The wait is over: a job ComfyUI still lists as waiting was not stopped. A
 * job still being sent carries its stop with it, and whether a running one
 * is still going is for its steps to say.
 */
async function stillWaiting(id: string, ask: Ask): Promise<void> {
  const job = stillStopping(id, ask)
  if (!job?.promptId) return
  let server: Awaited<ReturnType<typeof getJob>>
  try {
    server = await getJob(job.promptId)
  } catch {
    return // no answer is no evidence either way
  }
  if (server?.status === 'pending' && stillStopping(id, ask)) notTaken(id, 'waiting')
}

/** A step reported after the wait: the stop did not take. */
function stepAfterStop(id: string): void {
  const ask = asked.get(id)
  if (!ask || !stillStopping(id, ask) || Date.now() - ask.at < STOP_WAIT_MS) return
  notTaken(id, 'running')
}

/**
 * A desk's stop said it did not get through: the queue on the server was not
 * reached, or did not take it. A job waiting there has no prompt id to check
 * and sends no steps, so nothing else would ever take the stopping mark off,
 * and a job held for hours kept Stop out of reach for all of them.
 */
function notReached(id: string): void {
  stopAsking(id)
  // No Stop to hold again: the slug says why in its place.
  if (get(id)?.noStop) return
  postNotice({
    key: stopNotice(id),
    tone: 'error',
    title: 'Could not stop that job',
    body: 'The stop did not get through, so the job may still be going. Hold Stop again to ask once more.',
  })
}

/** Take the stopping mark off, and say why, so Stop can be held again. */
function notTaken(id: string, where: 'waiting' | 'running'): void {
  stopAsking(id)
  if (get(id)?.noStop) return
  postNotice({
    key: stopNotice(id),
    tone: 'warning',
    title: 'That job has not stopped yet',
    body:
      where === 'waiting'
        ? 'ComfyUI still has it waiting in its queue. Hold Stop again to ask once more.'
        : 'ComfyUI is still working on it. Hold Stop again to ask once more.',
  })
}

/**
 * A job has ended, or left the ledger. Its stop needs no more following up,
 * and a notice about that stop, saying it may still be running and to try
 * again, is no longer true.
 */
function ended(id: string): void {
  forgetAsk(id)
  dismissNotice(stopNotice(id))
}

/** Let go of everything kept for jobs no longer on the ledger. */
function forgetGone(): void {
  const kept = new Set(ledger.map((j) => j.id))
  for (const id of stoppers.keys()) if (!kept.has(id)) stoppers.delete(id)
  for (const id of asked.keys()) if (!kept.has(id)) ended(id)
}

/**
 * Ask ComfyUI to stop one prompt, and leave the ending to the job's desk.
 *
 * Asking is not stopping. A running job carries on until ComfyUI next checks
 * for the interrupt, and one on its last node can finish and save first, so
 * the desk reports whatever really happened: stopped, finished or failed.
 */
async function sendCancel(id: string, promptId: string): Promise<void> {
  const notice = stopNotice(id)
  let stopped: boolean
  try {
    stopped = await cancelJob(promptId)
  } catch (err) {
    stopAsking(id)
    // ComfyUI not answering is not ComfyUI declining: it is likely restarting,
    // which ends the job anyway, so "it may still be running" would mislead.
    const body =
      err instanceof ComfyError && err.unreachable
        ? `${err.message} If it comes back with this job still going, hold Stop again.`
        : `${err instanceof ComfyError ? err.message : 'We could not reach ComfyUI.'} It may still be running. Try again in a moment.`
    postNotice({ key: notice, tone: 'error', title: 'Could not stop that job', body })
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
  // A job with no Stop offered is not stopped from here either (see Job.noStop).
  if (!job || !isLive(job) || job.cancelling || job.noStop) return
  patch(id, (j) => ({ ...j, cancelling: true }))
  askedFor(id)
  const ask = asked.get(id)!
  const stop = stoppers.get(id)
  if (stop) {
    let told: void | Promise<boolean>
    try {
      told = stop()
    } catch {
      stopAsking(id)
      return
    }
    if (!told) return
    let reached: boolean
    try {
      reached = (await told) !== false
    } catch {
      reached = false
    }
    // Only for this stop, and only while the job still waits on it: a job that
    // ended meanwhile, or a stop asked for again, has news of its own.
    if (!reached && stillStopping(id, ask)) notReached(id)
    // A stop that got through clears an earlier one's refusal.
    else if (reached && asked.get(id) === ask) dismissNotice(stopNotice(id))
    return
  }
  // Nothing was submitted yet; `attach` sends the stop the moment it is.
  if (!job.promptId) return
  await sendCancel(id, job.promptId)
}

/** Take a finished job off the bar without touching the archive. */
function dismiss(id: string): void {
  stoppers.delete(id)
  ended(id)
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
  forgetGone()
  if (ledger.length !== before) emit()
}

export const jobs = {
  subscribe,
  snapshot: () => snapshot,
  start,
  attach,
  setStage,
  setNoStop,
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
export function progressOf(job: Pick<Job, 'value' | 'max' | 'pass'>): number | null {
  if (!job.max) return null
  const within = Math.min(1, Math.max(0, job.value / job.max))
  const pass = job.pass
  if (!pass || pass.count < 2) return within
  // Each pass counts its steps from one again, so step 5 of 10 is a quarter
  // of the way in the first pass and three quarters in the second.
  const index = Math.min(Math.max(pass.index, 1), pass.count)
  return Math.min(1, (index - 1 + within) / pass.count)
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
 *
 * A job that samples in passes is timed by the pass it is in: its steps so
 * far in this pass, over the time since the pass was first seen, give the
 * pace, and the steps left are this pass's and one pass's worth for each pass
 * still to come. Timed from the job's start instead, the rate counted model
 * loading and the first pass's load against the second, and the estimate
 * swung each time a pass began. It counts sampling only; the decode after it
 * is not included, which is what "at this rate" says.
 */
export function remainingOf(
  job: Pick<Job, 'status' | 'value' | 'max' | 'pass' | 'passAt' | 'passFrom' | 'startedAt'>,
  now = Date.now(),
): number | null {
  if (job.status !== 'running' || !job.max) return null
  const pass = job.pass
  if (pass && pass.count >= 2) {
    if (job.passAt === null || job.max <= 1) return null
    const stepsSeen = job.value - job.passFrom
    if (stepsSeen < 2) return null
    const perStep = (now - job.passAt) / stepsSeen
    const index = Math.min(Math.max(pass.index, 1), pass.count)
    const stepsLeft = Math.max(0, job.max - job.value) + (pass.count - index) * job.max
    const left = perStep * stepsLeft
    return left > 2000 ? left : null
  }
  if (job.value < 3) return null
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

/**
 * When the finished job the slug shows stops being news, or null when there
 * is no such job: something is still live, or nothing finished recently.
 */
export function newsUntil(snap: Pick<JobsSnapshot, 'active' | 'recent'>): number | null {
  if (snap.active.length) return null
  const last = snap.recent[0]
  return last?.finishedAt != null ? last.finishedAt + RECENT_MS : null
}

/**
 * Whether the slug's clock has anything to count: a live job, or a finished
 * one still shown as news. Every finished job the ledger keeps (up to fifty)
 * used to count, so after the first job of a session the clock ticked twice a
 * second for as long as the page was open, with nothing on screen changing.
 */
export function needsClock(snap: Pick<JobsSnapshot, 'active' | 'recent'>, now = Date.now()): boolean {
  if (snap.active.length) return true
  const until = newsUntil(snap)
  return until !== null && now < until
}
