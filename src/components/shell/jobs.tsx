/**
 * The press ledger.
 *
 * A module singleton that lives outside React, because a four-minute video
 * must not care that you walked over to the pictures desk. Every desk reports
 * its work here; the section bar reads it from wherever you happen to be.
 *
 * It also watches ComfyUI's own queue, so work started somewhere else — a
 * second tab, the ComfyUI page itself — is reported honestly rather than
 * pretended away. There is one graphics card and one queue.
 *
 * Desk integration is two lines:
 *
 *   const id = jobs.start({ desk: 'video', label: 'Wan 2.2 5B', prompt, kind: 'video' })
 *   const files = await run(workflow, jobs.handler(id, workflow))
 *   jobs.succeed(id)            // or jobs.fail(id, message)
 */
import { useSyncExternalStore } from 'react'
import {
  cancelJob,
  listJobs,
  watch,
  type ApiWorkflow,
  type ProgressEvent,
  type ServerJob,
} from '../../lib/comfy'
import type { DeskId } from '../../lib/session'

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

export type Job = {
  /** Local id, assigned before anything is submitted. */
  id: string
  /** ComfyUI's prompt id, once the queue has accepted it. */
  promptId: string | null
  desk: DeskId
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
  /** A stop has been asked for but the server has not confirmed it yet. */
  cancelling: boolean
  /** The archive record it produced, when the desk tells us. */
  entryId: string | null
}

export type JobInit = {
  desk: DeskId
  kind: 'image' | 'video'
  label: string
  prompt: string
  /** Set when the job was submitted before it was reported here. */
  promptId?: string | null
  /** Total sampler steps, so the rule can start at a sensible width. */
  steps?: number
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
const watchers = new Map<string, () => void>()

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
let misses = new Map<string, number>()

const BUSY_MS = 4000
const IDLE_MS = 12000

async function poll(): Promise<void> {
  try {
    const page = await listJobs({ status: ['pending', 'in_progress'], limit: 50 })
    const ours = new Set(ledger.filter(isLive).map((j) => j.promptId).filter(Boolean) as string[])
    let running = 0
    let pending = 0
    let foreign = 0
    const live = new Set<string>()
    for (const j of page.jobs as ServerJob[]) {
      live.add(j.id)
      if (j.status === 'in_progress') running += 1
      else pending += 1
      if (!ours.has(j.id)) foreign += 1
    }
    server = { running, pending, foreign, known: true }

    // A job the server has never heard of, twice running, is lost. Say so
    // rather than spinning a rule at somebody for an hour.
    for (const job of ledger) {
      if (!isLive(job) || !job.promptId) continue
      if (live.has(job.promptId)) {
        misses.delete(job.id)
        continue
      }
      const n = (misses.get(job.id) ?? 0) + 1
      misses.set(job.id, n)
      if (n >= 2) {
        misses.delete(job.id)
        patch(job.id, (j) =>
          isLive(j)
            ? {
                ...j,
                status: 'error',
                finishedAt: Date.now(),
                error:
                  'We lost track of this job. ComfyUI no longer lists it. Check the archive. It may have finished anyway.',
              }
            : j,
        )
      }
    }
    emit()
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
  emit()
  schedule()
  return job.id
}

/** The queue accepted it. If a stop was asked for first, honour it now. */
function attach(id: string, promptId: string): void {
  const job = patch(id, (j) => ({
    ...j,
    promptId,
    status: isLive(j) ? (j.status === 'submitting' ? 'queued' : j.status) : j.status,
  }))
  if (job && job.status === 'cancelled') void cancelJob(promptId).catch(() => {})
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
      patch(id, (j) => ({
        ...j,
        status: 'running',
        value: e.value,
        max: e.max || j.max,
        pass: pass ?? j.pass,
        stage,
      }))
      return
    }
    case 'preview':
      patch(id, (j) => ({ ...j, previewUrl: e.url }))
      return
    case 'done':
      patch(id, (j) =>
        isLive(j) ? { ...j, status: 'done', stage: 'Done', finishedAt: Date.now(), value: j.max } : j,
      )
      release(id)
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
            }
          : j,
      )
      release(id)
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

/** Follow a prompt that was submitted elsewhere. Returns an unsubscribe. */
function follow(id: string, promptId: string, graph?: ApiWorkflow): () => void {
  attach(id, promptId)
  release(id)
  const stop = watch(promptId, (e) => apply(id, e, graph))
  watchers.set(id, stop)
  return () => release(id)
}

function release(id: string): void {
  const stop = watchers.get(id)
  if (stop) {
    watchers.delete(id)
    try {
      stop()
    } catch {
      /* ignore */
    }
  }
}

/** The desk finished writing the record. */
function succeed(id: string, opts: { entryId?: string } = {}): void {
  patch(id, (j) => ({
    ...j,
    status: 'done',
    stage: 'Done',
    finishedAt: j.finishedAt ?? Date.now(),
    value: j.max,
    entryId: opts.entryId ?? j.entryId,
  }))
  release(id)
  schedule()
}

function fail(id: string, message: string, opts: { cancelled?: boolean } = {}): void {
  patch(id, (j) => ({
    ...j,
    status: opts.cancelled ? 'cancelled' : 'error',
    stage: opts.cancelled ? 'Stopped' : 'Stopped short',
    finishedAt: j.finishedAt ?? Date.now(),
    error: opts.cancelled ? null : message,
  }))
  release(id)
  schedule()
}

/**
 * Stop one job and only that job.
 *
 * `cancelJob` is prompt-targeted, so a picture stopped behind a four-minute
 * clip cannot take the clip with it.
 */
async function cancel(id: string): Promise<void> {
  const job = ledger.find((j) => j.id === id)
  if (!job || !isLive(job)) return
  if (!job.promptId) {
    // Nothing was submitted yet; `attach` will cancel it the moment it is.
    patch(id, (j) => ({ ...j, status: 'cancelled', stage: 'Stopped', finishedAt: Date.now() }))
    return
  }
  patch(id, (j) => ({ ...j, cancelling: true, stage: 'Stopping' }))
  try {
    await cancelJob(job.promptId)
  } catch {
    /* a cancel that did not land is reported by the next poll */
  }
  patch(id, (j) =>
    isLive(j) ? { ...j, status: 'cancelled', stage: 'Stopped', cancelling: false, finishedAt: Date.now() } : j,
  )
  release(id)
  schedule()
}

/** Take a finished job off the bar without touching the archive. */
function dismiss(id: string): void {
  release(id)
  misses.delete(id)
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
  misses = new Map()
  if (ledger.length !== before) emit()
}

export const jobs = {
  subscribe,
  snapshot: () => snapshot,
  start,
  attach,
  handler,
  follow,
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
