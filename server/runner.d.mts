import type { Plugin } from 'vite'
import type { ArchiveApi } from './archive.mjs'
import type { Comfy } from './runner/comfy.mjs'
import type { ApiWorkflow, OutputFile } from './runner/comfyRecord.mjs'

/** A desk that can hand its waiting work to the queue on the server. */
export type RunnerDesk = 'video' | 'images' | 'reel'

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

/** What a waiting job waits for, or null. */
export type Wait = { for: 'turn' | 'before' | 'heavy' | 'queue' | 'comfy' | 'held' | 'disk'; ahead?: number } | null

export type RunnerError = {
  code: 'refused' | 'failed' | 'lost' | 'unsent' | 'ended-unsent' | 'no-file' | 'no-frame' | 'stopped' | 'skipped' | 'internal'
  message: string | null
  node: string | null
  nodeType: string | null
  nodeErrors: Record<string, unknown> | null
  mayExist: boolean
  sent: boolean
  /** For a skipped job: the job whose ending ended its group, and that job's place in it (from 1). */
  after: { jobId: string; index: number } | null
}

export type Lane = {
  /**
   * Why work is held. Scope 'heavy': a heavy job was lost ('lost') or may
   * never have reached ComfyUI ('unsent'), and every heavy job waits. Scope
   * 'all': the machine restarted ('restart'), or the queue was off or stood
   * back while work waited ('paused'), and the jobs made no later than
   * `since` wait; work made after it goes.
   *
   * `since` names the hold: the lane word gives it back, and a word naming
   * another is refused.
   */
  held: null | {
    why: 'lost' | 'unsent' | 'restart' | 'paused'
    scope: 'heavy' | 'all'
    /** The heavy job whose loss holds the lane; null for a hold on all the work that no loss was added to. */
    jobId: string | null
    since: number
    /**
     * Scope 'all' only: when the heavy job `jobId` was lost or went unsent
     * while this hold stood. From then on every heavy job waits too, as
     * after any lost clip, while light work made after `since` still goes,
     * and `since` stays as it was. Once none of the work made by `since`
     * waits any more, the hold becomes the one for that loss:
     * {why: 'lost' | 'unsent', scope: 'heavy', jobId, since: heavyAfter}.
     */
    heavyAfter?: number | null
  }
}

export type JobView = {
  id: string
  groupId: string
  desk: RunnerDesk
  kind: 'image' | 'video'
  /** Order of intake across the whole queue. */
  seq: number
  /** Place in its group, from 1. */
  index: number
  total: number
  label: string
  prompt: string
  device: string
  heavy: boolean
  status: RunnerStatus
  wait: Wait
  stopRequested: boolean
  stopLanded: boolean
  /** Null until ComfyUI has the job queued. */
  promptId: string | null
  attempt: number
  createdAt: number
  sentAt: number | null
  /** ComfyUI's execution_start. */
  ranAt: number | null
  /** ComfyUI's end. */
  finishedAt: number | null
  endedAt: number | null
  files: OutputFile[]
  primary: OutputFile | null
  frame: OutputFile | null
  openedOn: string | null
  entryId: string | null
  entryNo: number | null
  repeatOf: string | null
  /** 0 = not measured. */
  durationMs: number
  error: RunnerError | null
  meta: Record<string, unknown> | null
  dismissed: boolean
}

export type GroupView = {
  id: string
  desk: RunnerDesk
  kind: 'batch' | 'clips' | 'pass'
  label: string
  device: string
  createdAt: number
  state: 'active' | 'ended'
  endedBy: { jobId: string; why: 'failed' | 'refused' | 'stopped' | 'lost' | 'unsent' | 'no-frame' } | null
  endedAt: number | null
  jobIds: string[]
  dismissed: boolean
}

export type Progress = {
  id: string
  value: number
  max: number
  node: string | null
  classType: string | null
  pass: { index: number; count: number } | null
  at: number
  previewN: number
}

/**
 * While the queue is not running (available false), the saved list read from
 * disk: its jobs and groups as they stand, a waiting job's wait null and the
 * lane never held, since nothing can be sent, stopped or let go until it runs.
 */
export type Snapshot = {
  v: 1
  available: boolean
  reason: string | null
  /** Random per runner instance: a new one means read everything again. */
  boot: string
  rev: number
  comfy: { answering: boolean | null; since: number }
  lane: Lane
  groups: GroupView[]
  jobs: JobView[]
  progress: Record<string, Progress>
}

/** A record as the page builds it, less the six fields the finished run decides. */
export type RecordTemplate = Record<string, unknown> & { desk: string; mode: string }

export type Status = { active: boolean; desks: RunnerDesk[]; reason: string | null }

export type RunnerOptions = {
  /** The folder for state.json, jobs/<id>.json and its lock; one queue runs from it at a time. */
  dir: string
  comfy: Comfy
  archive: ArchiveApi
  /** The outputs root, where a chained shot's frame must be found. */
  outputs: string
  desks?: RunnerDesk[]
  /** False keeps it off; SWITCHGEN_RUNNER=off keeps it off whatever this says. */
  enabled?: boolean
  settleMs?: number
  now?: () => number
  sleep?: (ms: number) => Promise<void>
  machineBoot?: () => string | null
  /** False: no timers, no early passes; step it with tick(). Default true. */
  autoTick?: boolean
  /** How long a stop from a page waits on ComfyUI's answer to its cancel before it answers. 3 s unless a test shortens it. */
  stopWaitMs?: number
}

export type Runner = {
  handler(req: unknown, res: unknown, next: () => void): Promise<unknown>
  /** One pass over the queue; a call made during a pass gets the pass after it. */
  tick(): Promise<void>
  status(): Status
  /**
   * Hand over: no more passes or commits, the socket closed, the pass under
   * way waited for. With `handover`, the next runner is one of this process
   * (Vite's config reloaded): the folder's lock is left for it to take over
   * in place, and counts as held until it does, so no server standing back
   * on the same folder sees it free in between. Without, the lock is let go.
   */
  retire(opts?: { handover?: boolean }): Promise<void>
  snapshot(): Snapshot
  closeSocket(): Promise<void>
}

export function createRunner(opts: RunnerOptions): Runner

/** The queue's folder for an archive file when SWITCHGEN_RUNNER_DIR names none: `runner` beside the default archive.json, `<name>.runner` beside any other. */
export function runnerDirFor(archiveFile: string): string

/** The queue as a Vite plugin: mounted for the dev server and the preview alike. */
export function switchgenRunner(): Plugin

/** The queue's word for /api/capabilities. */
export function runnerStatus(): Status

export type { ApiWorkflow }
