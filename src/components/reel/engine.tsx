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
 * shell unmounting the route.
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
 * the reader may be perfectly happy with them. So they are marked stale and the
 * strip says so, rather than being silently wiped or silently re-queued.
 */
import { useSyncExternalStore } from 'react'

import { cancelJob, run, type OutputFile, type ProgressEvent } from '../../lib/comfy'
import { faultBody, faultOf, faultWhere } from '../../lib/faults'
import { chainFrameOf, instantiateShot, type ShotJob } from '../../lib/continuation'
import { history } from '../../lib/history'
import { recordOf, type Composition } from '../../lib/session'

export type ShotStatus = 'waiting' | 'queued' | 'running' | 'done' | 'error' | 'stopped'

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
  /** True when this shot was rendered, and then the shot it follows changed. */
  stale: boolean
  /** Frames asked for, so the strip can weight progress by real work. */
  frames: number
}

export type RunStatus = 'idle' | 'running' | 'done' | 'stopped' | 'error'

export type RunState = {
  id: string
  status: RunStatus
  startedAt: number | null
  finishedAt: number | null
  /** Shot ids in the order the queue walks them. */
  order: string[]
  states: Record<string, ShotState>
  currentShotId: string | null
  /** How many shots this pass set out to render, for honest counting. */
  planned: number
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
    stale: false,
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
  planned: 0,
  stopRequested: false,
  note: null,
}

let state: RunState = IDLE
const listeners = new Set<() => void>()
/** One walk at a time. A second press while the queue moves is ignored. */
let walking = false

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

/**
 * Mark every shot downstream of `index` stale, stopping at the first shot that
 * opens on a frame of its own. Only shots that were actually rendered can be
 * out of date, so nothing else is touched.
 */
function markStaleAfter(index: number, order: string[], jobs: readonly ShotJob[]): void {
  const states = { ...state.states }
  let changed = false
  for (let k = index + 1; k < order.length; k++) {
    if (jobs[k]?.start.from !== 'previous') break
    const shotId = order[k]
    const shot = shotId ? states[shotId] : undefined
    if (!shotId || !shot) break
    if (shot.status === 'done' && !shot.stale) {
      states[shotId] = { ...shot, stale: true }
      changed = true
    }
  }
  if (changed) {
    state = { ...state, states }
    emit()
  }
}

// ---------------------------------------------------------------------------
// One shot
// ---------------------------------------------------------------------------

async function renderShot(
  shotId: string,
  job: ShotJob,
  previous: OutputFile | null,
  ctx: RunContext,
): Promise<'done' | 'error' | 'stopped'> {
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
    startedAt,
    finishedAt: null,
    previewUrl: null,
    stale: false,
  })

  const onEvent = (e: ProgressEvent) => {
    if (e.phase === 'queued') {
      setShot(shotId, { promptId: e.promptId, stage: 'Queued' })
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

    let entryId: string | null = null
    if (clip) {
      try {
        const entry = history.add(
          recordOf(ctx.compositionFor(job), {
            file: clip,
            files: files.length > 1 ? files : undefined,
            kind: 'video',
            promptId: shotOf(shotId)?.promptId ?? '',
            durationMs: finishedAt - startedAt,
            seed: job.params.seed,
            familyLabel: ctx.familyLabel,
            modelLabel: ctx.modelLabel,
            at: finishedAt,
          }),
        )
        entryId = entry.id
      } catch {
        // A full archive must never cost the reader the clip itself.
      }
    }

    setShot(shotId, {
      status: 'done',
      stage: 'Done',
      files,
      clip,
      frame,
      entryId,
      finishedAt,
      durationMs: finishedAt - startedAt,
      previewUrl: null,
      stale: false,
    })
    return 'done'
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
}

async function walk(pass: Pass): Promise<void> {
  walking = true
  const { order, jobs, ctx, indices } = pass
  let verdict: RunStatus = 'done'
  let note: string | null = null

  for (const index of indices) {
    if (state.stopRequested) {
      verdict = 'stopped'
      note = `Stopped before shot ${index + 1}. What was already rendered is on disk and in the archive.`
      break
    }
    const shotId = order[index]
    const job = jobs[index]
    if (!shotId || !job) continue

    setRun({ currentShotId: shotId })

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
        note = `Shot ${index + 1} had nothing to continue from, so the queue stopped there.`
        break
      }
    }

    const result = await renderShot(shotId, job, previous, ctx)

    if (result === 'done') {
      markStaleAfter(index, order, jobs)
      continue
    }
    if (result === 'stopped') {
      verdict = 'stopped'
      note = `Stopped during shot ${index + 1}. Everything before it is finished and filed.`
      break
    }
    verdict = 'error'
    note =
      index + 1 < order.length
        ? `Shot ${index + 1} failed, and every shot after it was going to open on its last frame. The queue stopped rather than carry on from a frame that does not exist.`
        : `Shot ${index + 1} failed.`
    break
  }

  // Anything the pass never reached goes back to waiting rather than sitting
  // on a stale "queued" that would read as work in progress.
  const states = { ...state.states }
  for (const shotId of order) {
    const shot = states[shotId]
    if (shot && (shot.status === 'queued' || shot.status === 'running')) {
      states[shotId] = { ...shot, status: 'waiting', stage: '', previewUrl: null }
    }
  }
  state = { ...state, states }

  walking = false
  setRun({
    status: verdict,
    finishedAt: Date.now(),
    currentShotId: null,
    stopRequested: false,
    note,
  })
}

// ---------------------------------------------------------------------------
// The public engine
// ---------------------------------------------------------------------------

function newRunId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `run_${Date.now().toString(36)}`
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

    // A shot needs rendering when it has no current clip of its own. Then the
    // need spreads forward: every shot that opens on the one before it has to
    // follow, because the frame it was going to open on is about to change.
    const need = new Set<number>()
    order.forEach((shotId, i) => {
      const shot = states[shotId]
      const current = !opts.force && shot?.status === 'done' && !shot.stale && shot.frame
      if (!current) need.add(i)
    })
    for (let i = 1; i < order.length; i++) {
      if (need.has(i - 1) && jobs[i]?.start.from === 'previous') need.add(i)
    }
    const queue = [...need].sort((a, b) => a - b)
    if (!queue.length) return

    for (const i of queue) {
      const shotId = order[i]
      const shot = shotId ? states[shotId] : undefined
      if (shotId && shot) states[shotId] = { ...shot, status: 'waiting', stage: '', error: null }
    }

    state = {
      id: newRunId(),
      status: 'running',
      startedAt: Date.now(),
      finishedAt: null,
      order,
      states,
      currentShotId: null,
      planned: queue.length,
      stopRequested: false,
      note: null,
    }
    emit()
    void walk({ order, jobs, ctx, indices: queue })
  },

  /** Render one shot, leaving every other shot exactly as it stands. */
  renderOne(index: number, order: string[], jobs: readonly ShotJob[], ctx: RunContext): void {
    if (walking) return
    const states = adopt(order, jobs)
    state = {
      id: newRunId(),
      status: 'running',
      startedAt: Date.now(),
      finishedAt: null,
      order,
      states,
      currentShotId: null,
      planned: 1,
      stopRequested: false,
      note: null,
    }
    emit()
    void walk({ order, jobs, ctx, indices: [index] })
  },

  /** Stop the running shot and abandon the rest of the queue. */
  stop(): void {
    if (state.status !== 'running') return
    setRun({ stopRequested: true })
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
    emit()
  },
}

export function useReelRun(): RunState {
  return useSyncExternalStore(reelRun.subscribe, reelRun.snapshot, reelRun.snapshot)
}
