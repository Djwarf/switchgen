/**
 * Warm and cold pictures.
 *
 * ComfyUI runs here with --disable-smart-memory, so its history cannot say
 * whether a model was already loaded. The lab goes by order instead: a
 * picture is warm when the job ComfyUI ran just before it, on any desk, was a
 * lab job on the same weight file doing the same operation; otherwise the
 * load is in its time and it counts as cold. A job ComfyUI answered from its
 * cache ran nothing, so it is passed over when looking back, and its own time
 * is never a cost figure (the ledger marks it cached).
 */

/** The fields of a runner job this reads. */
export type TimedJob = {
  id: string
  desk: string
  status?: string
  ranAt: number | null
  finishedAt: number | null
  primary?: { cached?: boolean } | null
  files?: readonly { cached?: boolean }[]
  meta?: Record<string, unknown> | null
}

/** What makes two lab pictures share a loaded model. */
export type CellFacts = { file: string; op: string }

/** The lab cell a job was for, from its meta, or null for any other job. */
export function labCellOf(job: Pick<TimedJob, 'desk' | 'meta'> | null | undefined): string | null {
  if (!job || job.desk !== 'lab') return null
  const lab = job.meta?.lab
  if (!lab || typeof lab !== 'object') return null
  const cell = (lab as { cell?: unknown }).cell
  return typeof cell === 'string' && cell ? cell : null
}

/** A job whose picture came from ComfyUI's cache: nothing was sampled. */
export function isCached(job: Pick<TimedJob, 'primary' | 'files'>): boolean {
  if (job.primary) return job.primary.cached === true
  const files = job.files ?? []
  return files.length > 0 && files.every((f) => f.cached === true)
}

/**
 * The job ComfyUI ran just before `job`: of those that ran and ended by the
 * time it started, the last to end, passing over cached ones. Null when none
 * is known (the runner may have pruned it, or it did not go through the runner).
 */
export function previousRan<T extends TimedJob>(jobs: readonly T[], job: TimedJob): T | null {
  if (job.ranAt == null) return null
  let best: T | null = null
  for (const o of jobs) {
    if (o.id === job.id || o.ranAt == null || o.finishedAt == null) continue
    if (o.finishedAt > job.ranAt) continue
    if (isCached(o)) continue
    if (!best || (o.finishedAt ?? 0) > (best.finishedAt ?? 0)) best = o
  }
  return best
}

/**
 * Cold unless the job before it (see previousRan) was a lab job on the same
 * weight file with the same operation. Unknown counts as cold, so a warm
 * figure is never one that might have carried a load.
 */
export function coldFlag(
  prevEndedJob: TimedJob | null | undefined,
  job: TimedJob,
  cellOf: (cellId: string) => CellFacts | null | undefined,
): boolean {
  if (!prevEndedJob) return true
  const prevCell = labCellOf(prevEndedJob)
  const cur = labCellOf(job)
  if (!prevCell || !cur) return true
  const a = cellOf(prevCell)
  const b = cellOf(cur)
  if (!a || !b) return true
  return !(a.file === b.file && a.op === b.op)
}
