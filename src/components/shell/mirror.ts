/**
 * The mirror between each desk's own job store and the press ledger.
 *
 * Every desk keeps its own job engine, because a four-minute clip must survive
 * walking over to the pictures desk, and the section bar reads one ledger. A
 * bridge flattens a desk's store into `Reported` rows; the mirror opens a
 * ledger job for each live one and passes on what the desk says about it.
 * App.tsx holds the bridges, since they reach into the desks; the mirror
 * lives here, beside the ledger it writes to.
 */
import { jobs, type JobDesk } from './jobs'

/** The shape every desk's store shares, once the differences are flattened out. */
export type Reported = {
  /** The desk's own id for the job, stable for its lifetime. */
  key: string
  status: 'submitting' | 'queued' | 'running' | 'done' | 'error' | 'cancelled'
  promptId: string | null
  label: string
  prompt: string
  value: number
  max: number
  entryId: string | null
  error: string | null
}

export type Bridge = {
  desk: JobDesk
  kind: 'image' | 'video'
  subscribe: (fn: () => void) => () => void
  read: () => Reported[]
  /**
   * The desk's own stop, given the desk's id for the job. Every desk has one,
   * because on every desk stopping means more than cancelling one prompt: a
   * batch or a reel has more to come, and a clip may not have a prompt yet.
   */
  stop?: (key: string) => void
  /**
   * Desk job id → ledger job id, kept on the bridge rather than inside the
   * mirror, so that a remount — StrictMode's double effect in development, or
   * the shell being torn down and rebuilt — adopts the jobs it already opened
   * instead of announcing the same clip twice. A blank value means "seen, and
   * already finished before we looked".
   */
  seen: Map<string, string>
}

const LIVE = new Set(['submitting', 'queued', 'running'])
const SEEN_LIMIT = 200

/**
 * Report one desk's jobs to the ledger until the function it returns is
 * called. Each live job is opened on the ledger once, its prompt id and
 * progress are passed on, and its ending is the one its desk reports.
 */
export function mirror(bridge: Bridge): () => void {
  const seen = bridge.seen

  const sync = () => {
    for (const report of bridge.read()) {
      let id = seen.get(report.key)

      if (id === undefined) {
        // Work that was already finished when the bridge first looked belongs
        // to the archive, not to the slug. Note it and leave it alone.
        if (!LIVE.has(report.status)) {
          seen.set(report.key, '')
          continue
        }
        const stop = bridge.stop
        const key = report.key
        id = jobs.start({
          desk: bridge.desk,
          kind: bridge.kind,
          label: report.label,
          prompt: report.prompt,
          promptId: report.promptId,
          steps: report.max || undefined,
          stop: stop ? () => stop(key) : undefined,
        })
        seen.set(report.key, id)
      }
      if (!id) continue

      const ledgerJob = jobs.get(id)
      if (!ledgerJob) continue

      if (report.promptId && ledgerJob.promptId !== report.promptId) {
        jobs.attach(id, report.promptId)
      }

      // The desk's ending is the job's ending. It holds the run that settled
      // it, so its word replaces anything the ledger shows, and the ledger
      // takes a new ending only when it differs from the one it has, with
      // one exception below.
      switch (report.status) {
        case 'running':
          if (
            LIVE.has(ledgerJob.status) &&
            (ledgerJob.status !== 'running' ||
              ledgerJob.value !== report.value ||
              ledgerJob.max !== report.max)
          ) {
            jobs.apply(id, {
              phase: 'running',
              node: null,
              value: report.value,
              max: report.max,
            })
          }
          break
        case 'done':
          // Never over a stop. When a pass ends, the reel puts a stopped
          // shot's earlier clip back, under the same key and marked done.
          // That is the shot as it stood before the pass, not how this job
          // ended: taken as news, it said Done over a shot the reader had
          // stopped, and See it opened the old clip.
          if (ledgerJob.status === 'cancelled') break
          if (ledgerJob.status !== 'done' || (report.entryId && report.entryId !== ledgerJob.entryId)) {
            jobs.succeed(id, report.entryId ? { entryId: report.entryId } : undefined)
          }
          break
        case 'error':
          if (ledgerJob.status !== 'error') {
            jobs.fail(id, report.error ?? 'The job stopped short.')
          }
          break
        case 'cancelled':
          if (ledgerJob.status !== 'cancelled') {
            jobs.fail(id, report.error ?? 'Stopped.', { cancelled: true })
          }
          break
        default:
          break
      }
    }

    // The map is a lookup for work in progress, not a second archive. A page
    // left open all day should not accumulate one entry per picture for ever.
    if (seen.size > SEEN_LIMIT) {
      const live = new Set(bridge.read().map((r) => r.key))
      for (const key of [...seen.keys()]) {
        if (seen.size <= SEEN_LIMIT / 2) break
        if (!live.has(key)) seen.delete(key)
      }
    }
  }

  sync()
  return bridge.subscribe(sync)
}
