/**
 * Progress across the whole reel.
 *
 * The thing a person actually wants to know while an hour of rendering happens
 * is not which sampler step the current shot is on. It is: how far through the
 * reel are we, and how much is left. So the rail is weighted by frames across
 * every shot, not reset to zero at each one.
 *
 * ON FINISH TIMES. A finish time is only printed when every remaining shot has
 * at least three measured runs of the same family at the same size and length
 * in the archive. Anything else is a guess wearing a number, and a guess that
 * says "4 minutes" over an hour of rendering is worse than saying nothing.
 * When the measurements are missing, the band says so and shows elapsed alone.
 */
import type { ShotJob } from '../../lib/continuation'
import type { HistoryEntry } from '../../lib/history'
import { Kicker, Mark, Quiet, Rail, duration, grouped, seconds } from './bits'
import type { RunState } from './engine'

/** The median of matching runs, and how many there were. */
export type Measured = { ms: number; runs: number }

const MIN_SAMPLES = 3

/**
 * Median wall clock time of past runs that match a shot exactly.
 *
 * Exactly, deliberately: a 121 frame shot at 1280 by 704 tells you nothing
 * useful about an 81 frame shot at 832 by 480. Fewer than three matches and
 * this returns null, which is the whole point.
 */
export function measuredFor(
  records: readonly HistoryEntry[],
  key: { familyId: string; mode: 't2v' | 'i2v'; length: number; width: number; height: number },
): Measured | null {
  const like = records
    .filter(
      (e) =>
        e.kind === 'video' &&
        e.familyId === key.familyId &&
        e.mode === key.mode &&
        e.length === key.length &&
        e.width === key.width &&
        e.height === key.height &&
        e.durationMs > 0,
    )
    .slice(0, 8)
  if (like.length < MIN_SAMPLES) return null
  const sorted = like.map((e) => e.durationMs).sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  const ms =
    sorted.length % 2 ? (sorted[mid] as number) : Math.round(((sorted[mid - 1] as number) + (sorted[mid] as number)) / 2)
  return { ms, runs: like.length }
}

export type ReelProgressProps = {
  run: RunState
  jobs: readonly ShotJob[]
  fps: number
  width: number
  height: number
  familyId: string
  records: readonly HistoryEntry[]
  /** A clock that ticks only while something is running. */
  now: number
  onStop: () => void
}

export function ReelProgress({
  run,
  jobs,
  fps,
  width,
  height,
  familyId,
  records,
  now,
  onStop,
}: ReelProgressProps) {
  if (run.status === 'idle') return null

  const running = run.status === 'running'
  const order = run.order
  const total = order.reduce((sum, id) => sum + (run.states[id]?.frames ?? 0), 0)

  let made = 0
  let done = 0
  let failed = 0
  order.forEach((id) => {
    const shot = run.states[id]
    if (!shot) return
    if (shot.status === 'done') {
      made += shot.frames
      done += 1
    } else if (shot.status === 'running' && shot.max > 1) {
      made += shot.frames * Math.min(1, shot.value / shot.max)
    } else if (shot.status === 'error') {
      failed += 1
    }
  })

  const currentIndex = run.currentShotId ? order.indexOf(run.currentShotId) : -1
  const left = order.filter((id) => {
    const s = run.states[id]
    return s && s.status !== 'done' && s.status !== 'error'
  }).length

  // --- the estimate, or the honest absence of one -------------------------
  let estimate: { ms: number; runs: number; shots: number } | null = null
  let unmeasured = 0
  if (running) {
    let ms = 0
    let runs = 0
    let shots = 0
    order.forEach((id, i) => {
      const shot = run.states[id]
      const job = jobs[i]
      if (!shot || !job) return
      if (shot.status === 'done' || shot.status === 'error') return
      const m = measuredFor(records, {
        familyId,
        mode: job.start.from === 'none' ? 't2v' : 'i2v',
        length: job.params.length ?? 0,
        width,
        height,
      })
      if (!m) {
        unmeasured += 1
        return
      }
      const fraction =
        shot.status === 'running' && shot.max > 1 ? Math.max(0, 1 - shot.value / shot.max) : 1
      ms += m.ms * fraction
      runs = Math.max(runs, m.runs)
      shots += 1
    })
    if (shots > 0 && unmeasured === 0) estimate = { ms, runs, shots }
  }

  const elapsed = run.startedAt ? now - run.startedAt : 0

  return (
    <section className="mb-6 border border-grey-300 bg-newsprint-aged px-4 py-3">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <span className="flex items-center gap-2">
          <Mark state={running ? 'live' : failed ? 'off' : 'ok'} />
          <Kicker tone={running ? 'burgundy' : 'ink'}>
            {running ? 'On the press' : run.status === 'stopped' ? 'Stopped' : failed ? 'Stopped short' : 'Reel complete'}
          </Kicker>
        </span>
        <span className="text-caption tabular-nums text-grey-700">
          {done} of {order.length} shots rendered
          {left > 0 ? ` · ${left} to go` : ''}
          {failed > 0 ? ` · ${failed} failed` : ''}
        </span>
      </div>

      <div className="my-2">
        <Rail value={made} max={total || 1} />
      </div>

      <div className="grid gap-x-6 gap-y-1 sm:grid-cols-2 lg:grid-cols-4">
        <Line
          label={running && currentIndex >= 0 ? `Shot ${currentIndex + 1}` : 'Shots'}
          value={
            running && currentIndex >= 0
              ? shotStage(run, currentIndex)
              : `${done} of ${order.length} on disk`
          }
        />
        <Line label="Elapsed" value={duration(elapsed)} />
        <Line
          label="Rendered"
          value={`${grouped(made)} of ${grouped(total)} frames · ${seconds(total, fps)} of screen time`}
        />
        <Line
          label="Finish"
          value={
            estimate
              ? `About ${duration(estimate.ms)} left`
              : running
                ? 'Not stated'
                : run.finishedAt && run.startedAt
                  ? duration(run.finishedAt - run.startedAt)
                  : 'Done'
          }
        />
      </div>

      {running ? (
        <p className="mt-2 text-caption italic text-grey-500">
          {estimate
            ? `From ${estimate.runs} measured runs of this family at this size, across the ${estimate.shots} shots still to render.`
            : unmeasured > 0
              ? `No finish time until this family has three measured runs at this size and length. ${unmeasured} of the remaining shots have none, so none is offered.`
              : 'Each shot is a separate generation. They run one at a time, in order.'}
        </p>
      ) : null}

      {run.note ? <p className="mt-2 border-l-2 border-ink pl-2 text-caption italic text-grey-700">{run.note}</p> : null}

      {running ? (
        <div className="mt-3 border-t border-grey-300 pt-2">
          <Quiet danger onClick={onStop}>
            Stop the reel
          </Quiet>
          <span className="ml-3 text-caption italic text-grey-500">
            The shot on the press is cancelled. Everything already rendered stays.
          </span>
        </div>
      ) : null}
    </section>
  )
}

function shotStage(run: RunState, index: number): string {
  const id = run.order[index]
  const shot = id ? run.states[id] : null
  if (!shot) return 'Waiting'
  if (shot.status === 'running' && shot.max > 1) return `${shot.stage} · step ${shot.value} of ${shot.max}`
  return shot.stage || 'Waiting'
}

function Line({ label, value }: { label: string; value: string }) {
  return (
    <p className="text-caption">
      <span className="mr-1 font-semibold uppercase tracking-[0.12em] text-grey-500">{label}</span>
      <span className="tabular-nums text-ink">{value}</span>
    </p>
  )
}
