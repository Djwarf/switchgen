/**
 * The running slug.
 *
 * In a print room the slug is the line of metal that identifies what is on the
 * press. This is the same idea: one line in the section bar that says what the
 * card is working on, from wherever you happen to be standing. A clip rendering
 * on the video desk stays visible while you write a picture prompt, and one
 * click takes you back to it.
 *
 * It never invents a time. The estimate comes from the run's own measured rate
 * and is labelled as such; before there is a rate, there is no estimate.
 */
import { useEffect, useState } from 'react'
import { goToSection, sectionForDesk } from './route'
import { useHoldToConfirm } from './hotkeys'
import {
  elapsedText,
  headline,
  jobs,
  progressOf,
  remainingOf,
  roughText,
  useJobs,
  type Job,
} from './jobs'

const DESK_LABEL = { images: 'Pictures', video: 'Video', reel: 'Reel' } as const

function jumpTo(job: Job): void {
  goToSection(sectionForDesk(job.desk))
}

/** A clock that only ticks while there is something to count. */
function useNow(live: boolean, ms = 500): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!live) return
    const t = setInterval(() => setNow(Date.now()), ms)
    return () => clearInterval(t)
  }, [live, ms])
  return now
}

export type RunningSlugProps = {
  className?: string
}

export function RunningSlug({ className = '' }: RunningSlugProps) {
  const snap = useJobs()
  const anyLive = snap.active.length > 0 || snap.recent.length > 0
  const now = useNow(anyLive)

  const job = headline(snap, now)

  // Nothing of ours is running, but the card is busy with work started
  // elsewhere. Say so plainly — there is one queue and one graphics card.
  if (!job) {
    if (snap.server.known && snap.server.running + snap.server.pending > 0) {
      const n = snap.server.running + snap.server.pending
      return (
        <p className={`flex items-center gap-2 kicker-quiet ${className}`}>
          <span className="sg-mark sg-mark-live" aria-hidden />
          <span>The card is busy</span>
          <span aria-hidden>·</span>
          <span>
            {n} {n === 1 ? 'job' : 'jobs'} started outside SwitchGen
          </span>
        </p>
      )
    }
    return null
  }

  const waiting = snap.active.filter((j) => j.id !== job.id).length
  return (
    <div className={`flex min-w-0 items-center gap-3 ${className}`}>
      <SlugBody job={job} waiting={waiting} now={now} />
      {/* Keyed by job, so a press armed for one job is not carried over to
          the next one the slug turns to. */}
      {(job.status === 'running' || job.status === 'queued' || job.status === 'submitting') && (
        <StopButton key={job.id} job={job} />
      )}
    </div>
  )
}

function SlugBody({ job, waiting, now }: { job: Job; waiting: number; now: number }) {
  const live = job.status === 'running' || job.status === 'queued' || job.status === 'submitting'
  const done = job.finishedAt !== null
  const pct = progressOf(job)
  const left = remainingOf(job, now)
  const spent = Math.max(0, (job.finishedAt ?? now) - job.startedAt)

  const markClass =
    job.status === 'error'
      ? 'sg-mark sg-mark-off'
      : live
        ? 'sg-mark sg-mark-live'
        : 'sg-mark sg-mark-idle'

  return (
    <button
      type="button"
      onClick={() => jumpTo(job)}
      title={job.prompt || undefined}
      className="ring flex min-w-0 items-center gap-2 border-0 bg-transparent p-0 text-left"
    >
      <span className={markClass} aria-hidden />
      <span className="kicker-quiet shrink-0">{DESK_LABEL[job.desk]}</span>
      <span className="hidden text-small text-grey-700 sm:inline">·</span>
      <span className="hidden max-w-[12rem] truncate text-small text-ink sm:inline">{job.label}</span>

      <span className="text-small text-grey-700">·</span>
      <span className="truncate text-small text-grey-700">
        {/* A stop asked for is not a stop made: the job may still finish, so
            it reads as stopping until its desk reports how it ended. */}
        {live && job.cancelling && 'Stopping'}
        {!job.cancelling && job.status === 'submitting' && 'Sending it over'}
        {!job.cancelling && job.status === 'queued' && (waiting > 0 ? 'Waiting its turn' : 'Queued')}
        {!job.cancelling &&
          job.status === 'running' &&
          (job.max > 0 ? `${job.stage} · step ${job.value} of ${job.max}` : job.stage)}
        {job.status === 'done' && 'Done'}
        {job.status === 'cancelled' && 'Stopped'}
        {job.status === 'error' && 'Stopped short'}
      </span>

      {live && pct !== null && (
        <span className="sg-progress hidden w-24 shrink-0 md:block" aria-hidden>
          <span className="sg-progress-fill" style={{ width: `${Math.round(pct * 100)}%` }} />
        </span>
      )}

      <span className="figures shrink-0 text-small text-grey-500">{elapsedText(spent)}</span>

      {live && left !== null && (
        <span className="hidden text-small text-grey-500 italic lg:inline">
          {roughText(left)} left, at this rate
        </span>
      )}

      {done && job.status === 'done' && (
        <span className="hidden text-small text-burgundy-900 underline lg:inline">See it →</span>
      )}

      {waiting > 0 && (
        <span className="figures shrink-0 kicker-quiet">
          +{waiting} waiting
        </span>
      )}
    </button>
  )
}

/**
 * What stopping does, per desk. A reel shot is one of a queue the reel walks
 * in order, and a picture may be one of a batch of two or four that ends when
 * one of them is stopped, so neither can promise that nothing else is
 * affected.
 */
const STOP_WHAT = {
  reel: {
    thing: 'the reel',
    title:
      'Hold for a moment to stop the reel. This shot stops and the shots after it are not made. Finished shots stay in the Archive.',
  },
  images: {
    thing: 'this picture',
    title:
      'Hold for a moment to stop this picture. If it is one of a batch, the rest of the batch is not made. Pictures already made stay in the Archive.',
  },
  video: {
    thing: 'this clip',
    title: 'Hold for a moment to stop this clip. Clips already made stay in the Archive.',
  },
} as const

function StopButton({ job }: { job: Job }) {
  const hold = useHoldToConfirm(() => {
    void jobs.cancel(job.id)
  })
  const what = STOP_WHAT[job.desk]
  // A screen reader presses with a bare click, which arms the stop rather
  // than firing it; the button and a polite announcement both say so, or the
  // first press would seem to do nothing.
  const armed = hold.armed && !job.cancelling
  return (
    <>
      <button
        type="button"
        {...hold.bind}
        disabled={job.cancelling}
        className="sg-quiet sg-hold ring shrink-0"
        aria-label={
          job.cancelling
            ? `Stopping ${what.thing}`
            : armed
              ? `Press again to stop ${what.thing}`
              : `Hold to stop ${what.thing}`
        }
        title={what.title}
      >
        <span className="sg-hold-wipe" style={{ width: `${Math.round(hold.progress * 100)}%` }} aria-hidden />
        <span className="relative">
          {job.cancelling ? 'Stopping' : armed ? 'Press again to stop' : 'Hold to stop'}
        </span>
      </button>
      <span aria-live="polite" className="sr-only">
        {armed ? `Stop armed. Press the button again within three seconds to stop ${what.thing}.` : ''}
      </span>
    </>
  )
}
