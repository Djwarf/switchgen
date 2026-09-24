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
  needsClock,
  progressOf,
  remainingOf,
  roughText,
  SENDING,
  useJobs,
  type Job,
  type JobsSnapshot,
} from './jobs'

const DESK_LABEL = { images: 'Pictures', video: 'Video', reel: 'Reel' } as const

function jumpTo(job: Job): void {
  goToSection(sectionForDesk(job.desk))
}

/**
 * A clock that only ticks while there is something to count: a live job, or
 * a finished one still shown as news (see needsClock). Once the news is old,
 * the next tick finds nothing to count and the clock stops until the ledger
 * changes again.
 */
function useClock(snap: JobsSnapshot, ms = 500): number {
  const [now, setNow] = useState(() => Date.now())
  const live = needsClock(snap, now)
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
  const now = useClock(snap)

  const job = headline(snap, now)

  // Nothing this page follows is on the press, but the card is busy. Say so
  // plainly, and say only what is known: this page is not following it. It
  // may be the reader's own, sent before a reload and not yet picked back up,
  // or from another tab or ComfyUI's own page; "started outside SwitchGen"
  // told the reader their own clip was someone else's.
  //
  // On a phone the sentence is wider than its row. Left to shrink, "The card
  // is busy" broke over two lines and the count was cut off before
  // "following", the word it turns on. So neither phrase breaks inside
  // itself, and below a wide screen the count takes a line of its own
  // instead; it is cut short only on a screen too narrow even for that.
  if (!job) {
    if (snap.server.known && snap.server.foreign > 0) {
      const n = snap.server.foreign
      return (
        <p
          className={`flex items-center gap-x-2 gap-y-0.5 whitespace-nowrap kicker-quiet max-lg:flex-wrap ${className}`}
        >
          <span className="sg-mark sg-mark-live" aria-hidden />
          <span>The card is busy</span>
          <span aria-hidden>·</span>
          <span className="truncate">
            {n} {n === 1 ? 'job' : 'jobs'} this page is not following
          </span>
        </p>
      )
    }
    return null
  }

  const waiting = snap.active.filter((j) => j.id !== job.id).length
  const live = job.status === 'running' || job.status === 'queued' || job.status === 'submitting'
  // A job that cannot be stopped from here just now (the queue on the server
  // takes no stop while it is off) is offered none, and the slug says why in
  // its place.
  const noStop = live ? job.noStop : null
  // Below a wide screen the slug has a row of its own, and Hold to stop goes
  // to the far end of it, apart from the line a tap on which opens the desk.
  // Only here: the busy line above is one sentence, and spread across the
  // row its pieces stood hundreds of pixels apart on a tablet.
  //
  // The reason in Stop's place is a sentence or two, far wider than the room
  // the job's own line leaves it on a phone or a tablet, where it was cut to
  // a few words and the rest was only in a title a touch screen cannot open.
  // So below a wide screen it takes a line of its own under the job's and
  // wraps there in full. On a wide screen it shares the bar with the rooms,
  // and is cut short with the rest in its title, which a pointer can show.
  return (
    <div
      className={`flex min-w-0 items-center gap-3 max-lg:justify-between ${noStop ? 'max-lg:flex-wrap max-lg:gap-y-0.5' : ''} ${className}`}
    >
      <SlugBody job={job} waiting={waiting} now={now} />
      {/* Keyed by job, so a press armed for one job is not carried over to
          the next one the slug turns to. */}
      {live &&
        (noStop ? (
          <span
            className="min-w-0 flex-1 text-small text-grey-700 italic max-lg:basis-full lg:truncate lg:text-right"
            title={noStop}
          >
            {noStop}
          </span>
        ) : (
          <StopButton key={job.id} job={job} />
        ))}
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
      {/* On a phone the slug has one row to itself, and the step it is on
          matters more than the desk's name, which a screen reader still gets. */}
      <span className="kicker-quiet shrink-0 max-sm:sr-only">{DESK_LABEL[job.desk]}</span>
      <span className="hidden text-small text-grey-700 sm:inline">·</span>
      <span className="hidden max-w-[12rem] truncate text-small text-ink sm:inline">{job.label}</span>

      <span className="text-small text-grey-700 max-sm:hidden">·</span>
      <span className="truncate text-small text-grey-700">
        {/* A stop asked for is not a stop made: the job may still finish, so
            it reads as stopping until its desk reports how it ended. */}
        {live && job.cancelling && 'Stopping'}
        {/* Not sent yet is not always being sent: a clip can be held behind a
            lost one until the reader answers, or wait for ComfyUI to come
            back, and its desk says which. */}
        {!job.cancelling && job.status === 'submitting' && (job.stage || SENDING)}
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
