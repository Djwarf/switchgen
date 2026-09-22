/**
 * The button. One of them, at full measure, doing the only irreversible thing
 * on the screen.
 *
 * While a job is running the same button becomes hold to stop: a 600ms
 * burgundy wipe, so a stop is a deliberate act and not a mis-click, and the
 * stage line underneath says what the machine is actually doing. Both of those
 * are lifted from the desk as it stood, unchanged, because neither was the
 * problem.
 *
 * The keyboard holds too. Enter or Space starts the same 600ms wipe and
 * letting go cancels it, and a key's auto-repeat is ignored. It used to stop on
 * the first keydown, and the button keeps focus as it turns from "Make the
 * picture" into "Hold to stop", so a second Enter, or one Enter held a moment
 * too long, stopped the job the first had just started.
 *
 * A screen reader activates a button with a bare click and no key or pointer
 * events at all, so it could not stop a job. A click with no pointer behind it
 * arms the stop instead, and a second one within a few seconds confirms it:
 * still two deliberate acts, and a mouse click still does nothing without the
 * hold.
 *
 * The receipt is the one number this component prints: how long the last run
 * took, measured by the desk's own clock. It is shown for a few seconds in
 * place of the label and then the label comes back.
 */
import {
  useEffect,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type MouseEvent as ReactMouseEvent,
} from 'react'

/** How long an armed stop waits for its confirming press. */
const ARMED_MS = 3000

const THIN = ' '

function seconds(ms: number): string {
  if (ms < 10_000) return `${(ms / 1000).toFixed(1)}${THIN}s`
  if (ms < 90_000) return `${Math.round(ms / 1000)}${THIN}s`
  const m = Math.floor(ms / 60_000)
  const s = Math.round((ms % 60_000) / 1000)
  return `${m} min ${s}${THIN}s`
}

/** What the desk knows about the job in front of it. All of it optional. */
export type RunJob = {
  stage?: string | null
  index?: number
  total?: number
  pct?: number
  status?: string
}

export function RunButton({
  label = 'Make the picture',
  disabled = false,
  why = '',
  running = false,
  queuedAhead = 0,
  lastMs = null,
  job = null,
  onRun,
  onStop,
  reduced = false,
}: {
  label?: string
  disabled?: boolean
  /** Why it is disabled. Printed under the button, quietly. */
  why?: string
  running?: boolean
  /** Jobs in the single queue ahead of this one. */
  queuedAhead?: number
  /** Duration of the last finished run, in ms. */
  lastMs?: number | null
  job?: RunJob | null
  onRun: () => void
  onStop?: () => void
  reduced?: boolean
}) {
  const [holding, setHolding] = useState(false)
  const [armed, setArmed] = useState(false)
  const [receipt, setReceipt] = useState<string | null>(null)
  const timer = useRef<number | null>(null)
  const seen = useRef<number | null>(null)

  useEffect(() => {
    if (lastMs == null || seen.current === lastMs) return
    seen.current = lastMs
    setReceipt(seconds(lastMs))
    const t = window.setTimeout(() => setReceipt(null), 2600)
    return () => window.clearTimeout(t)
  }, [lastMs])

  useEffect(() => () => {
    if (timer.current) window.clearTimeout(timer.current)
  }, [])

  // An armed stop lapses on its own.
  useEffect(() => {
    if (!armed) return
    const t = window.setTimeout(() => setArmed(false), ARMED_MS)
    return () => window.clearTimeout(t)
  }, [armed])

  const beginHold = () => {
    // One hold at a time: a key held down while the pointer is also down must
    // not start a second timer and stop twice.
    if (!onStop || timer.current) return
    setHolding(true)
    timer.current = window.setTimeout(() => {
      timer.current = null
      setHolding(false)
      onStop()
    }, 600)
  }

  const endHold = () => {
    setHolding(false)
    if (timer.current) window.clearTimeout(timer.current)
    timer.current = null
  }

  const isActivation = (e: ReactKeyboardEvent) => e.key === 'Enter' || e.key === ' '

  if (running) {
    return (
      <div>
        <button
          type="button"
          aria-label={armed ? 'Press again to stop this job' : 'Hold to stop this job'}
          onPointerDown={beginHold}
          onPointerUp={endHold}
          onPointerLeave={endHold}
          onPointerCancel={endHold}
          onKeyDown={(e: ReactKeyboardEvent) => {
            if (!isActivation(e)) return
            // Default prevented either way, so the key never turns into a click.
            e.preventDefault()
            if (e.repeat) return
            beginHold()
          }}
          onKeyUp={(e: ReactKeyboardEvent) => {
            if (!isActivation(e)) return
            e.preventDefault()
            endHold()
          }}
          onClick={(e: ReactMouseEvent) => {
            // A pointer click has a detail of one or more and is hold only.
            // Zero is a click with no pointer behind it: assistive technology.
            if (e.detail !== 0 || !onStop) return
            if (armed) {
              setArmed(false)
              onStop()
            } else {
              setArmed(true)
            }
          }}
          className="press sg-hold relative overflow-hidden"
          style={{ backgroundColor: 'var(--color-newsprint)', color: 'var(--color-burgundy-900)' }}
        >
          <span className="relative">{armed ? 'Press again to stop' : 'Hold to stop'}</span>
          <span
            aria-hidden
            className="absolute inset-0 grid place-items-center bg-burgundy-900 text-newsprint"
            style={{
              clipPath: holding ? 'inset(0 0 0 0)' : 'inset(0 100% 0 0)',
              transition: reduced ? 'none' : 'clip-path 600ms linear',
            }}
          >
            {armed ? 'Press again to stop' : 'Hold to stop'}
          </span>
        </button>
        <p aria-live="polite" className="sr-only">
          {armed ? 'Stop armed. Press the button again within three seconds to stop the job.' : ''}
        </p>
        <p className="mt-1.5 text-caption italic text-grey-700 tabular-nums">
          {job?.total && job.total > 1 ? `Picture ${job.index} of ${job.total} · ` : ''}
          {job?.stage ?? 'Working'}
          {job && (job.pct ?? 0) >= 0.97 && job.status === 'running'
            ? ' · running long, still working'
            : ''}
        </p>
      </div>
    )
  }

  return (
    <div>
      <button
        type="button"
        className="press"
        disabled={disabled}
        onClick={() => {
          // A stop armed for the last job is not carried into this one.
          setArmed(false)
          onRun()
        }}
      >
        {receipt ? (
          <span className="tabular-nums">{receipt}</span>
        ) : (
          <>
            {label}
            {queuedAhead > 0 ? ' · next in line' : ''}
          </>
        )}
      </button>
      {disabled && why && <p className="mt-1.5 text-caption italic text-grey-500">{why}</p>}
      {!disabled && queuedAhead > 0 && (
        <p className="mt-1.5 text-caption italic text-grey-700">
          There is one card and it is busy. Your picture starts when the job in front of it
          finishes.
        </p>
      )}
    </div>
  )
}
