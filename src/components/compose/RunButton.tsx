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
 * A stop that lands while the key or the pointer is still down must not turn
 * into a new job. The button stays the same element, with focus, as it turns
 * back into "Make the picture", so an Enter still held fires its auto-repeat
 * into it as clicks, a held Space clicks it on release, and a mouse let go
 * after the stop lands clicks it too. Each of those queued the same recipe
 * again, which turned a stop into a restart. The press that began on the
 * running button is followed until it is let go, and nothing it does reaches
 * the idle one. The first fresh press after that starts a job as usual.
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
import { WAITS_IN_PAGE, wakeLockAvailable } from '../../lib/wakeLock'

/** How long an armed stop waits for its confirming press. */
const ARMED_MS = 3000

/**
 * How long after the release of a stopping press its click is still taken for
 * part of that press. A mouse click follows its release at once; a tap's comes
 * after the browser has decided it was a tap. A fresh press clears the mark
 * before its own click arrives, so only the click that belongs to the release
 * is refused.
 */
const RELEASE_CLICK_MS = 500

const THIN = ' '

function seconds(ms: number): string {
  if (ms < 10_000) return `${(ms / 1000).toFixed(1)}${THIN}s`
  if (ms < 90_000) return `${Math.round(ms / 1000)}${THIN}s`
  const m = Math.floor(ms / 60_000)
  const s = Math.round((ms % 60_000) / 1000)
  return `${m} min ${s}${THIN}s`
}

/**
 * The stage a desk shows from the moment Stop is asked until the cancel lands.
 * Shared so the desk and this button cannot drift apart on the word.
 */
export const STOPPING = 'Stopping'

/**
 * The line under a batch while pictures still wait in the page, or null when
 * none do. Only the picture on the press is in ComfyUI; the rest are sent from
 * this page one at a time, and a phone that locks or hides the page sends
 * nothing more until it wakes. Nothing said so, and the card sat idle.
 */
export function waitingLine(job: RunJob | null, keepsScreenOn = wakeLockAvailable()): string | null {
  // A stop drops the rest of the batch at once, but the picture on the press
  // can take seconds to let go, and until it does the job still counts the
  // ones behind it. None of them will be sent, so none is said to wait.
  if (job?.stage === STOPPING) return null
  const left = job?.total && job.index ? job.total - job.index : 0
  if (left <= 0) return null
  const waits =
    left === 1
      ? 'One more picture waits in this page and is sent when this one is done.'
      : `${left} more pictures wait in this page and are sent one at a time.`
  const screen = keepsScreenOn ? ' The page asks the phone to keep the screen on until the last one is sent.' : ''
  return `${waits} ${WAITS_IN_PAGE}${screen}`
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
  /** An activation key went down on the running button and has not come up. */
  const keyHeld = useRef(false)
  /** A pointer went down on the running button and has not been let go. */
  const pointerHeld = useRef(false)
  /**
   * When the release of one of those presses landed on the idle button. The
   * click the browser fires after that release is not a request for a job. See
   * RELEASE_CLICK_MS for why it is a moment and not a flag.
   */
  const releasedAt = useRef<number | null>(null)

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
  const waiting = running ? waitingLine(job) : null

  /** The press that stopped the last job is over, wherever it ended. */
  const forgetPress = () => {
    keyHeld.current = false
    pointerHeld.current = false
  }

  if (running) {
    return (
      <div>
        <button
          type="button"
          aria-label={armed ? 'Press again to stop this job' : 'Hold to stop this job'}
          onPointerDown={() => {
            pointerHeld.current = true
            beginHold()
          }}
          onPointerUp={() => {
            pointerHeld.current = false
            endHold()
          }}
          onPointerLeave={() => {
            // Let go somewhere else, so no click will follow on this button.
            pointerHeld.current = false
            endHold()
          }}
          onPointerCancel={() => {
            pointerHeld.current = false
            endHold()
          }}
          onKeyDown={(e: ReactKeyboardEvent) => {
            if (!isActivation(e)) return
            // Default prevented either way, so the key never turns into a click.
            e.preventDefault()
            keyHeld.current = true
            if (e.repeat) return
            beginHold()
          }}
          onKeyUp={(e: ReactKeyboardEvent) => {
            if (!isActivation(e)) return
            e.preventDefault()
            keyHeld.current = false
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
        {waiting ? <p className="mt-1 text-caption leading-snug text-grey-700">{waiting}</p> : null}
      </div>
    )
  }

  return (
    <div>
      <button
        type="button"
        className="press"
        disabled={disabled}
        onKeyDown={(e: ReactKeyboardEvent) => {
          if (!isActivation(e)) return
          // The auto-repeat of a key that was held to stop the last job. Enter
          // clicks on every repeat, so each one is refused here.
          if (e.repeat) {
            e.preventDefault()
            return
          }
          // A key that goes down fresh is a new press, whatever came before.
          forgetPress()
          releasedAt.current = null
        }}
        onKeyUp={(e: ReactKeyboardEvent) => {
          if (!isActivation(e) || !keyHeld.current) return
          // Space clicks on release. This release ends the stop, not a start.
          keyHeld.current = false
          e.preventDefault()
          releasedAt.current = performance.now()
        }}
        onPointerDown={() => {
          forgetPress()
          releasedAt.current = null
        }}
        onPointerUp={() => {
          if (!pointerHeld.current) return
          pointerHeld.current = false
          releasedAt.current = performance.now()
        }}
        onPointerLeave={() => {
          pointerHeld.current = false
        }}
        onBlur={forgetPress}
        onClick={() => {
          const released = releasedAt.current
          releasedAt.current = null
          if (released !== null && performance.now() - released < RELEASE_CLICK_MS) return
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
