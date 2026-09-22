/**
 * Keyboard plumbing for the whole application.
 *
 * One rule governs everything here, and it is not negotiable: a single-key
 * shortcut is dead while a text field has focus. Typing "a rain-slicked tram
 * stop" must not toggle expert mode, star something and change desks three
 * times. `Ctrl/Cmd+Enter` is the one deliberate exception, because running the
 * thing you are writing is the one command you want from inside the field.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import type {
  FocusEvent as ReactFocusEvent,
  KeyboardEvent as ReactKeyboardEvent,
  MouseEvent as ReactMouseEvent,
  PointerEvent as ReactPointerEvent,
} from 'react'

// ---------------------------------------------------------------------------
// Typing guard
// ---------------------------------------------------------------------------

const FIELD = /^(INPUT|TEXTAREA|SELECT)$/

/** True when the event came from somewhere a person is writing. */
export function isTyping(e: Event): boolean {
  const t = e.target as HTMLElement | null
  if (!t) return false
  if (t.isContentEditable) return true
  if (FIELD.test(t.tagName)) return true
  // A field inside an open shadow root or a composed event still reports the
  // host as `target`; `composedPath` tells the truth.
  const path = typeof e.composedPath === 'function' ? e.composedPath() : []
  for (const node of path) {
    const el = node as HTMLElement
    if (el && el.tagName && FIELD.test(el.tagName)) return true
    if (el && el.isContentEditable) return true
  }
  return false
}

/** True while an IME is composing — never steal a key mid-composition. */
function isComposing(e: KeyboardEvent): boolean {
  return e.isComposing || e.keyCode === 229
}

// ---------------------------------------------------------------------------
// useHotkeys
// ---------------------------------------------------------------------------

export type Hotkey = {
  /** A single character (`'e'`), or a named key (`'Escape'`, `'Enter'`). Case-insensitive. */
  key: string
  /** Require Ctrl on Windows and Linux, Cmd on a Mac. */
  mod?: boolean
  /** Require Shift. Omitted means "must not be held", except for `?` and other shifted glyphs. */
  shift?: boolean
  alt?: boolean
  /** Fire even while a field has focus. Reserve this for Ctrl/Cmd combinations. */
  inFields?: boolean
  /** Leave the browser's own behaviour alone. Default is to prevent it. */
  passive?: boolean
  run: (e: KeyboardEvent) => void
}

/**
 * Bind a set of keys for as long as the component is mounted.
 *
 * The array may be rebuilt on every render; only the latest is ever consulted,
 * and the listener is attached exactly once.
 */
export function useHotkeys(keys: readonly Hotkey[], enabled = true): void {
  const latest = useRef(keys)
  useEffect(() => {
    latest.current = keys
  })

  useEffect(() => {
    if (!enabled) return
    const onKey = (e: KeyboardEvent) => {
      if (isComposing(e)) return
      // Someone nearer the key already acted on it. A canvas, a player or a
      // dialog that called preventDefault has claimed the key for its own
      // surface, and a global shortcut firing on top of that is how pressing
      // `e` inside the mask brush also collapsed the expert column and re laid
      // out the canvas underneath the stroke. The player and the archive open
      // their own window handlers with exactly this test.
      if (e.defaultPrevented) return
      const typing = isTyping(e)
      const mod = e.ctrlKey || e.metaKey
      for (const k of latest.current) {
        if (k.key.length === 1) {
          if (e.key.toLowerCase() !== k.key.toLowerCase()) continue
        } else if (e.key !== k.key) continue
        if (Boolean(k.mod) !== mod) continue
        if (k.shift !== undefined && Boolean(k.shift) !== e.shiftKey) continue
        if (Boolean(k.alt) !== e.altKey) continue
        if (typing && !k.inFields) continue
        if (!k.passive) e.preventDefault()
        k.run(e)
        return
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [enabled])
}

// ---------------------------------------------------------------------------
// Chords
// ---------------------------------------------------------------------------

/**
 * Two-key chords in the `g p` style, with a 300 ms window.
 *
 * Vim and Gmail both taught this, so a good number of people will try it. It
 * costs one timer.
 */
export function useChord(
  lead: string,
  map: Record<string, () => void>,
  window_ms = 300,
  enabled = true,
): void {
  const latest = useRef(map)
  useEffect(() => {
    latest.current = map
  })

  useEffect(() => {
    if (!enabled) return
    let armed = 0
    const onKey = (e: KeyboardEvent) => {
      if (isComposing(e) || isTyping(e) || e.ctrlKey || e.metaKey || e.altKey) return
      if (e.defaultPrevented) return
      const key = e.key.toLowerCase()
      if (armed && Date.now() - armed < window_ms) {
        armed = 0
        const fn = latest.current[key]
        if (fn) {
          e.preventDefault()
          fn()
        }
        return
      }
      armed = key === lead.toLowerCase() ? Date.now() : 0
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [lead, window_ms, enabled])
}

// ---------------------------------------------------------------------------
// Hold to confirm
// ---------------------------------------------------------------------------

export type HoldHandlers = {
  /** Spread onto the button. */
  bind: {
    onPointerDown: (e: ReactPointerEvent) => void
    onPointerUp: (e: ReactPointerEvent) => void
    onPointerLeave: (e: ReactPointerEvent) => void
    onPointerCancel: (e: ReactPointerEvent) => void
    onKeyDown: (e: ReactKeyboardEvent) => void
    onKeyUp: (e: ReactKeyboardEvent) => void
    onClick: (e: ReactMouseEvent) => void
    onBlur: (e: ReactFocusEvent) => void
  }
  /** 0 → 1. Drive the burgundy wipe with it. */
  progress: number
  holding: boolean
  /**
   * True for a few seconds after a click with no pointer or key behind it,
   * which is how assistive technology presses a button. A second such click
   * in that time confirms. Say so on the button while it is set: "Press again
   * to stop".
   */
  armed: boolean
}

/** How long an armed confirmation waits for its second press. */
const ARMED_MS = 3000

/**
 * Destroying four minutes of GPU time should cost more than a mis-click, so
 * stopping a job is a 600 ms hold with a wipe you can watch.
 *
 * Enter and Space hold too, and their auto-repeat is ignored: a key still down
 * after the hold has confirmed would otherwise start a second hold and confirm
 * again. Both keys have their default prevented on the way down and on the way
 * up, so neither turns into a click.
 *
 * A hold also ends when the button loses focus. A key held down while focus
 * moves away, to another window or another control, sends its key up
 * somewhere else, and without this the wipe would run on and fire a stop
 * nobody was still holding for.
 *
 * A screen reader presses a button with a bare click and no key or pointer
 * events at all, so it could never hold. Such a click arms the confirmation
 * instead and a second one within three seconds fires it: still two
 * deliberate acts, while a mouse click still does nothing without the hold.
 * The same fallback `RunButton` has.
 */
export function useHoldToConfirm(onConfirm: () => void, ms = 600): HoldHandlers {
  const [progress, setProgress] = useState(0)
  const [holding, setHolding] = useState(false)
  const [armed, setArmed] = useState(false)
  const raf = useRef(0)
  const started = useRef(0)
  const done = useRef(false)
  const fire = useRef(onConfirm)
  useEffect(() => {
    fire.current = onConfirm
  })

  const stop = useCallback(() => {
    cancelAnimationFrame(raf.current)
    raf.current = 0
    started.current = 0
    setHolding(false)
    setProgress(0)
  }, [])

  const begin = useCallback(() => {
    if (started.current) return
    done.current = false
    started.current = performance.now()
    setHolding(true)
    const tick = () => {
      const p = Math.min(1, (performance.now() - started.current) / ms)
      setProgress(p)
      if (p >= 1) {
        if (!done.current) {
          done.current = true
          // A press armed earlier is spent by the hold, not left to fire again.
          setArmed(false)
          fire.current()
        }
        stop()
        return
      }
      raf.current = requestAnimationFrame(tick)
    }
    raf.current = requestAnimationFrame(tick)
  }, [ms, stop])

  useEffect(() => () => cancelAnimationFrame(raf.current), [])

  // An armed confirmation lapses on its own.
  useEffect(() => {
    if (!armed) return
    const t = window.setTimeout(() => setArmed(false), ARMED_MS)
    return () => window.clearTimeout(t)
  }, [armed])

  const isActivation = (e: ReactKeyboardEvent) => e.key === ' ' || e.key === 'Enter'

  return {
    holding,
    progress,
    armed,
    bind: {
      onPointerDown: (e) => {
        e.preventDefault()
        ;(e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId)
        begin()
      },
      onPointerUp: stop,
      onPointerLeave: stop,
      onPointerCancel: stop,
      onKeyDown: (e) => {
        if (!isActivation(e)) return
        e.preventDefault()
        if (e.repeat) return
        begin()
      },
      onKeyUp: (e) => {
        if (!isActivation(e)) return
        e.preventDefault()
        stop()
      },
      onClick: (e) => {
        // A pointer click has a detail of one or more and is hold only. Zero
        // is a click with no pointer behind it: assistive technology.
        if (e.detail !== 0) return
        if (armed) {
          setArmed(false)
          fire.current()
        } else {
          setArmed(true)
        }
      },
      onBlur: stop,
    },
  }
}
