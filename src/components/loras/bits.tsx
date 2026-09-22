/**
 * The LoRA rack's small type.
 *
 * Nothing here knows what a LoRA is. It is the house furniture the three
 * components below share: the square focus ring, the uppercase kicker at
 * 0.625rem, the hairline rule, tabular figures, and one slider that can run
 * through zero because a size slider has to.
 *
 * ON PUNCTUATION. Clauses are set with rules, colons and full stops. A dash
 * joining two clauses is a layout failure, so there are none.
 */
import { useRef, type KeyboardEvent as ReactKeyboardEvent, type ReactNode } from 'react'

/** The house focus ring: square, burgundy, offset. Never rounded. */
export const RING =
  'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900'

export const clamp = (n: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, n))

/** Round to the slider's own step, so 0.7000000000000001 never reaches the UI. */
const toStep = (n: number, step: number) => Math.round(n / step) * step

export function Kicker({
  children,
  tone = 'quiet',
  className = '',
}: {
  children: ReactNode
  tone?: 'burgundy' | 'quiet'
  className?: string
}) {
  const colour = tone === 'burgundy' ? 'text-burgundy-900' : 'text-grey-700'
  return (
    <span
      className={`block text-[0.625rem] font-semibold uppercase tracking-[0.18em] ${colour} ${className}`}
    >
      {children}
    </span>
  )
}

/**
 * A compatibility badge. Three states, three weights of emphasis: a match is
 * quiet because it is the expected case, an untested crossing is bracketed,
 * and a wrong base is the only thing on this rack allowed to shout.
 */
export function Badge({
  tone,
  children,
  title,
}: {
  tone: 'match' | 'untested' | 'mismatch' | 'plain'
  children: ReactNode
  title?: string
}) {
  const style =
    tone === 'mismatch'
      ? 'border-error text-error'
      : tone === 'untested'
        ? 'border-warning text-warning'
        : tone === 'match'
          ? 'border-grey-300 text-grey-700'
          : 'border-grey-300 text-grey-500'
  return (
    <span
      title={title}
      className={`inline-block border px-1 py-px text-[0.5625rem] font-semibold uppercase tracking-[0.14em] ${style}`}
    >
      {children}
    </span>
  )
}

/** A small square button in the house voice: hairline box, uppercase label. */
export function Tap({
  children,
  onClick,
  disabled,
  label,
  active,
  className = '',
}: {
  children: ReactNode
  onClick: () => void
  disabled?: boolean
  /** Accessible name, when the visible label is a glyph. */
  label?: string
  active?: boolean
  className?: string
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      aria-pressed={active === undefined ? undefined : active}
      className={`border px-1.5 py-0.5 text-[0.5625rem] font-semibold uppercase tracking-[0.14em] ${RING} ${
        disabled
          ? 'cursor-not-allowed border-grey-200 text-grey-400'
          : active
            ? 'cursor-pointer border-burgundy-900 bg-burgundy-900 text-newsprint'
            : 'cursor-pointer border-grey-300 text-ink hover:border-burgundy-900 hover:text-burgundy-900'
      } ${className}`}
    >
      {children}
    </button>
  )
}

/**
 * The strength rail.
 *
 * It carries two marks the number field cannot: the author's recommended
 * value, and zero. Zero matters because a slider LoRA runs through it, and a
 * user dragging past zero is changing the direction of the effect rather than
 * turning it down. The rail is a div rather than an input so the marks can sit
 * on the track itself, and it carries the full slider role and keys.
 */
export function Rail({
  value,
  min,
  max,
  step,
  recommended,
  onChange,
  label,
  disabled,
}: {
  value: number
  min: number
  max: number
  step: number
  recommended?: number
  onChange: (n: number) => void
  label: string
  disabled?: boolean
}) {
  const rail = useRef<HTMLDivElement | null>(null)
  const span = max - min || 1
  const at = (n: number) => clamp((n - min) / span, 0, 1) * 100
  const zero = min < 0 && max > 0 ? at(0) : null

  const commit = (n: number) => {
    if (disabled) return
    onChange(clamp(Math.round(toStep(n, step) * 100) / 100, min, max))
  }

  const fromPointer = (clientX: number) => {
    const rect = rail.current?.getBoundingClientRect()
    if (!rect || rect.width === 0) return
    commit(min + ((clientX - rect.left) / rect.width) * span)
  }

  const onKey = (e: ReactKeyboardEvent) => {
    const big = step * 5
    if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') {
      e.preventDefault()
      commit(value - (e.shiftKey ? big : step))
    } else if (e.key === 'ArrowRight' || e.key === 'ArrowUp') {
      e.preventDefault()
      commit(value + (e.shiftKey ? big : step))
    } else if (e.key === 'Home') {
      e.preventDefault()
      commit(min)
    } else if (e.key === 'End') {
      e.preventDefault()
      commit(max)
    } else if (e.key === '0') {
      e.preventDefault()
      commit(0)
    }
  }

  // The filled part runs from zero on a two way rail, so a negative strength
  // reads as a direction rather than as an unusually short bar.
  const origin = zero === null ? 0 : zero
  const here = at(value)
  const left = Math.min(origin, here)
  const width = Math.abs(here - origin)

  return (
    <div
      ref={rail}
      role="slider"
      tabIndex={disabled ? -1 : 0}
      aria-label={label}
      aria-valuemin={min}
      aria-valuemax={max}
      aria-valuenow={value}
      aria-valuetext={value.toFixed(2)}
      aria-disabled={disabled || undefined}
      onKeyDown={onKey}
      onPointerDown={(e) => {
        if (disabled) return
        e.currentTarget.setPointerCapture(e.pointerId)
        fromPointer(e.clientX)
      }}
      onPointerMove={(e) => {
        if (e.buttons) fromPointer(e.clientX)
      }}
      className={`relative h-4 touch-none select-none ${RING} ${
        disabled ? 'cursor-not-allowed opacity-40' : 'cursor-pointer'
      }`}
    >
      <span className="absolute inset-x-0 top-1/2 block h-px -translate-y-1/2 bg-grey-300" />
      <span
        className="absolute top-1/2 block h-[2px] -translate-y-1/2 bg-burgundy-900"
        style={{ left: `${left}%`, width: `${width}%` }}
      />
      {zero !== null ? (
        <span
          aria-hidden
          className="absolute top-1/2 block h-2.5 w-px -translate-x-1/2 -translate-y-1/2 bg-grey-400"
          style={{ left: `${zero}%` }}
        />
      ) : null}
      {recommended !== undefined ? (
        <span
          aria-hidden
          title={`Author's recommendation: ${recommended}`}
          className="absolute top-1/2 block h-2 w-px -translate-x-1/2 -translate-y-1/2 bg-grey-400"
          style={{ left: `${at(recommended)}%` }}
        />
      ) : null}
      <span
        aria-hidden
        className="absolute top-1/2 block h-3.5 w-[2px] -translate-x-1/2 -translate-y-1/2 bg-burgundy-900"
        style={{ left: `${here}%` }}
      />
    </div>
  )
}

/** `12%`, `0%`. A bar that reports a download, not a generation. */
export function Meter({ pct, label }: { pct: number; label: string }) {
  return (
    <div className="mt-1">
      <div
        className="relative h-[3px] bg-grey-200"
        role="progressbar"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(clamp(pct, 0, 1) * 100)}
      >
        <span
          className="absolute inset-y-0 left-0 block bg-burgundy-900"
          style={{ width: `${clamp(pct, 0, 1) * 100}%` }}
        />
      </div>
    </div>
  )
}
