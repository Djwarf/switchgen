/**
 * The reel's small type. Nothing here knows about shots, graphs or ComfyUI.
 *
 * A storyboard is a grid of repeated parts, so the parts are written once. The
 * house rules they carry: square focus rings, tabular figures, uppercase
 * kickers at 0.625rem, hairline rules between rows and a heavy burgundy rule
 * under a section head.
 *
 * ON PUNCTUATION. This desk sets clauses with rules, columns and full stops.
 * Where a line wants a break it takes a colon or it becomes two sentences.
 */
import type { ReactNode } from 'react'

import { RING } from '../type'
export { RING }

import { clamp } from '../../lib/num'
export { clamp }

/** `4 min 10 s`, `42 s`, `0.9 s`. Never a bare decimal minute. */
export function duration(ms: number): string {
  const s = Math.max(0, ms) / 1000
  if (s < 10) return `${s.toFixed(1)} s`
  // Rounded whole first, so 59.6 s reads 1 min, and 119.6 s 2 min, never "60 s".
  const t = Math.round(s)
  if (t < 60) return `${t} s`
  const m = Math.floor(t / 60)
  const rest = t - m * 60
  if (rest === 0) return `${m} min`
  return `${m} min ${rest} s`
}

/** `5.0 s` of screen time, from a frame count and a rate. */
export function seconds(frames: number, fps: number): string {
  if (!fps) return `${frames} frames`
  return `${(frames / fps).toFixed(1)} s`
}

/** `1,284`. Grouped, so a long reel's frame count stays readable. */
export function grouped(n: number): string {
  return Math.round(n).toLocaleString('en-GB')
}

/** `1280 × 704`. */
export function times(w: number, h: number): string {
  return `${w} × ${h}`
}

// ---------------------------------------------------------------------------
// Type
// ---------------------------------------------------------------------------

export function Kicker({ children, tone = 'burgundy', className = '' }: {
  children: ReactNode
  tone?: 'burgundy' | 'quiet' | 'ink'
  className?: string
}) {
  const colour =
    tone === 'quiet' ? 'text-grey-500' : tone === 'ink' ? 'text-ink' : 'text-burgundy-900'
  return (
    <span
      className={`block text-[0.625rem] font-semibold uppercase tracking-[0.18em] ${colour} ${className}`}
    >
      {children}
    </span>
  )
}

/** A section head: kicker over a heavy rule, with an optional right-hand figure. */
export function Head({ title, figure, note }: { title: string; figure?: ReactNode; note?: string }) {
  return (
    <div className="mb-3 border-b-2 border-burgundy-900 pb-1.5">
      <div className="flex items-baseline justify-between gap-3">
        <Kicker>{title}</Kicker>
        {figure ? <span className="text-caption tabular-nums text-grey-700">{figure}</span> : null}
      </div>
      {note ? <p className="mt-1 text-caption italic text-grey-500">{note}</p> : null}
    </div>
  )
}

/** Label above, control below, hint in the same baseline as the label. */
export function Field({ label, hint, id, children }: {
  label: string
  hint?: ReactNode
  id?: string
  children: ReactNode
}) {
  return (
    <div className="mb-3">
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <label htmlFor={id} className="text-[0.625rem] font-semibold uppercase tracking-[0.16em] text-grey-700">
          {label}
        </label>
        {hint ? <span className="text-caption italic tabular-nums text-grey-500">{hint}</span> : null}
      </div>
      {children}
    </div>
  )
}

/** A leader line: label, dotted rule, figure. The classic tabular device. */
export function Leader({ label, value }: { label: ReactNode; value: ReactNode }) {
  return (
    <div className="sg-leader py-0.5 text-caption">
      <span className="text-grey-700">{label}</span>
      <span className="sg-leader-dots" aria-hidden="true" />
      <span className="tabular-nums text-ink">{value}</span>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

export function NumberField({ label, value, min, max, step, commit, id, italic, disabled }: {
  label: string
  value: number
  min: number
  max: number
  step: number
  commit: (n: number) => void
  id?: string
  italic?: boolean
  disabled?: boolean
}) {
  return (
    <input
      id={id}
      type="number"
      aria-label={label}
      className={`field tabular-nums disabled:cursor-not-allowed disabled:text-grey-500 ${italic ? 'italic text-grey-500' : ''}`}
      value={Number.isFinite(value) ? value : ''}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      onChange={(e) => {
        const n = Number(e.target.value)
        if (Number.isFinite(n)) commit(n)
      }}
    />
  )
}

export type ChipOption<T> = { value: T; label: string; title?: string; disabled?: boolean }

export function Chips<T extends string | number>({ ariaLabel, value, options, onChange, disabled }: {
  ariaLabel: string
  value: T
  options: readonly ChipOption<T>[]
  onChange: (v: T) => void
  /** Every chip at once, for a setting that must not move while the queue runs. */
  disabled?: boolean
}) {
  return (
    <div role="group" aria-label={ariaLabel} className="flex flex-wrap gap-1">
      {options.map((o) => {
        const on = o.value === value
        const off = disabled || o.disabled
        return (
          <button
            key={String(o.value)}
            type="button"
            title={o.title}
            disabled={off}
            aria-pressed={on}
            onClick={() => onChange(o.value)}
            className={`${RING} border px-2 py-1 text-caption tabular-nums transition-colors ${
              on
                ? 'border-ink bg-ink text-newsprint'
                : 'border-grey-300 text-grey-700 hover:border-ink hover:bg-newsprint-aged'
            } ${off ? 'cursor-not-allowed opacity-40' : ''}`}
          >
            {o.label}
          </button>
        )
      })}
    </div>
  )
}

/** A quiet bordered action. The desk's second voice, under the burgundy press. */
export function Quiet({ children, onClick, disabled, title, danger, className = '' }: {
  children: ReactNode
  onClick: () => void
  disabled?: boolean
  title?: string
  danger?: boolean
  className?: string
}) {
  return (
    <button
      type="button"
      title={title}
      disabled={disabled}
      onClick={onClick}
      className={`${RING} border px-2 py-1 text-[0.625rem] font-semibold uppercase tracking-[0.14em] transition-colors ${
        danger
          ? 'border-grey-300 text-error hover:border-error hover:bg-newsprint-aged'
          : 'border-grey-300 text-grey-700 hover:border-ink hover:bg-newsprint-aged'
      } disabled:cursor-not-allowed disabled:border-grey-200 disabled:text-grey-400 disabled:hover:bg-transparent ${className}`}
    >
      {children}
    </button>
  )
}

/** An underlined text action, for anything that is really a link in disguise. */
export function Link({ children, onClick, disabled }: {
  children: ReactNode
  onClick: () => void
  disabled?: boolean
}) {
  return (
    <button type="button" className={`sg-link ${RING}`} onClick={onClick} disabled={disabled}>
      {children}
    </button>
  )
}

/** The status pip. Breathes while live, flat otherwise. */
export function Mark({ state }: { state: 'idle' | 'live' | 'ok' | 'off' }) {
  const cls =
    state === 'live' ? 'sg-mark-live' : state === 'ok' ? 'sg-mark-ok' : state === 'off' ? 'sg-mark-off' : 'sg-mark-idle'
  return <span className={`sg-mark ${cls}`} aria-hidden="true" />
}

/** A hairline progress rail. `value` and `max` in any unit. */
export function Rail({ value, max }: { value: number; max: number }) {
  const pct = max > 0 ? clamp((value / max) * 100, 0, 100) : 0
  return (
    <div className="sg-progress" role="presentation">
      <div className="sg-progress-fill" style={{ width: `${pct}%` }} />
    </div>
  )
}

/** A short warning, set as a correction note rather than an alert box. */
export function Caution({ children }: { children: ReactNode }) {
  return (
    <p className="border-l-2 border-warning bg-paper-warning px-2 py-1 text-caption text-ink-warning">{children}</p>
  )
}
