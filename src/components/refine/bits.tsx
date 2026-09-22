/**
 * Small type and controls for the region refine surface.
 *
 * Deliberately local rather than imported from the reel's bits module: the
 * refine surface ships inside the Pictures desk, and a dependency pointing from
 * one desk into another desk's internals is a coupling nobody wants to unpick
 * later. Only the parts this folder actually uses are written here.
 *
 * House rules carried: square focus rings, no radius, no shadow, two rule
 * weights (1px grey-300, 2px burgundy-900), uppercase kickers at 0.625rem,
 * tabular figures on anything that changes.
 *
 * ON PUNCTUATION. No dash joins two clauses anywhere in this folder. A line
 * that wants a break takes a colon, a rule, a column or a full stop.
 */
import type { ReactNode } from 'react'

/** The house focus ring: square, burgundy, offset. */
export const RING =
  'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900'

export const clamp = (n: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, n))

/** `1,284`. Latent cell counts run into five figures, so they get grouped. */
export const grouped = (n: number) => Math.round(n).toLocaleString('en-GB')

/** `832 × 1216`. */
export const times = (w: number, h: number) => `${Math.round(w)} × ${Math.round(h)}`

/** `11.4x`, `1.0x`. The detail multiplier, always with one decimal. */
export const multiple = (n: number) => `${n.toFixed(1)}x`

// ---------------------------------------------------------------------------
// Type
// ---------------------------------------------------------------------------

export function Kicker({
  children,
  tone = 'quiet',
  className = '',
}: {
  children: ReactNode
  tone?: 'burgundy' | 'quiet' | 'ink'
  className?: string
}) {
  const colour =
    tone === 'burgundy' ? 'text-burgundy-900' : tone === 'ink' ? 'text-ink' : 'text-grey-700'
  return (
    <span
      className={`text-overline font-semibold uppercase tracking-[0.18em] ${colour} ${className}`}
    >
      {children}
    </span>
  )
}

/** A section head: kicker over a heavy rule, with an optional right hand figure. */
export function Head({
  title,
  figure,
  note,
}: {
  title: string
  figure?: ReactNode
  note?: ReactNode
}) {
  return (
    <div className="mb-3 border-b-2 border-burgundy-900 pb-1.5">
      <div className="flex items-baseline justify-between gap-3">
        <Kicker tone="burgundy">{title}</Kicker>
        {figure ? <span className="text-caption tabular-nums text-grey-700">{figure}</span> : null}
      </div>
      {note ? <p className="mt-1 text-caption italic text-grey-500">{note}</p> : null}
    </div>
  )
}

/** Label above, control below, hint on the label's baseline. */
export function Field({
  label,
  hint,
  id,
  children,
}: {
  label: string
  hint?: ReactNode
  id?: string
  children: ReactNode
}) {
  return (
    <div className="mb-4">
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <label
          htmlFor={id}
          className="text-overline font-semibold uppercase tracking-[0.18em] text-grey-700"
        >
          {label}
        </label>
        {hint ? (
          <span className="text-caption italic tabular-nums text-grey-500">{hint}</span>
        ) : null}
      </div>
      {children}
    </div>
  )
}

/** A leader line: label, dotted rule, figure. The expert margin is made of these. */
export function Leader({ label, value }: { label: ReactNode; value: ReactNode }) {
  return (
    <div className="sg-leader py-0.5 text-caption">
      <span className="text-grey-700">{label}</span>
      <span className="sg-leader-dots" aria-hidden="true" />
      <span className="tabular-nums text-ink">{value}</span>
    </div>
  )
}

/** An explanatory line under a control. Italic, grey, never a second heading. */
export function Note({ children }: { children: ReactNode }) {
  return <p className="mt-1.5 text-caption italic leading-snug text-grey-700">{children}</p>
}

/** A short caution, set as a printed note rather than an alert box. */
export function Caution({ children }: { children: ReactNode }) {
  return (
    <p className="border-l-2 border-warning bg-[#fffbeb] px-2 py-1 text-caption leading-snug text-[#78350f]">
      {children}
    </p>
  )
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

export type ChipOption<T> = { value: T; label: string; title?: string; disabled?: boolean }

export function Chips<T extends string | number>({
  ariaLabel,
  value,
  options,
  onChange,
  grow,
}: {
  ariaLabel: string
  value: T
  options: readonly ChipOption<T>[]
  onChange: (v: T) => void
  /** Fill the row, for a set of four equal buckets. */
  grow?: boolean
}) {
  return (
    <div role="group" aria-label={ariaLabel} className={grow ? 'flex' : 'flex flex-wrap gap-1'}>
      {options.map((o) => {
        const on = o.value === value
        const shared = `${RING} px-2 py-1 text-caption tabular-nums transition-colors ${
          o.disabled ? 'cursor-not-allowed opacity-40' : 'cursor-pointer'
        }`
        const skin = on
          ? 'bg-ink text-newsprint border-ink'
          : 'text-grey-700 hover:text-ink hover:bg-newsprint-aged border-grey-300'
        return (
          <button
            key={String(o.value)}
            type="button"
            title={o.title}
            disabled={o.disabled}
            aria-pressed={on}
            onClick={() => onChange(o.value)}
            className={
              grow
                ? `${shared} ${skin} flex-1 border-y border-r first:border-l`
                : `${shared} ${skin} border`
            }
          >
            {o.label}
          </button>
        )
      })}
    </div>
  )
}

/** A quiet bordered action. The second voice, under the burgundy press. */
export function Quiet({
  children,
  onClick,
  disabled,
  title,
  pressed,
  danger,
  className = '',
}: {
  children: ReactNode
  onClick: () => void
  disabled?: boolean
  title?: string
  pressed?: boolean
  danger?: boolean
  className?: string
}) {
  const skin = pressed
    ? 'border-ink bg-ink text-newsprint'
    : danger
      ? 'border-grey-300 text-error hover:border-error hover:bg-newsprint-aged'
      : 'border-grey-300 text-grey-700 hover:border-ink hover:bg-newsprint-aged'
  return (
    <button
      type="button"
      title={title}
      disabled={disabled}
      aria-pressed={pressed}
      onClick={onClick}
      className={`sg-tap ${RING} cursor-pointer border px-2 py-1 text-overline font-semibold uppercase tracking-[0.14em] transition-colors ${skin} disabled:cursor-not-allowed disabled:border-grey-200 disabled:text-grey-400 disabled:hover:bg-transparent ${className}`}
    >
      {children}
    </button>
  )
}

/** An underlined text action, for anything that is really a link in disguise. */
export function Link({
  children,
  onClick,
  disabled,
  title,
}: {
  children: ReactNode
  onClick: () => void
  disabled?: boolean
  title?: string
}) {
  return (
    <button type="button" title={title} className={`sg-link ${RING}`} onClick={onClick} disabled={disabled}>
      {children}
    </button>
  )
}

/** A hairline progress rail. */
export function Rail({ value }: { value: number }) {
  return (
    <div className="sg-progress" role="presentation">
      <div className="sg-progress-fill" style={{ width: `${clamp(value, 0, 1) * 100}%` }} />
    </div>
  )
}
