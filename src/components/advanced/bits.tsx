/**
 * The advanced panel's small type.
 *
 * Local rather than imported from ../refine/bits or ../loras/bits on purpose.
 * Those two folders are the internals of other surfaces, and a dependency
 * pointing from this panel into them is a coupling nobody wants to unpick when
 * one of them is next rewritten. Only the parts this folder uses are here.
 *
 * House rules carried from the branding guide: square focus rings, no radius,
 * no shadow, exactly two rule weights (1px grey-300 and 2px burgundy-900),
 * uppercase kickers at 0.625rem, tabular figures on anything that changes,
 * burgundy rationed to section heads and the one link voice.
 *
 * THE OVERRIDE MARK is the piece of furniture this panel exists for. The
 * default view is calm because every number was decided; this panel is where a
 * number can be taken back by hand, and the moment it is, the row has to say so
 * and offer the decided value back in one press. {@link Row} does that, and
 * every control in this folder is wrapped in one.
 *
 * ON PUNCTUATION. No dash joins two clauses anywhere in this folder. A line
 * that wants a break takes a colon, a rule, a column or a full stop.
 */
import type { ReactNode } from 'react'

/** The house focus ring: square, burgundy, offset. Never rounded. */
export const RING =
  'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900'

export const clamp = (n: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, n))

export const round2 = (n: number) => Math.round(n * 100) / 100

/** Latent sizes are multiples of 8; the desk has always snapped to 16. */
export const snap16 = (n: number) => Math.max(16, Math.round(n / 16) * 16)

const THIN = ' '

/** `832 × 1216`, with thin spaces so the figures read as one measurement. */
export const times = (w: number, h: number) => `${Math.round(w)}${THIN}×${THIN}${Math.round(h)}`

/** `1.14x`. Every ratio printed in this panel comes from a measured table. */
export const ratio = (n: number) => `${n.toFixed(2)}x`

// ---------------------------------------------------------------------------
// Type
// ---------------------------------------------------------------------------

export function Kicker({
  children,
  tone = 'quiet',
  className = '',
}: {
  children: ReactNode
  tone?: 'burgundy' | 'quiet' | 'warning' | 'error'
  className?: string
}) {
  const colour =
    tone === 'burgundy'
      ? 'text-burgundy-900'
      : tone === 'warning'
        ? 'text-warning'
        : tone === 'error'
          ? 'text-error'
          : 'text-grey-700'
  return (
    <span
      className={`block text-overline font-semibold uppercase tracking-[0.18em] ${colour} ${className}`}
    >
      {children}
    </span>
  )
}

/**
 * A section head: burgundy kicker over the heavy rule, an optional figure on
 * the right, an optional standfirst under both.
 */
export function Head({
  title,
  figure,
  note,
  action,
}: {
  title: string
  figure?: ReactNode
  note?: ReactNode
  action?: ReactNode
}) {
  return (
    <div className="mb-3 border-b-2 border-burgundy-900 pb-1.5">
      <div className="flex items-baseline justify-between gap-3">
        <Kicker tone="burgundy">{title}</Kicker>
        <span className="flex shrink-0 items-baseline gap-2 text-caption tabular-nums text-grey-700">
          {figure}
          {action}
        </span>
      </div>
      {note ? <p className="mt-1 text-caption italic leading-snug text-grey-500">{note}</p> : null}
    </div>
  )
}

/** An explanatory line under a control. Italic, grey, never a second heading. */
export function Note({ children }: { children: ReactNode }) {
  return <p className="mt-1 text-caption italic leading-snug text-grey-700">{children}</p>
}

/** A short caution, set as a printed note rather than an alert box. */
export function Caution({ children }: { children: ReactNode }) {
  return (
    <p className="border-l-2 border-warning bg-[#fffbeb] px-2 py-1 text-caption leading-snug text-[#78350f]">
      {children}
    </p>
  )
}

/** The same, for something that is simply broken rather than merely costly. */
export function Fault({ children }: { children: ReactNode }) {
  return (
    <p className="border-l-2 border-error bg-[#fef2f2] px-2 py-1 text-caption leading-snug text-[#7f1d1d]">
      {children}
    </p>
  )
}

/** Newspaper index line: label, leader dots, figure. */
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
// The link and button voices
// ---------------------------------------------------------------------------

/** An underlined text action. Anything that is really a link in disguise. */
export function Link({
  children,
  onClick,
  disabled,
  title,
  className = '',
}: {
  children: ReactNode
  onClick: () => void
  disabled?: boolean
  title?: string
  className?: string
}) {
  return (
    <button
      type="button"
      title={title}
      disabled={disabled}
      onClick={onClick}
      className={`sg-link ${RING} ${className}`}
    >
      {children}
    </button>
  )
}

/** A quiet bordered action. The second voice, under the burgundy press. */
export function Quiet({
  children,
  onClick,
  disabled,
  title,
  pressed,
  className = '',
}: {
  children: ReactNode
  onClick: () => void
  disabled?: boolean
  title?: string
  pressed?: boolean
  className?: string
}) {
  const skin = pressed
    ? 'border-ink bg-ink text-newsprint'
    : 'border-grey-300 text-grey-700 hover:border-ink hover:bg-newsprint-aged'
  return (
    <button
      type="button"
      title={title}
      disabled={disabled}
      aria-pressed={pressed}
      onClick={onClick}
      className={`${RING} cursor-pointer border px-2 py-1 text-overline font-semibold uppercase tracking-[0.14em] transition-colors ${skin} disabled:cursor-not-allowed disabled:border-grey-200 disabled:text-grey-400 ${className}`}
    >
      {children}
    </button>
  )
}

// ---------------------------------------------------------------------------
// The override row
// ---------------------------------------------------------------------------

/**
 * One labelled control, with the decided value attached to it.
 *
 * Three states, and the row looks different in each:
 *
 *   decided     the recipe chose it, the figure is printed as the hint
 *   overridden  a burgundy mark, the decided value named, and one press back
 *   fixed       no decided value exists, so the row is simply a control
 *
 * The mark is a 2px burgundy bar down the left edge rather than a badge. It is
 * the only thing in this panel that moves the text off the grid, which is what
 * makes a hand set row findable in a column of forty.
 */
export function Row({
  label,
  hint,
  id,
  overridden,
  decided,
  onRestore,
  children,
  note,
}: {
  label: string
  hint?: ReactNode
  id?: string
  /** True when the user has taken this value by hand. */
  overridden?: boolean
  /** What the recipe decided, printed when the row is overridden. */
  decided?: ReactNode
  onRestore?: () => void
  children: ReactNode
  note?: ReactNode
}) {
  return (
    <div
      className={`mb-3 border-b border-grey-300 pb-3 ${
        overridden ? 'border-l-2 border-l-burgundy-900 pl-2.5' : ''
      }`}
    >
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <label
          htmlFor={id}
          className="text-overline font-semibold uppercase tracking-[0.18em] text-grey-700"
        >
          {label}
        </label>
        {hint ? (
          <span className="shrink-0 text-caption italic tabular-nums text-grey-500">{hint}</span>
        ) : null}
      </div>
      {children}
      {overridden && onRestore ? (
        <p className="mt-1 flex flex-wrap items-baseline gap-x-2 text-caption text-grey-700">
          <span className="text-overline font-semibold uppercase tracking-[0.14em] text-burgundy-900">
            Set by hand
          </span>
          <span className="tabular-nums text-grey-500">
            {decided === undefined || decided === null || decided === ''
              ? 'The recipe left this alone.'
              : <>Decided: {decided}.</>}
          </span>
          <Link onClick={onRestore}>Put it back</Link>
        </p>
      ) : null}
      {note ? <Note>{note}</Note> : null}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

export function NumberInput({
  id,
  value,
  min,
  max,
  step,
  onChange,
  round,
}: {
  id?: string
  value: number
  min: number
  max: number
  step?: number
  onChange: (v: number) => void
  /** 'int' rounds to a whole number, 'two' to two places. */
  round?: 'int' | 'two'
}) {
  return (
    <input
      id={id}
      type="number"
      className="field tabular-nums"
      value={value}
      min={min}
      max={max}
      step={step}
      onChange={(e) => {
        const v = Number(e.target.value)
        if (!Number.isFinite(v)) return
        const c = clamp(v, min, max)
        onChange(round === 'int' ? Math.round(c) : round === 'two' ? round2(c) : c)
      }}
    />
  )
}

/**
 * A select over a list ComfyUI reported, and a plain text field when it did
 * not report one.
 *
 * The fallback matters. The sampler list comes from /object_info, which is a
 * live call; if it has not landed, a select with one option would make the
 * value look like the only one available. A text field says the opposite, and
 * it still sends whatever is typed.
 */
export function Choice({
  id,
  value,
  options,
  onChange,
}: {
  id?: string
  value: string
  options: readonly string[]
  onChange: (v: string) => void
}) {
  if (!options.length) {
    return (
      <input
        id={id}
        className="field"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        spellCheck={false}
      />
    )
  }
  const known = options.includes(value)
  return (
    <select id={id} className="field" value={value} onChange={(e) => onChange(e.target.value)}>
      {!known ? <option value={value}>{value}</option> : null}
      {options.map((o) => (
        <option key={o} value={o}>
          {o}
        </option>
      ))}
    </select>
  )
}

export type ChipOption<T> = { value: T; label: string; title?: string; disabled?: boolean }

/** A row of equal buttons behind one hairline box. */
export function Chips<T extends string | number>({
  ariaLabel,
  value,
  options,
  onChange,
}: {
  ariaLabel: string
  value: T
  options: readonly ChipOption<T>[]
  onChange: (v: T) => void
}) {
  return (
    <div role="group" aria-label={ariaLabel} className="flex border border-grey-300">
      {options.map((o, i) => {
        const on = o.value === value
        return (
          <button
            key={String(o.value)}
            type="button"
            title={o.title}
            disabled={o.disabled}
            aria-pressed={on}
            onClick={() => onChange(o.value)}
            className={`${RING} flex-1 cursor-pointer px-2 py-1.5 text-caption tabular-nums transition-colors ${
              i > 0 ? 'border-l border-grey-300' : ''
            } ${on ? 'bg-ink text-newsprint' : 'text-grey-700 hover:text-ink hover:bg-newsprint-aged'} ${
              o.disabled ? 'cursor-not-allowed opacity-40' : ''
            }`}
          >
            {o.label}
          </button>
        )
      })}
    </div>
  )
}

/** A checkbox with its explanation, set as a list item under a hairline. */
export function Check({
  on,
  onToggle,
  title,
  note,
  cost,
  disabled,
}: {
  on: boolean
  onToggle: () => void
  title: string
  note: ReactNode
  cost?: string
  disabled?: boolean
}) {
  return (
    <li className={`border-b border-grey-300 py-2 ${disabled ? 'opacity-50' : ''}`}>
      <label className={`flex items-start gap-2 ${disabled ? 'cursor-not-allowed' : 'cursor-pointer'}`}>
        <input
          type="checkbox"
          className={`mt-1 h-3 w-3 shrink-0 accent-burgundy-900 ${RING} ${disabled ? '' : 'cursor-pointer'}`}
          checked={on}
          disabled={disabled}
          onChange={onToggle}
        />
        <span className="min-w-0">
          <span className="block text-small font-semibold leading-tight">{title}</span>
          <span className="mt-0.5 block text-caption leading-snug text-grey-700">{note}</span>
          {cost ? (
            <span className="mt-0.5 block text-caption italic text-grey-500">Costs {cost}.</span>
          ) : null}
        </span>
      </label>
    </li>
  )
}

/** A provenance badge. Measured, the author's, or plainly not checked. */
export function Source({ measured }: { measured: boolean }) {
  return (
    <span
      title={
        measured
          ? 'This figure comes from a Laplacian variance run on this machine.'
          : 'This figure is the author’s recommendation from the catalogue, not a measurement taken here.'
      }
      className={`inline-block border px-1 py-px text-[0.5625rem] font-semibold uppercase tracking-[0.14em] ${
        measured ? 'border-grey-300 text-grey-700' : 'border-grey-300 text-grey-500'
      }`}
    >
      {measured ? 'Measured' : 'Author’s'}
    </span>
  )
}
