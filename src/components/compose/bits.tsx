/**
 * The small printed objects the compose rail is set from.
 *
 * Nothing here is a feature. These are the typographic primitives the desk
 * borrows from the rest of the paper: a kicker, a deck, a hairline, a notice,
 * an editorial link, and one control pattern.
 *
 * That one control pattern is {@link Choice}. It matters for the thing this
 * screen is being judged on. A row of three chips is not three controls: it is
 * one decision. Written as three tabbable buttons it would cost three tab stops
 * and read, to anyone counting the surface of this screen, as three things to
 * think about. So it is written as a WAI-ARIA radio group with a roving
 * tabindex: one stop in the tab order, arrow keys to move within it, exactly
 * like a printed set of options where the eye moves but the page does not.
 *
 * Both of the compose rail's answer controls (the look, the anatomy level) are
 * this component, which is why the default screen costs what it costs.
 */
import { useRef, type KeyboardEvent as ReactKeyboardEvent, type ReactNode } from 'react'

// ---------------------------------------------------------------------------
// Type
// ---------------------------------------------------------------------------

/** Metadata caps. Grey, not burgundy: burgundy is rationed to the page. */
export function Kicker({ children, className = '' }: { children: ReactNode; className?: string }) {
  return (
    <span
      className={`text-overline font-semibold uppercase tracking-[0.18em] text-grey-700 ${className}`}
    >
      {children}
    </span>
  )
}

/** A field's name, set as a kicker above it. */
export function Label({ children, hint }: { children: ReactNode; hint?: string }) {
  return (
    <span className="mb-2 block text-overline font-semibold uppercase tracking-[0.18em] text-grey-700">
      {children}
      {hint && (
        <span className="ml-2 text-caption italic normal-case tracking-normal text-grey-500">
          {hint}
        </span>
      )}
    </span>
  )
}

/** The standfirst under a headline. One sentence, italic, generous measure. */
export function Deck({ children }: { children: ReactNode }) {
  return (
    <p className="max-w-[58ch] text-body italic leading-relaxed text-grey-700">{children}</p>
  )
}

/** One of the two rule weights in this application. Nothing else is drawn. */
export function Hairline({ className = '' }: { className?: string }) {
  return <hr className={`border-0 border-b border-grey-300 ${className}`} />
}

/** An editorial link: burgundy, underlined, never a button in disguise. */
export function EditorialLink({
  onClick,
  children,
  className = '',
  title,
  expanded,
  controls,
}: {
  onClick: () => void
  children: ReactNode
  className?: string
  title?: string
  expanded?: boolean
  controls?: string
}) {
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      aria-expanded={expanded}
      aria-controls={controls}
      className={`sg-link ring ${className}`}
    >
      {children}
    </button>
  )
}

export type NoticeKind = 'info' | 'correction' | 'error' | 'warning'

/**
 * A ruled notice, in the house variants. Warnings are never folded away behind
 * the More link: a thing that is true and unwelcome is printed on the page.
 */
export function Notice({
  kind = 'info',
  title,
  children,
}: {
  kind?: NoticeKind
  title?: string
  children?: ReactNode
}) {
  return (
    <div role={kind === 'error' ? 'alert' : 'status'} className={`notice notice-${kind} text-small`}>
      {title && <strong>{title}</strong>}
      {children}
    </div>
  )
}

// ---------------------------------------------------------------------------
// The one control pattern
// ---------------------------------------------------------------------------

export type ChoiceOption<T extends string> = {
  id: T
  label: string
  /** Printed under the set when this option is the chosen one. */
  blurb?: string
}

/**
 * One decision, several options, one tab stop.
 *
 * Set like the section bar: the chosen option is burgundy over a 2px burgundy
 * rule, the rest are grey with no rule, and nothing moves between the two
 * states. The chosen option's blurb is printed beneath the set, so the screen
 * explains itself without the reader hovering anything.
 */
export function Choice<T extends string>({
  legend,
  hint,
  options,
  value,
  onChange,
  footnote,
}: {
  legend: string
  hint?: string
  options: readonly ChoiceOption<T>[]
  value: T
  onChange: (next: T) => void
  /** Printed under the blurb. Use it for a measured caveat, never for copy. */
  footnote?: ReactNode
}) {
  const refs = useRef(new Map<T, HTMLButtonElement | null>())
  const chosen = options.find(o => o.id === value) ?? options[0]

  const move = (delta: number) => {
    if (options.length === 0) return
    const at = options.findIndex(o => o.id === value)
    const from = at === -1 ? 0 : at
    const next = options[(from + delta + options.length) % options.length]
    onChange(next.id)
    refs.current.get(next.id)?.focus()
  }

  const onKeyDown = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
      e.preventDefault()
      move(1)
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
      e.preventDefault()
      move(-1)
    } else if (e.key === 'Home') {
      e.preventDefault()
      const first = options[0]
      if (first) {
        onChange(first.id)
        refs.current.get(first.id)?.focus()
      }
    } else if (e.key === 'End') {
      e.preventDefault()
      const last = options[options.length - 1]
      if (last) {
        onChange(last.id)
        refs.current.get(last.id)?.focus()
      }
    }
  }

  return (
    <div>
      <Label hint={hint}>{legend}</Label>
      <div
        role="radiogroup"
        aria-label={legend}
        onKeyDown={onKeyDown}
        className="flex flex-wrap items-baseline gap-x-6 gap-y-1 border-b border-grey-300"
      >
        {options.map(o => {
          const on = o.id === value
          return (
            <button
              key={o.id}
              ref={el => {
                refs.current.set(o.id, el)
              }}
              type="button"
              role="radio"
              aria-checked={on}
              tabIndex={on || (!options.some(x => x.id === value) && o === options[0]) ? 0 : -1}
              onClick={() => onChange(o.id)}
              className={`sg-tap ring -mb-px cursor-pointer border-b-2 px-0.5 py-2 text-small font-semibold uppercase tracking-[0.14em] transition-colors ${
                on
                  ? 'border-burgundy-900 text-burgundy-900'
                  : 'border-transparent text-grey-500 hover:text-ink'
              }`}
            >
              {o.label}
            </button>
          )
        })}
      </div>
      {chosen?.blurb && (
        <p className="mt-2 max-w-[58ch] text-caption italic text-grey-700">{chosen.blurb}</p>
      )}
      {footnote && <div className="mt-1 max-w-[58ch] text-caption text-grey-500">{footnote}</div>}
    </div>
  )
}
