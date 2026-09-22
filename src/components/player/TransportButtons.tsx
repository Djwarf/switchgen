/**
 * The typographic control row's parts: hand-drawn transport glyphs and the
 * three kinds of button the player uses.
 *
 * House rules, from the brand guide: no rounding, no shadow, two rule weights
 * only, and burgundy is spent on the scrub rail and the focus ring — not on
 * chrome. An active control is BLACK on newsprint with a black underline.
 * Strokes are 1.25 px, butt caps, mitre joins: film equipment cut in metal,
 * not a media player's rounded pills.
 */

import type { ReactNode, RefObject } from 'react'

/** The house focus ring. Square, 2 px burgundy, 2 px offset. */
export const RING =
  'focus-visible:[outline:2px_solid_var(--color-burgundy-900)] focus-visible:[outline-offset:2px]'

const LABEL = 'text-overline font-semibold uppercase tracking-[0.16em]'

export type GlyphName =
  | 'play'
  | 'pause'
  | 'first'
  | 'prev'
  | 'next'
  | 'last'
  | 'fullscreen'
  | 'exit-fullscreen'

const PATHS: Record<GlyphName, ReactNode> = {
  play: <path d="M4.5 3 L12.5 8 L4.5 13 Z" />,
  pause: (
    <>
      <path d="M5.5 3 L5.5 13" />
      <path d="M10.5 3 L10.5 13" />
    </>
  ),
  first: (
    <>
      <path d="M4 3 L4 13" />
      <path d="M13 3 L5.5 8 L13 13 Z" />
    </>
  ),
  prev: <path d="M12 3 L4.5 8 L12 13 Z" />,
  next: <path d="M4 3 L11.5 8 L4 13 Z" />,
  last: (
    <>
      <path d="M12 3 L12 13" />
      <path d="M3 3 L10.5 8 L3 13 Z" />
    </>
  ),
  fullscreen: (
    <>
      <path d="M2.5 6 L2.5 2.5 L6 2.5" />
      <path d="M10 2.5 L13.5 2.5 L13.5 6" />
      <path d="M13.5 10 L13.5 13.5 L10 13.5" />
      <path d="M6 13.5 L2.5 13.5 L2.5 10" />
    </>
  ),
  'exit-fullscreen': (
    <>
      <path d="M6 2.5 L6 6 L2.5 6" />
      <path d="M13.5 6 L10 6 L10 2.5" />
      <path d="M10 13.5 L10 10 L13.5 10" />
      <path d="M2.5 10 L6 10 L6 13.5" />
    </>
  ),
}

export function Glyph({ name }: { name: GlyphName }) {
  return (
    <svg
      viewBox="0 0 16 16"
      width="16"
      height="16"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.25}
      strokeLinecap="butt"
      strokeLinejoin="miter"
      aria-hidden="true"
      focusable="false"
    >
      {PATHS[name]}
    </svg>
  )
}

type IconButtonProps = {
  glyph: GlyphName
  /** Spoken label. Also the tooltip, with the key hint appended. */
  label: string
  /** Keyboard hint, e.g. `Space`. */
  hint?: string
  onClick: () => void
  disabled?: boolean
  /** Drawn as pressed — used for play while the clip is running. */
  pressed?: boolean
}

export function IconButton({ glyph, label, hint, onClick, disabled, pressed }: IconButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      aria-pressed={pressed === undefined ? undefined : pressed}
      title={hint ? `${label} (${hint})` : label}
      className={`inline-flex h-8 w-8 [@media(pointer:coarse)]:h-11 [@media(pointer:coarse)]:w-11 items-center justify-center border border-transparent text-ink transition-colors duration-100 hover:border-grey-300 hover:bg-newsprint-aged disabled:cursor-not-allowed disabled:text-grey-400 disabled:hover:border-transparent disabled:hover:bg-transparent ${RING}`}
    >
      <Glyph name={glyph} />
    </button>
  )
}

type TextButtonProps = {
  children: ReactNode
  label?: string
  hint?: string
  onClick: () => void
  active?: boolean
  disabled?: boolean
  /** Sets `aria-pressed`, for toggles such as LOOP. */
  toggle?: boolean
}

/**
 * A word, not a chip. Active state is a black underline — the one place the
 * spec permits a burgundy underline is the LOOP word, which the player passes
 * as `active` and styles itself.
 */
export function TextButton({
  children,
  label,
  hint,
  onClick,
  active,
  disabled,
  toggle,
}: TextButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      aria-pressed={toggle ? !!active : undefined}
      title={hint ? (label ? `${label} (${hint})` : hint) : label}
      className={`${LABEL} h-8 px-1.5 [@media(pointer:coarse)]:h-11 [@media(pointer:coarse)]:px-2.5 transition-colors duration-100 disabled:cursor-not-allowed disabled:text-grey-400 ${RING} ${
        active
          ? 'text-ink underline decoration-ink decoration-2 underline-offset-4'
          : 'text-grey-500 hover:text-ink'
      }`}
    >
      {children}
    </button>
  )
}

type LinkActionProps = {
  children: ReactNode
  onClick: () => void
  disabled?: boolean
  hint?: string
  /** Renders the menu affordance and reports its state to assistive tech. */
  expanded?: boolean
  /**
   * The underlying button, for a caller that opens a menu from this word. A
   * menu that takes focus has to be able to hand it back here on Escape, and
   * focus landing on the body instead is how a keyboard session ends up at the
   * top of the document with nothing selected.
   */
  buttonRef?: RefObject<HTMLButtonElement | null>
}

/** An editorial link: burgundy, underlined, in mixed case. Never a button. */
export function LinkAction({
  children,
  onClick,
  disabled,
  hint,
  expanded,
  buttonRef,
}: LinkActionProps) {
  return (
    <button
      ref={buttonRef}
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={hint}
      aria-haspopup={expanded === undefined ? undefined : 'menu'}
      aria-expanded={expanded}
      className={`text-caption text-burgundy-900 underline decoration-burgundy-900 underline-offset-2 transition-colors duration-100 hover:text-burgundy-700 disabled:cursor-not-allowed disabled:text-grey-400 disabled:no-underline ${RING}`}
    >
      {children}
    </button>
  )
}

/** The 1 px hairline the player uses between its bands. */
export function Hairline() {
  return <div className="h-px w-full bg-grey-300" aria-hidden="true" />
}

/** A vertical hairline separating clusters inside the transport row. */
export function Divider() {
  return <span className="mx-1 hidden h-4 w-px self-center bg-grey-300 sm:block" aria-hidden="true" />
}
