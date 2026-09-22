/**
 * The house type every room sets from.
 *
 * Six files each carried their own kicker, their own editorial link and their
 * own square focus ring, byte for byte the same. They live here now. A folder's
 * own `bits.tsx` still holds the pieces that genuinely differ from room to room
 * (a section head with a figure, a caution with a rule) and re-exports these.
 */
import type { ReactNode } from 'react'

export { Notice, type NoticeTone } from './shell/Notice'

/** The house focus ring: square, burgundy, offset. Never rounded. */
export const RING =
  'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900'

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

/** An editorial link: burgundy, underlined, never a button in disguise. */
export function Link({
  onClick,
  children,
  className = '',
  title,
  disabled,
  expanded,
  controls,
}: {
  onClick: () => void
  children: ReactNode
  className?: string
  title?: string
  disabled?: boolean
  expanded?: boolean
  controls?: string
}) {
  return (
    <button
      type="button"
      title={title}
      disabled={disabled}
      onClick={onClick}
      aria-expanded={expanded}
      aria-controls={controls}
      className={`sg-link ring ${className}`}
    >
      {children}
    </button>
  )
}
