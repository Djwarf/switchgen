/**
 * A film ruler drawn in newspaper rules: one hairline per frame directly under
 * the scrub rail, every eighth drawn taller.
 *
 * It is pure CSS — two repeating gradients over percentage periods — so an
 * eighty-one frame clip and a two-hundred frame clip both cost nothing to
 * draw, nothing to decode, and nothing to keep in memory. Its whole job is to
 * let you see, without reading a number, how long the clip is and how coarse a
 * single frame is against it.
 */

type Props = {
  /** How many marks the ruler stands for — frames, or seconds in time mode. */
  count: number
  /** Draw a taller mark every n marks. */
  major?: number
  /** Extra classes for the wrapper. */
  className?: string
}

export function FrameTicks({ count, major = 8, className = '' }: Props) {
  // Above 120 frames a mark per frame becomes a grey smear, so the ruler
  // thins out to every fourth and says the same thing more quietly.
  const step = count > 120 ? 4 : 1
  const marks = Math.max(1, Math.floor(count / step))
  const majorEvery = Math.max(2, major)

  if (count <= 1) return <div className={`h-2 w-full ${className}`} aria-hidden="true" />

  const hair = 'var(--color-grey-300)'
  const minor = `repeating-linear-gradient(to right, ${hair} 0, ${hair} 1px, transparent 1px, transparent calc(100% / ${marks}))`
  const tall = `repeating-linear-gradient(to right, ${hair} 0, ${hair} 1px, transparent 1px, transparent calc(100% * ${majorEvery} / ${marks}))`

  return (
    <div className={`relative h-2 w-full ${className}`} aria-hidden="true">
      <div className="absolute inset-x-0 top-0 h-1" style={{ backgroundImage: minor }} />
      <div className="absolute inset-x-0 top-0 h-2" style={{ backgroundImage: tall }} />
    </div>
  )
}
