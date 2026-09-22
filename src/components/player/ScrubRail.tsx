/**
 * The scrub rail: a 2 px rule with a printer's register mark for a playhead.
 *
 * Not an `<input type="range">` — a range input brings a browser thumb, a
 * rounded track and a focus ring that belong to somebody else's design. This
 * is a rule, the played part filled burgundy, and a 2 px × 14 px bar standing
 * on it. The visible rule is 2 px; the pointer target is the full 16 px band,
 * because nobody can hit a two-pixel line.
 *
 * It is a real slider to the rest of the world: `role="slider"`, arrow keys,
 * and `aria-valuetext` that says "frame 34 of 81" rather than a bare number.
 */

import { useCallback, useRef, useState } from 'react'
import { clamp, clock } from './Readout'
import { RING } from './TransportButtons'

type Props = {
  /** Current frame. */
  value: number
  /** Index of the last frame. */
  max: number
  fps: number
  /** False when the frame rate was never recorded: announce time, not frames. */
  showFrames: boolean
  disabled?: boolean
  inFrame?: number | null
  outFrame?: number | null
  /** Called continuously while dragging, and once per key press. */
  onSeek: (frame: number) => void
  onScrubStart?: () => void
  onScrubEnd?: () => void
  /** Frame under the pointer, or null when it leaves. */
  onHover?: (frame: number | null) => void
  /** Lets the player stand down its own arrow-key handling while the rail has focus. */
  onFocusChange?: (focused: boolean) => void
}

export function ScrubRail({
  value,
  max,
  fps,
  showFrames,
  disabled = false,
  inFrame = null,
  outFrame = null,
  onSeek,
  onScrubStart,
  onScrubEnd,
  onHover,
  onFocusChange,
}: Props) {
  const railRef = useRef<HTMLDivElement | null>(null)
  const draggingRef = useRef(false)
  const [dragging, setDragging] = useState(false)

  const frameFromClientX = useCallback(
    (clientX: number): number => {
      const el = railRef.current
      if (!el || max <= 0) return 0
      const rect = el.getBoundingClientRect()
      const ratio = clamp((clientX - rect.left) / Math.max(1, rect.width), 0, 1)
      return Math.round(ratio * max)
    },
    [max],
  )

  const pct = (frame: number): string => `${max > 0 ? (clamp(frame, 0, max) / max) * 100 : 0}%`

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (disabled || max <= 0 || e.button !== 0) return
    e.preventDefault()
    railRef.current?.focus()
    railRef.current?.setPointerCapture(e.pointerId)
    draggingRef.current = true
    setDragging(true)
    onScrubStart?.()
    onSeek(frameFromClientX(e.clientX))
  }

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (disabled || max <= 0) return
    const frame = frameFromClientX(e.clientX)
    if (draggingRef.current) onSeek(frame)
    else onHover?.(frame)
  }

  const endDrag = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!draggingRef.current) return
    draggingRef.current = false
    setDragging(false)
    try {
      railRef.current?.releasePointerCapture(e.pointerId)
    } catch {
      /* the capture is already gone; nothing to release */
    }
    onScrubEnd?.()
  }

  const onKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (disabled || max <= 0) return
    const second = Math.max(1, Math.round(fps))
    // A key press is a finished move, not a drag: seek and commit in one go,
    // so a clip that was playing carries on playing from the new position.
    const commit = (frame: number) => {
      e.preventDefault()
      e.stopPropagation()
      onSeek(frame)
      onScrubEnd?.()
    }
    const move = (delta: number) => commit(clamp(value + delta, 0, max))
    switch (e.key) {
      case 'ArrowLeft':
        return move(e.shiftKey ? -second : -1)
      case 'ArrowRight':
        return move(e.shiftKey ? second : 1)
      case 'ArrowDown':
        return move(-1)
      case 'ArrowUp':
        return move(1)
      case 'PageDown':
        return move(-second)
      case 'PageUp':
        return move(second)
      case 'Home':
        return commit(0)
      case 'End':
        return commit(max)
      default:
    }
  }

  const valueText = showFrames
    ? `frame ${value} of ${max + 1}`
    : `${clock(value / Math.max(1, fps))} of ${clock((max + 1) / Math.max(1, fps))}`

  const hasRegion = inFrame !== null && outFrame !== null && outFrame > inFrame

  return (
    <div
      ref={railRef}
      role="slider"
      tabIndex={disabled ? -1 : 0}
      aria-label="Position"
      aria-valuemin={0}
      aria-valuemax={Math.max(0, max)}
      aria-valuenow={clamp(value, 0, Math.max(0, max))}
      aria-valuetext={valueText}
      aria-disabled={disabled || undefined}
      aria-orientation="horizontal"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
      onPointerLeave={() => onHover?.(null)}
      onKeyDown={onKeyDown}
      onFocus={() => onFocusChange?.(true)}
      onBlur={() => onFocusChange?.(false)}
      className={`sg-tap relative h-4 w-full touch-none select-none ${
        disabled ? 'cursor-default' : dragging ? 'cursor-grabbing' : 'cursor-pointer'
      } ${RING}`}
    >
      {/* the rule */}
      <div className="absolute inset-x-0 top-1/2 h-0.5 -translate-y-1/2 bg-grey-300" />

      {/* the A–B region, bracketed on the rule itself */}
      {hasRegion && (
        <div
          className="absolute top-1/2 h-0.5 -translate-y-1/2 bg-burgundy-200"
          style={{ left: pct(inFrame), width: `calc(${pct(outFrame)} - ${pct(inFrame)})` }}
        />
      )}

      {/* the played portion */}
      <div
        className="absolute left-0 top-1/2 h-0.5 -translate-y-1/2 bg-burgundy-900"
        style={{ width: pct(value) }}
      />

      {/* in and out marks */}
      {inFrame !== null && (
        <div
          className="absolute top-1/2 h-2.5 w-px -translate-y-1/2 bg-burgundy-900"
          style={{ left: pct(inFrame) }}
          aria-hidden="true"
        />
      )}
      {outFrame !== null && (
        <div
          className="absolute top-1/2 h-2.5 w-px -translate-y-1/2 bg-burgundy-900"
          style={{ left: pct(outFrame) }}
          aria-hidden="true"
        />
      )}

      {/* the playhead: a register mark, not a knob */}
      <div
        className="pointer-events-none absolute top-1/2 h-3.5 w-0.5 -translate-x-1/2 -translate-y-1/2 bg-burgundy-900"
        style={{ left: pct(value) }}
        aria-hidden="true"
      />
    </div>
  )
}
