/**
 * Frame arithmetic, editorial number formatting, and the readout that prints
 * them in the transport row.
 *
 * Every number the player shows passes through here, so there is exactly one
 * place where "which frame am I on" is decided. The convention:
 *
 *   frame f occupies the half-open interval  [ f / fps , (f+1) / fps )
 *   the time to seek to in order to land on f is its MIDPOINT, (f + 0.5) / fps
 *   the frame a time t belongs to is         floor(t * fps)
 *
 * The build spec writes the reader as `Math.round(t * fps)`, which disagrees
 * with its own mid-frame seek — round((f + 0.5)) is f + 1, so every step
 * forward would read one frame ahead of where it landed. `floor` is the
 * inverse of the seek the spec specifies, so that is what is implemented here.
 */

import { clamp } from '../../lib/num'
export { clamp }

/** The frame a playback time belongs to. */
export function frameAt(time: number, fps: number, frames?: number): number {
  if (!Number.isFinite(time) || !Number.isFinite(fps) || fps <= 0) return 0
  // A hair of tolerance: decoders report mediaTime a few microseconds under
  // the boundary, which would otherwise read as the previous frame.
  const f = Math.floor(time * fps + 1e-4)
  const last = frames && frames > 0 ? frames - 1 : Number.MAX_SAFE_INTEGER
  return clamp(f, 0, last)
}

/** The time to seek to in order to land squarely inside frame `f`. */
export function timeOfFrame(frame: number, fps: number): number {
  if (!Number.isFinite(fps) || fps <= 0) return 0
  return (frame + 0.5) / fps
}

/** How many frames a clip of this duration holds. */
export function frameCount(duration: number, fps: number): number {
  if (!Number.isFinite(duration) || duration <= 0 || fps <= 0) return 0
  return Math.max(1, Math.round(duration * fps))
}

/** `0034` — zero padded to at least four figures, wider if the clip is longer. */
export function padFrame(n: number, total: number): string {
  const width = Math.max(4, String(Math.max(0, total - 1)).length)
  return String(Math.max(0, Math.round(n))).padStart(width, '0')
}

/** `02.125` — seconds to three decimal places, two figures before the point. */
export function seconds3(t: number): string {
  const safe = Number.isFinite(t) && t > 0 ? t : 0
  const whole = Math.floor(safe)
  const frac = Math.round((safe - whole) * 1000)
  const carry = frac === 1000
  return `${String(carry ? whole + 1 : whole).padStart(2, '0')}.${String(carry ? 0 : frac).padStart(3, '0')}`
}

/** `00:01.2` — the A–B readout, a minute-clock with one decimal. */
export function clock(t: number): string {
  const safe = Number.isFinite(t) && t > 0 ? t : 0
  const mins = Math.floor(safe / 60)
  const secs = safe - mins * 60
  return `${String(mins).padStart(2, '0')}:${secs.toFixed(1).padStart(4, '0')}`
}

/** `7.4 seconds` under a minute, `4 min 12 s` above it. */
export function humanDuration(ms: number): string {
  if (!Number.isFinite(ms) || ms <= 0) return ''
  const s = ms / 1000
  if (s < 60) return `${s < 10 ? s.toFixed(1) : Math.round(s)} seconds`
  const mins = Math.floor(s / 60)
  const rest = Math.round(s - mins * 60)
  return rest ? `${mins} min ${rest} s` : `${mins} min`
}

const DATE = new Intl.DateTimeFormat('en-GB', { day: 'numeric', month: 'long', year: 'numeric' })
const TIME = new Intl.DateTimeFormat('en-GB', { hour: '2-digit', minute: '2-digit', hour12: false })

/** `21 September 2026, 14:06`. */
export function britishDateTime(at: number): string {
  if (!Number.isFinite(at)) return ''
  const d = new Date(at)
  return `${DATE.format(d)}, ${TIME.format(d)}`
}

/** `1216 × 832`, thin-spaced, never an `x`. */
export function dimensions(w: number | null | undefined, h: number | null | undefined): string {
  if (!w || !h) return ''
  return `${w}  ×  ${h}`
}

/** A filename with its extension taken off, for building download names. */
export function stem(filename: string): string {
  const base = filename.split('/').pop() ?? filename
  const dot = base.lastIndexOf('.')
  return dot > 0 ? base.slice(0, dot) : base
}

/** The extension of a filename, with its dot. `.webm`, or '' when there is none. */
export function extension(filename: string): string {
  const base = filename.split('/').pop() ?? filename
  const dot = base.lastIndexOf('.')
  return dot > 0 ? base.slice(dot) : ''
}

type ReadoutProps = {
  /** Current frame index. */
  frame: number
  /** Total frames, when the record knows. */
  frames: number
  /** Current playback time, seconds. */
  time: number
  /** Clip duration, seconds. */
  duration: number
  /** Frames per second, from the record. `null` when it was never written down. */
  fps: number | null
  /** True when the numbers are read back from the decoder and can be trusted. */
  exact: boolean
}

/**
 * `0034 / 0081 · 02.125s / 05.063s · 24 fps`
 *
 * When the frame rate was never recorded, or the browser will not tell us which
 * frame it is showing, the frame half is dropped or daggered rather than
 * guessed. A confident wrong number is worse than an honest absence.
 */
export function Readout({ frame, frames, time, duration, fps, exact }: ReadoutProps) {
  const sep = <span className="px-2 text-grey-400" aria-hidden="true">·</span>
  return (
    <span className="flex items-baseline text-caption tabular-nums text-grey-700">
      {fps !== null && frames > 0 && (
        <>
          <span className="text-ink" title={exact ? 'Frame, read from the decoder' : 'Frame, approximate'}>
            {exact ? '' : '~'}
            {padFrame(frame, frames)}
            <span className="text-grey-400"> / </span>
            {padFrame(frames, frames)}
          </span>
          {sep}
        </>
      )}
      <span title="Time">
        {seconds3(time)}s<span className="text-grey-400"> / </span>
        {seconds3(duration)}s
      </span>
      {fps !== null && (
        <>
          {sep}
          <span title="Frame rate, from the record">{fps} fps</span>
        </>
      )}
    </span>
  )
}
