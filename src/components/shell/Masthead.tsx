/**
 * The masthead.
 *
 * Wordmark, standfirst, and a dateline strip that carries the facts worth
 * carrying everywhere: what day it is, what the machine is doing this second,
 * how many editions are filed, and whether the server is answering. Burgundy is
 * spent here and in very few other places.
 *
 * The hardware half of that line is live — a reading a second off
 * `lib/hardware.ts` — and falls back, without comment, to the one-shot probe
 * line `App.tsx` passes in.
 */
import { useEffect, useState, useSyncExternalStore, type ReactNode } from 'react'
import { gb, subscribeDeviceStatus, type DeviceStatus } from '../../lib/hardware'
import { count, subscribe as subscribeHistory } from '../../lib/history'
import { useConnection } from './Offline'

const DATE = new Intl.DateTimeFormat('en-GB', {
  weekday: 'long',
  day: 'numeric',
  month: 'long',
  year: 'numeric',
})

/** British, spelled out: "Sunday, 21 September 2026". */
export function britishDate(at: number | Date = Date.now()): string {
  return DATE.format(at instanceof Date ? at : new Date(at))
}

/** How many records are filed. Re-renders when the archive changes. */
export function useEntryCount(): number {
  return useSyncExternalStore(subscribeHistory, count, count)
}

// Burgundy is rationed: the live-job mark in the section bar is allowed one,
// and this is not it. Green for answering, red for not, grey while it tries.
const STATUS: Record<string, { label: string; className: string }> = {
  open: { label: 'On the wire', className: 'sg-mark sg-mark-ok' },
  connecting: { label: 'Connecting', className: 'sg-mark sg-mark-idle sg-mark-live' },
  closed: { label: 'Not answering', className: 'sg-mark sg-mark-off' },
}

export type MastheadProps = {
  /**
   * "RTX 5060 Ti · 16 GB", from the one-shot probe. Printed only while the
   * live stream is silent; left out when neither has answered.
   */
  gpu?: string | null
  standfirst?: string
  /** Extra dateline items, set in the same 10 px caps. */
  children?: ReactNode
}

export function Masthead({
  gpu,
  standfirst = 'Local image and video generation, printed on demand',
  children,
}: MastheadProps) {
  const filed = useEntryCount()
  const connection = useConnection()
  const status = STATUS[connection] ?? STATUS.closed

  return (
    <header className="shrink-0 bg-newsprint px-6 pt-3">
      <div className="mx-auto w-full max-w-[110rem]">
        <div className="border-t-2 border-burgundy-900 pt-3" />
        <h1 className="text-center text-[1.75rem] leading-none font-bold tracking-[0.2em] text-burgundy-900 sm:text-[2.6rem] sm:tracking-[0.3em]">
          SWITCHGEN
        </h1>
        <p className="mt-1 text-center text-small text-grey-700 italic">{standfirst}</p>

        <div className="mt-3 flex items-center justify-between gap-4 border-t border-b border-grey-300 py-1.5 kicker-quiet">
          <span className="truncate">{britishDate()}</span>
          <DeviceLine fallback={gpu} />
          {children}
          <span className="flex items-center gap-4">
            <span className="figures hidden md:inline">
              {filed.toLocaleString('en-GB')} {filed === 1 ? 'edition' : 'editions'} filed
            </span>
            <span className="flex items-center gap-2" title={`ComfyUI: ${status.label.toLowerCase()}`}>
              <span className={status.className} aria-hidden />
              {status.label}
            </span>
          </span>
        </div>
      </div>
    </header>
  )
}

// ---------------------------------------------------------------------------
// The device readout
// ---------------------------------------------------------------------------

/**
 * The machine, read aloud once a second.
 *
 * `lib/hardware.ts` streams CPU, RAM, the card and the disk over SSE. This is
 * a courtesy, not a fact the page depends on: when the stream never opens, or
 * stops answering, the dateline quietly carries the one-shot probe line it was
 * given instead, and nothing is said about the loss.
 */
function useDeviceStatus(): DeviceStatus | null {
  const [status, setStatus] = useState<DeviceStatus | null>(null)

  useEffect(() => {
    let live = true
    let last = 0

    const stop = subscribeDeviceStatus((s) => {
      if (!live) return
      last = Date.now()
      setStatus(s)
    })

    // A frozen readout is a lie told in figures. Five seconds of silence on a
    // one-second stream means the wire is gone, so the line falls back rather
    // than standing there insisting the card is still at 93%.
    const watch = setInterval(() => {
      if (live && last > 0 && Date.now() - last > 5000) {
        last = 0
        setStatus(null)
      }
    }, 2000)

    return () => {
      live = false
      clearInterval(watch)
      stop()
    }
  }, [])

  return status
}

/**
 * The wire sends CPU load as a fraction and nvidia-smi's utilisation as whole
 * percent. `pct()` in lib/hardware guesses between the two, which reads one
 * per cent of a quiet card as a hundred; here both shapes are known, so they
 * are said plainly.
 */
const percent = (n: number | null | undefined, scale: 1 | 100): string =>
  n === null || n === undefined ? '\u2014' : `${Math.round(n * scale)}%`

type Reading = { label: string; value: string }

function readingsFor(s: DeviceStatus): Reading[] {
  const out: Reading[] = []
  const g = s.gpu

  if (g) {
    const name = g.name.replace(/^NVIDIA\s+GeForce\s+/i, '').trim()
    if (name) out.push({ label: '', value: name })
    out.push({ label: 'GPU', value: percent(g.utilGpu, 1) })
    if (g.vramTotal > 0) {
      out.push({ label: 'VRAM', value: `${gb(g.vramUsed)} / ${gb(g.vramTotal)}` })
    }
  }

  out.push({ label: 'CPU', value: percent(s.cpu.overall, 100) })
  if (s.ram.total > 0) out.push({ label: 'RAM', value: `${gb(s.ram.used)} / ${gb(s.ram.total)}` })
  if (g && g.tempC !== null) out.push({ label: 'Temp', value: `${Math.round(g.tempC)}\u00b0C` })
  if (g && g.powerW !== null) out.push({ label: 'Power', value: `${Math.round(g.powerW)} W` })
  if (s.disk && s.disk.total > 0) out.push({ label: 'Disk', value: `${gb(s.disk.free)} free` })

  return out
}

/**
 * Where each reading joins the strip, narrowest measure first. A reading that
 * the machine cannot supply — no card, no disk figure — hands its place to the
 * next one, so the strip always opens with something and never opens with a
 * hairline. Nothing at all below 640px: the phone's dateline is the date.
 */
const STEPS = [
  'flex',
  'flex',
  'hidden lg:flex',
  'hidden lg:flex',
  'hidden xl:flex',
  'hidden xl:flex',
  'hidden min-[1600px]:flex',
  'hidden min-[1600px]:flex',
] as const

/**
 * The dateline's hardware line: overline caps, hairline separators, figures in
 * the tabular face, exactly as the rest of the strip is set. Not announced —
 * a number that changes every second has no business interrupting a reader —
 * but the full sentence is on the title, including the readings a narrow
 * measure has dropped.
 */
function DeviceLine({ fallback }: { fallback?: string | null }) {
  const status = useDeviceStatus()

  if (!status) {
    return fallback ? <span className="hidden truncate sm:inline">{fallback}</span> : null
  }

  const readings = readingsFor(status).slice(0, STEPS.length)
  if (readings.length === 0) {
    return fallback ? <span className="hidden truncate sm:inline">{fallback}</span> : null
  }

  const sentence = readings.map((r) => (r.label ? `${r.label} ${r.value}` : r.value)).join(' \u00b7 ')

  return (
    <span className="hidden min-w-0 shrink items-center gap-3 sm:flex" title={sentence}>
      {readings.map((r, i) => (
        <span
          key={r.label || 'card'}
          className={`${STEPS[i]} shrink-0 items-baseline gap-1.5 ${
            i > 0 ? 'border-l border-grey-300 pl-3' : ''
          }`}
        >
          {r.label && <span className="text-grey-500">{r.label}</span>}
          <span className="figures text-ink">{r.value}</span>
        </span>
      ))}
    </span>
  )
}
