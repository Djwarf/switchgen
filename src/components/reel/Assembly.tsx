/**
 * The cutting room.
 *
 * Ten clips laid end to end are a film only once something actually lays them
 * end to end. Clips written by the same encoder at the same size, codec and
 * frame rate meet the condition ffmpeg's concat demuxer needs to join them
 * without re-encoding, and the server then joins them by stream copy, which
 * costs nothing in quality. Clips that differ (a reel rendered partly on one
 * style and partly on another, or before and after a change of shape) cannot
 * be copied, and the server re-encodes them to match the first. The page says
 * which of the two happened, from the server's own answer, and never promises
 * a copy in advance.
 *
 * The button posts the ordered list to /api/reel/stitch, which resolves every
 * path inside ComfyUI's output root, runs ffmpeg there and hands back the
 * finished file's real duration and frame count, probed from the file rather
 * than predicted. The printed command stays underneath it. A reader who would
 * rather drive ffmpeg themselves, or whose server is not running, still has the
 * one line that works.
 */
import { useState } from 'react'

import { useServerCapabilities } from '../../lib/capabilities'
import { copyText } from '../../lib/clipboard'
import { fileUrl, relPath, type OutputFile } from '../../lib/comfy'
import { Head, Kicker, Quiet, duration, grouped, seconds } from './bits'

/** What the server says once the reel is one file. Probed, not predicted. */
type Cut = {
  out: string
  mode: 'copy' | 'encode' | 'crossfade'
  elapsed: number
  warnings: string[]
  /** Why the clips could not be copied, one line per disagreement. Empty for a copy. */
  reasons?: string[]
  actual: { seconds: number; frames: number; size: number; width: number; height: number }
}

export type AssemblyClip = {
  index: number
  label: string
  file: OutputFile
  /** Frames in the clip, and the rate and size it was written at. */
  frames: number
  fps: number
  width: number
  height: number
  durationMs: number
  /** True when the strip has moved on since this clip was made. */
  outOfDate: boolean
}

export type AssemblyProps = {
  clips: readonly AssemblyClip[]
  /** How many shots the reel has in total, so a partial cut says so. */
  shots: number
  /** Where the clips were written, for the command's working directory. */
  prefix: string
}

export function Assembly({ clips, shots, prefix }: AssemblyProps) {
  /** What the copy button last did, shown on it for a moment. */
  const [copied, setCopied] = useState<'copied' | 'failed' | null>(null)
  const [cutting, setCutting] = useState(false)
  // The cut is kept with the strip it was made from. Shots land while a cut
  // runs, and throwing the cut away when the strip moved on (the room used to
  // be remounted for it) lost the finished reel's link and summary and let
  // the button send a second cut the server was still busy with.
  const [made, setMade] = useState<{ cut: Cut; strip: string } | null>(null)
  // A failure is kept with its strip for the same reason, and so that once
  // the strip has moved on it is not read as the verdict on the strip as it
  // stands, which nobody has tried to cut.
  const [failed, setFailed] = useState<{ message: string; strip: string } | null>(null)
  const caps = useServerCapabilities()
  /** Null until the server answers; false when ffmpeg is not there to run. */
  const canCut = caps === null ? null : caps.stitch

  if (!clips.length) return null

  const frames = clips.reduce((n, c) => n + c.frames, 0)
  // Each clip at its own rate: a reel that changed style part way holds clips
  // at 24 and at 16 frames a second, and one rate for all of them misstates both.
  const runSeconds = clips.reduce((n, c) => n + (c.fps > 0 ? c.frames / c.fps : 0), 0)
  const spent = clips.reduce((n, c) => n + c.durationMs, 0)
  const head = clips[0]
  const mixed = head
    ? clips.some((c) => c.fps !== head.fps || c.width !== head.width || c.height !== head.height)
    : false
  const partial = clips.length < shots
  const folder = prefix.replace(/\/+$/, '')
  const out = `${folder}/reel.webm`
  const command = buildCommand(clips, out)
  const strip = clips.map((c) => relPath(c.file)).join('\n')
  const cut = made?.cut ?? null
  /** True when the strip has changed since the reel on show was cut. */
  const earlier = made !== null && made.strip !== strip
  /** True when the strip has changed since the cut on show failed. */
  const failedEarlier = failed !== null && failed.strip !== strip

  // navigator.clipboard exists only on a secure page, and this desk is mostly
  // reached over plain http from a phone, where the button did nothing at all
  // and said nothing either. copyText falls back to the older way, and the
  // button says when neither worked.
  const copy = () => {
    void copyText(command).then((ok) => {
      setCopied(ok ? 'copied' : 'failed')
      setTimeout(() => setCopied(null), 2500)
    })
  }

  /**
   * Cut it. One request, one answer: `json=1` asks the server for the summary
   * rather than a progress stream. The summary says whether the clips were
   * copied or re-encoded, and why, which is what the page reports.
   */
  const make = () => {
    const from = strip
    setCutting(true)
    setFailed(null)
    setMade(null)
    void fetch('/api/reel/stitch?json=1', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ clips: clips.map((c) => relPath(c.file)), prefix: `${folder}/reel` }),
    })
      .then(async (r) => {
        const body = (await r.json()) as Record<string, unknown>
        if (!r.ok) throw new Error(String(body.error ?? `HTTP ${r.status}`))
        setMade({ cut: body as unknown as Cut, strip: from })
      })
      .catch((e: unknown) => setFailed({ message: e instanceof Error ? e.message : String(e), strip: from }))
      .finally(() => setCutting(false))
  }

  return (
    <section className="mt-8">
      <Head
        title="The cutting room"
        figure={`${clips.length} clips · ${runSeconds.toFixed(1)} s`}
        note="The clips are numbered in cutting order. Clips that share one size, frame rate and codec are joined by stream copy, with nothing re-encoded. Clips that differ are re-encoded to match the first."
      />

      {mixed ? (
        <p className="mb-4 border-l-2 border-warning pl-2 text-caption text-ink-warning">
          These clips do not all share one size and frame rate, so cutting them re-encodes the reel to match shot{' '}
          {(head?.index ?? 0) + 1}. That costs some quality and takes longer than a copy. Rendering what is missing
          brings the older clips into line first.
        </p>
      ) : null}

      <div className="mb-5 flex flex-wrap items-baseline gap-x-4 gap-y-2 border-b border-grey-300 pb-4">
        {canCut === false ? (
          <p className="text-caption italic text-grey-500">
            ffmpeg is not on the server, so the reel cannot be joined here. The command below still
            works anywhere it is.
          </p>
        ) : (
          <button type="button" className="press" disabled={cutting || canCut === null} onClick={make}>
            {cutting ? 'Cutting' : 'Cut the reel'}
          </button>
        )}
        {canCut === false ? null : cut ? (
          <p className="text-caption text-grey-700">
            {earlier ? (
              <span className="mr-1 italic text-ink-warning">
                Cut before the strip last changed, so this is not the reel as it stands now.
              </span>
            ) : null}
            <a className="sg-link" href={`/comfy/view?filename=${encodeURIComponent(cut.out.split('/').pop() ?? '')}&subfolder=${encodeURIComponent(cut.out.split('/').slice(0, -1).join('/'))}&type=output`} target="_blank" rel="noreferrer">
              {cut.out}
            </a>{' '}
            <span className="tabular-nums text-grey-500">
              {cut.actual.seconds.toFixed(2)} s · {grouped(cut.actual.frames)} frames · {mb(cut.actual.size)} ·{' '}
              {cut.mode === 'copy'
                ? 'joined by stream copy, nothing re-encoded'
                : cut.mode === 'crossfade'
                  ? 'crossfaded, so re-encoded'
                  : 're-encoded to match the first clip'}
            </span>
          </p>
        ) : failed ? (
          <p className="text-caption text-ink-warning">
            {failedEarlier ? (
              <span className="mr-1 italic">
                This cut failed before the strip last changed. The strip as it stands has not been cut.
              </span>
            ) : null}
            {failed.message}
          </p>
        ) : (
          <p className="text-caption italic text-grey-500">
            The server does the joining. Nothing touches the card.
          </p>
        )}
      </div>

      {cut && cut.mode !== 'copy' && cut.reasons?.length ? (
        <div className="mb-4 border-l-2 border-warning pl-2 text-caption text-ink-warning">
          <p>Re-encoded, because the clips differ:</p>
          <ul>
            {cut.reasons.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        </div>
      ) : null}

      {cut?.warnings.length ? (
        <ul className="mb-4 border-l-2 border-warning pl-2 text-caption text-ink-warning">
          {cut.warnings.map((w) => (
            <li key={w}>{w}</li>
          ))}
        </ul>
      ) : null}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_18rem]">
        <div className="min-w-0">
          <Kicker tone="quiet" className="mb-2">
            One line, pasted into a terminal
          </Kicker>
          <pre className="overflow-x-auto border border-grey-300 bg-newsprint-aged p-3 text-caption leading-relaxed text-ink">
            <code>{command}</code>
          </pre>
          <div className="mt-2 flex flex-wrap items-center gap-3">
            <Quiet onClick={copy}>
              {copied === 'copied' ? 'Copied' : copied === 'failed' ? 'Could not copy' : 'Copy the command'}
            </Quiet>
            <span className="text-caption italic text-grey-500">
              For doing it yourself. Set COMFY_OUTPUT to ComfyUI's output folder first.
            </span>
          </div>
          {mixed ? (
            <p className="mt-2 text-caption italic text-grey-500">
              The command copies the clips as they are, so it only works once they all share one size, frame rate and
              codec.
            </p>
          ) : null}

          {partial ? (
            <p className="mt-3 border-l-2 border-warning pl-2 text-caption text-ink-warning">
              This cuts the {clips.length} shots that have rendered. {shots - clips.length} of the reel's shots are not
              in it yet.
            </p>
          ) : null}
        </div>

        <div>
          <Kicker tone="quiet" className="mb-2">
            In order
          </Kicker>
          <ol className="border-t border-grey-300">
            {clips.map((c) => (
              <li key={c.file.filename} className="flex items-baseline justify-between gap-2 border-b border-grey-300 py-1">
                <span className="min-w-0 flex-1 truncate text-caption text-grey-700">
                  <span className="mr-1 tabular-nums text-burgundy-900">{String(c.index + 1).padStart(2, '0')}</span>
                  <a className="sg-link" href={fileUrl(c.file)} target="_blank" rel="noreferrer">
                    {c.file.filename}
                  </a>
                  {c.outOfDate ? <span className="ml-1 italic text-ink-warning">out of date</span> : null}
                </span>
                <span className="shrink-0 text-caption tabular-nums text-grey-500">{seconds(c.frames, c.fps)}</span>
              </li>
            ))}
          </ol>
          <p className="mt-2 text-caption tabular-nums text-grey-500">
            {grouped(frames)} frames. {duration(spent)} of press time went into them.
          </p>
        </div>
      </div>
    </section>
  )
}

/**
 * The concat demuxer wants a list file whose paths are relative to its own
 * directory, so the command changes into ComfyUI's output root and names each
 * clip by the path ComfyUI already reports for it.
 */
/** `1.2 MB`. The reel's size, in the unit a reader thinks in. */
function mb(bytes: number): string {
  return `${(bytes / 1e6).toFixed(1)} MB`
}

function buildCommand(clips: readonly AssemblyClip[], out: string): string {
  const lines = clips.map((c) => `file '${relPath(c.file)}'`).join('\n')
  return [
    'cd "${COMFY_OUTPUT:-$HOME/ComfyUI/output}"',
    "cat > reel.txt <<'LIST'",
    lines,
    'LIST',
    `ffmpeg -f concat -safe 0 -i reel.txt -c copy "${out}"`,
  ].join('\n')
}
