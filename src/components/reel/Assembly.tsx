/**
 * The cutting room.
 *
 * Ten clips laid end to end are a film only once something actually lays them
 * end to end. Every clip in a reel is written by the same encoder at the same
 * size, codec and frame rate, which is the condition ffmpeg's concat demuxer
 * needs to join them without re encoding. So the join is a stream copy: it
 * takes about a second and costs nothing in quality.
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
import { fileUrl, relPath, type OutputFile } from '../../lib/comfy'
import { Head, Kicker, Quiet, duration, grouped, seconds } from './bits'

/** What the server says once the reel is one file. Probed, not predicted. */
type Cut = {
  out: string
  mode: 'copy' | 'encode' | 'crossfade'
  elapsed: number
  warnings: string[]
  actual: { seconds: number; frames: number; size: number; width: number; height: number }
}

export type AssemblyClip = {
  index: number
  label: string
  file: OutputFile
  frames: number
  durationMs: number
}

export type AssemblyProps = {
  clips: readonly AssemblyClip[]
  fps: number
  /** How many shots the reel has in total, so a partial cut says so. */
  shots: number
  /** Where the clips were written, for the command's working directory. */
  prefix: string
}

export function Assembly({ clips, fps, shots, prefix }: AssemblyProps) {
  const [copied, setCopied] = useState(false)
  const [cutting, setCutting] = useState(false)
  const [cut, setCut] = useState<Cut | null>(null)
  const [failed, setFailed] = useState<string | null>(null)
  const caps = useServerCapabilities()
  /** Null until the server answers; false when ffmpeg is not there to run. */
  const canCut = caps === null ? null : caps.stitch

  if (!clips.length) return null

  const frames = clips.reduce((n, c) => n + c.frames, 0)
  const spent = clips.reduce((n, c) => n + c.durationMs, 0)
  const partial = clips.length < shots
  const folder = prefix.replace(/\/+$/, '')
  const out = `${folder}/reel.webm`
  const command = buildCommand(clips, out)

  const copy = () => {
    void navigator.clipboard
      ?.writeText(command)
      .then(() => {
        setCopied(true)
        setTimeout(() => setCopied(false), 2500)
      })
      .catch(() => setCopied(false))
  }

  /**
   * Cut it. One request, one answer: `json=1` asks the server for the summary
   * rather than a progress stream, which is the right trade when the join is a
   * stream copy that lands in well under a second.
   */
  const make = () => {
    setCutting(true)
    setFailed(null)
    setCut(null)
    void fetch('/api/reel/stitch?json=1', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ clips: clips.map((c) => relPath(c.file)), prefix: `${folder}/reel` }),
    })
      .then(async (r) => {
        const body = (await r.json()) as Record<string, unknown>
        if (!r.ok) throw new Error(String(body.error ?? `HTTP ${r.status}`))
        setCut(body as unknown as Cut)
      })
      .catch((e: unknown) => setFailed(e instanceof Error ? e.message : String(e)))
      .finally(() => setCutting(false))
  }

  return (
    <section className="mt-8">
      <Head
        title="The cutting room"
        figure={`${clips.length} clips · ${seconds(frames, fps)}`}
        note="The clips are numbered in cutting order. Joining them is a stream copy, so nothing is re encoded."
      />

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
            <a className="sg-link" href={`/comfy/view?filename=${encodeURIComponent(cut.out.split('/').pop() ?? '')}&subfolder=${encodeURIComponent(cut.out.split('/').slice(0, -1).join('/'))}&type=output`} target="_blank" rel="noreferrer">
              {cut.out}
            </a>{' '}
            <span className="tabular-nums text-grey-500">
              {cut.actual.seconds.toFixed(2)} s · {grouped(cut.actual.frames)} frames · {mb(cut.actual.size)}
            </span>
          </p>
        ) : failed ? (
          <p className="text-caption text-ink-warning">{failed}</p>
        ) : (
          <p className="text-caption italic text-grey-500">
            The server does the joining. Nothing touches the card.
          </p>
        )}
      </div>

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
            <Quiet onClick={copy}>{copied ? 'Copied' : 'Copy the command'}</Quiet>
            <span className="text-caption italic text-grey-500">
              For doing it yourself. Set COMFY_OUTPUT to ComfyUI's output folder first.
            </span>
          </div>

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
                </span>
                <span className="shrink-0 text-caption tabular-nums text-grey-500">{seconds(c.frames, fps)}</span>
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
