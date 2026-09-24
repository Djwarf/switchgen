/**
 * The model picker, with the hardware verdicts.
 *
 * This is the ranking the recipe used, printed in full, with the file it chose
 * marked and every other file one press away. It is a picker because the old
 * desk had one, and it is a ranking rather than an alphabetical list because
 * the alphabet is the right order for finding a name and the wrong order for
 * choosing a checkpoint.
 *
 * WHY PICKING WRITES NO OVERRIDE. Choosing a different weight file changes far
 * more than the filename: the prompt prefix it was trained with, whether the
 * anatomy LoRAs can attach to it at all, which trigger tokens belong in the
 * positive, its sampler, its size, and which passes its graph can carry. All of
 * that is decide()'s work. So this panel pins the model and asks the desk to
 * decide again, rather than patching a name into a plan that was reasoned out
 * for a different file. The pin is shown as a pin, and dropping it hands the
 * choice back to the ranking.
 *
 * FOUR LISTS, NOT ONE. Ranked files can be chosen. Files that cannot load here
 * name the encoder, VAE or node pack they are missing. Blocked files will not
 * fit in this machine's memory and say by how much. Unrouted files are on disk
 * with no verified graph, which is a job for the registry rather than for the
 * reader. Collapsing them into one list is how a file that cannot run ends up
 * looking like a file nobody has tried.
 */
import { useState } from 'react'

import { gb } from '../../lib/hardware'
import { TAG_STYLE_NOTE, type IntentReport, type Recommendation } from '../../lib/intent'
import { Caution, Head, Kicker, Leader, Link, Note, Quiet } from './bits'

/** A weight file ComfyUI lists that cannot load here, and why, starting "needs". */
export type Unloadable = { model: string; label: string; why: string }

export function ModelPanel({
  report,
  familyId,
  model,
  label,
  pinned,
  onPin,
  measured,
  unloadable = [],
}: {
  report: IntentReport
  /** The family the recipe chose, or null when it could not choose one. */
  familyId: string | null
  model: string | null
  label: string | null
  /** The weight file the desk has pinned, when the reader pinned one. */
  pinned: string | null
  /**
   * Pin a weight file, or null to drop the pin. The desk answers by running
   * decide() again with that file as the only candidate, which is what keeps
   * the prefix, the LoRA fit and the trigger tokens correct for it.
   */
  onPin: (model: string | null) => void
  /** The checkpoint every sharpness ratio in this app was measured on. */
  measured: string
  /**
   * Files on disk that cannot load, each with what it lacks. Listed so the
   * sentence reaches the reader before a press, not as ComfyUI's refusal of
   * the job after it.
   */
  unloadable?: readonly Unloadable[]
}) {
  const [open, setOpen] = useState(false)
  const inUse = (r: Recommendation) => r.familyId === familyId && r.model === model

  const chosen = report.ranked.find(inUse) ?? null
  const shown = open ? report.ranked : report.ranked.slice(0, 4)
  const rest = report.ranked.length - shown.length

  return (
    <section className="mb-7">
      <Head
        title="Model"
        figure={report.ranked.length ? `${report.ranked.length} runnable` : undefined}
        note={report.note}
      />

      {pinned ? (
        <p className="mb-3 flex flex-wrap items-baseline gap-x-2 border-l-2 border-burgundy-900 pl-2.5 text-caption text-grey-700">
          <span className="text-overline font-semibold uppercase tracking-[0.14em] text-burgundy-900">
            Pinned
          </span>
          <span>{label ?? pinned} was chosen by hand, so the ranking is not deciding this.</span>
          <Link onClick={() => onPin(null)}>Let the recipe choose</Link>
        </p>
      ) : null}

      {chosen ? (
        <div className="mb-3">
          <Leader label="In use" value={chosen.label} />
          <Leader label="Family" value={chosen.familyLabel} />
          <Leader label="File" value={<span className="break-all">{chosen.model}</span>} />
          {chosen.verdict ? (
            <Leader
              label="Weights"
              value={`${gb(chosen.verdict.footprint.weightBytes)}, largest file ${gb(
                chosen.verdict.footprint.largestBytes,
              )}`}
            />
          ) : null}
          <Leader label="Prompting" value={<span className="normal-case">{chosen.tagStyle}</span>} />
          <Note>{TAG_STYLE_NOTE[chosen.tagStyle]}</Note>
          {chosen.model === measured ? (
            <Note>
              This is the checkpoint every sharpness ratio in this app was measured on, so the
              figures apply directly here.
            </Note>
          ) : (
            <Note>
              The sharpness figures were measured on {measured}, not on this file. Treat them as a
              starting point.
            </Note>
          )}
        </div>
      ) : null}

      <ol className="border-t border-grey-300">
        {shown.map((r) => (
          <Ranked
            key={`${r.familyId}:${r.model}`}
            rec={r}
            inUse={inUse(r)}
            pinned={pinned === r.model}
            onPin={() => onPin(r.model)}
          />
        ))}
      </ol>

      {rest > 0 ? (
        <p className="mt-2 text-caption">
          <Link onClick={() => setOpen(true)}>Show the other {rest} in the ranking</Link>
        </p>
      ) : null}
      {open && report.ranked.length > 4 ? (
        <p className="mt-2 text-caption">
          <Link onClick={() => setOpen(false)}>Show the top four only</Link>
        </p>
      ) : null}

      {unloadable.length > 0 ? (
        // Open when nothing ranks, because then this is the answer to why.
        <details className="mt-4" open={report.ranked.length === 0}>
          <summary className="cursor-pointer text-caption text-grey-500 [@media(pointer:coarse)]:min-h-11">
            Cannot load here ({unloadable.length})
          </summary>
          <ul className="mt-1 border-t border-grey-300">
            {unloadable.map((u) => (
              <li key={u.model} className="border-b border-grey-300 py-1.5">
                <span className="block text-caption text-ink">{u.label}</span>
                <span className="block text-caption italic leading-snug text-grey-500">
                  {u.why.charAt(0).toUpperCase()}
                  {u.why.slice(1)}
                  {u.why.endsWith('.') ? '' : '.'}
                </span>
              </li>
            ))}
          </ul>
        </details>
      ) : null}

      {report.blocked.length > 0 ? (
        <details className="mt-4">
          <summary className="cursor-pointer text-caption text-grey-500">
            Too large for this machine ({report.blocked.length})
          </summary>
          <ul className="mt-1 border-t border-grey-300">
            {report.blocked.map((b) => (
              <li key={b.model} className="border-b border-grey-300 py-1.5">
                <span className="block text-caption text-ink">{b.label}</span>
                <span className="block text-caption italic leading-snug text-grey-500">{b.why}</span>
              </li>
            ))}
          </ul>
        </details>
      ) : null}

      {report.unrouted.length > 0 ? (
        <details className="mt-3">
          <summary className="cursor-pointer text-caption text-grey-500">
            On disk, not wired up ({report.unrouted.length})
          </summary>
          <ul className="mt-1 border-t border-grey-300">
            {report.unrouted.map((u) => (
              <li key={u.model} className="border-b border-grey-300 py-1.5">
                <span className="block text-caption text-ink">{u.label}</span>
                <span className="block text-caption italic leading-snug text-grey-500">{u.why}</span>
              </li>
            ))}
          </ul>
          <Note>
            Offering one of these means a verified graph in src/lib/registry.ts, which is generated
            rather than written by hand.
          </Note>
        </details>
      ) : null}
    </section>
  )
}

function Ranked({
  rec,
  inUse,
  pinned,
  onPin,
}: {
  rec: Recommendation
  inUse: boolean
  pinned: boolean
  onPin: () => void
}) {
  const v = rec.verdict
  return (
    <li className={`border-b border-grey-300 py-2 ${inUse ? 'bg-newsprint-aged' : ''}`}>
      <div className="flex items-baseline justify-between gap-3">
        <span className="min-w-0 text-small font-semibold leading-tight">
          {rec.rank}. {rec.label}
          {inUse ? <span className="font-normal italic text-grey-500"> in use</span> : null}
          {pinned ? (
            <span className="ml-1 text-overline font-semibold uppercase tracking-[0.14em] text-burgundy-900">
              pinned
            </span>
          ) : null}
        </span>
        <span className="shrink-0 text-caption tabular-nums text-grey-500">{rec.score}</span>
      </div>

      <p className="mt-0.5 text-caption leading-snug text-grey-700">{rec.why}</p>

      {v && v.level !== 'ok' ? (
        <p className="mt-1">
          <Caution>{v.reason}</Caution>
        </p>
      ) : null}
      {v && v.level === 'ok' && v.offloads ? (
        <p className="mt-0.5 text-caption leading-snug text-grey-700">
          <Kicker>Heavy</Kicker>
          The largest file is {gb(v.footprint.largestBytes)}, more than the card holds, so it streams
          from memory and runs slower.
        </p>
      ) : null}
      {v && v.level === 'ok' && !v.offloads ? (
        <p className="mt-0.5 text-caption italic text-grey-500">{v.reason}</p>
      ) : null}

      {rec.warning ? <p className="mt-0.5 text-caption leading-snug text-warning">{rec.warning}</p> : null}
      {rec.caveat ? (
        <p className="mt-0.5 text-caption italic leading-snug text-grey-500">{rec.caveat}</p>
      ) : null}
      {!rec.def.verified ? (
        <p className="mt-0.5 text-caption italic leading-snug text-grey-500">
          These settings come from the model’s card and have not been checked against a live run.
        </p>
      ) : null}

      {!inUse || !pinned ? (
        <p className="mt-1.5">
          <Quiet onClick={onPin} disabled={pinned}>
            {pinned ? 'Pinned' : inUse ? 'Pin this one' : 'Use this one'}
          </Quiet>
        </p>
      ) : null}
    </li>
  )
}
