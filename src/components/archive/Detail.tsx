/**
 * One record, in full.
 *
 * Simple mode hides decisions, never history: whatever the desk is set to, the
 * detail view shows the complete parameter record. Reading a fact is not being
 * offered a choice.
 *
 * Every number here is a link that adopts itself into the desk this record
 * belongs to, so reuse works at the granularity of a single value as well as
 * the whole recipe.
 */
import { useEffect, useRef, useState } from 'react'
import { fileUrl, relPath } from '../../lib/comfy'
import { get as getEntry, update as updateEntry, type HistoryEntry } from '../../lib/history'
import { adoptValue, type DeskId, type TunableField } from '../../lib/session'
import { ArchiveVideo } from './ArchiveVideo'
import { CardActions, type EntryActions } from './CardActions'
import { Notice } from '../type'
import { duration, editionNo, fullDate, madeFrom } from './query'

type Props = EntryActions & {
  entry: HistoryEntry
  canDeleteFile: boolean
  hasPrev: boolean
  hasNext: boolean
  onPrev: () => void
  onNext: () => void
  onClose: () => void
  onOpenEntry: (id: string) => void
}

/**
 * `adoptValue` is generic over the field, and the table walks fields at
 * runtime. One cast, here, rather than a cast at every row.
 */
/** Everything a person can reach with Tab. Used to hold Tab inside the scrim. */
const FOCUSABLE =
  'a[href], button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex]:not([tabindex="-1"])'

const adopt = adoptValue as unknown as (
  desk: DeskId,
  field: TunableField,
  value: unknown,
) => () => void

type RowProps = {
  label: string
  value: string | number | null | undefined
  field?: TunableField
  raw?: unknown
  onAdopt?: (field: TunableField, raw: unknown, label: string) => void
  wrap?: boolean
}

function Row({ label, value, field, raw, onAdopt, wrap = false }: RowProps) {
  if (value === null || value === undefined || value === '') return null
  const text = typeof value === 'number' ? value.toLocaleString('en-GB') : value
  return (
    <div className="flex items-baseline gap-4 border-b border-grey-300 py-1.5">
      <dt className="w-36 shrink-0 text-[0.625rem] font-semibold tracking-[0.18em] text-grey-700 uppercase">
        {label}
      </dt>
      <dd className={`flex-1 text-small tabular-nums ${wrap ? 'break-words' : 'truncate'}`}>
        {field && onAdopt ? (
          <button
            type="button"
            onClick={() => onAdopt(field, raw ?? value, label)}
            title={`Use this ${label.toLowerCase()} on the desk`}
            className="text-left underline decoration-grey-300 underline-offset-4 hover:text-burgundy-900 hover:decoration-burgundy-900 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900"
          >
            {text}
          </button>
        ) : (
          text
        )}
      </dd>
    </div>
  )
}

export function Detail({
  entry,
  canDeleteFile,
  hasPrev,
  hasNext,
  onPrev,
  onNext,
  onClose,
  onOpenEntry,
  ...actions
}: Props) {
  const panel = useRef<HTMLDivElement | null>(null)
  const [note, setNote] = useState(entry.note ?? '')
  const [flash, setFlash] = useState<{ label: string; undo: () => void } | null>(null)

  // Only when the record itself changes — never on a keystroke, which would
  // pull focus out of the note the reader is writing.
  const shown = useRef(entry.id)
  useEffect(() => {
    if (shown.current === entry.id) return
    shown.current = entry.id
    setNote(entry.note ?? '')
    setFlash(null)
    panel.current?.focus()
  }, [entry.id, entry.note])

  // The first open needs the focus too, so Esc and j/k land somewhere sensible.
  // The page behind is held still while the record is up, so a stray wheel
  // gesture does not lose the reader's place in the run.
  //
  // Tab is held inside as well. This scrim is opaque and covers the viewport,
  // but the run underneath is still in the tab order, so without this one Tab
  // puts the caret somewhere nobody can see. Where the reader came from is kept
  // and handed back on close, for the same reason.
  useEffect(() => {
    const cameFrom = document.activeElement as HTMLElement | null
    panel.current?.focus()
    const behind = document.body.style.overflow
    document.body.style.overflow = 'hidden'

    const trap = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return
      const root = panel.current
      if (!root) return
      const focusable = root.querySelectorAll<HTMLElement>(FOCUSABLE)
      if (!focusable.length) return
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      const active = document.activeElement
      if (!active || !root.contains(active)) {
        e.preventDefault()
        first.focus()
      } else if (e.shiftKey && (active === first || active === root)) {
        e.preventDefault()
        last.focus()
      } else if (!e.shiftKey && active === last) {
        e.preventDefault()
        first.focus()
      }
    }

    document.addEventListener('keydown', trap, true)
    return () => {
      document.removeEventListener('keydown', trap, true)
      document.body.style.overflow = behind
      cameFrom?.focus?.()
    }
  }, [])

  useEffect(() => {
    if (!flash) return
    const t = setTimeout(() => setFlash(null), 4000)
    return () => clearTimeout(t)
  }, [flash])

  // The note is the reader's marginalia; it saves itself.
  useEffect(() => {
    if ((entry.note ?? '') === note) return
    const t = setTimeout(() => updateEntry(entry.id, { note: note || undefined }), 400)
    return () => clearTimeout(t)
  }, [note, entry.id, entry.note])

  const onAdopt = (field: TunableField, raw: unknown, label: string) => {
    const undo = adopt(entry.desk, field, raw)
    setFlash({ label, undo })
  }

  const source = entry.source
  const lineage = source?.fromEntryId ? getEntry(source.fromEntryId) : undefined

  return (
    <div
      className="fixed inset-0 z-40 overflow-y-auto bg-newsprint"
      role="dialog"
      aria-modal="true"
      aria-label={`Record ${entry.no}`}
      // The archive's own overlay. ArchivePage reads this to know its j/k/Esc
      // bindings should stay live here, and stand down under anything else.
      data-archive-overlay="true"
    >
      <div
        ref={panel}
        tabIndex={-1}
        className="mx-auto max-w-[92rem] px-6 pt-[calc(1rem+var(--sg-safe-t))] pb-[calc(4rem+var(--sg-safe-b))] focus:outline-none"
      >
        <div className="flex flex-wrap items-center justify-between gap-y-1 border-b-2 border-burgundy-900 pb-1">
          <h2 className="text-[0.625rem] font-semibold tracking-[0.18em] text-burgundy-900 uppercase tabular-nums">
            {editionNo(entry.no)}
            <span className="ml-3 text-grey-500">{madeFrom(entry)}</span>
          </h2>
          <div className="flex items-center gap-4">
            <button
              type="button"
              onClick={onPrev}
              disabled={!hasPrev}
              className="text-[0.625rem] font-semibold tracking-[0.16em] text-grey-700 uppercase hover:text-burgundy-900 disabled:text-grey-300 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900"
              title="The record before this one (k)"
            >
              ← Previous
            </button>
            <button
              type="button"
              onClick={onNext}
              disabled={!hasNext}
              className="text-[0.625rem] font-semibold tracking-[0.16em] text-grey-700 uppercase hover:text-burgundy-900 disabled:text-grey-300 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900"
              title="The next record (j)"
            >
              Next →
            </button>
            <button
              type="button"
              onClick={onClose}
              className="text-[0.625rem] font-semibold tracking-[0.16em] text-burgundy-900 uppercase underline underline-offset-4 hover:no-underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900"
              title="Close (Esc)"
            >
              Close
            </button>
          </div>
        </div>

        <div className="mt-6 grid gap-8 lg:grid-cols-[minmax(0,1.6fr)_minmax(22rem,1fr)]">
          <div>
            {entry.missing ? (
              <Notice tone="correction" title="Not on disk">
                This file has been moved or deleted. Everything that made it is still here, so you
                can make it again.
              </Notice>
            ) : entry.kind === 'video' ? (
              <ArchiveVideo entry={entry} />
            ) : (
              <div className="flex items-center justify-center border border-grey-300 bg-newsprint-aged">
                <img
                  src={fileUrl(entry.file)}
                  alt={entry.prompt || 'Archive picture'}
                  className="max-h-[70vh] w-auto max-w-full object-contain"
                />
              </div>
            )}

            {entry.recovered ? (
              <p className="mt-4 text-caption italic text-grey-700">
                {entry.prompt
                  ? "Settings read back from ComfyUI's history, not filed by the desk that made it."
                  : 'Filed from the outputs folder by name and date. ComfyUI no longer remembers how it was made.'}
              </p>
            ) : null}

            <blockquote className="mt-6 border-l-4 border-burgundy-900 pl-5 font-serif text-[1.5rem] leading-snug break-words italic">
              {entry.prompt || 'No prompt was recorded for this one.'}
            </blockquote>

            <p className="mt-3 text-small text-grey-700">
              Made by {entry.modelLabel || entry.familyLabel} · {fullDate(entry.at)} ·{' '}
              <span className="tabular-nums">{duration(entry.durationMs)}</span>
            </p>

            <div className="mt-4 border-t border-grey-300 pt-3">
              <CardActions
                entry={entry}
                canDeleteFile={canDeleteFile}
                selected={false}
                expanded
                {...actions}
              />
            </div>

            {flash && (
              <p className="mt-4 text-[0.625rem] font-semibold tracking-[0.18em] text-ink uppercase">
                {flash.label} adopted by the {entry.desk === 'video' ? 'Video' : 'Pictures'} desk
                <button
                  type="button"
                  onClick={() => {
                    flash.undo()
                    setFlash(null)
                  }}
                  className="ml-3 text-burgundy-900 underline underline-offset-4 hover:no-underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900"
                >
                  Undo
                </button>
              </p>
            )}
          </div>

          <div>
            <h3 className="border-b-2 border-burgundy-900 pb-1 text-[0.625rem] font-semibold tracking-[0.18em] text-burgundy-900 uppercase">
              The record
            </h3>
            <p className="mt-2 text-caption text-grey-500 italic">
              Every value here is a link. Click one and that setting alone moves to the desk.
            </p>

            <dl className="mt-3">
              <Row label="Style" value={entry.familyLabel} />
              <Row label="Model" value={entry.modelLabel} />
              <Row label="Weight file" value={entry.model} wrap />
              <Row label="Variant" value={entry.variant ?? undefined} />
              <Row label="Mode" value={entry.mode} />
              <Row
                label="Size"
                value={entry.width && entry.height ? `${entry.width} × ${entry.height}` : null}
              />
              <Row label="Width" value={entry.width} field="width" raw={entry.width} onAdopt={onAdopt} />
              <Row label="Height" value={entry.height} field="height" raw={entry.height} onAdopt={onAdopt} />
              <Row
                label="Output size"
                value={entry.megapixels ? `${entry.megapixels} MP` : null}
                field="megapixels"
                raw={entry.megapixels}
                onAdopt={onAdopt}
              />
              <Row label="Steps" value={entry.steps} field="steps" raw={entry.steps} onAdopt={onAdopt} />
              <Row label="CFG" value={entry.cfg.toFixed(1)} field="cfg" raw={entry.cfg} onAdopt={onAdopt} />
              <Row label="Sampler" value={entry.sampler} field="sampler" raw={entry.sampler} onAdopt={onAdopt} />
              <Row
                label="Scheduler"
                value={entry.scheduler}
                field="scheduler"
                raw={entry.scheduler}
                onAdopt={onAdopt}
              />
              <Row label="Seed" value={entry.seed} field="seed" raw={entry.seed} onAdopt={onAdopt} />
              <Row
                label="How much changed"
                value={entry.denoise !== undefined ? entry.denoise.toFixed(2) : null}
                field="denoise"
                raw={entry.denoise}
                onAdopt={onAdopt}
              />
              <Row label="Frames" value={entry.length} field="length" raw={entry.length} onAdopt={onAdopt} />
              <Row label="Frame rate" value={entry.fps ? `${entry.fps} fps` : null} field="fps" raw={entry.fps} onAdopt={onAdopt} />
              <Row label="Shift" value={entry.shift} field="shift" raw={entry.shift} onAdopt={onAdopt} />
              <Row label="Clip skip" value={entry.clipSkip} field="clipSkip" raw={entry.clipSkip} onAdopt={onAdopt} />
              <Row
                label="Prefix"
                value={entry.positivePrefix}
                field="positivePrefix"
                raw={entry.positivePrefix}
                onAdopt={onAdopt}
                wrap
              />
              <Row
                label="Negative"
                value={entry.negative ?? 'The house wording'}
                field={entry.negative ? 'negative' : undefined}
                raw={entry.negative}
                onAdopt={entry.negative ? onAdopt : undefined}
                wrap
              />
              <Row label="Took" value={duration(entry.durationMs)} />
              <Row label="File" value={relPath(entry.file)} wrap />
              <Row label="Job" value={entry.promptId} wrap />
            </dl>

            {source && (
              <div className="mt-6">
                <h3 className="border-b border-grey-300 pb-1 text-[0.625rem] font-semibold tracking-[0.18em] text-grey-700 uppercase">
                  Made from
                </h3>
                <p className="mt-2 text-small text-grey-700">
                  {source.ref ? relPath(source.ref) : source.name}
                  {source.fromFrame !== undefined && (
                    <span className="tabular-nums"> · frame {source.fromFrame}</span>
                  )}
                </p>
                {lineage && (
                  <button
                    type="button"
                    onClick={() => onOpenEntry(lineage.id)}
                    className="mt-1 text-small text-burgundy-900 underline underline-offset-4 hover:no-underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900"
                  >
                    Open {editionNo(lineage.no)}, which it came from
                  </button>
                )}
              </div>
            )}

            <div className="mt-6">
              <label
                htmlFor="record-note"
                className="block border-b border-grey-300 pb-1 text-[0.625rem] font-semibold tracking-[0.18em] text-grey-700 uppercase"
              >
                Your note
              </label>
              <textarea
                id="record-note"
                value={note}
                onChange={(e) => setNote(e.target.value)}
                rows={3}
                placeholder="What you would change next time"
                className="mt-2 w-full resize-y border border-grey-300 bg-transparent px-2 py-1.5 font-serif text-small leading-relaxed focus:border-burgundy-900 focus:outline-none placeholder:text-grey-400 placeholder:italic"
              />
              <p className="mt-1 text-caption text-grey-500 italic">
                Saved as you type, and searchable.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
