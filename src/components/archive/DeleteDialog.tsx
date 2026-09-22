/**
 * The one dialog in the archive, and it exists for exactly one action:
 * deleting a file from disk.
 *
 * Removing a record never comes through here — that is cheap and undoable, and
 * asking about it would teach people to click past the question that matters.
 *
 * The copy states what actually happens on this machine. `POST /api/delete`
 * unlinks the file; there is no trash folder to fall back on, so the dialog
 * does not offer one.
 */
import { useEffect, useRef, useState } from 'react'
import type { HistoryEntry } from '../../lib/history'
import { relPath } from '../../lib/comfy'

type Props = {
  records: readonly HistoryEntry[]
  busy: boolean
  error: string | null
  onCancel: () => void
  onConfirm: () => void
}

const OUTPUTS = '/mnt/storage/ai/outputs'
/** Above this many files, the count has to be typed out. */
const TYPE_THRESHOLD = 5

export function DeleteDialog({ records, busy, error, onCancel, onConfirm }: Props) {
  const panel = useRef<HTMLDivElement | null>(null)
  const cancel = useRef<HTMLButtonElement | null>(null)
  const [typed, setTyped] = useState('')

  const many = records.length > 1
  const needsTyping = records.length > TYPE_THRESHOLD
  const ready = !busy && (!needsTyping || typed.trim() === String(records.length))

  useEffect(() => {
    cancel.current?.focus()
    const key = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault()
        e.stopPropagation()
        if (!busy) onCancel()
        return
      }
      if (e.key !== 'Tab') return
      const focusable = panel.current?.querySelectorAll<HTMLElement>(
        'button:not(:disabled), input:not(:disabled), [href]',
      )
      if (!focusable || !focusable.length) return
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault()
        last.focus()
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault()
        first.focus()
      }
    }
    document.addEventListener('keydown', key, true)
    return () => document.removeEventListener('keydown', key, true)
  }, [busy, onCancel])

  const shown = records.slice(0, 5)

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink/40 px-4 pt-[calc(1rem+var(--sg-safe-t))] pb-[calc(1rem+var(--sg-safe-b))]"
      onPointerDown={(e) => {
        if (e.target === e.currentTarget && !busy) onCancel()
      }}
    >
      <div
        ref={panel}
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="delete-title"
        className="flex max-h-[calc(100dvh-2rem-var(--sg-safe-t)-var(--sg-safe-b))] w-full max-w-xl flex-col overflow-y-auto border-2 border-burgundy-900 bg-newsprint p-6"
      >
        <h2
          id="delete-title"
          className="border-b-2 border-burgundy-900 pb-1 text-[0.625rem] font-semibold tracking-[0.18em] text-burgundy-900 uppercase"
        >
          {many ? `Delete ${records.length} files` : 'Delete the file'}
        </h2>

        <p className="mt-4 text-body leading-relaxed">
          {many ? (
            <>
              This removes {records.length} files from <span className="text-grey-700">{OUTPUTS}</span>.
            </>
          ) : (
            <>
              This removes{' '}
              <span className="font-semibold">{relPath(records[0].file)}</span> from{' '}
              <span className="text-grey-700">{OUTPUTS}</span>.
            </>
          )}
        </p>

        <p className="mt-2 text-body leading-relaxed">
          The {many ? 'files go' : 'file goes'} at once and for good. There is no trash folder on
          this machine, so we cannot put {many ? 'them' : 'it'} back.{' '}
          {many ? 'Their records leave' : 'The record leaves'} the archive at the same time.
        </p>

        {many && (
          <ul className="mt-3 border-t border-grey-300 pt-2 text-caption text-grey-700 tabular-nums">
            {shown.map((r) => (
              <li key={r.id} className="truncate">
                {relPath(r.file)}
              </li>
            ))}
            {records.length > shown.length && (
              <li className="text-grey-500 italic">
                and {records.length - shown.length} more
              </li>
            )}
          </ul>
        )}

        <p className="mt-3 text-small text-grey-700 italic">
          To keep the {many ? 'files' : 'file'} and only stop seeing{' '}
          {many ? 'these records' : 'this record'}, cancel and choose{' '}
          <span className="not-italic">Remove from the archive</span> instead.
        </p>

        {needsTyping && (
          <label className="mt-4 block">
            <span className="block text-[0.625rem] font-semibold tracking-[0.18em] text-grey-700 uppercase">
              Type {records.length} to confirm
            </span>
            <input
              value={typed}
              onChange={(e) => setTyped(e.target.value)}
              inputMode="numeric"
              autoComplete="off"
              className="mt-1 w-24 border border-grey-300 bg-transparent px-2 py-1 font-serif text-body tabular-nums focus:border-burgundy-900 focus:outline-none"
            />
          </label>
        )}

        {error && (
          <p className="mt-4 border-l-4 border-error bg-paper-error px-4 py-3 text-small text-ink-error">
            <strong className="mr-2 font-bold tracking-[0.05em] uppercase">
              We could not delete that
            </strong>
            {error}
          </p>
        )}

        <div className="mt-6 flex items-center gap-4">
          <button
            type="button"
            onClick={onConfirm}
            disabled={!ready}
            className="border border-error bg-error px-4 py-2 text-[0.75rem] font-semibold tracking-[0.16em] text-newsprint uppercase hover:bg-newsprint hover:text-error disabled:border-grey-300 disabled:bg-grey-300 disabled:text-grey-500 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900"
          >
            {busy ? 'Deleting…' : many ? `Delete ${records.length} files` : 'Delete the file'}
          </button>
          <button
            ref={cancel}
            type="button"
            onClick={onCancel}
            disabled={busy}
            className="text-[0.75rem] font-semibold tracking-[0.16em] text-grey-700 uppercase underline underline-offset-4 hover:text-burgundy-900 hover:no-underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  )
}
