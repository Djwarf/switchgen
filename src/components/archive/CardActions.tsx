/**
 * What you can do with a record, in plain verbs.
 *
 * The two destructive verbs are deliberately unlike each other. Removing a
 * record is cheap, undoable and sits in the open. Deleting a file is a
 * different sentence in a different place, it always asks first, and it has no
 * key binding at all.
 */
import { useEffect, useRef, useState } from 'react'
import type { DeskId } from '../../lib/session'
import type { HistoryEntry } from '../../lib/history'

export type EntryActions = {
  /** Open the full record. */
  onOpen: () => void
  /** Load everything back into its desk, ready to run. */
  onReuse: () => void
  /** The same, with a fresh seed. */
  onAnother: () => void
  /** Send only the picture to a desk. */
  onSource: (desk: DeskId) => void
  /**
   * Open the region bench on this picture, over on the Pictures desk.
   * Absent on a video record, where there is no single frame to paint on.
   */
  onRegion?: () => void
  onStar: () => void
  onDownload: () => void
  /** Remove the record. The file stays. */
  onRemove: () => void
  /** Remove the record and the file. Always confirms first. */
  onDeleteFile: () => void
  onSelect: () => void
}

type Props = EntryActions & {
  entry: HistoryEntry
  canDeleteFile: boolean
  selected: boolean
  /** Rendered flat, in a detail view, rather than as a card's hover row. */
  expanded?: boolean
}

const link =
  'inline-flex items-center text-[0.625rem] font-semibold tracking-[0.16em] uppercase text-grey-700 hover:text-burgundy-900 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900 [@media(pointer:coarse)]:min-h-11'

const item =
  'block w-full px-3 py-1.5 text-left text-small text-ink hover:bg-newsprint-aged focus-visible:bg-newsprint-aged focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-burgundy-900 disabled:text-grey-400 disabled:hover:bg-transparent'

export function CardActions({
  entry,
  canDeleteFile,
  selected,
  expanded = false,
  onOpen,
  onReuse,
  onAnother,
  onSource,
  onRegion,
  onStar,
  onDownload,
  onRemove,
  onDeleteFile,
  onSelect,
}: Props) {
  const [open, setOpen] = useState(false)
  const wrap = useRef<HTMLDivElement | null>(null)
  const menu = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!open) return
    const away = (e: PointerEvent) => {
      if (!wrap.current?.contains(e.target as Node)) setOpen(false)
    }
    // Capture, so nothing on `window` sees these first. While the menu is up it
    // owns the keyboard: the archive binds bare single keys there, `Backspace`
    // among them, and focus sitting on a `<button role="menuitem">` does not
    // look like typing, so without this the keys fire behind the open menu at
    // whichever record is focused. Tab, Enter, Space and the arrows are left
    // alone, because they are how the menu itself is worked.
    const key = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation()
        setOpen(false)
        return
      }
      if (e.ctrlKey || e.metaKey || e.altKey) return
      if (e.key === 'Backspace' || (e.key.length === 1 && e.key !== ' ')) e.stopPropagation()
    }
    document.addEventListener('pointerdown', away)
    document.addEventListener('keydown', key, true)
    menu.current?.querySelector<HTMLButtonElement>('button:not(:disabled)')?.focus()
    return () => {
      document.removeEventListener('pointerdown', away)
      document.removeEventListener('keydown', key, true)
    }
  }, [open])

  const run = (fn: () => void) => () => {
    setOpen(false)
    fn()
  }

  const isVideo = entry.kind === 'video'

  return (
    <div ref={wrap} className="relative flex items-center gap-3">
      <button type="button" className={link} onClick={onReuse} title="Load these settings (r)">
        Use these settings
      </button>
      <span className="text-grey-300" aria-hidden>
        ·
      </span>
      <button type="button" className={link} onClick={onAnother} title="Load it with a fresh seed (Shift+R)">
        Make another
      </button>

      {expanded && (
        <>
          <span className="text-grey-300" aria-hidden>
            ·
          </span>
          <button type="button" className={link} onClick={onRemove} title="Remove the record (Backspace)">
            Remove
          </button>
        </>
      )}

      <span className="flex-1" />

      <button
        type="button"
        className={`${link} px-1`}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label="More actions"
        onClick={() => setOpen((v) => !v)}
      >
        ⋯
      </button>

      {open && (
        <div
          ref={menu}
          role="menu"
          className="absolute top-full right-0 z-30 mt-1 w-64 border border-grey-300 bg-newsprint py-1"
        >
          <button role="menuitem" className={item} onClick={run(onOpen)}>
            Open the full record
          </button>
          <button role="menuitem" className={item} onClick={run(onStar)}>
            {entry.starred ? 'Remove the star' : 'Star it'}
          </button>
          <button role="menuitem" className={item} onClick={run(onSelect)}>
            {selected ? 'Take it out of the selection' : 'Add it to the selection'}
          </button>

          <div className="my-1 border-t border-grey-300" />

          {isVideo ? (
            <button role="menuitem" className={item} onClick={run(onOpen)}>
              Take a frame from it…
              <span className="block text-caption text-grey-500 italic">
                Open it and use the player's frame picker.
              </span>
            </button>
          ) : (
            <>
              <button role="menuitem" className={item} onClick={run(() => onSource('images'))}>
                Use as a source on the Pictures desk
              </button>
              <button role="menuitem" className={item} onClick={run(() => onSource('video'))}>
                Use as a start frame on the Video desk
              </button>
              {onRegion ? (
                <button
                  role="menuitem"
                  className={item}
                  onClick={run(onRegion)}
                  disabled={entry.missing}
                >
                  Change part of it…
                  <span className="block text-caption text-grey-500 italic">
                    Paint over an area and have just that redrawn.
                  </span>
                </button>
              ) : null}
            </>
          )}

          <button role="menuitem" className={item} onClick={run(onDownload)} disabled={entry.missing}>
            {isVideo ? 'Save the clip' : 'Save the picture'}
          </button>

          <div className="my-1 border-t border-grey-300" />

          <button role="menuitem" className={item} onClick={run(onRemove)}>
            Remove from the archive
            <span className="block text-caption text-grey-500 italic">
              The file stays on disk. Undoable.
            </span>
          </button>

          {canDeleteFile && !entry.missing && (
            <button
              role="menuitem"
              className={`${item} text-error`}
              onClick={run(onDeleteFile)}
            >
              Delete the file…
              <span className="block text-caption text-grey-500 italic">
                Removes the file itself. We ask first.
              </span>
            </button>
          )}
        </div>
      )}
    </div>
  )
}
