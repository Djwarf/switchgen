/**
 * What you can do to several records at once. It appears only when something
 * is selected, and it names the count in every verb so nothing is ambiguous.
 */
type Props = {
  count: number
  starred: number
  deletable: number
  canDeleteFiles: boolean
  onStar: (starred: boolean) => void
  onRemove: () => void
  onDeleteFiles: () => void
  onSelectAll: () => void
  onClear: () => void
}

const action =
  'inline-flex items-center text-[0.625rem] font-semibold tracking-[0.16em] uppercase underline underline-offset-4 hover:no-underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900 [@media(pointer:coarse)]:min-h-11'

export function SelectionBar({
  count,
  starred,
  deletable,
  canDeleteFiles,
  onStar,
  onRemove,
  onDeleteFiles,
  onSelectAll,
  onClear,
}: Props) {
  if (!count) return null
  const allStarred = starred === count

  return (
    <div className="sticky bottom-0 z-30 -mx-1 mt-6 border-t-2 border-burgundy-900 bg-newsprint px-1 pt-2 pb-[calc(0.5rem+var(--sg-safe-b))]">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
        <span className="text-[0.625rem] font-semibold tracking-[0.18em] text-ink uppercase tabular-nums">
          {count.toLocaleString('en-GB')} selected
        </span>

        <button type="button" className={`${action} text-grey-700`} onClick={() => onStar(!allStarred)}>
          {allStarred ? 'Remove the stars' : 'Star them'}
        </button>

        <button type="button" className={`${action} text-grey-700`} onClick={onSelectAll}>
          Select everything shown
        </button>

        <button type="button" className={`${action} text-grey-700`} onClick={onRemove}>
          Remove {count === 1 ? 'the record' : `${count} records`}
        </button>

        {canDeleteFiles && deletable > 0 && (
          <button type="button" className={`${action} text-error`} onClick={onDeleteFiles}>
            Delete {deletable === 1 ? 'the file' : `${deletable} files`}…
          </button>
        )}

        <span className="flex-1" />

        <button type="button" className={`${action} text-grey-700`} onClick={onClear}>
          Clear the selection
        </button>
      </div>
      <p className="mt-1 text-caption text-grey-500 italic">
        Removing a record leaves the file on disk. Deleting a file cannot be undone.
      </p>
    </div>
  )
}
