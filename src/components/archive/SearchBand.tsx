/**
 * One line, the full width of the page, under a rule. No pill, no magnifying
 * glass, no button: you type and the archive narrows.
 *
 * The box is always the truth about the current query — the facets write into
 * it — so it is worth reading, and it is hand-editable.
 */
import type { RefObject } from 'react'

export type KindFilter = 'all' | 'image' | 'video'

type Props = {
  value: string
  onChange: (value: string) => void
  /** How many records the query left, and how many there are altogether. */
  shown: number
  total: number
  inputRef?: RefObject<HTMLInputElement | null>
  /** Pictures, clips or everything. Writes `is:image` / `is:video` into the query. */
  kind: KindFilter
  onKind: (kind: KindFilter) => void
  view: 'grid' | 'list'
  onView: (view: 'grid' | 'list') => void
  sort: 'newest' | 'oldest'
  onSort: (sort: 'newest' | 'oldest') => void
}

const KINDS: { id: KindFilter; label: string }[] = [
  { id: 'all', label: 'Everything' },
  { id: 'image', label: 'Pictures' },
  { id: 'video', label: 'Clips' },
]

const chip =
  'px-2 py-1 text-[0.625rem] font-semibold tracking-[0.18em] uppercase focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900'

export function SearchBand({
  value,
  onChange,
  shown,
  total,
  inputRef,
  kind,
  onKind,
  view,
  onView,
  sort,
  onSort,
}: Props) {
  const narrowed = value.trim().length > 0

  return (
    <div className="bg-newsprint pt-1">
      <div className="flex flex-col items-stretch gap-3 lg:flex-row lg:items-end lg:gap-4">
        <div className="min-w-0 flex-1">
          <label
            htmlFor="archive-search"
            className="block text-[0.625rem] font-semibold tracking-[0.18em] text-grey-700 uppercase"
          >
            Search the archive
          </label>
          <div className="relative">
            <input
              id="archive-search"
              ref={inputRef}
              type="search"
              value={value}
              spellCheck={false}
              autoComplete="off"
              placeholder="Anything you wrote, or a name, or a number"
              onChange={(e) => onChange(e.target.value)}
              className="w-full border-b border-grey-300 bg-transparent py-2 pr-8 font-serif text-[1.125rem] text-ink outline-none placeholder:text-grey-400 placeholder:italic focus:border-b-2 focus:border-burgundy-900 focus:pb-[calc(0.5rem-1px)]"
            />
            {narrowed && (
              <button
                type="button"
                onClick={() => onChange('')}
                aria-label="Clear the search"
                title="Clear the search (Esc)"
                className="absolute right-0 bottom-2 px-2 text-grey-500 hover:text-burgundy-900 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900"
              >
                ×
              </button>
            )}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3 lg:shrink-0 lg:gap-4 lg:pb-2">
          <span className="flex items-center border border-grey-300" role="group" aria-label="What to show">
            {KINDS.map((option, i) => (
              <button
                key={option.id}
                type="button"
                onClick={() => onKind(option.id)}
                aria-pressed={kind === option.id}
                className={`${chip} ${i > 0 ? 'border-l border-grey-300' : ''} ${
                  kind === option.id ? 'bg-ink text-newsprint' : 'text-grey-700'
                }`}
              >
                {option.label}
              </button>
            ))}
          </span>

          <span className="flex items-center border border-grey-300">
            <button
              type="button"
              onClick={() => onView('grid')}
              aria-pressed={view === 'grid'}
              className={`${chip} ${view === 'grid' ? 'bg-ink text-newsprint' : 'text-grey-700'}`}
            >
              Grid
            </button>
            <button
              type="button"
              onClick={() => onView('list')}
              aria-pressed={view === 'list'}
              className={`${chip} border-l border-grey-300 ${
                view === 'list' ? 'bg-ink text-newsprint' : 'text-grey-700'
              }`}
            >
              List
            </button>
          </span>

          <button
            type="button"
            onClick={() => onSort(sort === 'newest' ? 'oldest' : 'newest')}
            className={`${chip} text-grey-700 hover:text-burgundy-900`}
            title="Change the order"
          >
            {sort === 'newest' ? 'Newest first' : 'Oldest first'}
          </button>
        </div>
      </div>

      <p className="mt-2 text-[0.625rem] font-semibold tracking-[0.18em] text-grey-500 uppercase tabular-nums">
        {narrowed
          ? `${shown.toLocaleString('en-GB')} of ${total.toLocaleString('en-GB')}`
          : `${total.toLocaleString('en-GB')} ${total === 1 ? 'record' : 'records'}`}
        {!narrowed && total > 0 && (
          <span className="ml-3 tracking-[0.06em] text-grey-500 normal-case italic">
            Try <code>is:video</code>, <code>model:krea</code>, <code>steps:&gt;20</code> or{' '}
            <code>after:2026-09-01</code>
          </span>
        )}
      </p>
    </div>
  )
}
