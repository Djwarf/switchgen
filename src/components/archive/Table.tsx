/**
 * The dense view: one line per record, every number in tabular figures, and
 * sortable heads. This is the view you use when you are hunting for the run
 * where you set CFG to 3.5, not the one where you are looking at pictures.
 *
 * Memoised for the same reason as the grid: typing in the search box must not
 * redraw every row before the results have even changed.
 */
import { memo, type MouseEvent } from 'react'
import type { HistoryEntry } from '../../lib/history'
import { CardActions, type EntryActions } from './CardActions'
import { Highlight, clockTime, dimensions } from './query'

export type SortKey = 'no' | 'at' | 'steps' | 'cfg' | 'seed' | 'model'

type Props = {
  entries: readonly HistoryEntry[]
  terms: readonly string[]
  sortKey: SortKey
  ascending: boolean
  onSort: (key: SortKey) => void
  focusedId: string | null
  selected: ReadonlySet<string>
  canDeleteFile: boolean
  onActivate: (entry: HistoryEntry, e: MouseEvent) => void
  onFocused: (entry: HistoryEntry) => void
  onToggleSelect: (entry: HistoryEntry, e: MouseEvent) => void
  registerRef: (entry: HistoryEntry, el: HTMLElement | null) => void
  actionsFor: (entry: HistoryEntry) => EntryActions
}

const head =
  'sticky top-[calc(var(--sg-bar-h)+var(--sg-safe-t))] z-10 bg-newsprint px-2 py-2 text-left text-[0.625rem] font-semibold tracking-[0.18em] uppercase text-grey-700'

function Head({
  label,
  k,
  sortKey,
  ascending,
  onSort,
  align = 'left',
}: {
  label: string
  k?: SortKey
  sortKey: SortKey
  ascending: boolean
  onSort: (key: SortKey) => void
  align?: 'left' | 'right'
}) {
  const active = k !== undefined && k === sortKey
  return (
    <th scope="col" className={`${head} ${align === 'right' ? 'text-right' : ''}`} aria-sort={
      active ? (ascending ? 'ascending' : 'descending') : undefined
    }>
      {k ? (
        <button
          type="button"
          onClick={() => onSort(k)}
          className={`tracking-[0.18em] uppercase focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900 ${
            active ? 'text-ink underline underline-offset-4' : 'hover:text-ink'
          }`}
        >
          {label}
          {active && <span aria-hidden>{ascending ? ' ↑' : ' ↓'}</span>}
        </button>
      ) : (
        label
      )}
    </th>
  )
}

export const Table = memo(function Table({
  entries,
  terms,
  sortKey,
  ascending,
  onSort,
  focusedId,
  selected,
  canDeleteFile,
  onActivate,
  onFocused,
  onToggleSelect,
  registerRef,
  actionsFor,
}: Props) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-small">
        <thead>
          <tr className="border-b-2 border-burgundy-900">
            <Head label="No." k="no" sortKey={sortKey} ascending={ascending} onSort={onSort} />
            <Head label="When" k="at" sortKey={sortKey} ascending={ascending} onSort={onSort} />
            <Head label="Style" k="model" sortKey={sortKey} ascending={ascending} onSort={onSort} />
            <Head label="Prompt" sortKey={sortKey} ascending={ascending} onSort={onSort} />
            <Head label="Size" sortKey={sortKey} ascending={ascending} onSort={onSort} align="right" />
            <Head label="Steps" k="steps" sortKey={sortKey} ascending={ascending} onSort={onSort} align="right" />
            <Head label="CFG" k="cfg" sortKey={sortKey} ascending={ascending} onSort={onSort} align="right" />
            <Head label="Seed" k="seed" sortKey={sortKey} ascending={ascending} onSort={onSort} align="right" />
            <Head label="" sortKey={sortKey} ascending={ascending} onSort={onSort} />
          </tr>
        </thead>
        <tbody>
          {entries.map((entry) => {
            const isSelected = selected.has(entry.id)
            return (
              <tr
                key={entry.id}
                ref={(el) => registerRef(entry, el)}
                data-archive-record
                tabIndex={focusedId === entry.id ? 0 : -1}
                onFocus={() => onFocused(entry)}
                onClick={(e) => onActivate(entry, e)}
                className={`border-b border-grey-300 align-top focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-burgundy-900 ${
                  isSelected || entry.missing ? 'bg-newsprint-aged' : 'hover:bg-newsprint-aged'
                }`}
              >
                <td className="px-2 py-2 text-caption text-grey-500 tabular-nums">
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation()
                      onToggleSelect(entry, e)
                    }}
                    aria-pressed={isSelected}
                    aria-label={isSelected ? 'Take out of the selection' : 'Add to the selection'}
                    className={`sg-tap mr-2 inline-block h-3 w-3 border border-grey-500 align-middle text-[0.5rem] leading-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900 ${
                      isSelected ? 'bg-ink text-newsprint' : 'bg-newsprint'
                    }`}
                  >
                    {isSelected ? '×' : ''}
                  </button>
                  {entry.no.toLocaleString('en-GB')}
                  {entry.starred && <span className="ml-1 text-ink">★</span>}
                </td>
                <td className="px-2 py-2 whitespace-nowrap text-grey-700 tabular-nums">
                  {clockTime(entry.at)}
                </td>
                <td className="px-2 py-2 text-grey-700">
                  {entry.modelLabel || entry.familyLabel}
                  {entry.kind === 'video' && (
                    <span className="ml-1 text-caption text-grey-500">clip</span>
                  )}
                </td>
                <td className="max-w-[28rem] px-2 py-2">
                  <span className="line-clamp-2 font-serif">
                    {entry.prompt ? (
                      <Highlight text={entry.prompt} terms={terms} />
                    ) : (
                      <span className="text-grey-500 italic">No prompt was recorded</span>
                    )}
                  </span>
                  {entry.missing && (
                    <span className="text-caption text-grey-700 italic">Not on disk</span>
                  )}
                </td>
                <td className="px-2 py-2 text-right whitespace-nowrap text-grey-700 tabular-nums">
                  {dimensions(entry.width, entry.height) ??
                    (entry.megapixels ? `${entry.megapixels} MP` : '—')}
                </td>
                <td className="px-2 py-2 text-right text-grey-700 tabular-nums">{entry.steps}</td>
                <td className="px-2 py-2 text-right text-grey-700 tabular-nums">
                  {entry.cfg.toFixed(1)}
                </td>
                <td className="px-2 py-2 text-right text-grey-700 tabular-nums">{entry.seed}</td>
                <td className="px-2 py-2" onClick={(e) => e.stopPropagation()}>
                  <CardActions
                    entry={entry}
                    canDeleteFile={canDeleteFile}
                    selected={isSelected}
                    {...actionsFor(entry)}
                  />
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
})
