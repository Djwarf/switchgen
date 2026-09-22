/**
 * The picture view: the records under their day headings.
 *
 * Memoised, and handed only props that hold still while the reader types. The
 * search box changes on every keystroke and the results only after a pause, so
 * without this every keystroke re-rendered every card on the page just to
 * redraw the box above them, and after a long scroll that is thousands.
 */
import { memo, type MouseEvent } from 'react'
import type { HistoryEntry } from '../../lib/history'
import { Card } from './Card'
import type { EntryActions } from './CardActions'
import { DateHead } from './DateHead'

type Props = {
  groups: readonly { label: string; entries: readonly HistoryEntry[] }[]
  terms: readonly string[]
  focusedId: string | null
  selected: ReadonlySet<string>
  canDeleteFile: boolean
  onActivate: (entry: HistoryEntry, e: MouseEvent) => void
  onFocused: (entry: HistoryEntry) => void
  onToggleSelect: (entry: HistoryEntry, e: MouseEvent) => void
  registerRef: (entry: HistoryEntry, el: HTMLElement | null) => void
  actionsFor: (entry: HistoryEntry) => EntryActions
}

export const Grid = memo(function Grid({
  groups,
  terms,
  focusedId,
  selected,
  canDeleteFile,
  onActivate,
  onFocused,
  onToggleSelect,
  registerRef,
  actionsFor,
}: Props) {
  const selecting = selected.size > 0
  return (
    <div>
      {groups.map((group) => (
        <section key={group.label}>
          <DateHead label={group.label} count={group.entries.length} />
          <ul
            role="list"
            className="grid grid-cols-1 gap-x-6 gap-y-8 min-[780px]:grid-cols-2 min-[1100px]:grid-cols-3 min-[1400px]:grid-cols-4"
          >
            {group.entries.map((entry) => (
              <li key={entry.id}>
                <Card
                  entry={entry}
                  terms={terms}
                  focused={focusedId === entry.id}
                  selected={selected.has(entry.id)}
                  selecting={selecting}
                  canDeleteFile={canDeleteFile}
                  onActivate={(e) => onActivate(entry, e)}
                  onFocused={() => onFocused(entry)}
                  onToggleSelect={(e) => onToggleSelect(entry, e)}
                  registerRef={(el) => registerRef(entry, el)}
                  {...actionsFor(entry)}
                />
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  )
})
