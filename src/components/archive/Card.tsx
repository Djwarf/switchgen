/**
 * One record, as it appears in the run.
 *
 * The prompt is set as the headline, because that is what you remember and
 * that is what you are scanning for. Everything else — the style, the edition
 * number, the time — is metadata and is coloured like metadata.
 *
 * Hover does not lift and casts no shadow. The rule beneath thickens and turns
 * burgundy, and the actions appear. Print-native.
 */
import type { MouseEvent } from 'react'
import type { HistoryEntry } from '../../lib/history'
import { CardActions, type EntryActions } from './CardActions'
import { Poster } from './Poster'
import { Highlight, clockTime, editionNo, madeFrom } from './query'

type Props = EntryActions & {
  entry: HistoryEntry
  terms: readonly string[]
  focused: boolean
  selected: boolean
  selecting: boolean
  canDeleteFile: boolean
  onActivate: (e: MouseEvent) => void
  onFocused: () => void
  onToggleSelect: (e: MouseEvent) => void
  registerRef: (el: HTMLElement | null) => void
}

export function Card({
  entry,
  terms,
  focused,
  selected,
  selecting,
  canDeleteFile,
  onActivate,
  onFocused,
  onToggleSelect,
  registerRef,
  ...actions
}: Props) {
  return (
    <article
      ref={registerRef}
      tabIndex={focused ? 0 : -1}
      onFocus={onFocused}
      aria-label={`${editionNo(entry.no)}. ${entry.prompt || 'no prompt'}`}
      className={`group relative flex flex-col focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900 ${
        entry.missing ? 'bg-newsprint-aged' : ''
      } ${selected ? 'bg-newsprint-aged' : ''}`}
    >
      <button
        type="button"
        onClick={onActivate}
        aria-label="Open the full record"
        className="relative block aspect-[4/3] w-full border border-grey-300 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900"
      >
        <Poster entry={entry} className="h-full w-full" />
      </button>

      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation()
          onToggleSelect(e)
        }}
        aria-pressed={selected}
        aria-label={selected ? 'Take out of the selection' : 'Add to the selection'}
        className={`sg-tap absolute top-1.5 left-1.5 h-4 w-4 border border-grey-500 bg-newsprint text-[0.5rem] leading-none text-ink transition-opacity focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900 ${
          selecting || selected
            ? 'opacity-100'
            : 'sg-touch-reveal opacity-0 group-hover:opacity-100 group-focus-within:opacity-100'
        } ${selected ? 'bg-ink text-newsprint' : ''}`}
      >
        {selected ? '×' : ''}
      </button>

      {entry.starred && (
        <span
          className="absolute top-1.5 right-1.5 bg-newsprint/90 px-1 text-caption text-ink"
          title="Starred"
          aria-label="Starred"
        >
          ★
        </span>
      )}

      <div className="pt-2">
        <p className="text-[0.625rem] font-semibold tracking-[0.18em] text-grey-700 uppercase">
          {entry.modelLabel || entry.familyLabel}
          <span className="ml-2 font-normal tracking-[0.12em] text-grey-500">{madeFrom(entry)}</span>
        </p>

        <h3 className="mt-1 font-serif text-[1.125rem] leading-snug font-semibold break-words text-ink">
          <span className="line-clamp-3">
            {entry.prompt ? (
              <Highlight text={entry.prompt} terms={terms} />
            ) : (
              <span className="text-grey-500 italic">No prompt was recorded</span>
            )}
          </span>
        </h3>

        {entry.missing && (
          <div className="mt-2 border-l-4 border-ink bg-[#F5F5F5] px-3 py-2 text-caption leading-relaxed">
            <p className="italic">
              <strong className="mr-1 text-[0.625rem] font-bold tracking-[0.05em] uppercase not-italic">
                Not on disk
              </strong>
              This file has been moved or deleted. The settings are still here, so you can make it
              again.
            </p>
            <p className="mt-1 flex gap-3">
              <button
                type="button"
                onClick={actions.onReuse}
                className="text-[0.625rem] font-semibold tracking-[0.16em] text-burgundy-900 uppercase underline underline-offset-4 hover:no-underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900"
              >
                Use these settings
              </button>
              <button
                type="button"
                onClick={actions.onRemove}
                className="text-[0.625rem] font-semibold tracking-[0.16em] text-grey-700 uppercase underline underline-offset-4 hover:text-burgundy-900 hover:no-underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900"
              >
                Remove this record
              </button>
            </p>
          </div>
        )}

        <p className="mt-1 text-caption text-grey-500 tabular-nums">
          {editionNo(entry.no)}
          <span className="mx-1.5 text-grey-300" aria-hidden>
            ·
          </span>
          {clockTime(entry.at)}
        </p>

        <div className="sg-touch-reveal mt-1 h-6 opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100 motion-reduce:transition-none sm:h-6 [@media(hover:none)]:min-h-11">
          <CardActions
            entry={entry}
            canDeleteFile={canDeleteFile}
            selected={selected}
            {...actions}
          />
        </div>
      </div>

      <div className="mt-1 h-[2px] w-full">
        <div
          className={`w-full bg-grey-300 group-hover:h-[2px] group-hover:bg-burgundy-900 group-focus-within:h-[2px] group-focus-within:bg-burgundy-900 ${
            focused ? 'h-[2px] bg-burgundy-900' : 'h-px'
          }`}
        />
      </div>
    </article>
  )
}
