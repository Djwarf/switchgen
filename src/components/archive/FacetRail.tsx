/**
 * The index down the left-hand side.
 *
 * Every facet writes its token into the search box rather than holding hidden
 * state of its own. The box stays the single truth about what is being shown,
 * clicking teaches the syntax, and a reader who has learned it can type faster
 * than they can click.
 *
 * Counts are taken over the whole archive, not the filtered set, so you can
 * see what dropping a filter would buy you before you drop it.
 */
import type { ArchiveSyncState } from '../../lib/archiveSync'
import { useMemo, useRef } from 'react'
import type { HistoryEntry } from '../../lib/history'
import { hasToken } from './query'

type Props = {
  entries: readonly HistoryEntry[]
  query: string
  onToggle: (token: string) => void
  onClear: () => void
  onExport: () => void
  onImport: (file: File) => void
  onCheckMissing: () => void
  checking: boolean
  persistent: boolean
  /** False when the app cannot reach the local server that deletes files. */
  canDeleteFiles: boolean
  /** Where the archive lives right now, from the sync. Null before it has said. */
  sync: ArchiveSyncState | null
  /** File the outputs no record describes. */
  onRecover: () => void
  recovering: boolean
  /** The tagger is installed on the server, so pictures can be read in bulk. */
  canTag: boolean
  /** Tag every picture that has no tags yet. */
  onTagAll: () => void
  /** Progress of that pass, or null when it is not running. */
  tagging: { done: number; total: number } | null
}

function syncLine(sync: ArchiveSyncState | null): string {
  if (!sync || sync.mode === 'starting') return 'Checking the server.'
  const waiting = sync.pending
    ? ` ${sync.pending} ${sync.pending === 1 ? 'change is' : 'changes are'} waiting for the server.`
    : ''
  if (sync.mode === 'server') return `Shared with every device on this server.${waiting || ' Up to date.'}`
  // A server that answered with a reason (another server holds the archive)
  // is quoted, so the reader knows what to stop rather than waiting for it.
  if (sync.mode === 'offline' && sync.refusal) return `This browser only for now. ${sync.refusal}${waiting}`
  if (sync.mode === 'offline') return `This browser only for now: the server did not answer.${waiting}`
  return 'This browser only. There is no local server to share it through.'
}

type Row = { label: string; token: string; count: number }

const DAY = 86_400_000
const startOfDay = (ms: number) => {
  const d = new Date(ms)
  d.setHours(0, 0, 0, 0)
  return d.getTime()
}

function tally(entries: readonly HistoryEntry[], now: number = Date.now()) {
  const today = startOfDay(now)
  const counts = {
    images: 0,
    videos: 0,
    words: 0,
    picture: 0,
    instruction: 0,
    starred: 0,
    missing: 0,
    today: 0,
    yesterday: 0,
    week: 0,
    month: 0,
  }
  const families = new Map<string, { label: string; count: number }>()
  const tags = new Map<string, number>()
  let untagged = 0

  for (const e of entries) {
    if (e.kind === 'image' && !e.missing) {
      if (e.tags?.length) for (const t of e.tags.slice(0, 20)) tags.set(t, (tags.get(t) ?? 0) + 1)
      else untagged++
    }
    if (e.kind === 'video') counts.videos++
    else counts.images++

    if (e.mode === 'edit') counts.instruction++
    else if (e.source) counts.picture++
    else counts.words++

    if (e.starred) counts.starred++
    if (e.missing) counts.missing++

    const day = startOfDay(e.at)
    if (day === today) counts.today++
    else if (day === today - DAY) counts.yesterday++
    if (e.at >= today - 6 * DAY) counts.week++
    if (e.at >= today - 29 * DAY) counts.month++

    const seen = families.get(e.familyId)
    if (seen) seen.count++
    else families.set(e.familyId, { label: e.familyLabel || e.familyId, count: 1 })
  }

  return { counts, families, tags, untagged }
}

function Group({ title, rows, query, onToggle }: {
  title: string
  rows: Row[]
  query: string
  onToggle: (token: string) => void
}) {
  const useful = rows.filter((r) => r.count > 0)
  if (!useful.length) return null
  return (
    <section className="mb-6">
      <h3 className="mb-1 border-b border-grey-300 pb-1 text-[0.625rem] font-semibold tracking-[0.18em] text-grey-700 uppercase">
        {title}
      </h3>
      <ul>
        {useful.map((row) => {
          const active = hasToken(query, row.token)
          return (
            <li key={row.token}>
              <button
                type="button"
                onClick={() => onToggle(row.token)}
                aria-pressed={active}
                className={`flex w-full items-baseline justify-between gap-2 px-1 py-[3px] text-left text-small focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900 ${
                  active
                    ? 'bg-newsprint-aged font-semibold text-ink'
                    : 'text-grey-700 hover:text-ink'
                }`}
              >
                <span className="truncate">{row.label}</span>
                <span className="shrink-0 text-caption text-grey-500 tabular-nums">{row.count}</span>
              </button>
            </li>
          )
        })}
      </ul>
    </section>
  )
}

export function FacetRail({
  entries,
  query,
  onToggle,
  onClear,
  onExport,
  onImport,
  onCheckMissing,
  checking,
  persistent,
  canDeleteFiles,
  sync,
  onRecover,
  recovering,
  canTag,
  onTagAll,
  tagging,
}: Props) {
  const file = useRef<HTMLInputElement | null>(null)
  const { counts, families, tags, untagged } = useMemo(() => tally(entries), [entries])

  const styleRows: Row[] = useMemo(
    () =>
      [...families.entries()]
        .map(([id, f]) => ({ label: f.label, token: `family:${id}`, count: f.count }))
        .sort((a, b) => b.count - a.count || a.label.localeCompare(b.label)),
    [families],
  )

  const contentRows: Row[] = useMemo(
    () =>
      [...tags.entries()]
        .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
        .slice(0, 12)
        .map(([tag, count]) => ({ label: tag.replace(/_/g, ' '), token: `tag:${tag}`, count })),
    [tags],
  )

  const filtering = query.trim().length > 0

  return (
    <aside className="order-last w-full border-t border-grey-300 pt-6 lg:order-none lg:w-60 lg:shrink-0 lg:border-t-0 lg:pt-0" aria-label="Filters">
      <div className="mb-4 flex items-baseline justify-between border-b-2 border-burgundy-900 pb-1">
        <h2 className="text-[0.625rem] font-semibold tracking-[0.18em] text-burgundy-900 uppercase">
          Index
        </h2>
        {filtering && (
          <button
            type="button"
            onClick={onClear}
            className="text-caption text-burgundy-900 underline underline-offset-2 hover:no-underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900"
          >
            Clear
          </button>
        )}
      </div>

      <Group
        title="Desk"
        query={query}
        onToggle={onToggle}
        rows={[
          { label: 'Pictures', token: 'is:image', count: counts.images },
          { label: 'Clips', token: 'is:video', count: counts.videos },
        ]}
      />

      <Group title="Style" query={query} onToggle={onToggle} rows={styleRows} />

      <Group
        title="Made from"
        query={query}
        onToggle={onToggle}
        rows={[
          { label: 'Words', token: '-is:source', count: counts.words },
          { label: 'A picture', token: 'is:source', count: counts.picture },
          { label: 'An instruction', token: 'mode:edit', count: counts.instruction },
        ]}
      />

      <Group
        title="When"
        query={query}
        onToggle={onToggle}
        rows={[
          { label: 'Today', token: 'on:today', count: counts.today },
          { label: 'Yesterday', token: 'on:yesterday', count: counts.yesterday },
          { label: 'This week', token: 'on:week', count: counts.week },
          { label: 'This month', token: 'on:month', count: counts.month },
        ]}
      />

      <Group
        title="Marks"
        query={query}
        onToggle={onToggle}
        rows={[
          { label: 'Starred', token: 'is:starred', count: counts.starred },
          { label: 'Not on disk', token: 'is:missing', count: counts.missing },
        ]}
      />

      {contentRows.length ? <Group title="Content" query={query} onToggle={onToggle} rows={contentRows} /> : null}

      <section className="border-t border-grey-300 pt-3">
        <h3 className="mb-2 text-[0.625rem] font-semibold tracking-[0.18em] text-grey-700 uppercase">
          The archive
        </h3>
        <p className="mb-2 text-caption italic text-grey-700">{syncLine(sync)}</p>
        <ul className="space-y-1 text-small">
          <li>
            <button
              type="button"
              onClick={onExport}
              className="text-burgundy-900 underline underline-offset-2 hover:no-underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900"
            >
              Save a copy
            </button>
          </li>
          {canTag && (untagged > 0 || tagging) ? (
            <li>
              <button
                type="button"
                onClick={onTagAll}
                disabled={!!tagging}
                className="text-burgundy-900 underline underline-offset-2 hover:no-underline disabled:text-grey-500 disabled:no-underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900"
              >
                {tagging
                  ? `Tagging, ${tagging.done} of ${tagging.total}`
                  : `Tag the ${untagged} ${untagged === 1 ? 'picture' : 'pictures'} with no tags`}
              </button>
            </li>
          ) : null}
          {sync?.mode === 'server' ? (
            <li>
              <button
                type="button"
                onClick={onRecover}
                disabled={recovering}
                className="text-burgundy-900 underline underline-offset-2 hover:no-underline disabled:text-grey-500 disabled:no-underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900"
              >
                {recovering ? 'Looking through the outputs folder' : 'Look for files with no record'}
              </button>
            </li>
          ) : null}
          <li>
            <button
              type="button"
              onClick={() => file.current?.click()}
              className="text-burgundy-900 underline underline-offset-2 hover:no-underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900"
            >
              Restore from a file
            </button>
            <input
              ref={file}
              type="file"
              accept="application/json,.json"
              hidden
              onChange={(e) => {
                const chosen = e.target.files?.[0]
                if (chosen) onImport(chosen)
                e.target.value = ''
              }}
            />
          </li>
          <li>
            <button
              type="button"
              onClick={onCheckMissing}
              disabled={checking}
              className="text-burgundy-900 underline underline-offset-2 hover:no-underline disabled:text-grey-400 disabled:no-underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900"
            >
              {checking ? 'Checking the files…' : 'Check the files are still there'}
            </button>
          </li>
        </ul>

        {!persistent && (
          <p className="mt-3 text-caption leading-relaxed text-grey-700 italic">
            This browser will not let us save the archive, so these records last only as long as
            the tab. Your files are untouched.
          </p>
        )}
        {!canDeleteFiles && (
          <p className="mt-3 text-caption leading-relaxed text-grey-700 italic">
            Deleting files needs the SwitchGen dev or preview server. Right now we can only remove
            records; your files stay on disk.
          </p>
        )}
      </section>
    </aside>
  )
}
