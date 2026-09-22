/**
 * The section bar.
 *
 * Four rooms, named as rooms: Pictures, Video, Reel, Archive. Not a mode
 * dropdown. A dropdown says "these are the same job with a setting changed",
 * and making a five-second clip is not making a picture with a setting
 * changed. Each desk keeps its own prompt, its own source picture and its own
 * controls, and this bar is the door between them.
 *
 * The middle of the bar is shared: the section's standfirst when the card is
 * idle, and the running job the moment there is one. That is how a clip
 * rendering on the video desk stays visible while you write on Pictures.
 */
import { useEntryCount } from './Masthead'
import { RunningSlug } from './RunningSlug'
import { SettingsToggle } from './SettingsToggle'
import { openShortcuts } from './Shortcuts'
import { headline, useJobs } from './jobs'
import {
  SECTIONS,
  SECTION_LABEL,
  SECTION_STANDFIRST,
  sectionHref,
  useSection,
  type Section,
} from './route'

const KEY: Record<Section, string> = { pictures: '1', video: '2', reel: '3', archive: '4' }

export function SectionBar() {
  const here = useSection()
  const filed = useEntryCount()
  const snap = useJobs()
  const busy = headline(snap) !== null || (snap.server.known && snap.server.running > 0)

  return (
    <div className="sg-sticky-top z-40 border-b border-grey-300 bg-newsprint px-6">
      <div className="mx-auto flex w-full max-w-[110rem] flex-wrap items-center gap-x-4 gap-y-1 sm:flex-nowrap sm:gap-6">
        <nav aria-label="Sections" className="flex shrink-0 items-baseline gap-6">
          {SECTIONS.map((s) => (
            <a
              key={s}
              href={sectionHref(s)}
              className="sg-tab ring"
              aria-current={here === s ? 'page' : undefined}
              title={`${SECTION_STANDFIRST[s]} (${KEY[s]})`}
            >
              {SECTION_LABEL[s]}
              {s === 'archive' && filed > 0 && (
                <span className="figures ml-2 text-grey-500">{filed.toLocaleString('en-GB')}</span>
              )}
            </a>
          ))}
        </nav>

        <div className="flex min-w-0 flex-1 items-center justify-center">
          {busy ? (
            <RunningSlug className="min-w-0" />
          ) : (
            <p className="hidden truncate text-small text-grey-500 italic lg:block">
              {SECTION_STANDFIRST[here]}
            </p>
          )}
        </div>

        <div className="flex shrink-0 items-center gap-3">
          <SettingsToggle />
          <button
            type="button"
            className="sg-quiet ring"
            onClick={openShortcuts}
            title="Keyboard shortcuts (?)"
            aria-label="Keyboard shortcuts"
          >
            ?
          </button>
        </div>
      </div>
    </div>
  )
}
