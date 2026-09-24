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
 *
 * Below a wide screen the running job gets a row of its own, under the tabs.
 * Squeezed in beside them it had about 30 pixels on a phone, where the step
 * it was on could not be read and Hold to stop ran off the right edge and
 * pushed the whole page sideways, and about 130 on a tablet, where its words
 * were drawn over Hold to stop.
 */
import { useEffect, useLayoutEffect, useReducer, useRef, type RefObject } from 'react'
import { useEntryCount } from './Masthead'
import { RunningSlug } from './RunningSlug'
import { SettingsToggle } from './SettingsToggle'
import { openShortcuts } from './Shortcuts'
import { headline, newsUntil, useJobs, type JobsSnapshot } from './jobs'
import {
  SECTIONS,
  SECTION_LABEL,
  SECTION_STANDFIRST,
  sectionHref,
  useSection,
  type Section,
} from './route'

const KEY: Record<Section, string> = { pictures: '1', video: '2', reel: '3', archive: '4' }

/**
 * Draw again when the finished job the slug shows stops being news. Nothing
 * in the ledger changes at that moment, so without this the bar kept the
 * slug's row open, empty, until the next change.
 */
function useNewsExpiry(snap: JobsSnapshot): void {
  const [, redraw] = useReducer((n: number) => n + 1, 0)
  const until = newsUntil(snap)
  useEffect(() => {
    if (until === null) return
    const wait = until - Date.now()
    if (wait <= 0) return
    const t = setTimeout(redraw, wait + 50)
    return () => clearTimeout(t)
  }, [until])
}

/**
 * Tell the page how tall the bar really is. The archive's day headings and
 * the list view's column heads stick beneath it at --sg-bar-h, and the bar is
 * one row only on a wide screen: on a phone it is two rows idle and three
 * with a job running, and the headings parked under it, out of sight. The
 * values in index.css stay as the first paint's guess.
 */
function useBarHeight(bar: RefObject<HTMLDivElement | null>): void {
  useLayoutEffect(() => {
    const el = bar.current
    if (!el || typeof ResizeObserver === 'undefined') return
    const root = document.documentElement
    const write = () => root.style.setProperty('--sg-bar-h', `${Math.ceil(el.getBoundingClientRect().height)}px`)
    write()
    const watch = new ResizeObserver(write)
    watch.observe(el)
    return () => {
      watch.disconnect()
      root.style.removeProperty('--sg-bar-h')
    }
  }, [bar])
}

export function SectionBar() {
  const here = useSection()
  const filed = useEntryCount()
  const snap = useJobs()
  const bar = useRef<HTMLDivElement>(null)
  useNewsExpiry(snap)
  useBarHeight(bar)
  // Exactly when the slug has something to say, so a phone never gets an
  // empty row for it.
  const busy = headline(snap) !== null || (snap.server.known && snap.server.foreign > 0)

  return (
    <div ref={bar} className="sg-sticky-top z-40 border-b border-grey-300 bg-newsprint px-6">
      <div className="mx-auto flex w-full max-w-[110rem] flex-wrap items-center gap-x-4 gap-y-1 sm:gap-x-6 lg:flex-nowrap">
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

        <div
          className={`flex min-w-0 flex-1 items-center justify-center ${busy ? 'max-lg:order-last max-lg:basis-full' : ''}`}
        >
          {busy ? (
            <RunningSlug className="min-w-0 max-lg:w-full" />
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
