/**
 * The shell.
 *
 * Masthead, section bar, the page, and the five things that must be reachable
 * from every section: the running job, work the server holds until you say,
 * notices, the undo window and the shortcuts card. `App.tsx` renders exactly
 * one route inside it.
 *
 *   <Shell gpu={hardware?.gpu}>
 *     {route.name === 'pictures' ? <Pictures /> : …}
 *   </Shell>
 */
import { useEffect, useRef, type ReactNode } from 'react'
import { clearLoadIssue, clearQuotaIssue, isPersistent, loadIssue, quotaIssue } from '../../lib/history'
import { draftIssue, subscribeDraftIssue } from '../../lib/session'
import { Masthead, useEntryCount } from './Masthead'
import { NoticeRail, dismissNotice, postNotice } from './Notice'
import { Offline } from './Offline'
import { RunnerHold } from './RunnerHold'
import { SectionBar } from './SectionBar'
import { Shortcuts, closeShortcuts, shortcutsOpen, toggleShortcuts, useShortcutsOpen } from './Shortcuts'
import { UndoBar, undoLast } from './UndoBar'
import { toggleExpert } from './SettingsToggle'
import { isTyping, useChord, useHotkeys } from './hotkeys'
import {
  currentRoute,
  go,
  goToSection,
  normaliseHash,
  requestSearchFocus,
  setArchiveQuery,
  useRoute,
} from './route'

/** One stable key, so a second desk failing replaces the notice, not stacks it. */
const DRAFT_NOTICE = 'draft-quota'

export type ShellProps = {
  /** "RTX 5060 Ti · 16 GB", once the hardware probe answers. */
  gpu?: string | null
  children: ReactNode
}

export function Shell({ gpu, children }: ShellProps) {
  const route = useRoute()
  const filed = useEntryCount()
  const redirected = useRef(false)

  // A bare `#` or a mistyped hash lands on the pictures desk without leaving a
  // history entry to walk back through.
  useEffect(() => {
    normaliseHash()
  }, [])

  // An empty archive is never the landing screen. Once, though — if you go
  // there deliberately afterwards, that is your business.
  useEffect(() => {
    if (redirected.current) return
    if (route.name !== 'archive' || filed > 0) return
    redirected.current = true
    go({ name: 'pictures' }, { replace: true })
    postNotice({
      key: 'first-run',
      tone: 'info',
      title: 'Nothing filed yet',
      body: 'Everything you make is filed in the archive with the settings that made it. Make something first.',
      actions: [{ label: 'Open the archive anyway', run: () => goToSection('archive') }],
      ttl: 12000,
    })
  }, [route.name, filed])

  // A new section starts at the top of the page, the way turning to a section
  // of a newspaper does. Within a section, scrolling is the reader's business.
  useEffect(() => {
    window.scrollTo({ top: 0 })
  }, [route.name])

  // Corrections the archive raised while loading or writing.
  useEffect(() => {
    const load = loadIssue()
    if (load) {
      postNotice({ key: 'archive-load', tone: 'correction', title: 'Correction', body: load })
      clearLoadIssue()
    }
    const quota = quotaIssue()
    if (quota) {
      postNotice({ key: 'archive-quota', tone: 'warning', title: 'Archive', body: quota })
      clearQuotaIssue()
    }
    if (!isPersistent()) {
      postNotice({
        key: 'archive-memory',
        tone: 'correction',
        title: 'Correction',
        body: 'This browser will not let us save your archive, so it lasts only until you close the tab. Your files are still written to disk as usual.',
      })
    }
  }, [])

  // A desk whose draft the browser had no room to save. The session says so
  // once when it starts and once when a save lands again, so the notice
  // stands for as long as the trouble does and goes when it clears.
  useEffect(() => {
    const show = () => {
      const message = draftIssue()
      if (message) postNotice({ key: DRAFT_NOTICE, tone: 'warning', title: 'Draft not saved', body: message })
      else dismissNotice(DRAFT_NOTICE)
    }
    show()
    return subscribeDraftIssue(show)
  }, [])

  useGlobalKeys()

  return (
    <div className="flex min-h-full flex-col bg-newsprint">
      <Masthead gpu={gpu} />
      <SectionBar />
      <Offline />
      {/* The queue's hold is one for every desk, so its word is given from any room. */}
      <RunnerHold />
      {/* tabIndex -1 so focus has somewhere to land when an overlay closes and
          the element it was opened from has since been unmounted. Without it
          focus falls back to the body, and a keyboard session then restarts at
          the top of the document, which on a television is a long way back. */}
      <main id="page" tabIndex={-1} className="flex-1 pb-16 focus:outline-none">
        <div className="mx-auto w-full max-w-[110rem]">{children}</div>
      </main>
      <NoticeRail />
      <UndoBar />
      <Shortcuts />
    </div>
  )
}

/**
 * The global keys. Every one of them is dead while a field has focus, with the
 * single exception of `Ctrl/Cmd+Z`, which is deliberately *not* claimed inside
 * a field so that undoing your typing still undoes your typing.
 */
function useGlobalKeys(): void {
  // `guard` covers the keys in THIS array and nothing else. Four other modules
  // register their own unguarded window listeners, and the archive's includes
  // Backspace on a record, so this guard was never the thing keeping the page
  // still behind a modal. What does that job is the capture phase listener the
  // shortcuts card installs while it is open, which stops every key but the
  // ones that close it or scroll it. This guard stays as the belt to that
  // brace: it costs nothing and it works even if the card's listener is not
  // yet attached on the frame the card opens.
  const overlay = useShortcutsOpen()
  const guard = (fn: () => void) => () => {
    if (overlay) return
    fn()
  }

  useHotkeys([
    { key: '1', run: guard(() => goToSection('pictures')) },
    { key: '2', run: guard(() => goToSection('video')) },
    { key: '3', run: guard(() => goToSection('reel')) },
    { key: '4', run: guard(() => goToSection('archive')) },
    { key: 'e', run: guard(toggleExpert) },
    { key: '/', run: guard(requestSearchFocus) },
    { key: '?', run: toggleShortcuts },
    { key: 'z', mod: true, run: guard(() => void undoLast()) },
    {
      key: 'Escape',
      inFields: true,
      passive: true,
      run: (e) => {
        if (shortcutsOpen()) {
          closeShortcuts()
          return
        }
        const here = currentRoute()
        if (here.name === 'archive' && here.q) {
          setArchiveQuery('')
          return
        }
        if (isTyping(e)) (e.target as HTMLElement).blur?.()
      },
    },
  ])

  useChord(
    'g',
    {
      p: () => goToSection('pictures'),
      v: () => goToSection('video'),
      r: () => goToSection('reel'),
      a: () => goToSection('archive'),
    },
    300,
    !overlay,
  )
}
