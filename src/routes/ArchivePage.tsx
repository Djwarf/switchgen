/**
 * The archive: everything this machine has made, searchable, reusable and
 * removable.
 *
 * Three ideas hold this screen together.
 *
 *   1. The search box is the only state. Facets write tokens into it, the
 *      address bar mirrors it, and it is hand-editable — so what you see can
 *      always be explained by what is written in one line.
 *   2. The record and the file are different things. Removing a record is
 *      cheap and undoable; deleting a file asks first, happens once, and has
 *      no keyboard shortcut at all.
 *   3. Reuse lands in the right room. A clip restores into the Video desk and
 *      a picture into Pictures, with every parameter the record carried.
 */
import { capabilities as visionCapabilities, tagImages } from '../lib/vision'
import { useArchiveSync } from '../lib/archiveSync'
import { recoverUnfiled } from '../lib/recover'
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
  type MouseEvent,
} from 'react'
import { fileUrl, relPath } from '../lib/comfy'
import { serverCapabilities } from '../lib/capabilities'
import {
  all,
  checkMissing,
  clearLoadIssue,
  clearQuotaIssue,
  count as countRecords,
  deleteFiles,
  exportJson,
  get as getRecord,
  importJson,
  isPersistent,
  loadIssue,
  quotaIssue,
  removeMany,
  restore,
  star as starRecord,
  subscribe,
  type HistoryEntry,
  update
} from '../lib/history'
import { modelFiles } from '../lib/hardware'
import {
  adoptSource,
  requestRegionEdit,
  reuseIntoDesk,
  settings,
  type DeskId,
  type ReuseNote,
} from '../lib/session'
import {
  FOCUS_SEARCH_EVENT,
  currentRoute,
  goToSection,
  setArchiveQuery,
} from '../components/shell/route'
import { Card } from '../components/archive/Card'
import type { EntryActions } from '../components/archive/CardActions'
import { DateHead } from '../components/archive/DateHead'
import { DeleteDialog } from '../components/archive/DeleteDialog'
import { Detail } from '../components/archive/Detail'
import { FacetRail } from '../components/archive/FacetRail'
import { Notice } from '../components/type'
import { offerUndo } from '../components/shell/UndoBar'
import { SearchBand, type KindFilter } from '../components/archive/SearchBand'
import { SelectionBar } from '../components/archive/SelectionBar'
import { Table, type SortKey } from '../components/archive/Table'
import { dayHeading, hasToken, isTyping, runQuery, toggleToken } from '../components/archive/query'

export type ArchivePageProps = {
  /** The query from the address bar, when the shell parses it. */
  q?: string
  /** Called when the reader changes the query, so the shell can mirror it. */
  onQueryChange?: (q: string) => void
  /** Where to send the reader after a reuse. Falls back to the hash route. */
  onNavigate?: (desk: DeskId) => void
}

const PAGE = 120

// ---------------------------------------------------------------------------
// Small helpers that only this screen needs
// ---------------------------------------------------------------------------

/**
 * Whether the local server will delete a file on request.
 *
 * Asked of /api/capabilities, which probes rather than assumes. Without the
 * SwitchGen middleware (a static host, say) the answer is false and the
 * archive offers to remove the record only, which is the honest offer.
 */
async function probeDeleteCapability(): Promise<boolean> {
  return (await serverCapabilities()).deleteFiles
}

/** ComfyUI's /view sets no Content-Disposition, so a plain link would open it. */
async function saveToDisk(entry: HistoryEntry): Promise<void> {
  const res = await fetch(fileUrl(entry.file))
  if (!res.ok) throw new Error(`The file could not be read (HTTP ${res.status}).`)
  const blob = await res.blob()
  const url = URL.createObjectURL(blob)
  const ext = entry.file.filename.includes('.')
    ? entry.file.filename.slice(entry.file.filename.lastIndexOf('.'))
    : entry.kind === 'video'
      ? '.webm'
      : '.png'
  const a = document.createElement('a')
  a.href = url
  a.download = `switchgen-${entry.familyId}-${entry.seed}${ext}`
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 10_000)
}

function saveText(name: string, text: string): void {
  const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }))
  const a = document.createElement('a')
  a.href = url
  a.download = name
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 10_000)
}

/**
 * Is something layered over the page that is not the archive's own record view?
 *
 * This screen binds bare single keys on `window`, and one of them, `Backspace`,
 * removes records. Nothing else in the app stops those keys reaching us: the
 * shortcuts card, the desks' archive pickers and the reel's keyframe picker are
 * all scrims whose focus sits on a `<div tabIndex={-1}>`, so the typing guard
 * says false and every binding stays live behind them. On a television
 * `Backspace` is the natural back key, which makes that a destructive default.
 *
 * So the test is made here, at the moment the key arrives, over the DOM rather
 * than over a list of overlays this file would have to be told about. The
 * archive's own record view marks itself `data-archive-overlay`, because `j`,
 * `k` and `Escape` are how you read through it.
 */
function foreignOverlayIsUp(): boolean {
  if (typeof document === 'undefined') return false
  const modals = document.querySelectorAll('[aria-modal="true"]')
  for (let i = 0; i < modals.length; i++) {
    if (!modals[i].hasAttribute('data-archive-overlay')) return true
  }
  return false
}

/** The query in the address bar, when the shell has not passed one in. */
function queryFromRoute(): string {
  const route = currentRoute()
  return route.name === 'archive' ? route.q : ''
}

const sorters: Record<SortKey, (a: HistoryEntry, b: HistoryEntry) => number> = {
  no: (a, b) => a.no - b.no,
  at: (a, b) => a.at - b.at,
  steps: (a, b) => a.steps - b.steps,
  cfg: (a, b) => a.cfg - b.cfg,
  seed: (a, b) => a.seed - b.seed,
  model: (a, b) => (a.modelLabel || a.model).localeCompare(b.modelLabel || b.model),
}

type Banner = {
  variant: 'info' | 'correction' | 'error' | 'success'
  title: string
  text: string
  notes?: ReuseNote[]
  actions?: { label: string; run: () => void }[]
}

let auditedThisSession = false

// ---------------------------------------------------------------------------

export function ArchivePage({ q, onQueryChange, onNavigate }: ArchivePageProps = {}) {
  const records = useSyncExternalStore(subscribe, all)
  const prefs = useSyncExternalStore(settings.subscribe, settings.get)

  const [query, setQuery] = useState<string>(() => q ?? queryFromRoute())
  const [applied, setApplied] = useState(query)
  const [limit, setLimit] = useState(PAGE)
  const [focusedId, setFocusedId] = useState<string | null>(null)
  const [openId, setOpenId] = useState<string | null>(null)
  const [selected, setSelected] = useState<ReadonlySet<string>>(() => new Set())
  const [banner, setBanner] = useState<Banner | null>(null)
  const [pendingDelete, setPendingDelete] = useState<HistoryEntry[] | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [deleteError, setDeleteError] = useState<string | null>(null)
  const [canDeleteFiles, setCanDeleteFiles] = useState(false)
  const [checking, setChecking] = useState(false)
  const [recovering, setRecovering] = useState(false)
  const sync = useArchiveSync()
  const [canTag, setCanTag] = useState(false)
  const [tagging, setTagging] = useState<{ done: number; total: number } | null>(null)
  const tagStop = useRef(false)

  useEffect(() => {
    let live = true
    void visionCapabilities().then((c) => {
      if (live) setCanTag(c.tagger)
    })
    return () => {
      live = false
      tagStop.current = true
    }
  }, [])
  const [sortKey, setSortKey] = useState<SortKey>('at')
  const [ascending, setAscending] = useState(false)
  const [issue, setIssue] = useState<string | null>(() => loadIssue())
  const [quota, setQuota] = useState<string | null>(() => quotaIssue())

  const searchInput = useRef<HTMLInputElement | null>(null)
  const cardRefs = useRef(new Map<string, HTMLElement>())
  const anchor = useRef<number | null>(null)
  const installed = useRef<string[] | null>(null)
  const sentinel = useRef<HTMLDivElement | null>(null)

  // -- query plumbing -------------------------------------------------------

  // A query pushed in from outside — the back button, or a link someone sent —
  // is adopted during the render that brings it, not in an effect afterwards,
  // so the results never flash the old query first.
  const [lastGivenQuery, setLastGivenQuery] = useState(q)
  if (q !== undefined && q !== lastGivenQuery) {
    setLastGivenQuery(q)
    setQuery(q)
    setApplied(q)
    setLimit(PAGE)
  }

  useEffect(() => {
    const t = setTimeout(() => {
      setApplied(query)
      setLimit(PAGE)
    }, 120)
    return () => clearTimeout(t)
  }, [query])

  useEffect(() => {
    if (onQueryChange) {
      onQueryChange(applied)
      return
    }
    // Mirrored into the address bar, replacing rather than pushing, so the
    // back button does not have to walk through every keystroke.
    if (queryFromRoute() !== applied.trim()) setArchiveQuery(applied.trim())
  }, [applied, onQueryChange])

  // `/` from anywhere navigates here and then asks for the caret.
  useEffect(() => {
    const focus = () => {
      searchInput.current?.focus()
      searchInput.current?.select()
    }
    window.addEventListener(FOCUS_SEARCH_EVENT, focus)
    return () => window.removeEventListener(FOCUS_SEARCH_EVENT, focus)
  }, [])

  // -- results --------------------------------------------------------------

  const { results, terms } = useMemo(() => runQuery(applied, records), [applied, records])

  const ordered = useMemo(() => {
    if (prefs.view === 'list') {
      const sorted = [...results].sort(sorters[sortKey])
      return ascending ? sorted : sorted.reverse()
    }
    return prefs.sort === 'oldest' ? [...results].reverse() : results
  }, [results, prefs.view, prefs.sort, sortKey, ascending])

  const visible = useMemo(() => ordered.slice(0, limit), [ordered, limit])

  const groups = useMemo(() => {
    const out: { label: string; entries: HistoryEntry[] }[] = []
    for (const entry of visible) {
      const label = dayHeading(entry.at)
      const last = out[out.length - 1]
      if (last && last.label === label) last.entries.push(entry)
      else out.push({ label, entries: [entry] })
    }
    return out
  }, [visible])

  // Reveal more as the reader reaches the bottom, rather than mounting five
  // thousand cards at once.
  useEffect(() => {
    const el = sentinel.current
    if (!el || typeof IntersectionObserver === 'undefined') return
    const io = new IntersectionObserver(
      (rows) => {
        if (rows.some((r) => r.isIntersecting)) setLimit((n) => n + PAGE)
      },
      { rootMargin: '600px' },
    )
    io.observe(el)
    return () => io.disconnect()
  }, [ordered.length, limit])

  // -- once-per-session housekeeping ---------------------------------------

  useEffect(() => {
    probeDeleteCapability().then(setCanDeleteFiles)
    modelFiles()
      .then((map) => {
        installed.current = [...map.keys()]
      })
      .catch(() => {
        installed.current = null
      })
  }, [])

  useEffect(() => {
    if (auditedThisSession || !records.length) return
    auditedThisSession = true
    const run = () => {
      void checkMissing(all().slice(0, 200), 20)
    }
    const idle = (
      window as unknown as { requestIdleCallback?: (cb: () => void) => number }
    ).requestIdleCallback
    if (idle) idle(run)
    else setTimeout(run, 2000)
  }, [records.length])

  // -- selection helpers ----------------------------------------------------

  const focused = focusedId ? (records.find((e) => e.id === focusedId) ?? null) : null
  const open = openId ? (getRecord(openId) ?? null) : null
  const openIndex = open ? ordered.findIndex((e) => e.id === open.id) : -1

  const focusCard = useCallback((id: string | null) => {
    setFocusedId(id)
    if (!id) return
    requestAnimationFrame(() => {
      const el = cardRefs.current.get(id)
      el?.focus({ preventScroll: false })
    })
  }, [])

  const move = useCallback(
    (delta: number) => {
      if (!ordered.length) return
      const at = focusedId ? ordered.findIndex((e) => e.id === focusedId) : -1
      const next = Math.min(Math.max(at + delta, 0), Math.min(ordered.length, limit) - 1)
      const entry = ordered[at === -1 ? 0 : next]
      if (entry) focusCard(entry.id)
    },
    [ordered, focusedId, limit, focusCard],
  )

  const toggleSelection = useCallback(
    (entry: HistoryEntry, e: MouseEvent | null) => {
      const index = ordered.findIndex((r) => r.id === entry.id)
      setSelected((current) => {
        const next = new Set(current)
        if (e?.shiftKey && anchor.current !== null && index >= 0) {
          const [from, to] = [anchor.current, index].sort((a, b) => a - b)
          for (const r of ordered.slice(from, to + 1)) next.add(r.id)
          return next
        }
        if (next.has(entry.id)) next.delete(entry.id)
        else next.add(entry.id)
        anchor.current = index
        return next
      })
    },
    [ordered],
  )

  // -- the verbs ------------------------------------------------------------

  const goToDesk = useCallback(
    (desk: DeskId) => {
      if (onNavigate) onNavigate(desk)
      else goToSection(desk === 'video' ? 'video' : 'pictures')
    },
    [onNavigate],
  )

  const reuse = useCallback(
    (entry: HistoryEntry, freshSeed: boolean) => {
      const applyReuse = reuseIntoDesk(entry, {
        freshSeed,
        installedModels: installed.current ?? undefined,
      })
      const deskName = applyReuse.desk === 'video' ? 'the Video desk' : 'the Pictures desk'
      setOpenId(null)

      if (!applyReuse.clobbered && !applyReuse.notes.length) {
        goToDesk(applyReuse.desk)
        return
      }

      setBanner({
        variant: applyReuse.notes.length ? 'correction' : 'info',
        title: applyReuse.notes.length ? 'Correction' : 'Settings loaded',
        text: applyReuse.clobbered
          ? `From No. ${entry.no.toLocaleString('en-GB')}. What you had written on ${deskName} was replaced. Nothing has run.`
          : `From No. ${entry.no.toLocaleString('en-GB')}, onto ${deskName}. Nothing has run.`,
        notes: applyReuse.notes,
        actions: [
          { label: `Go to ${deskName}`, run: () => goToDesk(applyReuse.desk) },
          {
            label: applyReuse.clobbered ? 'Keep what I was writing' : 'Undo this',
            run: () => {
              applyReuse.undo()
              setBanner(null)
            },
          },
        ],
      })
    },
    [goToDesk],
  )

  const sendAsSource = useCallback(
    (entry: HistoryEntry, desk: DeskId) => {
      const undoSource = adoptSource(entry, desk)
      setOpenId(null)
      setBanner({
        variant: 'info',
        title: 'Picture sent',
        text: `${entry.file.filename} is standing by on the ${
          desk === 'video' ? 'Video desk as a start frame' : 'Pictures desk as a source'
        }. Your prompt there is untouched.`,
        actions: [
          { label: 'Go to the desk', run: () => goToDesk(desk) },
          {
            label: 'Undo this',
            run: () => {
              undoSource()
              setBanner(null)
            },
          },
        ],
      })
    },
    [goToDesk],
  )

  const removeRecords = useCallback((targets: readonly HistoryEntry[]) => {
    if (!targets.length) return
    const removed = removeMany(targets.map((t) => t.id))
    if (!removed.length) return
    setSelected(new Set())
    setOpenId(null)
    // The shell's undo bar, so Ctrl+Z works from anywhere and the archive does
    // not keep a second bar of its own.
    offerUndo({
      body:
        removed.length === 1 ? (
          <>
            One record removed. The file is still on disk at <code>{relPath(removed[0].file)}</code>.
          </>
        ) : (
          `${removed.length} records removed. The files are still on disk.`
        ),
      undo: () => restore(...removed),
    })
  }, [])

  const confirmDelete = useCallback(async () => {
    if (!pendingDelete) return
    setDeleting(true)
    setDeleteError(null)
    const outcome = await deleteFiles(pendingDelete)
    setDeleting(false)

    const failed = outcome.filter((o) => !o.result.ok)
    const gone = outcome.length - failed.length
    const unsupported = failed.some((o) => !o.result.ok && o.result.unsupported)
    if (unsupported) setCanDeleteFiles(false)

    if (failed.length) {
      const first = failed[0].result
      setDeleteError(first.ok ? '' : first.reason)
      if (gone > 0) {
        setPendingDelete(failed.map((f) => f.entry))
      }
      return
    }

    setPendingDelete(null)
    setSelected(new Set())
    setOpenId(null)
    setBanner({
      variant: 'success',
      title: 'Deleted',
      text:
        gone === 1
          ? 'One file has gone from the outputs folder, and its record with it.'
          : `${gone} files have gone from the outputs folder, and their records with them.`,
    })
  }, [pendingDelete])

  const runMissingAudit = useCallback(async () => {
    setChecking(true)
    const missing = await checkMissing(all(), 20)
    setChecking(false)
    setBanner({
      variant: missing ? 'correction' : 'success',
      title: missing ? 'Correction' : 'All present',
      text: missing
        ? `${missing} ${missing === 1 ? 'file is' : 'files are'} no longer on disk. Their settings are still here. Search is:missing to see them.`
        : 'Every file in the archive is still on disk.',
    })
  }, [])

  const runRecover = useCallback(async () => {
    setRecovering(true)
    try {
      const { filed, fromHistory } = await recoverUnfiled()
      setBanner({
        variant: filed ? 'success' : 'info',
        title: filed ? 'Filed' : 'Nothing to file',
        text: filed
          ? `${filed} ${filed === 1 ? 'file' : 'files'} in the outputs folder had no record. ${
              fromHistory
                ? `${fromHistory} came back with ${fromHistory === 1 ? 'its' : 'their'} settings from ComfyUI's history; the rest are filed by name and date.`
                : 'ComfyUI no longer remembers how they were made, so they are filed by name and date.'
            }`
          : 'Every file in the outputs folder already has a record.',
      })
    } catch (err) {
      setBanner({
        variant: 'error',
        title: 'We could not read the outputs folder',
        text: err instanceof Error ? err.message : String(err),
      })
    } finally {
      setRecovering(false)
    }
  }, [])

  /**
   * Read every picture that has no tags, 24 at a time, on the server's CPU.
   * Stops when the page is left; what was tagged stays tagged.
   */
  const runTagging = useCallback(async () => {
    const todo = all().filter((e) => e.kind === 'image' && !e.missing && !e.tags?.length)
    if (!todo.length) return
    tagStop.current = false
    setTagging({ done: 0, total: todo.length })
    let tagged = 0
    try {
      for (let i = 0; i < todo.length && !tagStop.current; i += 24) {
        const slice = todo.slice(i, i + 24)
        const rows = await tagImages(slice.map((e) => ({ kind: 'output' as const, rel: relPath(e.file) })))
        for (const row of rows) {
          if (row.error) continue
          const e = slice.find((x) => relPath(x.file) === row.rel)
          if (!e) continue
          update(e.id, { tags: row.general.slice(0, 40).map((t) => t.tag), rating: row.rating ?? undefined })
          tagged++
        }
        setTagging({ done: Math.min(todo.length, i + slice.length), total: todo.length })
      }
      setBanner({
        variant: 'success',
        title: 'Tagged',
        text: `${tagged} ${tagged === 1 ? 'picture' : 'pictures'} read. Search tag:something, or use the Content facets.`,
      })
    } catch (err) {
      setBanner({
        variant: 'error',
        title: 'The tagger stopped',
        text: err instanceof Error ? err.message : String(err),
      })
    } finally {
      setTagging(null)
    }
  }, [])

  const importFile = useCallback(async (file: File) => {
    try {
      const added = await file.text().then(importJson)
      setBanner({
        variant: added ? 'success' : 'info',
        title: added ? 'Restored' : 'Nothing to add',
        text: added
          ? `${added} ${added === 1 ? 'record' : 'records'} merged into the archive.`
          : 'That copy held nothing this archive did not already have.',
      })
    } catch (err) {
      setBanner({
        variant: 'error',
        title: 'We could not read that file',
        text: err instanceof Error ? err.message : 'That file is not a SwitchGen archive.',
      })
    }
  }, [])

  const actionsFor = useCallback(
    (entry: HistoryEntry): EntryActions => ({
      onOpen: () => setOpenId(entry.id),
      onReuse: () => reuse(entry, false),
      onAnother: () => reuse(entry, true),
      onSource: (desk) => sendAsSource(entry, desk),
      // Video records have no single frame to paint on, so the bench is not
      // offered for them; the frame picker is the route in for those.
      onRegion:
        entry.kind === 'video'
          ? undefined
          : () => {
              requestRegionEdit(entry)
              goToDesk('images')
            },
      onStar: () => starRecord(entry.id, !entry.starred),
      onDownload: () => {
        saveToDisk(entry).catch((err: unknown) =>
          setBanner({
            variant: 'error',
            title: 'We could not save that file',
            text: err instanceof Error ? err.message : 'ComfyUI did not return the file.',
          }),
        )
      },
      onRemove: () => removeRecords([entry]),
      onDeleteFile: () => {
        setDeleteError(null)
        setPendingDelete([entry])
      },
      onSelect: () => toggleSelection(entry, null),
    }),
    [reuse, sendAsSource, removeRecords, toggleSelection, goToDesk],
  )

  // -- keyboard -------------------------------------------------------------

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // The confirmation owns the keyboard while it is up, and so does anything
      // else drawn over the page. `DeleteDialog` also swallows keys in the
      // capture phase, so this is the belt to its braces.
      if (pendingDelete && pendingDelete.length) return
      if (foreignOverlayIsUp()) return
      // Someone nearer the key already acted on it. The player does the same.
      if (e.defaultPrevented) return
      const typing = isTyping(e.target)

      if (typing) {
        if (e.key === 'Escape') {
          if (query) setQuery('')
          else (e.target as HTMLElement).blur()
        }
        return
      }

      // While focus is inside the player, the player owns the keyboard. They
      // genuinely collide: `k` is play and pause there and the previous record
      // here, `r` is loop there and reuse here, `s` saves the clip there and
      // stars the record here, and `u` opens the frame menu there and sends the
      // picture to a desk here. The thing you are looking at should win.
      // Escape still closes the record.
      const inPlayer = (e.target as HTMLElement | null)?.closest?.('[data-archive-player]')
      if (inPlayer && e.key !== 'Escape') return

      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'a' && ordered.length) {
        e.preventDefault()
        setSelected(new Set(visible.map((r) => r.id)))
        return
      }
      if (e.ctrlKey || e.metaKey || e.altKey) return

      const target = open ?? focused

      switch (e.key) {
        case '/':
          e.preventDefault()
          searchInput.current?.focus()
          searchInput.current?.select()
          return
        case 'Escape':
          if (open) setOpenId(null)
          else if (selected.size) setSelected(new Set())
          else if (query) setQuery('')
          return
        case 'j':
          e.preventDefault()
          if (open) {
            const next = ordered[openIndex + 1]
            if (next) setOpenId(next.id)
          } else move(1)
          return
        case 'k':
          e.preventDefault()
          if (open) {
            const prev = ordered[openIndex - 1]
            if (prev) setOpenId(prev.id)
          } else move(-1)
          return
        case 'Enter':
          if (focused && !open) {
            e.preventDefault()
            setOpenId(focused.id)
          }
          return
        case 'v':
          settings.patch({ view: prefs.view === 'grid' ? 'list' : 'grid' })
          return
        default:
          break
      }

      if (!target) return

      switch (e.key) {
        case 'r':
          e.preventDefault()
          reuse(target, false)
          break
        case 'R':
          e.preventDefault()
          reuse(target, true)
          break
        case 'u':
          e.preventDefault()
          if (target.kind === 'video') setOpenId(target.id)
          else sendAsSource(target, target.desk === 'video' ? 'video' : 'images')
          break
        case 's':
          e.preventDefault()
          starRecord(target.id, !target.starred)
          break
        case 'x':
          e.preventDefault()
          toggleSelection(target, null)
          break
        case 'Backspace':
          e.preventDefault()
          removeRecords(selected.size ? records.filter((r) => selected.has(r.id)) : [target])
          break
        default:
          break
      }
    }

    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [
    pendingDelete,
    query,
    open,
    openIndex,
    focused,
    ordered,
    visible,
    selected,
    records,
    prefs.view,
    move,
    reuse,
    sendAsSource,
    removeRecords,
    toggleSelection,
  ])

  // -- render ---------------------------------------------------------------

  const selectedRecords = useMemo(
    () => records.filter((r) => selected.has(r.id)),
    [records, selected],
  )

  // Pictures, clips or everything. It is a view of the query, not a second
  // piece of state, so the search line stays the whole truth.
  const kind: KindFilter = hasToken(query, 'is:video')
    ? 'video'
    : hasToken(query, 'is:image')
      ? 'image'
      : 'all'

  const setKind = useCallback((next: KindFilter) => {
    setQuery((current) => {
      let out = current
      if (hasToken(out, 'is:image')) out = toggleToken(out, 'is:image')
      if (hasToken(out, 'is:video')) out = toggleToken(out, 'is:video')
      if (next !== 'all') out = toggleToken(out, next === 'video' ? 'is:video' : 'is:image')
      return out.trim()
    })
  }, [])

  const registerRef = useCallback((entry: HistoryEntry, el: HTMLElement | null) => {
    if (el) cardRefs.current.set(entry.id, el)
    else cardRefs.current.delete(entry.id)
  }, [])

  const total = countRecords()
  const empty = total === 0
  const nothingMatches = !empty && ordered.length === 0

  return (
    <div className="px-6 pb-24">
      <div className="flex flex-col gap-8 lg:flex-row">
        <FacetRail
          entries={records}
          query={query}
          onToggle={(token) => setQuery((current) => toggleToken(current, token))}
          onClear={() => setQuery('')}
          onExport={() =>
            saveText(`switchgen-archive-${new Date().toISOString().slice(0, 10)}.json`, exportJson())
          }
          onImport={importFile}
          onCheckMissing={runMissingAudit}
          checking={checking}
          persistent={isPersistent()}
          canDeleteFiles={canDeleteFiles}
          sync={sync}
          onRecover={() => void runRecover()}
          recovering={recovering}
          canTag={canTag}
          onTagAll={() => void runTagging()}
          tagging={tagging}
        />

        <main className="min-w-0 flex-1">
          <SearchBand
            value={query}
            onChange={setQuery}
            shown={ordered.length}
            total={total}
            inputRef={searchInput}
            kind={kind}
            onKind={setKind}
            view={prefs.view}
            onView={(view) => settings.patch({ view })}
            sort={prefs.sort}
            onSort={(sort) => settings.patch({ sort })}
          />

          <div className="mt-4 space-y-3">
            {issue && (
              <Notice
                tone="correction"
                title="Correction"
                onDismiss={() => {
                  clearLoadIssue()
                  setIssue(null)
                }}
              >
                {issue}
              </Notice>
            )}
            {quota && (
              <Notice
                tone="correction"
                title="Correction"
                onDismiss={() => {
                  clearQuotaIssue()
                  setQuota(null)
                }}
              >
                {quota}
              </Notice>
            )}
            {banner && (
              <Notice tone={banner.variant} title={banner.title} onDismiss={() => setBanner(null)}>
                {banner.text}
                {banner.notes?.map((note) => (
                  <span key={note.field} className="mt-1 block text-caption not-italic">
                    {note.reason}
                  </span>
                ))}
                {banner.actions && (
                  <span className="mt-2 flex flex-wrap gap-4 not-italic">
                    {banner.actions.map((action) => (
                      <button
                        key={action.label}
                        type="button"
                        onClick={action.run}
                        className="text-[0.625rem] font-semibold tracking-[0.16em] text-burgundy-900 uppercase underline underline-offset-4 hover:no-underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900"
                      >
                        {action.label}
                      </button>
                    ))}
                  </span>
                )}
              </Notice>
            )}
          </div>

          {empty && (
            <section className="mx-auto mt-16 max-w-2xl">
              <h2 className="font-serif text-[2.25rem] leading-tight font-bold">Nothing here yet</h2>
              <p className="mt-4 text-body leading-relaxed">
                Everything you make is filed here automatically, with the settings that made it. You
                can search it, reuse it, and clear it out.
              </p>
              <p className="mt-3 text-body leading-relaxed">
                Start on the{' '}
                <button
                  type="button"
                  onClick={() => goToDesk('images')}
                  className="text-burgundy-900 underline underline-offset-4 hover:no-underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900"
                >
                  Pictures desk
                </button>
                , where a picture takes seconds, or on the{' '}
                <button
                  type="button"
                  onClick={() => goToDesk('video')}
                  className="text-burgundy-900 underline underline-offset-4 hover:no-underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900"
                >
                  Video desk
                </button>
                , which takes minutes.
              </p>
              <p className="mt-6 border-t border-grey-300 pt-3 text-small text-grey-700 italic">
                Already have an archive saved? Restore it from the index on the left.
              </p>
            </section>
          )}

          {nothingMatches && (
            <section className="mt-16 max-w-2xl">
              <h2 className="font-serif text-[1.5rem] font-semibold">Nothing matches that</h2>
              <p className="mt-2 text-body leading-relaxed text-grey-700">
                {total.toLocaleString('en-GB')} records are filed, but none of them answer to{' '}
                <span className="text-ink italic">{applied}</span>. Try fewer words, or{' '}
                <button
                  type="button"
                  onClick={() => setQuery('')}
                  className="text-burgundy-900 underline underline-offset-4 hover:no-underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900"
                >
                  clear the search
                </button>
                .
              </p>
            </section>
          )}

          {!empty && !nothingMatches && prefs.view === 'list' && (
            <div className="mt-4">
              <Table
                entries={visible}
                terms={terms}
                sortKey={sortKey}
                ascending={ascending}
                onSort={(key) => {
                  if (key === sortKey) setAscending((v) => !v)
                  else {
                    setSortKey(key)
                    setAscending(false)
                  }
                }}
                focusedId={focusedId}
                selected={selected}
                canDeleteFile={canDeleteFiles}
                onActivate={(entry, e) => {
                  if (e.shiftKey || selected.size) toggleSelection(entry, e)
                  else setOpenId(entry.id)
                }}
                onFocused={(entry) => setFocusedId(entry.id)}
                onToggleSelect={(entry, e) => toggleSelection(entry, e)}
                registerRef={registerRef}
                actionsFor={actionsFor}
              />
            </div>
          )}

          {!empty && !nothingMatches && prefs.view === 'grid' && (
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
                          selecting={selected.size > 0}
                          canDeleteFile={canDeleteFiles}
                          onActivate={(e) => {
                            if (e.shiftKey || selected.size) toggleSelection(entry, e)
                            else setOpenId(entry.id)
                          }}
                          onFocused={() => setFocusedId(entry.id)}
                          onToggleSelect={(e) => toggleSelection(entry, e)}
                          registerRef={(el) => registerRef(entry, el)}
                          {...actionsFor(entry)}
                        />
                      </li>
                    ))}
                  </ul>
                </section>
              ))}
            </div>
          )}

          {!empty && ordered.length > visible.length && (
            <div ref={sentinel} className="py-10 text-center">
              <button
                type="button"
                onClick={() => setLimit((n) => n + PAGE)}
                className="text-[0.625rem] font-semibold tracking-[0.18em] text-burgundy-900 uppercase underline underline-offset-4 hover:no-underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900"
              >
                Show {Math.min(PAGE, ordered.length - visible.length)} more
              </button>
              <p className="mt-1 text-caption text-grey-500 tabular-nums">
                {visible.length.toLocaleString('en-GB')} of {ordered.length.toLocaleString('en-GB')}{' '}
                shown
              </p>
            </div>
          )}

          <SelectionBar
            count={selected.size}
            starred={selectedRecords.filter((r) => r.starred).length}
            deletable={selectedRecords.filter((r) => !r.missing).length}
            canDeleteFiles={canDeleteFiles}
            onStar={(on) => selectedRecords.forEach((r) => starRecord(r.id, on))}
            onRemove={() => removeRecords(selectedRecords)}
            onDeleteFiles={() => {
              const withFiles = selectedRecords.filter((r) => !r.missing)
              setDeleteError(null)
              if (!withFiles.length) {
                setBanner({
                  variant: 'correction',
                  title: 'Correction',
                  text: 'None of those files are on disk any more. You can remove their records instead. The settings go with them.',
                })
                return
              }
              setPendingDelete(withFiles)
            }}
            onSelectAll={() => setSelected(new Set(visible.map((r) => r.id)))}
            onClear={() => setSelected(new Set())}
          />
        </main>
      </div>

      {open && (
        <Detail
          entry={open}
          canDeleteFile={canDeleteFiles}
          hasPrev={openIndex > 0}
          hasNext={openIndex >= 0 && openIndex < ordered.length - 1}
          onPrev={() => {
            const prev = ordered[openIndex - 1]
            if (prev) setOpenId(prev.id)
          }}
          onNext={() => {
            const next = ordered[openIndex + 1]
            if (next) setOpenId(next.id)
          }}
          onClose={() => {
            setOpenId(null)
            focusCard(open.id)
          }}
          onOpenEntry={(id) => setOpenId(id)}
          {...actionsFor(open)}
        />
      )}

      {pendingDelete && pendingDelete.length > 0 && (
        <DeleteDialog
          records={pendingDelete}
          busy={deleting}
          error={deleteError}
          onCancel={() => {
            setPendingDelete(null)
            setDeleteError(null)
          }}
          onConfirm={() => void confirmDelete()}
        />
      )}

    </div>
  )
}

export default ArchivePage
