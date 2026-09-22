/**
 * The LoRA library.
 *
 * Two populations in one list, and the difference is stated rather than
 * implied: files on disk, which can be added to the stack now, and verified
 * catalogue entries, which have to be fetched first. Hiding the second group
 * would leave the user with an empty picker and no idea that thirty anatomy
 * LoRAs exist for the checkpoint they have loaded.
 *
 * The compatibility filter defaults to on. A picker that lists every file for
 * every checkpoint is how a Pony LoRA ends up on a Flux model, which fails
 * silently: LoraLoader patches whatever keys happen to match, reports nothing,
 * and hands back a model that renders noise. The filter can be turned off, and
 * what it was hiding is then labelled rather than merely present.
 */
import { useEffect, useMemo, useRef, useState } from 'react'

import {
  ARCH_LABEL,
  CATEGORY_LABEL,
  CATEGORY_NOTE,
  fetchLora,
  fitFor,
  size,
  type FetchProgress,
  type LoraCategory,
  type LoraInfo,
  type LoraLibrary,
  type LoraTarget,
} from '../../lib/loras'
import { useServerCapabilities } from '../../lib/capabilities'
import { Badge, Kicker, Meter, RING, Tap } from './bits'

type Job = { progress: FetchProgress; abort: AbortController }

export function Picker({
  lib,
  target,
  inStack,
  onAdd,
  onFetched,
  onClose,
}: {
  lib: LoraLibrary
  target: LoraTarget
  /** Files already in the stack, so they are offered as present rather than twice. */
  inStack: ReadonlySet<string>
  onAdd: (info: LoraInfo) => void
  /** A file landed on disk. The parent re reads /api/models. */
  onFetched: () => void
  onClose: () => void
}) {
  const [query, setQuery] = useState('')
  const [fitsOnly, setFitsOnly] = useState(true)
  const [installedOnly, setInstalledOnly] = useState(false)
  const [jobs, setJobs] = useState<Record<string, Job>>({})
  const search = useRef<HTMLInputElement | null>(null)
  const caps = useServerCapabilities()
  /** False once the server has said aria2c is not there to run. */
  const canFetch = caps === null ? true : caps.downloads

  // The picker opens because the user asked for it, so the caret belongs in
  // the search field: seventy four rows is a list you type at, not scroll.
  // Closing hands focus back to whatever opened it, because the Close button
  // is inside the panel and unmounts with it, which otherwise drops focus onto
  // the body and sends a keyboard session back to the top of the page.
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null
    search.current?.focus()
    return () => {
      if (opener && opener.isConnected) opener.focus()
    }
  }, [])

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase()
    const out: { info: LoraInfo; fit: ReturnType<typeof fitFor> }[] = []
    for (const info of lib.all) {
      const fit = fitFor(info, target)
      if (fitsOnly && fit.level === 'mismatch') continue
      if (installedOnly && !info.installed) continue
      if (
        q &&
        !info.label.toLowerCase().includes(q) &&
        !info.file.toLowerCase().includes(q) &&
        !info.does.toLowerCase().includes(q) &&
        !info.trigger.toLowerCase().includes(q)
      ) {
        continue
      }
      out.push({ info, fit })
    }
    return out
  }, [lib, target, query, fitsOnly, installedOnly])

  const groups = useMemo(() => {
    const by = new Map<LoraCategory, typeof rows>()
    for (const row of rows) {
      const list = by.get(row.info.category) ?? []
      list.push(row)
      by.set(row.info.category, list)
    }
    return [...by.entries()]
  }, [rows])

  const hidden = lib.all.length - rows.length

  const fetchOne = (info: LoraInfo) => {
    const abort = new AbortController()
    const set = (progress: FetchProgress) =>
      setJobs((j) => ({ ...j, [info.file]: { progress, abort } }))
    set({ state: 'starting', pct: 0, done: 0, total: info.bytes, speed: 0, etaSec: null, error: null })
    void fetchLora(info, lib.folder, set, abort.signal)
      .then(() => {
        setJobs((j) => {
          const next = { ...j }
          delete next[info.file]
          return next
        })
        onFetched()
      })
      .catch((err: unknown) => {
        const message = abort.signal.aborted ? 'Stopped. The partial file is kept.' : String((err as Error)?.message ?? err)
        set({
          state: abort.signal.aborted ? 'cancelled' : 'error',
          pct: 0,
          done: 0,
          total: info.bytes,
          speed: 0,
          etaSec: null,
          error: message,
        })
      })
  }

  return (
    <div
      className="mt-3 border border-grey-300 p-3"
      // The shortcuts card promises that Escape closes what is open, and the
      // picker is open. It is an inline panel rather than a modal, so it takes
      // the key only while focus is inside it, and it stops the key there so
      // the shell does not also blur the search field on the way out.
      onKeyDown={(e) => {
        if (e.key !== 'Escape') return
        e.preventDefault()
        e.stopPropagation()
        onClose()
      }}
    >
      <div className="flex items-baseline justify-between gap-2">
        <Kicker tone="burgundy">All add-ons</Kicker>
        <button type="button" className={`sg-link text-caption ${RING}`} onClick={onClose}>
          Close
        </button>
      </div>

      <input
        ref={search}
        type="search"
        className="field mt-2 w-full"
        placeholder="Search: skin, hands, eyes, lighting, film grain"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        aria-label="Search all add-ons"
      />

      <div className="mt-2 flex flex-wrap items-center gap-2">
        <Tap active={fitsOnly} onClick={() => setFitsOnly((v) => !v)}>
          fits this model
        </Tap>
        <Tap active={installedOnly} onClick={() => setInstalledOnly((v) => !v)}>
          downloaded
        </Tap>
        <span className="ml-auto text-caption tabular-nums text-grey-500">
          {rows.length} of {lib.all.length}
        </span>
      </div>

      {fitsOnly && hidden > 0 ? (
        <p className="mt-2 text-caption italic text-grey-500">
          {hidden} hidden: made for a different kind of model than {ARCH_LABEL[target.arch]}.
        </p>
      ) : null}

      {!canFetch ? (
        <p className="mt-2 text-caption italic text-warning">
          aria2c is not on the server, so nothing can be fetched from here. Files already on disk
          can still be added.
        </p>
      ) : null}

      {!rows.length ? (
        <p className="mt-3 text-caption italic text-grey-700">
          Nothing matches. Clear the search, or turn off the filters above to see what was hidden.
        </p>
      ) : null}

      <div className="mt-2 max-h-[26rem] overflow-y-auto">
        {groups.map(([category, list]) => (
          <section key={category} className="mb-3">
            <div className="border-b border-grey-300 pb-1">
              <Kicker>{CATEGORY_LABEL[category]}</Kicker>
            </div>
            <p className="mt-1 text-caption italic text-grey-500">{CATEGORY_NOTE[category]}</p>
            <ul>
              {list.map(({ info, fit }) => {
                const job = jobs[info.file]
                const present = inStack.has(info.file)
                return (
                  <li key={info.file} className="border-b border-grey-200 py-2">
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
                          <span className="text-small leading-tight text-ink">{info.label}</span>
                          <Badge tone={fit.level} title={fit.why}>
                            {fit.level === 'match'
                              ? 'fits'
                              : fit.level === 'untested'
                                ? 'untested'
                                : 'wrong model'}
                          </Badge>
                          {info.installed ? null : <Badge tone="plain">download</Badge>}
                        </div>
                        <p className="mt-0.5 text-[0.6875rem] text-grey-500">
                          {ARCH_LABEL[info.arch]}
                          <span className="px-1 text-grey-300">|</span>
                          <span className="tabular-nums">{size(info.bytes)}</span>
                          {info.trigger ? (
                            <>
                              <span className="px-1 text-grey-300">|</span>
                              <span className="italic">say: {info.trigger}</span>
                            </>
                          ) : null}
                        </p>
                        <p className="mt-1 text-caption leading-snug text-grey-700">{info.does}</p>
                        {fit.level !== 'match' ? (
                          <p
                            className={`mt-1 text-caption leading-snug ${
                              fit.level === 'mismatch' ? 'text-error' : 'text-warning'
                            }`}
                          >
                            {fit.why}
                          </p>
                        ) : null}
                        {info.caution ? (
                          <p className="mt-1 text-caption italic text-warning">{info.caution}</p>
                        ) : null}
                      </div>

                      <div className="flex shrink-0 flex-col items-end gap-1">
                        {present ? (
                          <Badge tone="plain">in use</Badge>
                        ) : info.installed ? (
                          <Tap onClick={() => onAdd(info)}>add</Tap>
                        ) : job && (job.progress.state === 'starting' || job.progress.state === 'downloading') ? (
                          <Tap onClick={() => job.abort.abort()}>stop</Tap>
                        ) : (
                          <Tap onClick={() => fetchOne(info)} disabled={!info.url || !canFetch}>
                            fetch
                          </Tap>
                        )}
                      </div>
                    </div>

                    {job ? (
                      <div className="mt-1">
                        {job.progress.state === 'error' || job.progress.state === 'cancelled' ? (
                          <p className="text-caption text-error">{job.progress.error}</p>
                        ) : (
                          <>
                            <Meter pct={job.progress.pct} label={`Fetching ${info.label}`} />
                            <p className="mt-0.5 text-caption tabular-nums text-grey-500">
                              {Math.round(job.progress.pct * 100)}% of {size(job.progress.total || info.bytes)}
                              {job.progress.speed ? ` at ${size(job.progress.speed)} per second` : ''}
                            </p>
                          </>
                        )}
                      </div>
                    ) : null}
                  </li>
                )
              })}
            </ul>
          </section>
        ))}
      </div>
    </div>
  )
}
