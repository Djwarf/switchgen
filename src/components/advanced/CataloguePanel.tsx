/**
 * THE CATALOGUE: what this desk could run if it fetched the files.
 *
 * The model panel above lists what is installed. This lists the rest of what
 * this app has a verified graph for, what each is missing, how big that is,
 * and whether the server thinks it will fit. Fetching runs through the same
 * download server the add-on picker uses, and keeps going if the reader
 * leaves this panel: the run lives in a store, not in this component, and
 * the fetch itself at the server, which a reloaded page takes up again.
 *
 * Two facts shape the copy. The catalogue and the registry were written
 * apart, so a family is matched to its catalogue entry by the files it loads
 * and not by its name; and the catalogue never lists a community finetune,
 * so a family it has no entry for is named with its file and left to be
 * placed by hand, and is counted installed from the model tree rather than
 * from a catalogue that cannot know.
 */
import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useServerCapabilities } from '../../lib/capabilities'
import {
  bytesText,
  fetchCatalog,
  fetchPlan,
  forgetCatalog,
  heldBack,
  noToken,
  type CatalogFamily,
  type CatalogFile,
  type CatalogPlan,
} from '../../lib/catalog'
import { cancelPlan, forgetPlan, startPlan, useDownloads } from '../../lib/downloads'
import { modelFiles } from '../../lib/hardware'
import { FAMILIES, modelsOf, sidecarsOf, type FamilyDef } from '../../lib/workflows'
import { Meter } from '../loras/bits'
import { Caution, Head, Link, Note } from './bits'

/** What a family still lacks: main weights with no usable one on disk, and the other files its graph loads. */
type Lack = { mains: string[]; files: string[] }

type Row = { def: FamilyDef; cat: CatalogFamily | null; installed: boolean; lack: Lack }

const base = (file: string) => file.split('/').pop() ?? file

/**
 * What stands between a family and a render, read off the files its graph
 * loads. The text encoders, the VAE and any add-on the graph names are all
 * needed; of the main weights, one of the listed alternatives is enough,
 * except in a two-model family, which loads both halves.
 *
 * A name on disk is not proof on its own, because aria2c writes under the
 * final name from the first byte, so a file a fetch has not finished does not
 * count. The catalogue's own size check is not used for this: it cannot tell
 * a partial from a different quant saved under the same name, and calls a
 * complete mixed fp8 qwen_3_4b.safetensors short of the bf16 file's size.
 */
function lackOf(def: FamilyDef, onDisk: ReadonlySet<string>, unfinished: ReadonlySet<string>): Lack {
  const whole = (file: string) => onDisk.has(base(file)) && !unfinished.has(base(file))
  const { clip, vae } = sidecarsOf(def)
  const addOns = Object.values(def.graph)
    .map((n) => n.inputs['lora_name'])
    .filter((l): l is string => typeof l === 'string')
  const files = [...new Set([...clip, ...(vae ? [vae] : []), ...addOns])].filter((f) => !whole(f))
  const mains = def.dualModel
    ? def.models.filter((m) => !whole(m))
    : def.models.some(whole)
      ? []
      : def.models
  return { mains, files }
}

/** Does this family load the file: one of its main weights, or a file its graph names? */
function loads(def: FamilyDef, file: string): boolean {
  const name = base(file)
  if (def.models.some((m) => base(m) === name)) return true
  return Object.values(def.graph).some((n) =>
    Object.values(n.inputs).some((v) => typeof v === 'string' && base(v) === name),
  )
}

/**
 * The catalogue entry for a registry family: the same id, or failing that
 * the entry that lists the most of the family's weight files, preferring one
 * of the same mode when two tie (Qwen 2.1 lists its one file under an image
 * entry and an edit entry).
 */
function matchCatalogue(def: FamilyDef, families: readonly CatalogFamily[]): CatalogFamily | null {
  const direct = families.find((f) => f.id === def.id)
  if (direct) return direct
  const wanted = new Set(modelsOf(def).map(base))
  let best: { fam: CatalogFamily; score: number } | null = null
  for (const fam of families) {
    const listed = [...(fam.models ?? []), ...(fam.deps ?? [])]
    const hits = listed.filter((m) => wanted.has(base(m))).length
    if (!hits) continue
    const score = hits * 2 + (fam.mode === def.mode ? 1 : 0)
    if (!best || score > best.score) best = { fam, score }
  }
  return best?.fam ?? null
}

export function CataloguePanel({
  modes,
  onInstalled,
}: {
  /** Which families belong on this desk. */
  modes: readonly FamilyDef['mode'][]
  /** A family's files all landed; the desk should re-read what is installed. */
  onInstalled: () => void
}) {
  const caps = useServerCapabilities()
  const runs = useDownloads()
  const [catalog, setCatalog] = useState<{ families: CatalogFamily[]; total: number } | null>(null)
  const [onDisk, setOnDisk] = useState<ReadonlySet<string> | null>(null)
  /** Reading the catalogue failed. A later read that succeeds clears it. */
  const [catalogError, setCatalogError] = useState<string | null>(null)
  const [plans, setPlans] = useState<Record<string, CatalogPlan | 'loading' | undefined>>({})
  /**
   * Asking whether a family fits failed, by catalogue id. Kept apart from the
   * catalogue's own error: a re-read runs whenever another family lands, and
   * clearing everything then wiped the one line saying this row's question
   * went unanswered, while the row itself was back on its Fetch link. Only
   * asking about the same row again clears it.
   */
  const [askFailed, setAskFailed] = useState<Record<string, string | undefined>>({})
  const mounted = useRef(false)
  const installedRef = useRef(onInstalled)
  /** Catalogue ids whose landing is being re-read, so it is handled once. */
  const settling = useRef(new Set<string>())

  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
    }
  }, [])

  useEffect(() => {
    installedRef.current = onInstalled
  })

  useEffect(() => {
    if (!caps?.downloads) return
    let live = true
    void fetchCatalog()
      .then((c) => {
        if (live) setCatalog({ families: c.families, total: c.counts.families })
      })
      .catch((e: unknown) => {
        if (live) setCatalogError(e instanceof Error ? e.message : String(e))
      })
    void modelFiles()
      .then((m) => {
        if (live) setOnDisk(new Set([...m.keys()].map(base)))
      })
      .catch(() => {
        if (live) setOnDisk(new Set())
      })
    return () => {
      live = false
    }
  }, [caps?.downloads])

  const { rows, unmatchedTotal } = useMemo(() => {
    if (!catalog) return { rows: [] as Row[], unmatchedTotal: 0 }
    // Files a run in the store set out to fetch and has not finished: the one
    // being written now, the ones queued behind it, and the one a failed run
    // stopped on, which the server keeps so the next fetch can resume it.
    const unfinished = new Set<string>()
    for (const run of runs.values()) {
      if (run.state === 'done') continue
      for (const f of run.files) if (!run.finished.includes(f.filename)) unfinished.add(base(f.filename))
    }
    const matchedIds = new Set<string>()
    const all = FAMILIES.map((def) => {
      const cat = matchCatalogue(def, catalog.families)
      if (cat) matchedIds.add(cat.id)
      const lack = onDisk ? lackOf(def, onDisk, unfinished) : { mains: def.models, files: [] }
      const installed = onDisk ? !lack.mains.length && !lack.files.length : cat?.installed.ready ?? false
      return { def, cat, installed, lack }
    })
    return {
      rows: all.filter((r) => modes.includes(r.def.mode)),
      unmatchedTotal: catalog.total - matchedIds.size,
    }
  }, [catalog, onDisk, modes, runs])

  /** Ask the server again what is in the catalogue and on disk. Never rejects; a failure is said in the panel. */
  const reread = useCallback(async () => {
    forgetCatalog()
    await Promise.all([
      fetchCatalog(true).then(
        (c) => {
          if (!mounted.current) return
          setCatalog({ families: c.families, total: c.counts.families })
          setCatalogError(null)
        },
        (e: unknown) => {
          if (mounted.current) setCatalogError(e instanceof Error ? e.message : String(e))
        },
      ),
      modelFiles().then(
        (m) => {
          if (mounted.current) setOnDisk(new Set([...m.keys()].map(base)))
        },
        () => {},
      ),
    ])
  }, [])

  // A family on this desk landed. The run lives in the store, so this may be
  // the panel that started it or a later one, mounted after the reader left
  // and came back: either way, re-read what is installed, tell the desk that
  // is on screen now, and only then forget the run, so its row says it landed
  // until the fresh reading takes its place.
  useEffect(() => {
    for (const row of rows) {
      const id = row.cat?.id
      if (!id || runs.get(id)?.state !== 'done' || settling.current.has(id)) continue
      settling.current.add(id)
      void reread().then(() => {
        settling.current.delete(id)
        if (!mounted.current) return
        setPlans((p) => ({ ...p, [id]: undefined }))
        forgetPlan(id)
        installedRef.current()
      })
    }
  }, [rows, runs, reread])

  if (!caps) return null
  if (!caps.downloads) {
    // A reason means the server never answered, so nothing was checked and
    // aria2c may well be there; the hook asks again, and this gives way to
    // the catalogue once it does. Without one, the server looked and found
    // no aria2c.
    return (
      <section className="mb-7">
        <Head title="The catalogue" />
        <Note>
          {caps.reason !== null
            ? `${caps.reason.charAt(0).toUpperCase()}${caps.reason.slice(1)}. This panel asks again on its own.`
            : 'aria2c is not on the server, so nothing can be fetched from here.'}{' '}
          Files placed under the models folder by hand are picked up the next time the desk reads what
          is installed.
        </Note>
      </section>
    )
  }

  const installedCount = rows.filter((r) => r.installed).length
  // A row with a run in the store stays on screen whatever the files say, so
  // its progress and its Stop link cannot vanish mid-fetch. Otherwise a row
  // is offered for fetching when the catalogue has something to fetch or
  // names what it cannot; a family lacking only files the catalogue does not
  // list for it is left to be placed by hand.
  const fetchable = rows.filter(
    (r) => r.cat && (runs.has(r.cat.id) || (!r.installed && (r.cat.installed.missing.length || r.cat.incomplete?.length))),
  )
  const byHand = rows.filter((r) => !r.installed && !fetchable.includes(r))
  // Files that installed families on this desk use as they are on disk,
  // named once each however many families load them. An installed family is
  // not listed above (unless a run keeps it there, and then its row says
  // this), so without this nothing would say that one of its files is not
  // the catalogue's.
  const usedAsIs = new Map<string, { file: CatalogFile; labels: string[] }>()
  for (const r of rows) {
    if (!r.installed || !r.cat || fetchable.includes(r)) continue
    for (const f of r.cat.installed.files) {
      if (!f.conflict || !loads(r.def, f.filename)) continue
      const seen = usedAsIs.get(f.filename)
      if (seen) seen.labels.push(r.def.label)
      else usedAsIs.set(f.filename, { file: f, labels: [r.def.label] })
    }
  }

  const ask = (row: Row) => {
    const cat = row.cat!
    setAskFailed((f) => ({ ...f, [cat.id]: undefined }))
    setPlans((p) => ({ ...p, [cat.id]: 'loading' }))
    void fetchPlan(cat.id, cat.installed.chosenModel ?? null)
      .then((plan) => {
        setPlans((p) => ({ ...p, [cat.id]: plan }))
        // Fits, and nothing gated is missing a token: go without another question.
        if (plan.fits && !(plan.gated.files.length && !plan.gated.tokenPresent)) {
          startPlan({ family: cat.id, model: plan.chosenModel })
        }
      })
      .catch((e: unknown) => {
        if (!mounted.current) return
        // No plan came back, so the row returns to its Fetch link and asking
        // again is one click; left on 'loading' it would wait for an answer
        // that is not coming. The reason is said on the row, above that link.
        const why = (e instanceof Error ? e.message : String(e)).replace(/\.$/, '')
        setPlans((p) => ({ ...p, [cat.id]: undefined }))
        setAskFailed((f) => ({ ...f, [cat.id]: `Could not ask the server whether it fits: ${why}.` }))
      })
  }

  return (
    <section className="mb-7">
      <Head
        title="The catalogue"
        figure={catalog ? `${installedCount} of ${rows.length} installed` : null}
        note={
          catalog
            ? `${unmatchedTotal} families in the catalogue have no verified graph in this app and are not offered.`
            : 'Reading the catalogue.'
        }
      />
      {catalogError ? <Caution>{catalogError}</Caution> : null}

      {catalog && !fetchable.length && !byHand.length ? (
        <Note>Every family this desk has a graph for is installed.</Note>
      ) : null}

      <ul className="divide-y divide-grey-300">
        {fetchable.map((row) => {
          const cat = row.cat!
          const id = cat.id
          const run = runs.get(id)
          const plan = plans[id]
          const files = cat.installed.missing
          const gated = cat.installed.gatedMissing.length > 0
          // The verdict of a plan that fits quotes the server's reasons, and
          // those name each file kept as it is, so a row held back only for a
          // missing token leaves them to it. A plan that will not run here
          // quotes only what blocks it, so there the row still names them:
          // otherwise nothing would say, as the reader decides whether to
          // fetch anyway, that the fetch leaves those files alone.
          const verdictNamesThem =
            !run &&
            plan !== undefined &&
            plan !== 'loading' &&
            plan.fits &&
            plan.gated.files.length > 0 &&
            !plan.gated.tokenPresent
          const keptAsIs = verdictNamesThem ? [] : cat.installed.files.filter((f) => f.conflict && loads(row.def, f.filename))
          const failed = askFailed[id]
          return (
            <li key={row.def.id} className="py-2">
              <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
                <span className="text-small text-ink">{row.def.label}</span>
                <span className="text-caption tabular-nums text-grey-500">
                  {files.length} {files.length === 1 ? 'file' : 'files'} · {bytesText(cat.installed.missingBytes)}
                  {gated ? ' · gated' : ''}
                </span>
              </div>
              {files.length || cat.incomplete?.length ? (
                <p className="mt-0.5 text-caption text-grey-700">
                  {files.length ? `Needs ${files.join(', ')}.` : ''}
                  {cat.incomplete?.length ? ` No verified download exists for ${cat.incomplete.join(', ')}; that file has to be placed by hand.` : ''}
                </p>
              ) : null}
              {keptAsIs.map((f) => (
                <p key={f.filename} className="mt-0.5 text-caption text-grey-700">
                  {f.filename} is on disk {sizeAgainstCatalogue(f)}. {NOT_THE_CATALOGUES} It is used as it is, and
                  the catalogue’s own is not fetched. {TO_FETCH_IT}
                </p>
              ))}

              {run && (run.state === 'starting' || run.state === 'running') ? (
                <div className="mt-1">
                  <p className="text-caption italic text-grey-700">
                    {run.current
                      ? `Fetching ${run.current.filename}, ${run.current.index} of ${run.current.count}: ${bytesText(run.current.done)} of ${bytesText(run.current.total)}${run.current.etaSec ? `, about ${Math.ceil(run.current.etaSec / 60)} min left` : ''}.`
                      : 'Starting.'}{' '}
                    <Link onClick={() => void cancelPlan(id)}>Stop</Link>
                  </p>
                  <Meter pct={run.current?.pct ?? 0} label={`Fetching ${row.def.label}`} />
                  {/* Until the server says the fetch has begun it is still on its
                      checks, and a page that hangs up then calls the fetch off
                      (see POST /api/download). So only a run that has begun is
                      promised to outlive the page. */}
                  <p className="mt-0.5 text-caption text-grey-500">
                    {run.outOfTouch
                      ? 'The server is not answering. These figures are from its last answer, and this page keeps asking.'
                      : run.state === 'running'
                        ? 'The server carries on with it if this page is closed or the phone is locked.'
                        : 'Keep this page open until the fetch begins. From then on the server carries on with it if the page is closed or the phone is locked.'}
                  </p>
                </div>
              ) : run && run.state === 'error' ? (
                <p className="mt-1 text-caption text-ink-error">
                  {run.error}{' '}
                  <Link
                    onClick={() => {
                      // A failed run may have left a partial behind, or have
                      // landed after all when it was lost track of, so the
                      // row is read again from the disk.
                      forgetPlan(id)
                      void reread()
                    }}
                  >
                    Dismiss
                  </Link>
                </p>
              ) : run && run.state === 'cancelled' ? (
                <p className="mt-1 text-caption italic text-grey-700">
                  Stopped. <Link onClick={() => forgetPlan(id)}>Dismiss</Link>
                </p>
              ) : run && run.state === 'done' ? (
                <p className="mt-1 text-caption italic text-grey-700">Landed. The desk is re-reading what is installed.</p>
              ) : plan === 'loading' ? (
                <p className="mt-1 text-caption italic text-grey-500">Asking the server whether it fits.</p>
              ) : plan && (!plan.fits || (plan.gated.files.length && !plan.gated.tokenPresent)) ? (
                <div className="mt-1 space-y-1">
                  <Caution>{plan.verdict}</Caution>
                  {plan.blockers.map((b) => (
                    <p key={b} className="text-caption text-grey-700">{b}</p>
                  ))}
                  {noToken(plan) ? <p className="text-caption text-grey-700">{noToken(plan)}</p> : null}
                  {plan.incomplete.length ? null : (
                    <p className="text-caption">
                      <Link
                        onClick={() => {
                          if (window.confirm(`${heldBack(plan)}\n\nFetch ${row.def.label} anyway?`)) {
                            startPlan({ family: id, model: plan.chosenModel, force: true })
                          }
                        }}
                      >
                        Fetch anyway
                      </Link>
                    </p>
                  )}
                </div>
              ) : (
                <>
                  {failed ? <p className="mt-1 text-caption text-ink-error">{failed}</p> : null}
                  <p className="mt-1 text-caption">
                    {cat.incomplete?.length ? null : (
                      <Link onClick={() => ask(row)}>
                        Fetch {files.length === 1 ? 'the file' : `${files.length} files`} ({bytesText(cat.installed.missingBytes)})
                      </Link>
                    )}
                  </p>
                </>
              )}
            </li>
          )
        })}
      </ul>

      {byHand.length ? (
        <div className="mt-2">
          <Note>
            No download source in the catalogue, so these have to be placed under the models folder by
            hand:
          </Note>
          <ul className="mt-1 space-y-0.5 text-caption text-grey-700">
            {byHand.map((r) => (
              <li key={r.def.id}>
                {r.def.label}: <Needs def={r.def} lack={r.lack} />
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {[...usedAsIs.values()].map(({ file, labels }) => (
        <p key={file.filename} className="mt-2 text-caption text-grey-700">
          {labels.join(', ')} {labels.length === 1 ? 'uses' : 'use'} {file.filename} as it is on disk,{' '}
          {sizeAgainstCatalogue(file)}. {NOT_THE_CATALOGUES} {TO_FETCH_IT}
        </p>
      ))}
    </section>
  )
}

// A same-name file of another size, which the server counts as installed and
// will not fetch over. On a row it explains why the row's list and size leave
// the file out; under the list it names what installed families load. Both
// say the size is all that was compared: nothing here loads the file.

/** `at 5.2 GB, where the catalogue lists 7.5 GB`. */
function sizeAgainstCatalogue(f: CatalogFile): string {
  return `at ${bytesText(f.installedBytes)}${f.sizeBytes ? `, where the catalogue lists ${bytesText(f.sizeBytes)}` : ''}`
}

const NOT_THE_CATALOGUES =
  'It is not a fetch that stopped part way, so it is probably another build under the same name, and nothing ' +
  'here has checked that it loads.'

const TO_FETCH_IT = 'To fetch the catalogue’s own, move this file aside first.'

/** The files a family lacks, named for placing by hand. */
function Needs({ def, lack }: { def: FamilyDef; lack: Lack }) {
  // One of several alternative main weights will do, so name the first and
  // count the rest; a two-model family names each half it lacks.
  const oneOf = !def.dualModel && lack.mains.length > 1
  const named = oneOf ? lack.mains.slice(0, 1) : lack.mains
  const others = lack.mains.length - 1
  return (
    <>
      {[...named, ...lack.files].map((f, i) => (
        <Fragment key={f}>
          {i ? ', ' : ''}
          <code>{f}</code>
          {oneOf && i === 0 ? ` or ${others} other file${others > 1 ? 's' : ''} the registry names` : ''}
        </Fragment>
      ))}
    </>
  )
}
