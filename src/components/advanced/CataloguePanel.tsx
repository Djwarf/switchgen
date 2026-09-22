/**
 * THE CATALOGUE: what this desk could run if it fetched the files.
 *
 * The model panel above lists what is installed. This lists the rest of what
 * this app has a verified graph for, what each is missing, how big that is,
 * and whether the server thinks it will fit. Fetching runs through the same
 * download server the add-on picker uses, and keeps going if the reader
 * leaves this panel: the run lives in a store, not in this component.
 *
 * Two facts shape the copy. The catalogue and the registry were written
 * apart, so a family is matched to its catalogue entry by the files it loads
 * and not by its name; and the catalogue never lists a community finetune,
 * so a family it has no entry for is named with its file and left to be
 * placed by hand, and is counted installed from the model tree rather than
 * from a catalogue that cannot know.
 */
import { useEffect, useMemo, useState } from 'react'
import { useServerCapabilities } from '../../lib/capabilities'
import { bytesText, fetchCatalog, fetchPlan, forgetCatalog, type CatalogFamily, type CatalogPlan } from '../../lib/catalog'
import { cancelPlan, forgetPlan, startPlan, useDownloads } from '../../lib/downloads'
import { modelFiles } from '../../lib/hardware'
import { FAMILIES, modelsOf, type FamilyDef } from '../../lib/workflows'
import { Meter } from '../loras/bits'
import { Caution, Head, Link, Note } from './bits'

type Row = { def: FamilyDef; cat: CatalogFamily | null; installed: boolean }

const base = (file: string) => file.split('/').pop() ?? file

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
  const [error, setError] = useState<string | null>(null)
  const [plans, setPlans] = useState<Record<string, CatalogPlan | 'loading' | undefined>>({})

  useEffect(() => {
    if (!caps?.downloads) return
    let live = true
    void fetchCatalog()
      .then((c) => {
        if (live) setCatalog({ families: c.families, total: c.counts.families })
      })
      .catch((e: unknown) => {
        if (live) setError(e instanceof Error ? e.message : String(e))
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
    const matchedIds = new Set<string>()
    const all = FAMILIES.map((def) => {
      const cat = matchCatalogue(def, catalog.families)
      if (cat) matchedIds.add(cat.id)
      const installed = onDisk ? def.models.some((m) => onDisk.has(base(m))) : cat?.installed.ready ?? false
      return { def, cat, installed }
    })
    return {
      rows: all.filter((r) => modes.includes(r.def.mode)),
      unmatchedTotal: catalog.total - matchedIds.size,
    }
  }, [catalog, onDisk, modes])

  if (!caps) return null
  if (!caps.downloads) {
    return (
      <section className="mb-7">
        <Head title="The catalogue" />
        <Note>
          aria2c is not on the server, so nothing can be fetched from here. Files placed under the
          models folder by hand are picked up the next time the desk reads what is installed.
        </Note>
      </section>
    )
  }

  const installedCount = rows.filter((r) => r.installed).length
  const fetchable = rows.filter((r) => r.cat && !r.installed)
  const byHand = rows.filter((r) => !r.cat && !r.installed)

  const refresh = () => {
    forgetCatalog()
    void fetchCatalog(true).then((c) => setCatalog({ families: c.families, total: c.counts.families }))
    void modelFiles().then((m) => setOnDisk(new Set([...m.keys()].map(base)))).catch(() => {})
  }

  const finished = (catId: string) => {
    refresh()
    setPlans((p) => ({ ...p, [catId]: undefined }))
    onInstalled()
  }

  const ask = (row: Row) => {
    const cat = row.cat!
    setPlans((p) => ({ ...p, [cat.id]: 'loading' }))
    void fetchPlan(cat.id, cat.installed.chosenModel ?? null)
      .then((plan) => {
        setPlans((p) => ({ ...p, [cat.id]: plan }))
        // Fits, and nothing gated is missing a token: go without another question.
        if (plan.fits && !(plan.gated.files.length && !plan.gated.tokenPresent)) {
          startPlan({ family: cat.id, model: plan.chosenModel }, () => finished(cat.id))
        }
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
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
      {error ? <Caution>{error}</Caution> : null}

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
          return (
            <li key={row.def.id} className="py-2">
              <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
                <span className="text-small text-ink">{row.def.label}</span>
                <span className="text-caption tabular-nums text-grey-500">
                  {files.length} {files.length === 1 ? 'file' : 'files'} · {bytesText(cat.installed.missingBytes)}
                  {gated ? ' · gated' : ''}
                </span>
              </div>
              <p className="mt-0.5 text-caption text-grey-700">
                Needs {files.join(', ')}.
                {cat.incomplete?.length ? ` No verified download exists for ${cat.incomplete.join(', ')}; that file has to be placed by hand.` : ''}
              </p>

              {run && (run.state === 'starting' || run.state === 'running') ? (
                <div className="mt-1">
                  <p className="text-caption italic text-grey-700">
                    {run.current
                      ? `Fetching ${run.current.filename}, ${run.current.index} of ${run.current.count}: ${bytesText(run.current.done)} of ${bytesText(run.current.total)}${run.current.etaSec ? `, about ${Math.ceil(run.current.etaSec / 60)} min left` : ''}.`
                      : 'Starting.'}{' '}
                    <Link onClick={() => void cancelPlan(id)}>Stop</Link>
                  </p>
                  <Meter pct={run.current?.pct ?? 0} label={`Fetching ${row.def.label}`} />
                </div>
              ) : run && run.state === 'error' ? (
                <p className="mt-1 text-caption text-ink-error">
                  {run.error} <Link onClick={() => forgetPlan(id)}>Dismiss</Link>
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
                  {plan.gated.files.length && !plan.gated.tokenPresent ? (
                    <p className="text-caption text-grey-700">
                      {plan.gated.files.join(', ')} {plan.gated.files.length === 1 ? 'is' : 'are'} gated on HuggingFace and no
                      token is on the server, so the fetch would be refused.
                    </p>
                  ) : null}
                  {plan.incomplete.length ? null : (
                    <p className="text-caption">
                      <Link
                        onClick={() => {
                          if (window.confirm(`The server says this will not fit. Fetch ${row.def.label} anyway?`)) {
                            startPlan({ family: id, model: plan.chosenModel, force: true }, () => finished(id))
                          }
                        }}
                      >
                        Fetch anyway
                      </Link>
                    </p>
                  )}
                </div>
              ) : (
                <p className="mt-1 text-caption">
                  {cat.incomplete?.length ? null : (
                    <Link onClick={() => ask(row)}>
                      Fetch {files.length === 1 ? 'the file' : `${files.length} files`} ({bytesText(cat.installed.missingBytes)})
                    </Link>
                  )}
                </p>
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
                {r.def.label}: <code>{r.def.models[0]}</code>
                {r.def.models.length > 1 ? ` or ${r.def.models.length - 1} other file${r.def.models.length > 2 ? 's' : ''} the registry names` : ''}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </section>
  )
}
