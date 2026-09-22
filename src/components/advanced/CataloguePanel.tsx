/**
 * THE CATALOGUE: what this desk could run if it fetched the files.
 *
 * The model panel above lists what is installed. This lists the rest of what
 * this app has a verified graph for, what each is missing, how big that is,
 * and whether the server thinks it will fit. Fetching runs through the same
 * download server the add-on picker uses, and keeps going if the reader
 * leaves this panel: the run lives in a store, not in this component.
 *
 * The catalogue itself knows far more families than this app carries a graph
 * for. Those are not offered, and the panel says how many there are rather
 * than pretending the list is the whole world.
 */
import { useEffect, useMemo, useState } from 'react'
import { useServerCapabilities } from '../../lib/capabilities'
import { bytesText, fetchCatalog, fetchPlan, forgetCatalog, type CatalogFamily, type CatalogPlan } from '../../lib/catalog'
import { cancelPlan, forgetPlan, startPlan, useDownloads } from '../../lib/downloads'
import { FAMILIES, type FamilyDef } from '../../lib/workflows'
import { Meter } from '../loras/bits'
import { Caution, Head, Link, Note } from './bits'

type Row = { def: FamilyDef; cat: CatalogFamily | null }

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
  const [catalog, setCatalog] = useState<{ byId: Map<string, CatalogFamily>; total: number } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [plans, setPlans] = useState<Record<string, CatalogPlan | 'loading' | undefined>>({})

  useEffect(() => {
    if (!caps?.downloads) return
    let live = true
    void fetchCatalog()
      .then((c) => {
        if (!live) return
        setCatalog({ byId: new Map(c.families.map((f) => [f.id, f])), total: c.counts.families })
      })
      .catch((e: unknown) => {
        if (live) setError(e instanceof Error ? e.message : String(e))
      })
    return () => {
      live = false
    }
  }, [caps?.downloads])

  const rows: Row[] = useMemo(() => {
    if (!catalog) return []
    return FAMILIES.filter((d) => modes.includes(d.mode)).map((def) => ({ def, cat: catalog.byId.get(def.id) ?? null }))
  }, [catalog, modes])

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

  const missing = rows.filter((r) => r.cat && !r.cat.installed.ready)
  const installed = rows.filter((r) => r.cat?.installed.ready).length
  const unlisted = rows.filter((r) => !r.cat)

  const finished = (family: string) => {
    forgetCatalog()
    void fetchCatalog(true).then((c) => setCatalog({ byId: new Map(c.families.map((f) => [f.id, f])), total: c.counts.families }))
    setPlans((p) => ({ ...p, [family]: undefined }))
    onInstalled()
  }

  const ask = (row: Row) => {
    const id = row.def.id
    setPlans((p) => ({ ...p, [id]: 'loading' }))
    void fetchPlan(id, row.cat?.installed.chosenModel ?? null)
      .then((plan) => {
        setPlans((p) => ({ ...p, [id]: plan }))
        // Fits, and nothing gated is missing a token: go without another question.
        if (plan.fits && !(plan.gated.files.length && !plan.gated.tokenPresent)) {
          startPlan({ family: id, model: plan.chosenModel }, () => finished(id))
        }
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
  }

  return (
    <section className="mb-7">
      <Head
        title="The catalogue"
        figure={catalog ? `${installed} of ${rows.length} installed` : null}
        note={
          catalog
            ? `${catalog.total - rows.length} more families are in the catalogue with no verified graph in this app, so they are not offered.`
            : 'Reading the catalogue.'
        }
      />
      {error ? <Caution>{error}</Caution> : null}

      {catalog && !missing.length ? (
        <Note>Every family this desk has a graph for is installed.</Note>
      ) : null}

      <ul className="divide-y divide-grey-300">
        {missing.map((row) => {
          const id = row.def.id
          const cat = row.cat!
          const run = runs.get(id)
          const plan = plans[id]
          const files = cat.installed.missing
          const gated = cat.installed.gatedMissing.length > 0
          return (
            <li key={id} className="py-2">
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

      {unlisted.length ? (
        <Note>
          Not in the catalogue, so not fetchable from here: {unlisted.map((r) => r.def.label).join(', ')}.
        </Note>
      ) : null}
    </section>
  )
}
