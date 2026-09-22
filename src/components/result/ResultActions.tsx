/**
 * WHAT NEXT: the strip that prints under a finished picture.
 *
 * This component exists to take controls OFF the compose screen. Everything it
 * offers used to be a checkbox the reader had to answer before there was a
 * picture to answer it about: run a face pass, run a hand pass, run a hires
 * pass. Here the question is answerable, because the picture is on screen. The
 * hands are either wrong or they are not.
 *
 * SO IT MUST NOT READ AS A SECOND CONTROL PANEL.
 *
 * No switches. No sliders. No settings that persist and have to be remembered
 * next time. Every row is one decision the reader makes once, by looking at the
 * picture, and it takes effect immediately. The layout is a printed list: name,
 * a sentence, a figure in the right hand column. A classified column rather
 * than a dashboard, which is also the house style getting what it wants.
 *
 * WHAT IS PRINTED AND WHAT IS NOT.
 *
 * A row appears when the graph that made this picture can carry the pass, and
 * is absent otherwise. There is no disabled state for an unsupported pass, no
 * tooltip explaining why the button is grey. The one exception is the press
 * being busy, which blocks everything at once and is therefore said once, at
 * the top, rather than six times.
 *
 * Every row carries its cost before it runs, because these are GPU minutes on a
 * card the reader is also using for other things. The figures come from
 * offers.tsx, which reads them off the derived graphs.
 */
import { useMemo } from 'react'
import { Caution, Kicker, RING } from '../refine/bits'
import type { DerivedDef } from '../../lib/refine'
import type { ImageFacts } from '../../lib/vision'
import type { FamilyDef } from '../../lib/workflows'
import {
  SHARPNESS_CAVEAT,
  offerFigures,
  offersFor,
  type OfferOptions,
  type ResultActionId,
  type ResultOffer,
} from './offers'

export type ResultActionsProps = {
  /**
   * The finished picture. Null prints nothing at all: this surface is the step
   * after a picture, and an empty version of it would be one more thing on a
   * screen that is being emptied.
   */
  picture: { url: string; width?: number; height?: number } | null
  /**
   * The graph the picture was made with. On a recipe this is `plan.def`, with
   * image to image and the LoRA chain already applied, which is the graph whose
   * capabilities actually govern what can happen next.
   */
  def: FamilyDef | DerivedDef | null
  /** False removes the image to image row. See OfferOptions.canSource. */
  canSource?: boolean
  /** What the detectors found, when the picture has been read. See OfferOptions.facts. */
  facts?: ImageFacts | null
  /** The press is working. Every row is held, and the reason is printed once. */
  busy?: boolean
  /** A reason nothing can run, from the desk. Printed in place of the rows' own copy. */
  blocked?: string | null
  onAction: (id: ResultActionId, offer: ResultOffer) => void
}

export function ResultActions({
  picture,
  def,
  canSource = true,
  facts = null,
  busy = false,
  blocked = null,
  onAction,
}: ResultActionsProps) {
  const width = picture?.width ?? 0
  const height = picture?.height ?? 0

  const url = picture?.url ?? null

  // Each offer derives a graph, and every derivation deep clones one. Cheap in
  // absolute terms and wrong to repeat on every render of a desk this sits
  // inside, so it is keyed on the things that can change the answer: the graph,
  // the picture, and the size the hires row quotes.
  const offers = useMemo(() => {
    if (!url) return []
    const opts: OfferOptions = {
      size: width && height ? { width, height } : null,
      canSource,
      facts,
    }
    return offersFor(def, opts)
  }, [url, def, width, height, canSource, facts])

  if (!picture || !offers.length) return null

  const improve = offers.filter(o => o.group === 'improve')
  const carry = offers.filter(o => o.group === 'carry')
  const held = busy || !!blocked
  const anyMeasured = improve.some(o => o.measured !== null)

  return (
    <section aria-label="What to do with this picture" className="mt-6">
      <div className="mb-3 flex items-baseline justify-between gap-3 border-b-2 border-burgundy-900 pb-1.5">
        <Kicker tone="burgundy">What next</Kicker>
        {improve.length ? (
          <span className="text-caption italic text-grey-500">
            Cost is a multiple of one ordinary generation.
          </span>
        ) : null}
      </div>

      {blocked ? (
        <div className="mb-3">
          <Caution>{blocked}</Caution>
        </div>
      ) : busy ? (
        <p className="mb-3 text-caption italic text-grey-700">
          The press is working. These wait until it is free.
        </p>
      ) : null}

      {improve.length ? (
        <>
          <p className="mb-2 text-caption leading-snug text-grey-700">
            <Kicker className="mr-2">This picture</Kicker>
            None of these overwrite what you have. The picture in front of you stays in the
            archive and the pass arrives as a new one beside it.
          </p>
          <Rows offers={improve} held={held} onAction={onAction} />
        </>
      ) : null}

      {carry.length ? (
        <div className={improve.length ? 'mt-5' : ''}>
          <p className="mb-2 text-caption leading-snug text-grey-700">
            <Kicker className="mr-2">The next one</Kicker>
            Carries this recipe forward. Nothing to fill in again.
          </p>
          <Rows offers={carry} held={held} onAction={onAction} />
        </div>
      ) : null}

      {anyMeasured ? <p className="mt-3 text-caption italic leading-snug text-grey-500">{SHARPNESS_CAVEAT}</p> : null}
    </section>
  )
}

function Rows({
  offers,
  held,
  onAction,
}: {
  offers: ResultOffer[]
  held: boolean
  onAction: (id: ResultActionId, offer: ResultOffer) => void
}) {
  return (
    <ul className="border-t border-grey-300">
      {offers.map(o => (
        <li key={o.id} className="border-b border-grey-300">
          <button
            type="button"
            disabled={held}
            title={o.costNote ?? undefined}
            onClick={() => onAction(o.id, o)}
            className={`sg-tap ${RING} group flex w-full cursor-pointer items-baseline gap-4 py-2 text-left transition-colors hover:bg-newsprint-aged disabled:cursor-not-allowed disabled:opacity-45 disabled:hover:bg-transparent`}
          >
            <span className="min-w-0 flex-1">
              <span className="block text-small font-semibold text-ink group-hover:text-burgundy-900">
                {o.label}
              </span>
              <span className="mt-0.5 block text-caption leading-snug text-grey-700">{o.what}</span>
              {o.measured ? (
                <span className="mt-0.5 block text-caption italic leading-snug text-grey-500">
                  {o.measured}
                </span>
              ) : null}
            </span>
            <span className="shrink-0 text-right">
              {/* A blank cost column is the honest entry for a row that only
                  moves the picture into a slot. It spends nothing until the
                  reader presses the button on the compose screen. */}
              {o.cost === null ? null : (
                <>
                  <span className="block text-small tabular-nums text-ink">
                    {offerFigures.costLabel(o.cost)}
                  </span>
                  <span className="block text-overline font-semibold uppercase tracking-[0.18em] text-grey-500">
                    cost
                  </span>
                </>
              )}
            </span>
          </button>
        </li>
      ))}
    </ul>
  )
}

export default ResultActions
