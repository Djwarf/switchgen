/**
 * Add-ons the wording matched, offered rather than applied, and the ones the
 * reader has added, with a way to take each off again.
 *
 * These used to be pushed straight into the chain: you asked for a picture and
 * silently got two extra add-ons plus their trigger words in your prompt. The
 * app decided, printed what it had decided, and gave you no moment to disagree
 * before the press. This is that moment.
 *
 * An added one stays on after the wording stops matching it, because a
 * decision should not evaporate on the next keystroke. That is exactly why it
 * is listed here with "Take off": it leaves the offers as it joins the chain,
 * and before this list the main screen had no way to remove it at all.
 *
 * Nothing here is jargon. An add-on is a thing that changes how the picture
 * comes out; "Add" applies it; "Not this time" means stop offering it for this
 * composition; "Take off" withdraws an add. The decisions live in the
 * composition, so none of them is undone by the next keystroke, and a new
 * composition starts without them.
 *
 * Printed only when there is something to offer or something added, so a desk
 * with neither never mentions add-ons at all.
 */
import type { RecipeLora } from '../../lib/recipe'
import { EditorialLink, Kicker } from './bits'

export function AddOnOffers({
  offers,
  applied = [],
  onAccept,
  onDecline,
  onRemove,
}: {
  offers: readonly RecipeLora[]
  /** Add-ons the reader added that are in the chain now. */
  applied?: readonly RecipeLora[]
  onAccept: (file: string) => void
  onDecline: (file: string) => void
  /** Take an added one off. Omit it and the added ones are not listed. */
  onRemove?: (file: string) => void
}) {
  const added = onRemove ? applied : []
  if (!offers.length && !added.length) return null

  return (
    <section aria-label="Add-ons">
      {added.length ? (
        <div className={offers.length ? 'mb-4' : ''}>
          <Kicker>Added by you</Kicker>
          <ul className="mt-2 space-y-2">
            {added.map((a) => (
              <li
                key={a.file}
                className="flex flex-wrap items-baseline gap-x-3 gap-y-1 border-l-2 border-burgundy-900 pl-3"
              >
                <span className="text-body text-grey-900">{a.label}</span>
                <span className="min-w-0 flex-1 text-caption text-grey-500">
                  Stays on whatever the wording, until you take it off or clear the prompt.
                </span>
                <span className="flex shrink-0 items-baseline gap-3">
                  <EditorialLink onClick={() => onRemove?.(a.file)}>Take off</EditorialLink>
                </span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {offers.length ? (
        <>
          <Kicker>Might suit this</Kicker>
          <ul className="mt-2 space-y-2">
            {offers.map((o) => (
              <li
                key={o.file}
                className="flex flex-wrap items-baseline gap-x-3 gap-y-1 border-l-2 border-grey-200 pl-3"
              >
                <span className="text-body text-grey-900">{o.label}</span>
                <span className="min-w-0 flex-1 text-caption text-grey-500">{o.why}</span>
                <span className="flex shrink-0 items-baseline gap-3">
                  <EditorialLink onClick={() => onAccept(o.file)}>Add</EditorialLink>
                  <EditorialLink onClick={() => onDecline(o.file)}>Not this time</EditorialLink>
                </span>
              </li>
            ))}
          </ul>
          <p className="mt-2 max-w-[62ch] text-caption text-grey-500">
            Matched against what each add-on was trained on, not measured. Nothing is applied until
            you add it.
          </p>
        </>
      ) : null}
    </section>
  )
}
