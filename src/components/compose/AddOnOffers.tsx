/**
 * Add-ons the wording matched, offered rather than applied.
 *
 * These used to be pushed straight into the chain: you asked for a picture and
 * silently got two extra add-ons plus their trigger words in your prompt. The
 * app decided, printed what it had decided, and gave you no moment to disagree
 * before the press. This is that moment.
 *
 * Nothing here is jargon. An add-on is a thing that changes how the picture
 * comes out; "Add" applies it; "Not this time" means stop offering it for this
 * composition. Both decisions live in the composition, so neither is undone by
 * the next keystroke.
 *
 * Printed only when there is something to offer, so a desk with no matches
 * never mentions add-ons at all.
 */
import type { RecipeLora } from '../../lib/recipe'
import { EditorialLink, Kicker } from './bits'

export function AddOnOffers({
  offers,
  onAccept,
  onDecline,
}: {
  offers: readonly RecipeLora[]
  onAccept: (file: string) => void
  onDecline: (file: string) => void
}) {
  if (!offers.length) return null

  return (
    <section aria-label="Suggested add-ons">
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
        Matched against what each add-on was trained on, not measured. Nothing is applied until you
        add it.
      </p>
    </section>
  )
}
