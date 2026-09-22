/**
 * What the desk decided, printed in prose under the button.
 *
 * This paragraph is the whole trade. The old screen kept the reader honest by
 * making them set everything: the model, the LoRAs, the strengths, the passes.
 * This one keeps the app honest instead. Every decision recipe.ts made is
 * readable here without touching a control, so a reader who disagrees knows
 * exactly what to go and change behind More, and a reader who does not can
 * stop reading and press the button.
 *
 * The sentences are {@link explain} verbatim. Nothing is composed in this file
 * and no number is formatted here, because a number formatted twice is a
 * number that can disagree with itself.
 *
 * Warnings are printed, never folded away. A thing that is true and unwelcome
 * belongs on the page: the stack that could not attach, the LoRA that is not
 * downloaded, the level that measured below base.
 */
import { MEASURED, explain, type Recipe } from '../../lib/recipe'
import { Hairline, Kicker, Notice } from './bits'

export function RecipeProse({ recipe }: { recipe: Recipe }) {
  const text = explain(recipe)

  return (
    <div className="mt-6">
      <Hairline className="mb-3" />
      <Kicker>{recipe.ok ? 'What the desk chose' : 'Nothing to run'}</Kicker>

      {recipe.ok ? (
        <p className="mt-2 max-w-[62ch] text-body leading-relaxed text-ink">{text}</p>
      ) : (
        <div className="mt-2 max-w-[62ch]">
          <Notice tone="correction" title="Correction">
            <p>{text}</p>
          </Notice>
        </div>
      )}

      {recipe.ok && recipe.sharpness && (
        <p className="mt-2 max-w-[62ch] text-caption text-grey-500">{MEASURED.metricCaveat}</p>
      )}

      {recipe.warnings.length > 0 && (
        <div className="mt-3 max-w-[62ch] space-y-2">
          {recipe.warnings.map((w, i) => (
            <Notice key={i} tone="warning" title="Note">
              <p>{w}</p>
            </Notice>
          ))}
        </div>
      )}
    </div>
  )
}
