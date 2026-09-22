/**
 * The third answer: how much anatomy.
 *
 * This is the control that replaces the rack. The old screen offered a LoRA
 * list, a strength slider per row, an enable box per row, a pass selector and
 * two detail-pass checkboxes, and asked the reader to know that anatomy-helper
 * falls apart above 0.4. All of that still exists, resolved by recipe.ts from
 * the measured table and reachable in full behind More. Here it is three
 * words.
 *
 * Every figure on this control is measured. The option blurbs are
 * {@link ANATOMY_LEVELS} verbatim and the emphasised verdict is
 * {@link MEASURED} verbatim: nothing is written in this file that a run did
 * not produce. The caveat that Laplacian variance is sharpness and not
 * anatomical correctness is printed once per page, under the button, beside
 * the ratio the chosen stack actually measured.
 *
 * Emphasised is printed with its verdict attached rather than as a better
 * setting, because it is not one: three anatomy LoRAs measured below the base
 * model on the same seed. A picker that implied more is better would be lying
 * about the only numbers we have.
 */
import { ANATOMY_LEVELS, MEASURED, type AnatomyLevel } from '../../lib/recipe'
import { Choice } from './bits'

export function AnatomyPicker({
  value,
  onChange,
}: {
  value: AnatomyLevel
  onChange: (next: AnatomyLevel) => void
}) {
  return (
    <Choice
      legend="Anatomy"
      options={ANATOMY_LEVELS}
      value={value}
      onChange={onChange}
      footnote={value === 'emphasised' ? MEASURED.stacks.emphasised.verdict : null}
    />
  )
}
