/**
 * The second answer: what kind of picture it is.
 *
 * Three words, one tab stop. This replaces the style picker, which listed
 * every installed weight file by name and asked the reader to know which of
 * them was trained on booru tags.
 *
 * The options are not written here. They are {@link LOOKS} out of recipe.ts,
 * because that is the module that has to route them, and a list that lives in
 * two places drifts. `cartoon` is deliberately not among them: intent.ts
 * carries it, the More panel reaches it, and it does not earn a chip on a
 * screen with three of them.
 */
import { LOOKS, type Look } from '../../lib/recipe'
import { Choice } from './bits'

export function LookPicker({
  value,
  onChange,
}: {
  value: Look
  onChange: (next: Look) => void
}) {
  return <Choice legend="The look" options={LOOKS} value={value} onChange={onChange} />
}
