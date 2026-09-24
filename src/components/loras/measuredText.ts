/**
 * The line an add-on row prints under "measured here". Kept in a file of its
 * own, apart from the components, so it can be read without rendering one.
 */
import { ratioText, type MeasuredPoint } from './measured'

/**
 * The measured points as the row prints them: `0.81x as sharp as no add-ons
 * at 0.3, 0.72x at 0.5`. Every single here was run on one held prompt with
 * no add-on's word in it, so for an add-on that takes a word the line says
 * so: add-micro-details measured 1.625x without its word and 2.132x with it,
 * and the offer beside the row quotes the second.
 */
export function measuredText(points: readonly MeasuredPoint[], takesWord: boolean): string {
  const text = points
    .map((p, i) => `${ratioText(p.ratio)}${i ? '' : ' as sharp as no add-ons'} at ${p.strength.toFixed(1)}`)
    .join(', ')
  return takesWord ? `${text}, without its word` : text
}
