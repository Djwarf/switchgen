/**
 * WHAT NEXT, for a clip.
 *
 * The picture list offers quality passes with measured figures. Nothing of
 * the kind has been measured on a clip, so this list offers only what carries
 * the work forward, and says so once at the top rather than pretending.
 */
import type { HistoryEntry } from '../../lib/history'
import type { ResultOffer } from './offers'

export function videoOffersFor(entry: HistoryEntry | null): ResultOffer[] {
  const out: ResultOffer[] = [
    {
      id: 'continue',
      label: 'Continue from the last frame',
      what: 'Lifts the final frame out of this clip and puts it in the start-frame slot, so the next clip opens where this one closed.',
      measured: null,
      cost: null,
      costNote: null,
      group: 'carry',
    },
  ]
  if (entry) {
    out.push(
      {
        id: 'again',
        label: 'Make another like this',
        what: 'The same settings, a new seed, straight to the press.',
        measured: null,
        cost: 1,
        costNote: 'One ordinary clip.',
        group: 'carry',
      },
      {
        id: 'settings',
        label: 'Use these settings',
        what: 'Puts everything that made this clip back on the desk, add-ons included, and waits for you.',
        measured: null,
        cost: null,
        costNote: null,
        group: 'carry',
      },
    )
  }
  out.push(
    {
      id: 'toPictures',
      label: 'Send the last frame to the Pictures desk',
      what: 'Lifts the final frame and opens the Pictures desk on it, to work from it as a still.',
      measured: null,
      cost: null,
      costNote: null,
      group: 'carry',
    },
    {
      id: 'save',
      label: 'Save the clip',
      what: 'Downloads the file as it is, named after the family and the seed.',
      measured: null,
      cost: null,
      costNote: null,
      group: 'carry',
    },
  )
  return out
}
