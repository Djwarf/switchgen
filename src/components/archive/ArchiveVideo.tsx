/**
 * The archive's window onto the player.
 *
 * Clips in the archive play in the house player — scrub rail, frame ticks,
 * frame stepping, loop, save — and never in a bare `<video>`. This file is the
 * one seam between the two, so the archive holds a single import rather than
 * fifty, and it hands the player the record, which is where the frame rate and
 * the frame count honestly come from.
 *
 * The wrapper carries `data-archive-player` so the archive's own single-key
 * shortcuts stand down while focus is inside the player. Five keys mean two
 * things: `k` is play and pause there, the previous record here; `r` is loop
 * there, reuse here; `s` saves the clip there, stars the record here; `u`
 * opens the frame menu there, sends the picture to a desk here; and `Escape`
 * closes whichever is nearer. The thing with focus should win.
 */
import { Player } from '../player/Player'
import type { HistoryEntry } from '../../lib/history'
import { posterUrl } from './Poster'

export function ArchiveVideo({
  entry,
  className = '',
}: {
  entry: HistoryEntry
  className?: string
}) {
  return (
    <div data-archive-player="true" className={className}>
      <Player entry={entry} poster={posterUrl(entry.file) ?? null} keyboard="scoped" />
    </div>
  )
}

export default ArchiveVideo
