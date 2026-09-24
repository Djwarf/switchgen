/**
 * Work the queue on the server holds until the reader says.
 *
 * The server holds its waiting work when sending it the same way could go
 * wrong the same way again: a heavy job was lost, or may never have reached
 * ComfyUI, which usually means ComfyUI restarted, as it does when memory runs
 * out; or the machine restarted, or the queue was off, while the work waited.
 * The hold is one for the whole server, whichever desk or device the work
 * came from, and so is the word that ends it. A desk shows the hold over its
 * own work, but a reader standing at another room, or holding work from a
 * desk with no word of its own, had nowhere to give it. So the shell says it
 * on every room, as long as the hold stands, with the two answers there are:
 * send the work, or call it off.
 *
 * The word names the hold it answers (its since), so a notice left standing
 * on a device that slept cannot answer a newer hold it never showed: the
 * server refuses it, the state is read again, and the hold that stands now is
 * shown with its own reason and count, and a line saying the word was for the
 * one before.
 *
 * What it counts is what the server holds: every waiting job the hold covers,
 * which is every waiting job for a hold over all the work, and every heavy one
 * for a hold over heavy work, whatever else its wait says (in a batch or a
 * pass only the first job waiting says "held"; the rest wait for the one
 * before them, and are held all the same). The model lab's pictures (lab/ in
 * the repo) are counted too, by name: no desk of the app shows them, but the
 * hold covers them, and so does the word given here.
 */
import { useState } from 'react'
import { holdCovers, laneWord, runnerStore, useRunner, type RunnerDesk, type RunnerSnapshot } from '../../lib/runner'

/**
 * How many waiting jobs a hold covers, desk by desk. `lab` is there only
 * when the hold covers some of the model lab's pictures: the server sends
 * them through the same lane, though no desk of the app shows them.
 */
export type HeldCounts = Record<RunnerDesk, number> & { lab?: number }

/**
 * The waiting jobs the lane's hold covers, by desk, counted as the server's
 * engine counts them (covers() in server/runner/engine.mjs, which holdCovers
 * repeats for the page). All nought when nothing is held.
 */
// oxlint-disable-next-line react/only-export-components -- the count is shared with its tests and any desk that shows the same hold
export function heldCounts(snap: Pick<RunnerSnapshot, 'lane' | 'jobs'>): HeldCounts {
  const counts: HeldCounts = { images: 0, video: 0, reel: 0 }
  const held = snap.lane.held
  if (!held) return counts
  for (const j of snap.jobs) {
    if (j.status !== 'waiting' || !holdCovers(held, j)) continue
    // The page's type names the app's three desks; the server's list holds the lab's too.
    const desk: string = j.desk
    if (desk === 'lab') counts.lab = (counts.lab ?? 0) + 1
    else if (desk in counts) counts[j.desk]++
  }
  return counts
}

/** Why the work is held, in one plain sentence, by the hold's reason. */
const WHY: Record<string, string> = {
  lost: 'A heavy job was lost, which usually means ComfyUI restarted, so the server holds the work waiting behind it until you say.',
  unsent:
    'A heavy job may never have reached ComfyUI, which usually means ComfyUI restarted, so the server holds the work waiting behind it until you say.',
  restart: 'The machine restarted while this work waited, so the server holds it until you say.',
  paused: 'The queue on the server was off while this work waited, so the server holds it until you say.',
}
const WHY_ANY = 'The server holds its waiting work until you say.'

/** What each desk's work is called, one and many, in the order the rooms go, the lab's last. */
const NOUNS: readonly (readonly [keyof HeldCounts, string, string])[] = [
  ['images', 'picture', 'pictures'],
  ['video', 'clip', 'clips'],
  ['reel', 'shot', 'shots'],
  ['lab', 'lab picture', 'lab pictures'],
]

/** "10 pictures, 2 clips and 1 lab picture", or null for none. */
function countLine(counts: HeldCounts): string | null {
  const parts = NOUNS.flatMap(([desk, one, many]) => {
    const n = counts[desk] ?? 0
    return n > 0 ? [`${n} ${n === 1 ? one : many}`] : []
  })
  if (!parts.length) return null
  return parts.length === 1 ? parts[0] : `${parts.slice(0, -1).join(', ')} and ${parts[parts.length - 1]}`
}

/** What came of a word to the hold. */
export type HoldAnswer =
  /** The server took it; what it changed comes on the stream. */
  | { is: 'taken' }
  /** Refused, and no hold stands now (another device answered it, say). */
  | { is: 'gone' }
  /** Refused, and another hold stands now: the word was for the one before. */
  | { is: 'moved'; since: number }
  /** Refused, or not answered, and the same hold still stands. */
  | { is: 'refused' }

/**
 * Give the word for the hold whose since is `since`. The server refuses a
 * word for a hold that is not the one standing, so a notice left on a device
 * that slept cannot answer a newer hold it never showed. A refused word may
 * also have come after another device answered, so the state is read again
 * before anything is said of it, and the answer says which of those it was.
 *
 * The read is one that leaves after the refusal. A read already out may have
 * left before the hold changed: taken as the answer, it showed the old hold,
 * and the notice said the word did not get through when it was for the hold
 * before.
 */
// oxlint-disable-next-line react/only-export-components -- the notice's one word, shared with its tests
export async function answerHold(action: 'send' | 'stop', since: number): Promise<HoldAnswer> {
  const ok = await laneWord(action, since).catch(() => false)
  if (ok) return { is: 'taken' }
  await runnerStore.refresh({ fresh: true }).catch(() => {})
  const now = runnerStore.snapshot().lane.held
  if (!now) return { is: 'gone' }
  return now.since !== since ? { is: 'moved', since: now.since } : { is: 'refused' }
}

/** A control a thumb can find: 44 px on a coarse pointer. */
const CONTROL = 'sg-link ring inline-flex items-center [@media(pointer:coarse)]:min-h-11 [@media(pointer:coarse)]:min-w-11'

export function RunnerHold() {
  const snap = useRunner()
  const held = snap.lane.held
  // The hold that stood when a word given for the one before it was refused:
  // the notice for it says so, and no other does.
  const [movedTo, setMovedTo] = useState<number | null>(null)
  if (!held) return null
  // Keyed by the hold, so a word that failed for one hold is not said of the next.
  return (
    <HeldBand
      key={`${held.why}:${held.since}`}
      why={held.why}
      since={held.since}
      counts={heldCounts(snap)}
      available={snap.available}
      moved={movedTo === held.since}
      onMoved={setMovedTo}
    />
  )
}

type BandProps = {
  why: string
  /** The since of the hold this notice shows, which the word names. */
  since: number
  counts: HeldCounts
  available: boolean
  /** A word given on the notice for the hold before this one was refused. */
  moved: boolean
  /** Tells the shell which hold stands after a refused word (null: none to tell of). */
  onMoved: (since: number | null) => void
}

function HeldBand({ why, since, counts, available, moved, onMoved }: BandProps) {
  const [asking, setAsking] = useState<'send' | 'stop' | null>(null)
  const [refused, setRefused] = useState(false)
  const count = countLine(counts)

  const say = (action: 'send' | 'stop') => {
    if (asking) return
    setAsking(action)
    setRefused(false)
    if (moved) onMoved(null)
    // The buttons stay busy until the answer is known, a refused word's read
    // included, so the same word cannot go to whatever hold comes next.
    void answerHold(action, since).then((a) => {
      // Another hold stands: this notice goes (it is keyed by its hold), and
      // the one for the new hold says the word was for the one before.
      if (a.is === 'moved') onMoved(a.since)
      // What a word that was taken changed comes on the stream, and a hold
      // that is gone takes this notice with it.
      if (a.is === 'refused') setRefused(true)
      setAsking(null)
    })
  }

  return (
    <section aria-label="Work held on the server" className="border-b border-grey-300 bg-newsprint px-6 py-3">
      <div className="mx-auto w-full max-w-[110rem]">
        <div className="notice notice-warning text-small" role="status">
          <p className="m-0">
            <strong className="block not-italic">Work held on the server</strong>
            {WHY[why] ?? WHY_ANY}
            {count ? ` Waiting: ${count}.` : ''}
          </p>
          {moved ? (
            <p className="mt-2 mb-0">
              The hold changed while you answered. Your word was for the one before, so this one still waits for yours.
            </p>
          ) : null}
          {/* Said, not enforced: the page's copy of the queue can be behind
              the server's, and this is the one place the word can be given. */}
          {!available ? (
            <p className="mt-2 mb-0 italic text-grey-700">
              The queue on the server is not running just now, so it cannot take your word until it is back.
            </p>
          ) : null}
          {refused ? <p className="mt-2 mb-0">Your word did not get through. Try again in a moment.</p> : null}
          <p className="mt-2 mb-0 flex flex-wrap items-baseline gap-x-4 gap-y-1 not-italic">
            <button
              type="button"
              className={CONTROL}
              disabled={asking !== null}
              aria-busy={asking === 'send'}
              onClick={() => say('send')}
            >
              Send them
            </button>
            <button
              type="button"
              className={CONTROL}
              disabled={asking !== null}
              aria-busy={asking === 'stop'}
              onClick={() => say('stop')}
            >
              Call them off
            </button>
          </p>
        </div>
      </div>
    </section>
  )
}
