/**
 * Pairs: two whole grids side by side (Flip between them), one question.
 *
 * - Tie-break pairs are made during judging, once a set's grids are all
 *   scored, among the grids that share the top step. They are deterministic:
 *   the same set and the same tied grids always give the same pairs, whatever
 *   order the grids are listed in, and there are never more than six.
 * - Chain, sweep, sampler and prompt-style pairs are fixed at sealing, with
 *   their sides drawn at random there (see seal.ts).
 */
import { createHash } from 'node:crypto'

export const MAX_TIE_PAIRS = 6

export const PAIR_QUESTIONS = {
  /** Tie-break and chain pairs. */
  block: 'Which would you rather have, for what this block asks?',
  /** Step sweep and sampler pairs. */
  overall: 'Which is the better picture overall?',
  /** Sentence prompt against tag prompt. */
  brief: 'Which follows the brief better?',
} as const

export type PairKind = 'tie' | 'chain' | 'sweep' | 'sampler' | 'promptstyle'

/** A small deterministic generator (mulberry32) seeded from the sha256 of a text. */
export function seededRandom(seedText: string): () => number {
  let a = createHash('sha256').update(seedText).digest().readUInt32LE(0)
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Fisher-Yates with the given generator; returns a new array. */
export function shuffled<T>(xs: readonly T[], rnd: () => number): T[] {
  const out = xs.slice()
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1))
    ;[out[i], out[j]] = [out[j], out[i]]
  }
  return out
}

/**
 * The tie-break pairs for the grids sharing a set's top step.
 * 2 grids give 1 pair, 3 give all 3, 4 give 4 in a cycle, 5 or more give
 * neighbours in a cycle, at most 6. The order and the sides of each pair are
 * drawn from a generator seeded by the set id, over the sorted ids, so the
 * answer does not depend on the order the ids arrive in.
 */
export function tieBreakPairs(setId: string, topItemIds: readonly string[]): [string, string][] {
  const ids = [...new Set(topItemIds)].sort()
  if (ids.length < 2) return []
  const rnd = seededRandom(`tie:${setId}:${ids.join(',')}`)
  const x = shuffled(ids, rnd)
  let pairs: [string, string][]
  if (x.length === 2) pairs = [[x[0], x[1]]]
  else if (x.length === 3) pairs = [[x[0], x[1]], [x[1], x[2]], [x[2], x[0]]]
  else {
    pairs = []
    for (let i = 0; i < x.length && pairs.length < MAX_TIE_PAIRS; i++) pairs.push([x[i], x[(i + 1) % x.length]])
  }
  return pairs.map(([a, b]) => (rnd() < 0.5 ? [a, b] : [b, a]) as [string, string])
}

/** The id of the k-th tie pair of a set (0-based). */
export const tieItemId = (setId: string, k: number): string => `${setId}:tie${k}`
/** The id of a set's "your pick" item. */
export const pickItemId = (setId: string): string => `${setId}:pick`

/**
 * What a pair answer means for one side. `winner` is 1 (the first grid is
 * better), 2 (the second) or 0 (can't tell).
 */
export function outcomeFor(winner: 0 | 1 | 2, side: 'a' | 'b'): 'won' | 'lost' | 'tied' {
  if (winner === 0) return 'tied'
  return (winner === 1) === (side === 'a') ? 'won' : 'lost'
}
