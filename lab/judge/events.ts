/**
 * The judging log: runs/<run>/judging/events.jsonl.
 *
 * Every tap on the phone becomes one event with an id made on the phone, so an
 * outbox can resend without doubling anything. The file is append-only; the
 * state is always recomputed from it by {@link reduceEvents}. Nothing here knows
 * which model made what: events name blind item ids only.
 */
import { closeSync, existsSync, fstatSync, fsyncSync, mkdirSync, openSync, readFileSync, readSync, statSync, writeSync } from 'node:fs'
import { join } from 'node:path'

export const EVENT_KINDS = ['open-set', 'score', 'pair', 'pick', 'content', 'skip', 'undo', 'note', 'reveal', 'pairs-made'] as const
export type EventKind = (typeof EVENT_KINDS)[number]
/** Kinds only the lab server writes; the phone may not send them. */
export const SERVER_KINDS: readonly EventKind[] = ['reveal', 'pairs-made']
/** Kinds that answer an item. The latest one of these decides the item's state. */
export const ANSWER_KINDS: readonly EventKind[] = ['score', 'pair', 'pick', 'content', 'skip']

export type ScoreValue = {
  step: 1 | 2 | 3 | 4 | 5
  /** Tile (or row) positions marked Failed, 0-based. */
  fail: number[]
  /** The single tile (or row) marked Best, or null. */
  best: number | null
  chips: string[]
  /** "I think I know this model". */
  recognised: boolean
  /** The second card, when the set has one (va, sh or ly). */
  second?: { card: string; step: 1 | 2 | 3 | 4 | 5 }
  /** Description chips on the defaults card. */
  defaults?: { look?: string | null; who?: string | null; framing?: string | null; extra?: string[] }
}
export type PairValue = { winner: 0 | 1 | 2 }
export type PickValue = { letter: string }
export type ContentValue = { agree: 'yes' | 'no' | 'unsure' }
export type OpenSetValue = { lookThrough: boolean }
export type UndoValue = { target: string }
export type NoteValue = { text: string }
export type PairsMadeValue = { pairs: [string, string][] }

export type JudgeEvent = {
  v: 1
  id: string
  /** Milliseconds since the epoch (an ISO string is accepted and read as such). */
  at: number | string
  judge: string
  session: string
  device: { w: number; h: number; dpr: number }
  run: string
  /**
   * What the event is about: a grid, pair or check item id for most kinds;
   * the set id for open-set and pairs-made; '<set>:pick' for a pick and
   * '<set>:tie<k>' for a tie-break pair (pairs.ts pickItemId, tieItemId);
   * the study id for reveal.
   */
  item: string
  card?: string
  kind: EventKind
  value?: unknown
  dwellMs: number
  afterReveal?: boolean
}

export const EVENTS_FILE = 'events.jsonl'
const MAX_VALUE_BYTES = 8 * 1024
const ID_RE = /^[A-Za-z0-9._:~-]{1,80}$/
const NAME_RE = /^[A-Za-z0-9._:@~ -]{1,80}$/

export function atOf(e: Pick<JudgeEvent, 'at'>): number {
  if (typeof e.at === 'number') return e.at
  const t = Date.parse(e.at)
  return Number.isFinite(t) ? t : 0
}

const isStep = (n: unknown): n is 1 | 2 | 3 | 4 | 5 => n === 1 || n === 2 || n === 3 || n === 4 || n === 5
const isObj = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v)

/** Why an event is malformed, or null when it is fine. The value is checked per kind. */
export function eventProblem(e: unknown, opts: { allowServerKinds?: boolean } = {}): string | null {
  if (!isObj(e)) return 'not an object'
  if (e.v !== 1) return 'v must be 1'
  if (typeof e.id !== 'string' || !ID_RE.test(e.id)) return 'bad id'
  if (!(typeof e.at === 'number' && Number.isFinite(e.at)) && !(typeof e.at === 'string' && Number.isFinite(Date.parse(e.at)))) return 'bad at'
  if (typeof e.judge !== 'string' || !NAME_RE.test(e.judge)) return 'bad judge'
  if (typeof e.session !== 'string' || !NAME_RE.test(e.session)) return 'bad session'
  if (typeof e.run !== 'string' || !NAME_RE.test(e.run)) return 'bad run'
  if (typeof e.item !== 'string' || !ID_RE.test(e.item)) return 'bad item'
  if (e.card !== undefined && (typeof e.card !== 'string' || e.card.length > 16)) return 'bad card'
  if (typeof e.kind !== 'string' || !(EVENT_KINDS as readonly string[]).includes(e.kind)) return 'bad kind'
  if (!opts.allowServerKinds && SERVER_KINDS.includes(e.kind as EventKind)) return `the phone may not send ${e.kind}`
  if (typeof e.dwellMs !== 'number' || !Number.isFinite(e.dwellMs) || e.dwellMs < 0) return 'bad dwellMs'
  const d = e.device
  if (!isObj(d) || typeof d.w !== 'number' || typeof d.h !== 'number' || typeof d.dpr !== 'number') return 'bad device'
  if (e.value !== undefined && JSON.stringify(e.value).length > MAX_VALUE_BYTES) return 'value too large'
  const v = e.value
  switch (e.kind as EventKind) {
    case 'score': {
      if (!isObj(v) || !isStep(v.step)) return 'score needs a step from 1 to 5'
      if (!Array.isArray(v.fail) || !v.fail.every(n => Number.isInteger(n) && (n as number) >= 0 && (n as number) < 8)) return 'bad fail list'
      if (!(v.best === null || (Number.isInteger(v.best) && (v.best as number) >= 0 && (v.best as number) < 8))) return 'bad best'
      if (!Array.isArray(v.chips) || !v.chips.every(c => typeof c === 'string' && c.length <= 40)) return 'bad chips'
      if (typeof v.recognised !== 'boolean') return 'bad recognised'
      if (v.second !== undefined && (!isObj(v.second) || typeof v.second.card !== 'string' || !isStep(v.second.step))) return 'bad second card'
      return null
    }
    case 'pair':
      return isObj(v) && (v.winner === 0 || v.winner === 1 || v.winner === 2) ? null : 'pair needs winner 0, 1 or 2'
    case 'pick':
      return isObj(v) && typeof v.letter === 'string' && /^[A-Z]{1,2}$/.test(v.letter) ? null : 'pick needs a letter'
    case 'content':
      return isObj(v) && (v.agree === 'yes' || v.agree === 'no' || v.agree === 'unsure') ? null : 'content needs yes, no or unsure'
    case 'undo':
      return isObj(v) && typeof v.target === 'string' && ID_RE.test(v.target) ? null : 'undo needs a target id'
    case 'open-set':
      return isObj(v) && typeof v.lookThrough === 'boolean' ? null : 'open-set needs lookThrough'
    case 'note':
      return isObj(v) && typeof v.text === 'string' && v.text.length <= 2000 ? null : 'note needs text'
    case 'pairs-made':
      return isObj(v) && Array.isArray(v.pairs) && v.pairs.every(p => Array.isArray(p) && p.length === 2 && p.every(x => typeof x === 'string')) ? null : 'bad pairs'
    default:
      return null
  }
}

/** Parse a jsonl file, skipping a torn last line (a crash mid-append) and any line that is not an object. */
export function readJsonl<T>(file: string): T[] {
  if (!existsSync(file)) return []
  const out: T[] = []
  for (const line of readFileSync(file, 'utf8').split('\n')) {
    if (!line.trim()) continue
    try {
      const v = JSON.parse(line)
      if (v && typeof v === 'object') out.push(v as T)
    } catch { /* a torn line from a crash: ignore it */ }
  }
  return out
}

export function readEvents(dir: string): JudgeEvent[] {
  return readJsonl<JudgeEvent>(join(dir, EVENTS_FILE))
}

/** Known ids per events file, checked against the file size so another writer is noticed. */
const known = new Map<string, { size: number; ids: Set<string> }>()

function idsOf(file: string): Set<string> {
  const size = existsSync(file) ? statSync(file).size : 0
  const hit = known.get(file)
  if (hit && hit.size === size) return hit.ids
  const ids = new Set(readJsonl<JudgeEvent>(file).map(e => e.id))
  known.set(file, { size, ids })
  return ids
}

export type AppendResult = { accepted: number; duplicate: number; rejected: { index: number; error: string }[] }

/**
 * Append events to <dir>/events.jsonl, dropping any whose id is already there.
 * The phone's afterReveal field is never trusted: the server says whether the
 * study is revealed, and every event it appends after that is stamped.
 * Writes are synced before returning, so an accepted event survives a crash.
 */
export function appendEvents(
  dir: string,
  events: unknown[],
  opts: { afterReveal?: boolean; allowServerKinds?: boolean } = {},
): AppendResult {
  mkdirSync(dir, { recursive: true })
  const file = join(dir, EVENTS_FILE)
  const ids = idsOf(file)
  const res: AppendResult = { accepted: 0, duplicate: 0, rejected: [] }
  const lines: string[] = []
  const fresh: string[] = []
  events.forEach((raw, index) => {
    const why = eventProblem(raw, opts)
    if (why) { res.rejected.push({ index, error: why }); return }
    const e = { ...(raw as JudgeEvent) }
    delete e.afterReveal
    if (opts.afterReveal) e.afterReveal = true
    if (ids.has(e.id) || fresh.includes(e.id)) { res.duplicate++; return }
    fresh.push(e.id)
    lines.push(JSON.stringify(e) + '\n')
    res.accepted++
  })
  if (lines.length) {
    const fd = openSync(file, 'a+', 0o600)
    try {
      // A line left torn by a crash mid-write is closed off first, so the next event starts clean.
      const size = fstatSync(fd).size
      let lead = ''
      if (size > 0) {
        const last = Buffer.alloc(1)
        readSync(fd, last, 0, 1, size - 1)
        if (last[0] !== 0x0a) lead = '\n'
      }
      writeSync(fd, lead + lines.join(''))
      fsyncSync(fd)
    } finally {
      closeSync(fd)
    }
    for (const id of fresh) ids.add(id)
    known.set(file, { size: statSync(file).size, ids })
  }
  return res
}

/** Append one server event (reveal, pairs-made) with a server-made id. */
export function appendServerEvent(dir: string, e: JudgeEvent): AppendResult {
  return appendEvents(dir, [e], { allowServerKinds: true, afterReveal: e.afterReveal })
}

/** Keep a stray non-append writer out: tests and tools may call this to drop the id cache. */
export function forgetEventCache(): void {
  known.clear()
}

export type Answer = {
  kind: EventKind
  event: JudgeEvent
  at: number
  afterReveal: boolean
}

export type JudgeState = {
  /** Every distinct event (duplicates by id dropped), in log order, with afterReveal worked out. */
  events: JudgeEvent[]
  /** Event ids that an undo cancelled. */
  undone: Set<string>
  /** `${judge}\u0000${item}\u0000${kind}` → the latest event of that kind that was not undone. */
  latest: Map<string, JudgeEvent>
  /** judge → item → the latest answer (score, pair, pick, content or skip) that was not undone. */
  answers: Map<string, Map<string, Answer>>
  /** set id → the tie pairs fixed by the server's pairs-made event. */
  pairsMade: Map<string, [string, string][]>
  /** judge → set id → when the set was first opened. */
  opened: Map<string, Map<string, number>>
  /** When the first reveal event was logged, or null. */
  revealedAt: number | null
  sessions: Set<string>
  duplicates: number
}

export const keyOf = (judge: string, item: string, kind: string): string => `${judge}\u0000${item}\u0000${kind}`

/**
 * Fold the log into the judging state.
 * - Ids are deduplicated (the first copy wins).
 * - An undo cancels its target, whatever order they arrive in.
 * - Per (judge, item, kind) the latest event that was not undone wins; latest
 *   means the later `at`, and the later line when two share an `at`.
 * - An item's answer is the latest of its answering kinds, so a skip after a
 *   score leaves the item skipped, and skipping twice is still one skip.
 * - Every event logged after its own run's reveal event, or stamped
 *   afterReveal, is kept and flagged; the results leave flagged events out.
 *   The reveal is tracked run by run because each run's log gets its own
 *   reveal event: a study's logs joined one after another (as the report
 *   reads them) must not flag the second run's judging as after the first
 *   run's reveal.
 */
export function reduceEvents(input: JudgeEvent[]): JudgeState {
  const seen = new Set<string>()
  const events: JudgeEvent[] = []
  let duplicates = 0
  let revealedAt: number | null = null
  const revealedRuns = new Set<string>()
  for (const raw of input) {
    if (!raw || typeof raw.id !== 'string') continue
    if (seen.has(raw.id)) { duplicates++; continue }
    seen.add(raw.id)
    const e: JudgeEvent = { ...raw }
    if (revealedRuns.has(e.run)) e.afterReveal = true
    else if (e.kind === 'reveal') {
      revealedRuns.add(e.run)
      revealedAt ??= atOf(e)
    }
    events.push(e)
  }
  const undone = new Set<string>()
  for (const e of events) {
    if (e.kind !== 'undo') continue
    const t = (e.value as UndoValue | undefined)?.target
    if (typeof t === 'string') undone.add(t)
  }
  const order = events.map((e, i) => ({ e, i, at: atOf(e) }))
  order.sort((a, b) => a.at - b.at || a.i - b.i)
  const latest = new Map<string, JudgeEvent>()
  const answers = new Map<string, Map<string, Answer>>()
  const pairsMade = new Map<string, [string, string][]>()
  const opened = new Map<string, Map<string, number>>()
  const sessions = new Set<string>()
  for (const { e, at } of order) {
    if (!SERVER_KINDS.includes(e.kind)) sessions.add(e.session)
    if (e.kind === 'undo' || undone.has(e.id)) continue
    if (e.kind === 'pairs-made') {
      if (!pairsMade.has(e.item)) {
        const pairs = (e.value as PairsMadeValue | undefined)?.pairs
        pairsMade.set(e.item, Array.isArray(pairs) ? pairs.map(p => [p[0], p[1]] as [string, string]) : [])
      }
      continue
    }
    if (e.kind === 'open-set') {
      const m = opened.get(e.judge) ?? new Map<string, number>()
      if (!m.has(e.item)) m.set(e.item, at)
      opened.set(e.judge, m)
    }
    latest.set(keyOf(e.judge, e.item, e.kind), e)
    if (ANSWER_KINDS.includes(e.kind)) {
      const m = answers.get(e.judge) ?? new Map<string, Answer>()
      m.set(e.item, { kind: e.kind, event: e, at, afterReveal: !!e.afterReveal })
      answers.set(e.judge, m)
    }
  }
  return { events, undone, latest, answers, pairsMade, opened, revealedAt, sessions, duplicates }
}

/** The judge's answer for an item, or undefined. `includeAfterReveal` defaults to false. */
export function answerOf(state: JudgeState, judge: string, item: string, includeAfterReveal = false): Answer | undefined {
  const a = state.answers.get(judge)?.get(item)
  if (!a) return undefined
  if (a.afterReveal && !includeAfterReveal) return undefined
  return a
}

/** Every judge who answered anything. */
export function judgesOf(state: JudgeState): string[] {
  return [...state.answers.keys()]
}
