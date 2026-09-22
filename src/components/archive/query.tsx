/**
 * The archive's query layer and its small typographic helpers.
 *
 * `history.search()` already understands the text terms and the flag tokens
 * (`is:video`, `model:krea`, `family:wan`, `seed:…`). It does not understand
 * dates or numeric comparisons, and an unknown prefix falls through to a text
 * match — so `after:2026-09-01` would quietly find nothing rather than filter.
 *
 * Rather than edit a file this desk does not own, the extra tokens are lifted
 * out of the query here, turned into predicates, and everything else is handed
 * to `history.search()` untouched. One search line, one syntax, no duplication
 * of the matching that already exists.
 */
import type { ReactNode } from 'react'
import { search as historySearch, type HistoryEntry } from '../../lib/history'

// ---------------------------------------------------------------------------
// Tokens
// ---------------------------------------------------------------------------

const TOKEN = /"[^"]*"|\S+/g

/** Numeric fields, in the spellings a person actually types. */
const NUMERIC: Record<string, (e: HistoryEntry) => number | null | undefined> = {
  steps: (e) => e.steps,
  cfg: (e) => e.cfg,
  mp: (e) => e.megapixels ?? megapixelsOf(e),
  megapixels: (e) => e.megapixels ?? megapixelsOf(e),
  denoise: (e) => e.denoise,
  frames: (e) => e.length,
  length: (e) => e.length,
  fps: (e) => e.fps,
  width: (e) => e.width,
  height: (e) => e.height,
  no: (e) => e.no,
  seconds: (e) => e.durationMs / 1000,
}

const DATE_FIELDS = new Set(['on', 'after', 'before'])

function megapixelsOf(e: HistoryEntry): number | null {
  if (e.width && e.height) return Math.round((e.width * e.height) / 1e4) / 100
  return null
}

const startOfDay = (ms: number): number => {
  const d = new Date(ms)
  d.setHours(0, 0, 0, 0)
  return d.getTime()
}

const DAY = 86_400_000

/** `2026-09-18`, or a day word. Returns epoch ms of the day's start, or null. */
function parseDay(value: string, now: number): number | null {
  const word = value.toLowerCase()
  if (word === 'today') return startOfDay(now)
  if (word === 'yesterday') return startOfDay(now) - DAY
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value)
  if (!m) return null
  const d = new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]))
  if (Number.isNaN(d.getTime())) return null
  d.setHours(0, 0, 0, 0)
  return d.getTime()
}

type Predicate = (e: HistoryEntry) => boolean

function numericPredicate(
  read: (e: HistoryEntry) => number | null | undefined,
  raw: string,
): Predicate | null {
  const m = /^(>=|<=|>|<|=)?\s*(-?\d+(?:\.\d+)?)$/.exec(raw)
  if (!m) return null
  const op = m[1] ?? '='
  const n = Number(m[2])
  return (e) => {
    const v = read(e)
    if (v === null || v === undefined || Number.isNaN(v)) return false
    switch (op) {
      case '>':
        return v > n
      case '<':
        return v < n
      case '>=':
        return v >= n
      case '<=':
        return v <= n
      default:
        return Math.abs(v - n) < 1e-9
    }
  }
}

function datePredicate(field: string, value: string, now: number): Predicate | null {
  const day = parseDay(value, now)
  if (day === null) {
    if (field !== 'on') return null
    const word = value.toLowerCase()
    if (word === 'week') return (e) => e.at >= startOfDay(now) - 6 * DAY
    if (word === 'month') return (e) => e.at >= startOfDay(now) - 29 * DAY
    return null
  }
  if (field === 'on') return (e) => e.at >= day && e.at < day + DAY
  if (field === 'after') return (e) => e.at >= day
  return (e) => e.at < day
}

export type SplitQuery = {
  /** What is left for `history.search()`. */
  rest: string
  /** The date and numeric tokens, already compiled. */
  pred: Predicate
  /** Plain text the reader typed, for highlighting inside a headline. */
  terms: string[]
  /** Every token, in the order typed. Used to toggle facets. */
  tokens: string[]
}

/**
 * Pull the tokens this layer owns out of a query, leaving the rest verbatim —
 * quotes, minus signs and all — so the two matchers never disagree about what
 * the reader wrote.
 */
export function splitQuery(query: string, now: number = Date.now()): SplitQuery {
  const rest: string[] = []
  const extra: Predicate[] = []
  const terms: string[] = []
  const tokens: string[] = []

  TOKEN.lastIndex = 0
  let m: RegExpExecArray | null
  while ((m = TOKEN.exec(query)) !== null) {
    const raw = m[0]
    tokens.push(raw)
    const negated = raw.startsWith('-') && raw.length > 1 && !raw.startsWith('-"')
    const body = negated ? raw.slice(1) : raw
    const quoted = body.startsWith('"')
    const colon = quoted ? -1 : body.indexOf(':')

    if (colon > 0 && colon < body.length - 1) {
      const field = body.slice(0, colon).toLowerCase()
      const value = body.slice(colon + 1)
      let pred: Predicate | null = null
      if (DATE_FIELDS.has(field)) pred = datePredicate(field, value, now)
      else if (field in NUMERIC) pred = numericPredicate(NUMERIC[field], value)
      if (pred) {
        const compiled = pred
        extra.push(negated ? (e) => !compiled(e) : compiled)
        continue
      }
      rest.push(raw)
      continue
    }

    rest.push(raw)
    if (!negated) {
      const text = quoted ? body.slice(1, -1) : body
      if (text.trim().length > 1) terms.push(text.trim())
    }
  }

  return {
    rest: rest.join(' '),
    pred: extra.length ? (e) => extra.every((p) => p(e)) : () => true,
    terms,
    tokens,
  }
}

/** Filter the archive by a query. Newest first, as `history.all()` keeps it. */
export function runQuery(
  query: string,
  within: readonly HistoryEntry[],
  now: number = Date.now(),
): { results: HistoryEntry[]; terms: string[] } {
  const q = query.trim()
  if (!q) return { results: [...within], terms: [] }
  const { rest, pred, terms } = splitQuery(q, now)
  const text = historySearch(rest, within)
  return { results: text.filter(pred), terms }
}

/** True when a record satisfies a whole query, extra tokens included. */
export function matchesQuery(e: HistoryEntry, query: string, now: number = Date.now()): boolean {
  const q = query.trim()
  if (!q) return true
  const { rest, pred } = splitQuery(q, now)
  if (!pred(e)) return false
  return historySearch(rest, [e]).length > 0
}

/** Add a facet's token, or take it away again if it is already there. */
export function toggleToken(query: string, token: string): string {
  const { tokens } = splitQuery(query)
  const without = tokens.filter((t) => t.toLowerCase() !== token.toLowerCase())
  if (without.length !== tokens.length) return without.join(' ')
  return [...tokens, token].join(' ').trim()
}

/** True when a facet's token is currently in the query. */
export function hasToken(query: string, token: string): boolean {
  return splitQuery(query).tokens.some((t) => t.toLowerCase() === token.toLowerCase())
}

// ---------------------------------------------------------------------------
// Highlighting
// ---------------------------------------------------------------------------

const escapeRe = (s: string): string => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

/**
 * The reader's words, marked burgundy inside a headline. Nothing else in the
 * card is coloured, so the eye lands on the match and not on the furniture.
 */
export function Highlight({ text, terms }: { text: string; terms: readonly string[] }): ReactNode {
  const useful = terms.filter((t) => t.length > 1)
  if (!useful.length || !text) return text
  let re: RegExp
  try {
    re = new RegExp(`(${useful.map(escapeRe).join('|')})`, 'gi')
  } catch {
    return text
  }
  const parts = text.split(re)
  return (
    <>
      {parts.map((part, i) =>
        i % 2 === 1 ? (
          <mark key={i} className="bg-transparent text-burgundy-900">
            {part}
          </mark>
        ) : (
          <span key={i}>{part}</span>
        ),
      )}
    </>
  )
}

// ---------------------------------------------------------------------------
// Typography helpers — British spelling, thin spaces, tabular numbers
// ---------------------------------------------------------------------------

const THIN = ' '

const DAY_NAMES = [
  'Sunday',
  'Monday',
  'Tuesday',
  'Wednesday',
  'Thursday',
  'Friday',
  'Saturday',
] as const

const MONTHS = [
  'January',
  'February',
  'March',
  'April',
  'May',
  'June',
  'July',
  'August',
  'September',
  'October',
  'November',
  'December',
] as const

/** `Thursday, 18 September` — and `Today` / `Yesterday` where that reads better. */
export function dayHeading(ms: number, now: number = Date.now()): string {
  const day = startOfDay(ms)
  const today = startOfDay(now)
  if (day === today) return 'Today'
  if (day === today - DAY) return 'Yesterday'
  const d = new Date(ms)
  const year = d.getFullYear() === new Date(now).getFullYear() ? '' : ` ${d.getFullYear()}`
  return `${DAY_NAMES[d.getDay()]}, ${d.getDate()} ${MONTHS[d.getMonth()]}${year}`
}

/** `18 September 2026, 14:06` */
export function fullDate(ms: number): string {
  const d = new Date(ms)
  const hh = String(d.getHours()).padStart(2, '0')
  const mm = String(d.getMinutes()).padStart(2, '0')
  return `${d.getDate()} ${MONTHS[d.getMonth()]} ${d.getFullYear()}, ${hh}:${mm}`
}

/** `14:06` */
export function clockTime(ms: number): string {
  const d = new Date(ms)
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
}

/** `7.4 seconds`, `4 min 10 s`. */
export function duration(ms: number): string {
  if (!Number.isFinite(ms) || ms <= 0) return '—'
  const s = ms / 1000
  if (s < 60) return `${s.toFixed(1)} seconds`
  const min = Math.floor(s / 60)
  const rest = Math.round(s - min * 60)
  return `${min} min ${rest} s`
}

/** `1024 × 1216`, thin-spaced. */
export function dimensions(w: number | null, h: number | null): string | null {
  if (!w || !h) return null
  return `${w}${THIN}×${THIN}${h}`
}

/** `0:05 · 81f` — the corner slug on a clip's poster. */
export function clipSlug(length?: number, fps?: number): string | null {
  if (!length) return null
  if (!fps) return `${length}f`
  const s = length / fps
  const mins = Math.floor(s / 60)
  const secs = Math.round(s - mins * 60)
  return `${mins}:${String(secs).padStart(2, '0')} · ${length}f`
}

/** `No. 1,284` */
export function editionNo(no: number): string {
  return `No. ${no.toLocaleString('en-GB')}`
}

export function count(n: number): string {
  return n.toLocaleString('en-GB')
}

/** What made it, in words: `From words`, `From a picture`, `An instruction`. */
export function madeFrom(e: HistoryEntry): string {
  switch (e.mode) {
    case 'i2i':
      return 'From a picture'
    case 'i2v':
      return 'From a start frame'
    case 'edit':
      return 'An instruction'
    case 't2v':
      return 'From words'
    default:
      return 'From words'
  }
}

/** Guard every single-key shortcut: none of them fire while you are typing. */
export function isTyping(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null
  return (
    !!el &&
    (el.isContentEditable === true || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName ?? ''))
  )
}
