/**
 * vision.ts
 *
 * What is in a picture, and which LoRAs that picture actually calls for.
 *
 *
 * THE HALF OF THE PROBLEM loraIndex.ts CANNOT REACH.
 *
 * loraIndex.ts recovers each LoRA's training vocabulary from its own safetensors
 * header, and matches a prompt against it. On the generate path that is the
 * whole job. On the edit and refine paths it is nearly useless, because there
 * the prompt is a handful of words or nothing at all and the picture carries the
 * intent. This file reads the picture.
 *
 * The tagger is WD14 (wd-vit-tagger-v3), and the reason is exact rather than
 * aesthetic: it emits danbooru tags, these LoRAs were captioned in danbooru
 * tags, and loraIndex.ts `normalise` already folds underscores to spaces. So
 * `large_breasts` off the tagger becomes `large breasts`, which is character for
 * character the string in the index. Joining the two is an equality test. No
 * embedding, no similarity threshold, no translation table that can silently
 * drift. server/vision.mjs records what the alternatives would have cost.
 *
 *
 * THE TRAP THIS FILE EXISTS TO AVOID.
 *
 * The obvious wiring is tagImage -> matchPrompt -> missingTriggers, and it is
 * wrong. Measured here on real outputs from /mnt/storage/ai/outputs, feeding raw
 * WD14 tags straight into matchPrompt ranked `gaping-pussy-illustrious-xl` and
 * `meaty-labia-slutty-pussy` top for a clothed-below-the-waist nude with no
 * genital tag anywhere in its tags, and missingTriggers would then have appended
 * "gaping pussy" and "5lutty pussy" to the prompt. On a picture of a soldier
 * holding a rifle it proposed `rnct` and `v4n1lla`. That is not a weak
 * suggestion, it is an instruction to draw something the user did not ask for.
 *
 * The cause is that matchPrompt weights a hit by `share`, the tag's frequency
 * inside that one LoRA's training set, which says nothing about whether the tag
 * distinguishes that LoRA from the other 38. Measured across the 39 indexed
 * LoRAs that carry tag data:
 *
 *     breasts        in 23 of 39 vocabularies    idf 0.53
 *     large breasts  in 21 of 39                 idf 0.62
 *     nipples        in 19 of 39                 idf 0.72
 *     nude           in 19 of 39                 idf 0.72
 *     pussy          in 15 of 39                 idf 0.96
 *     gaping pussy   in  1 of 39                 idf 3.66
 *     sports bra     in  1 of 39                 idf 3.66
 *
 * Nearly every LoRA here was trained on nudes, so "breasts, nipples, nude" is
 * the background radiation of the whole folder and carries almost no
 * information. Weighting by inverse document frequency across the index, which
 * needs no new data because it is computed from LORA_INDEX itself, separates the
 * two by a factor of seven.
 *
 *
 * A SECOND TRAP, WHICH IDF ALONE DOES NOT FIX.
 *
 * A LoRA's concept list describes its training set, not its purpose. The
 * clearest case on disk is good-hands-for-pony: its top concepts are blush 0.83,
 * breasts 0.73, nipples 0.55, and the word "hands" appears nowhere in them. Its
 * subject lives in its trigger, `good_hands`. So concept overlap can never on its
 * own establish that a LoRA is wanted. It can only establish, by absence, that a
 * LoRA is not.
 *
 * Hence the two sided rule below. A LoRA's DEFINING tags are the ones that are
 * both high share (most of its training set) and high idf (rare across the
 * folder): for gaping-pussy those are "gaping pussy" 1.00, "pussy" 0.96, "spread
 * pussy" 0.95. If an image shows none of a LoRA's defining tags, that LoRA is
 * vetoed outright, and the veto is the reliable half. What survives is then
 * ordered by idf weighted overlap, and that ordering is reported as an ordering,
 * never as detection. A LoRA with no defining tags at all can only ever come
 * back as 'incidental' and must not be auto applied.
 *
 * Measured on a 14 image sample of real outputs, this vetoes 15 of 39 candidates
 * on clothed pictures and returns nothing at all for most of them, ranks
 * perfect-pussy-pony at 3 of 3 defining tags covered on an explicit one, and
 * puts the three realism LoRAs top on the photoreal ones. The abstentions are
 * the point: silence is a correct answer and is what the old wiring could not
 * produce.
 *
 *
 * WHAT THE DETECTORS ADD.
 *
 * The YOLO weights the Impact Pack already installed answer the question WD14
 * answers worst: where the faces and hands are, and how much of the frame they
 * occupy. That is exactly the gate a hands LoRA needs and exactly what its
 * concept list cannot supply. They cost no download. They cannot name a style or
 * a concept, so they supplement the tagger rather than replacing it.
 *
 *
 * DEGRADING HONESTLY.
 *
 * Every entry point here works when the tagger is absent, and says so. It never
 * invents a tag. `capabilities()` reports what is really on disk, and when the
 * tagger is missing it carries the verified URL and byte count so the caller can
 * offer the download through the existing POST /api/download. A stub returning
 * plausible tags would be worse than an honest absence, because every number
 * downstream of it would be fiction.
 */

import type { IndexedBase, LoraIndexEntry } from './loraIndex'
import { LORA_INDEX, } from './loraIndex'
import { askUntilAnswered, RETRY_FAILED_MS } from './capabilities'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Where an image lives, as server/vision.mjs confines it. */
export type ImageKind = 'output' | 'input' | 'model'

/** An image the server can reach by path, rather than one the browser holds. */
export type ImageRef = { kind: ImageKind; rel: string }

/**
 * Anything callers have in hand at the point they want tags: a path under the
 * outputs root, a full reference, or the bytes themselves for a picture the user
 * has only just dropped in and which is not saved anywhere yet.
 */
export type ImageSource = string | ImageRef | Blob

export type ImageTag = {
  /** Booru spelling, underscores intact, exactly as the tagger emits it. */
  tag: string
  /** 0 to 1, the tagger's own sigmoid output. Not calibrated across tags. */
  confidence: number
}

/** WD14's four way content rating. Its own vocabulary, not this project's. */
export type ImageRating = 'general' | 'sensitive' | 'questionable' | 'explicit'

/** One detector's finding. `areaShare` is the fraction of the frame covered. */
export type Detection = {
  confidence: number
  box: [number, number, number, number]
  areaShare: number
}

export type ImageFacts = {
  face: Detection[]
  hand: Detection[]
  person: Detection[]
}

export type ImageTags = {
  width: number
  height: number
  rating: ImageRating | null
  /** All four ratings with their scores, because the top one is often close. */
  ratings: ImageTag[]
  /** Content tags above the threshold, strongest first. */
  general: ImageTag[]
  /** Named characters, at a deliberately high threshold. Usually empty. */
  character: ImageTag[]
}

/** Deliberately the same members as `AnatomyLevel` in recipe.ts. */
export type AnatomyLevel = 'off' | 'natural' | 'emphasised'

/**
 * How much the evidence is worth.
 *
 *   'strong'      the LoRA has defining tags and the image shows most of them.
 *   'likely'      it has defining tags and the image shows at least one.
 *   'incidental'  overlap only, or the LoRA has no defining tags to test
 *                 against. Show it, let a person pick it, never auto apply it.
 */
export type SuggestionConfidence = 'strong' | 'likely' | 'incidental'

export type SuggestionEvidence = {
  /** The matched tag, in the index's normalised spelling. */
  tag: string
  /** The tagger's confidence that it is in the picture. */
  confidence: number
  /** Share of this LoRA's training images carrying it. */
  share: number
  /** How rare it is across the indexed folder. Higher discriminates more. */
  idf: number
  via: 'trigger' | 'concept'
  /** True when this tag is one of the LoRA's defining tags. */
  defining: boolean
}

export type LoraSuggestion = {
  entry: LoraIndexEntry
  score: number
  confidence: SuggestionConfidence
  /** Defining tags present over defining tags the LoRA has. Null when it has none. */
  coverage: { present: number; total: number } | null
  evidence: SuggestionEvidence[]
  /** One sentence of UI ready copy saying why this is here. */
  why: string
}

export type VetoedLora = {
  file: string
  stem: string
  /** The defining tags the image would have had to show. */
  needed: string[]
  why: string
}

export type VisionReport = {
  tags: ImageTags | null
  facts: ImageFacts | null
  suggestions: LoraSuggestion[]
  vetoed: VetoedLora[]
  /** A prompt fragment built from the tags, for feeding matchPrompt or a box. */
  promptFromImage: string
  rating: ImageRating | null
  anatomy: AnatomyLevel
  /**
   * How many installed LoRAs this ranking could see, and how many it could not.
   *
   * Of the 61 files on disk, 22 ship no ss_tag_frequency, so they have no
   * vocabulary to match against and were never candidates. That is not the same
   * as having been considered and rejected, and a UI that shows a ranking
   * without saying so implies the whole folder was weighed. `withoutTriggerData`
   * in loraIndex.ts names them.
   */
  considered: { ranked: number; noTagData: number }
  /** Present when nothing could be read, with the reason in plain words. */
  unavailable?: string
}

export type VisionCapabilities = {
  /** Null when the vision endpoint did not answer: the rest is then a stand-in, not a check. */
  server: string | null
  python: string | null
  /** True only when an interpreter, the model and the tag list are all present. */
  tagger: boolean
  taggerModel: string | null
  taggerBytes: number | null
  taggerVocabulary: string
  detect: boolean
  detectors: string[]
  device: string
  roots: Record<string, string>
  /** What to fetch, when the tagger is missing. Verified URLs and byte counts. */
  install: {
    repo: string
    files: { filename: string; dest: string; url: string; sizeBytes: number }[]
    missing: string[]
  } | null
  reason: string | null
}

// ---------------------------------------------------------------------------
// Tuning, all of it measured rather than picked
// ---------------------------------------------------------------------------

/**
 * A tag has to be in most of a LoRA's training set before it can define it.
 * Below this it is background: real-nipples saw `pubic hair` in 15 percent of
 * its images and is not a pubic hair LoRA.
 */
const DEFINING_SHARE = 0.6

/**
 * And it has to be rare enough across the folder to distinguish anything. At
 * 1.2 this admits `pussy` (in 15 of 39, idf 0.96) only as support and never as a
 * definition, which is right: a pussy tag does not choose between the six LoRAs
 * that have one.
 */
const DEFINING_IDF = 1.2

/** Below this a tag is background radiation and is ignored for scoring. */
const SUPPORT_IDF = 1.0

/** The tagger's confidence floor for a tag to count as present in the image. */
const PRESENT_AT = 0.35

/** A LoRA with no defining hit needs this much plain overlap to be worth showing. */
const INCIDENTAL_FLOOR = 0.35

/** Most of the defining tags present, so 'strong' rather than 'likely'. */
const STRONG_COVERAGE = 0.5

// ---------------------------------------------------------------------------
// The index side: inverse document frequency over LORA_INDEX
// ---------------------------------------------------------------------------

/**
 * loraIndex.ts normalises the same way but does not export it, and the two must
 * agree character for character or the join silently misses. Kept identical on
 * purpose; if that file's `normalise` changes, this has to change with it.
 */
function normalise(text: string): string {
  return text
    .toLowerCase()
    .replace(/[_]+/g, ' ')
    .replace(/[():]/g, ' ')
    .replace(/[^a-z0-9'\- ]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

type Vocab = {
  entry: LoraIndexEntry
  /** Normalised tag to its share and how it appears in this LoRA. */
  tags: Map<string, { share: number; via: 'trigger' | 'concept' }>
  /** The tags that say what this LoRA is for. Empty when nothing qualifies. */
  defining: string[]
}

let built: { vocab: Vocab[]; idf: Map<string, number>; corpus: number } | null = null

/**
 * Build the vocabulary table and the document frequencies once.
 *
 * The corpus is only the LoRAs that carry tag data. Counting the 22 that say
 * nothing would inflate every idf uniformly and, worse, imply those files had
 * been consulted and found wanting. They were not consulted, because they cannot
 * be.
 */
function build() {
  if (built) return built
  const vocab: Vocab[] = []
  const df = new Map<string, number>()
  for (const entry of LORA_INDEX) {
    if (!entry.hasTagFrequency) continue
    const tags = new Map<string, { share: number; via: 'trigger' | 'concept' }>()
    for (const c of entry.concepts) {
      const key = normalise(c.tag)
      if (key) tags.set(key, { share: c.share, via: 'concept' })
    }
    // A trigger reading beats a concept reading of the same string.
    for (const t of entry.triggers) {
      const key = normalise(t.tag)
      if (key) tags.set(key, { share: t.share, via: 'trigger' })
    }
    for (const key of tags.keys()) df.set(key, (df.get(key) ?? 0) + 1)
    vocab.push({ entry, tags, defining: [] })
  }
  const corpus = vocab.length
  const idf = new Map<string, number>()
  for (const [tag, n] of df) idf.set(tag, Math.log(corpus / n))
  for (const v of vocab) {
    v.defining = [...v.tags.entries()]
      .filter(([tag, t]) => t.share >= DEFINING_SHARE && (idf.get(tag) ?? 0) >= DEFINING_IDF)
      .sort((a, b) => b[1].share - a[1].share)
      .map(([tag]) => tag)
  }
  built = { vocab, idf, corpus }
  return built
}

/** The number of LoRAs this ranking could have drawn on. The rest carry no tags. */
function indexedCorpusSize(): number {
  return build().corpus
}

/**
 * Installed LoRAs that carry no tag frequency and so can never be ranked here.
 *
 * Worth showing next to any ranking. These files are not bad and they are not
 * irrelevant; they simply do not say what they were trained on, so no amount of
 * looking at the picture can decide whether they apply.
 */
function unrankableCount(): number {
  return LORA_INDEX.length - build().corpus
}

// ---------------------------------------------------------------------------
// Ranking
// ---------------------------------------------------------------------------

export type SuggestOptions = {
  /** Restrict to LoRAs trained for these bases. Pass the chosen checkpoint's arch. */
  bases?: IndexedBase[]
  limit?: number
  /** Include 'incidental' matches. Default true; pass false for auto apply. */
  includeIncidental?: boolean
  /** Tagger confidence floor for a tag to count as present. */
  presentAt?: number
}

function confidenceOf(coverage: { present: number; total: number } | null): SuggestionConfidence {
  if (!coverage || coverage.total === 0) return 'incidental'
  if (coverage.present === 0) return 'incidental'
  return coverage.present / coverage.total >= STRONG_COVERAGE ? 'strong' : 'likely'
}

function whyFor(entry: LoraIndexEntry, conf: SuggestionConfidence, evidence: SuggestionEvidence[]): string {
  const defining = evidence.filter(e => e.defining).map(e => e.tag)
  const top = evidence.slice(0, 3).map(e => e.tag)
  if (conf === 'strong') {
    return `The picture shows ${defining.slice(0, 3).join(', ')}, which is what this add-on was trained on.`
  }
  if (conf === 'likely') {
    return `The picture shows ${defining[0]}, one of the things this add-on was trained on.`
  }
  if (entry.triggers.length && !defining.length) {
    // The good-hands-for-pony case. Say plainly that the overlap is not proof.
    return `Its training set overlaps this picture on ${top.join(', ')}, but those tags do not describe what ` +
      `this add-on is for, so treat this as a loose suggestion.`
  }
  return `Loose overlap only, on ${top.join(', ')}.`
}

/**
 * Rank the installed LoRAs against a set of image tags.
 *
 * Pure: it takes tags and returns a ranking, so it works equally on tags from
 * the tagger, tags a user typed, or tags from somewhere else entirely. Nothing
 * here touches the network.
 *
 * The veto list comes back alongside the suggestions because it is the more
 * trustworthy output of the two and a UI that wants to explain why a LoRA the
 * user expected is absent needs it.
 */
function suggestLorasForTags(
  tags: ImageTag[],
  options: SuggestOptions = {},
): { suggestions: LoraSuggestion[]; vetoed: VetoedLora[] } {
  const { vocab, idf } = build()
  const presentAt = options.presentAt ?? PRESENT_AT
  const wantBases = options.bases ? new Set(options.bases) : null
  const includeIncidental = options.includeIncidental !== false

  const seen = new Map<string, number>()
  for (const t of tags) {
    const key = normalise(t.tag)
    if (!key) continue
    const prev = seen.get(key)
    if (prev === undefined || t.confidence > prev) seen.set(key, t.confidence)
  }
  if (seen.size === 0) return { suggestions: [], vetoed: [] }

  const suggestions: LoraSuggestion[] = []
  const vetoed: VetoedLora[] = []

  for (const v of vocab) {
    if (wantBases && !wantBases.has(v.entry.base)) continue

    const presentDefining = v.defining.filter(d => (seen.get(d) ?? 0) >= presentAt)
    if (v.defining.length > 0 && presentDefining.length === 0) {
      vetoed.push({
        file: v.entry.file,
        stem: v.entry.stem,
        needed: v.defining.slice(0, 4),
        why: `Nothing in the picture matches what this add-on was trained on (${v.defining.slice(0, 3).join(', ')}).`,
      })
      continue
    }

    let score = 0
    const evidence: SuggestionEvidence[] = []
    for (const [tag, t] of v.tags) {
      const confidence = seen.get(tag)
      if (confidence === undefined || confidence < presentAt) continue
      const rarity = idf.get(tag) ?? 0
      if (rarity < SUPPORT_IDF) continue
      score += t.share * rarity * confidence * (t.via === 'trigger' ? 1.25 : 1)
      evidence.push({
        tag,
        confidence,
        share: t.share,
        idf: Number(rarity.toFixed(2)),
        via: t.via,
        defining: presentDefining.includes(tag),
      })
    }

    if (evidence.length === 0) continue
    if (presentDefining.length === 0 && score < INCIDENTAL_FLOOR) continue

    const coverage = v.defining.length ? { present: presentDefining.length, total: v.defining.length } : null
    const confidence = confidenceOf(coverage)
    if (!includeIncidental && confidence === 'incidental') continue

    evidence.sort((a, b) => Number(b.defining) - Number(a.defining) || b.share * b.idf - a.share * a.idf)
    suggestions.push({
      entry: v.entry,
      score: Number(score.toFixed(3)),
      confidence,
      coverage,
      evidence,
      why: whyFor(v.entry, confidence, evidence),
    })
  }

  const rank = { strong: 0, likely: 1, incidental: 2 }
  suggestions.sort((a, b) =>
    rank[a.confidence] - rank[b.confidence] ||
    b.score - a.score ||
    a.entry.file.localeCompare(b.entry.file))

  return {
    suggestions: options.limit ? suggestions.slice(0, options.limit) : suggestions,
    vetoed,
  }
}

// ---------------------------------------------------------------------------
// Turning tags back into prompt material
// ---------------------------------------------------------------------------

/**
 * The image as a prompt fragment: booru underscores turned into spaces, ordered
 * by the tagger's confidence.
 *
 * This is what makes the image usable by everything already written against
 * text. `matchPrompt(promptFromTags(tags), ...)` works, though the ranking here
 * is better for this purpose because matchPrompt cannot veto.
 */
function promptFromTags(tags: ImageTag[], limit = 40): string {
  return tags
    .slice(0, limit)
    .map(t => t.tag.replace(/_/g, ' '))
    .join(', ')
}

// ---------------------------------------------------------------------------
// Reading the rating
// ---------------------------------------------------------------------------

/**
 * WD14's rating mapped onto the project's anatomy levels.
 *
 * The mapping is a judgement, not a measurement, and it is deliberately
 * conservative in one direction: 'questionable' means partial nudity or
 * suggestive framing, which is 'natural' and not 'emphasised', because raising
 * the level changes the picture and guessing upward is the more annoying error.
 * recipe.ts `suggestAnatomy` reads the prompt for the same thing; when both have
 * an opinion the caller should prefer whichever had real evidence, which on the
 * edit path is this one.
 */
function anatomyFromRating(rating: ImageRating | null): AnatomyLevel {
  switch (rating) {
    case 'explicit': return 'emphasised'
    case 'questionable': return 'natural'
    case 'sensitive': return 'natural'
    case 'general': return 'off'
    default: return 'off'
  }
}

// ---------------------------------------------------------------------------
// The server side
// ---------------------------------------------------------------------------

function refBody(source: ImageSource): { json: string } | { bytes: Blob } {
  if (source instanceof Blob) return { bytes: source }
  const ref: ImageRef = typeof source === 'string' ? { kind: 'output', rel: source } : source
  return { json: JSON.stringify({ image: ref }) }
}

async function post(path: string, source: ImageSource, extra?: Record<string, unknown>, signal?: AbortSignal) {
  const body = refBody(source)
  const init: RequestInit = 'bytes' in body
    ? { method: 'POST', headers: { 'Content-Type': (source as Blob).type || 'application/octet-stream' }, body: body.bytes, signal }
    : {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: extra ? JSON.stringify({ ...JSON.parse(body.json), ...extra }) : body.json,
        signal,
      }
  const res = await fetch(path, init)
  const text = await res.text()
  let parsed: unknown
  try { parsed = JSON.parse(text) } catch {
    // An HTML body here means the middleware is not mounted and Vite's SPA
    // fallback answered instead. Say that, rather than reporting a parse error.
    throw new Error('the vision endpoint is not mounted on this server')
  }
  const data = parsed as Record<string, unknown>
  if (!res.ok) throw Object.assign(new Error(String(data.error ?? res.statusText)), { status: res.status, data })
  return data
}

/** What a probe that got no answer stands in for. Nothing in it was checked. */
const UNANSWERED: VisionCapabilities = {
  server: null,
  python: null,
  tagger: false,
  taggerModel: null,
  taggerBytes: null,
  taggerVocabulary: 'danbooru-v3',
  detect: false,
  detectors: [],
  device: 'cpu',
  roots: {},
  install: null,
  reason: 'the vision endpoint did not answer, so no picture can be read until it does',
}

let capsCache: Promise<VisionCapabilities> | null = null

/**
 * What this machine can actually see, straight from the server's own checks.
 *
 * An answer is cached, because it is read on every render of anything that
 * offers the feature and it only changes when a file appears on disk; call
 * `refreshCapabilities` after a download. A failure is not an answer, so it
 * is held only as long as the server probe in capabilities.ts holds one: the
 * server may have been restarting, and a failure kept for good would hide
 * reading, the tagger fetch and archive tagging until the page was reloaded.
 */
export function capabilities(): Promise<VisionCapabilities> {
  if (capsCache) return capsCache
  const probe: Promise<VisionCapabilities> = fetch('/api/vision/capabilities')
    .then(async r => {
      if (!r.ok) throw new Error(`vision capabilities: ${r.status}`)
      return await r.json() as VisionCapabilities
    })
    .catch(() => {
      setTimeout(() => {
        if (capsCache === probe) capsCache = null
      }, RETRY_FAILED_MS)
      return UNANSWERED
    })
  capsCache = probe
  return probe
}

/**
 * Hand each answer to `onAnswer`, asking again while the server has not
 * answered, so a reading on screen since a restart finds the server once it
 * is back. Returns the function that stops asking.
 */
export function watchCapabilities(onAnswer: (caps: VisionCapabilities) => void): () => void {
  return askUntilAnswered(capabilities, c => c.server === null, onAnswer)
}

export function refreshCapabilities(): Promise<VisionCapabilities> {
  capsCache = null
  return capabilities()
}

type TagRow = ImageTags & { index: number; error?: string }
type DetectRow = { index: number; detections: Partial<ImageFacts>; error?: string }

function factsFrom(row: DetectRow | undefined): ImageFacts | null {
  if (!row) return null
  return {
    face: row.detections.face ?? [],
    hand: row.detections.hand ?? [],
    person: row.detections.person ?? [],
  }
}

/**
 * One call, one child process: tags, detections, ranked LoRAs and the vetoes.
 *
 * This is the entry point the edit and refine paths want. It never throws for a
 * missing model; it comes back with `unavailable` set and everything else empty,
 * so a caller can render the rest of its UI and say why this part is blank.
 */
export async function inspectImage(
  file: ImageSource,
  options: SuggestOptions & { detect?: boolean; signal?: AbortSignal } = {},
): Promise<VisionReport> {
  const considered = { ranked: indexedCorpusSize(), noTagData: unrankableCount() }
  const empty: VisionReport = {
    tags: null, facts: null, suggestions: [], vetoed: [],
    promptFromImage: '', rating: null, anatomy: 'off', considered,
  }
  const caps = await capabilities()
  if (!caps.tagger && !caps.detect) {
    return { ...empty, unavailable: caps.reason ?? 'no image understanding is installed' }
  }

  const wantDetect = options.detect !== false && caps.detect
  const path = caps.tagger
    ? (wantDetect ? '/api/vision/inspect' : '/api/vision/tag')
    : '/api/vision/detect'

  let data: Record<string, unknown>
  try {
    data = await post(path, file, undefined, options.signal)
  } catch (err) {
    if ((err as { name?: string }).name === 'AbortError') throw err
    return { ...empty, unavailable: String((err as Error).message ?? err) }
  }

  const tagRows = (data.tag as { rows?: TagRow[] } | undefined)?.rows ?? []
  const detectRows = (data.detect as { rows?: DetectRow[] } | undefined)?.rows ?? []
  const row = tagRows[0]
  const facts = factsFrom(detectRows[0])

  if (!row || row.error) {
    return {
      ...empty,
      facts,
      unavailable: row?.error ?? (caps.tagger ? 'the tagger returned nothing' : caps.reason ?? 'no tagger installed'),
    }
  }

  const tags: ImageTags = {
    width: row.width,
    height: row.height,
    rating: row.rating,
    ratings: row.ratings ?? [],
    general: row.general ?? [],
    character: row.character ?? [],
  }
  const { suggestions, vetoed } = suggestLorasForTags(tags.general, options)

  return {
    tags,
    facts,
    suggestions,
    vetoed,
    promptFromImage: promptFromTags(tags.general),
    rating: tags.rating,
    anatomy: anatomyFromRating(tags.rating),
    considered,
  }
}

/** One tagged row of a batch, with the reference it was asked about. */
export type TaggedRow = ImageTags & { kind: ImageKind; rel: string; error?: string }

/**
 * Tag up to 24 images the server can reach, in one child process.
 *
 * For the archive's "tag everything that has no tags" pass. Rows come back
 * with the caller's own reference attached, so a batch is reassembled by
 * reference rather than by trusting array order. A row with `error` set is a
 * file the server could not read; the others are still good.
 */
export async function tagImages(refs: readonly ImageRef[], signal?: AbortSignal): Promise<TaggedRow[]> {
  if (!refs.length) return []
  const res = await fetch('/api/vision/tag', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ images: refs.slice(0, 24) }),
    signal,
  })
  const text = await res.text()
  let data: Record<string, unknown>
  try { data = JSON.parse(text) as Record<string, unknown> } catch {
    throw new Error('the vision endpoint is not mounted on this server')
  }
  if (!res.ok) throw new Error(String(data.error ?? res.statusText))
  const rows = (data.tag as { rows?: (TagRow & { kind?: ImageKind; rel?: string })[] } | undefined)?.rows ?? []
  return rows.map(row => ({
    width: row.width,
    height: row.height,
    rating: row.rating,
    ratings: row.ratings ?? [],
    general: row.general ?? [],
    character: row.character ?? [],
    kind: row.kind ?? 'output',
    rel: row.rel ?? '',
    error: row.error,
  }))
}

