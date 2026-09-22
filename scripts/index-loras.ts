/**
 * index-loras.ts
 *
 * Reads the trigger words back out of the installed LoRAs and writes them into
 * src/lib/loraIndex.ts as a typed, browser-safe module.
 *
 * Run it with:   npm run index-loras
 *
 *
 * WHY THIS EXISTS.
 *
 * A LoRA without its trigger word in the prompt is substantially weaker. That is
 * not folklore, it was measured on this machine: Pony V6, same seed,
 * add-micro-details at strength 0.6, whole-frame Laplacian variance.
 *
 *     base, no LoRA                    165.9   1.000x
 *     add-micro-details, no trigger    269.5   1.625x
 *     add-micro-details, with trigger  353.8   2.132x
 *
 * The trigger alone is worth 1.313x on top of loading the weights, and it changes
 * 67 percent of the pixels. So the difference between "the LoRA is on" and "the
 * LoRA is on and told what it is" is roughly a third of the whole effect.
 *
 * Trigger words are usually copied off a web page by hand, which is how
 * src/lib/loras.ts ended up recording `rnat` for the real-nipples LoRA when the
 * file itself says `rnct`. A wrong trigger is worse than a missing one: it
 * silently poisons the prompt with a token the model has no meaning for, and
 * nothing anywhere reports it.
 *
 * The files know the answer. Every LoRA trained with kohya-ss sd-scripts carries
 * ss_tag_frequency in its safetensors header: the actual caption tag distribution
 * of the training set, with counts. A tag present in nearly every training image
 * and absent from ordinary booru vocabulary is, by construction, the token the
 * author minted to address this LoRA. Nothing is guessed here.
 *
 *
 * HOW THE HEADER IS READ.
 *
 * safetensors is a header-first format: 8 bytes little endian header length, then
 * that many bytes of JSON, then the tensors. The metadata lives in the JSON under
 * `__metadata__`. So a 1.8 GB LoRA costs one seek and a few tens of kilobytes to
 * index. Nothing is loaded onto the GPU and nothing large is read.
 *
 *
 * HOW A TRIGGER IS TOLD APART FROM A DESCRIPTION.
 *
 * Four signals, in decreasing order of how much they are trusted:
 *
 *   1. Near universal.  Counts are summed across dataset buckets, the largest sum
 *      is taken as the image count, and a tag at TRIGGER_SHARE or more of it is a
 *      candidate. An author's token is in every caption; "blonde hair" is not.
 *
 *   2. Unique to this LoRA.  Document frequency is computed across all indexed
 *      LoRAs: how many of them use this tag at all. Real booru vocabulary is
 *      shared, so "1girl" lands in 30 of 39 and "blush" in 22. An invented token
 *      lands in exactly one. This replaces a hardcoded booru wordlist with a
 *      measurement taken from the folder being indexed, which is better, because
 *      it adapts when new LoRAs arrive. COMMON_DF is the line.
 *
 *   3. Minted.  p3p05y, addmicrodetails, rnct, good_hands: a single token shaped
 *      like an invention rather than like English, which no other LoRA uses even
 *      as a word inside a longer tag. See isMinted, which lists what it accepts
 *      and what it rejects in this folder. A minted token near universal in the
 *      captions is the only thing rated `strong`, and the narrower digit
 *      substituted form of it is the only signal allowed to rescue a trigger that
 *      is NOT near universal, see RESCUE_SHARE.
 *
 *   4. The dataset folder name.  kohya names buckets `<repeats>_<concept>`, so
 *      `29_pfbk` and `7_5lutty_pussy` carry the author's token even when the
 *      captions do not. Used only as a fallback, because folder names are also
 *      where `10_asd` and `10_bdf` live.
 *
 * Everything rejected as a trigger but still informative becomes a `concept` with
 * its frequency. That list is what prompt matching scores against: it is the
 * LoRA's own statement of what it was shown.
 *
 *
 * WHAT IT REFUSES TO DO.
 *
 * A third of the folder ships no usable metadata: 11 files have no `__metadata__`
 * block at all (mostly the tiny slider LoRAs, which are steered by weight rather
 * than by a word) and 11 more have metadata but no ss_tag_frequency, leaving 39
 * of 61 taggable. Those are recorded with hasMetadata / hasTagFrequency false
 * and an empty trigger list. They are not broken. The honest statement about them
 * is "no trigger data in the file", never "no trigger needed", and the index says
 * so rather than letting a consumer infer the wrong one.
 *
 * Likewise every entry carries `confidence` and `notes`. A weak extraction that
 * admits it is weak is worth more than a confident wrong token.
 */
import { createHash } from 'node:crypto'
import { closeSync, existsSync, openSync, readFileSync, readSync, readdirSync, statSync, writeFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

// ---------------------------------------------------------------------------
// Tuning. Every one of these is a judgement call, so each says what it costs.
// ---------------------------------------------------------------------------

/**
 * Where the LoRAs live: the Lora folder of the same models root the server
 * reads (SWITCHGEN_MODELS, as in server/api.mjs), unless SWITCHGEN_LORA_DIR
 * names it outright. It used to ignore SWITCHGEN_MODELS, so on any machine
 * but the author's the indexer went looking in a folder that is not there.
 */
const LORA_DIR =
  process.env.SWITCHGEN_LORA_DIR ?? join(process.env.SWITCHGEN_MODELS ?? '/mnt/storage/ai/models', 'Lora')

/**
 * A tag must appear in this share of training images to be a trigger candidate.
 * 0.85 rather than 1.0 because captions get hand-edited and a few images in any
 * set lose the token. Lowering it pulls in ordinary content tags; raising it
 * loses triggers on sets with sloppy captioning.
 */
const TRIGGER_SHARE = 0.85

/**
 * How many indexed LoRAs must use a tag before it counts as ordinary booru
 * vocabulary rather than one author's invention. Measured against this folder:
 * `1girl` is in 30 of 39, `blush` in 22, `spread pussy` in 12, while `p3p05y`,
 * `rnct`, `penisquiron`, `pfbk` and `addmicrodetails` are each in exactly 1.
 * The gap is wide, so the exact value is not delicate.
 */
const COMMON_DF = 6

/**
 * An orthographically invented token is allowed to be a trigger from this share
 * upward, well below TRIGGER_SHARE. This exists for one measured case:
 * pony-nsfw-explicit-realistic-photography captions `v4n1lla` on 698 of its 1339
 * images, 52 percent, because only half the set is the styled half. Without the
 * rescue that trigger is lost entirely. The rule is kept narrow, digits
 * substituted into a single word used by no other LoRA, because a loose version
 * of it would start inventing triggers out of rare content tags.
 */
const RESCUE_SHARE = 0.25

/** Longer than this and it is a caption sentence, not a token. Kept as a concept. */
const MAX_TRIGGER_WORDS = 6

/** Concepts kept per LoRA, most frequent first. Bounds the generated file. */
const MAX_CONCEPTS = 48

/** A concept below this share is noise from a single stray caption. */
const MIN_CONCEPT_SHARE = 0.02

/**
 * Tags excluded from trigger candidacy outright. The document frequency rule
 * above already catches all of these in the current folder, this list is the
 * belt to its braces: it keeps the heuristic sane on a folder of two LoRAs,
 * where every tag would look unique.
 */
const GENERIC_TAGS: ReadonlySet<string> = new Set([
  '1girl', '1boy', '2girls', 'multiple girls', 'multiple boys', 'solo', 'solo focus',
  'looking at viewer', 'looking_at_viewer', 'looking back', 'looking up', 'looking down',
  'breasts', 'large breasts', 'medium breasts', 'small breasts', 'huge breasts', 'flat chest',
  'nipples', 'nude', 'completely nude', 'ass', 'pussy', 'anus', 'penis', 'navel', 'thighs',
  'blush', 'smile', 'open mouth', 'closed mouth', 'closed eyes', 'teeth', 'tongue',
  'long hair', 'short hair', 'black hair', 'blonde hair', 'brown hair', 'blue eyes',
  'brown eyes', 'green eyes', 'red eyes', 'purple eyes', 'bangs', 'collarbone',
  'simple background', 'white background', 'grey background', 'transparent background',
  'blurry', 'blurry background', 'depth of field', 'close-up', 'closeup', 'close up',
  'upper body', 'lower body', 'full body', 'standing', 'sitting', 'lying', 'spread legs',
  'indoors', 'outdoors', 'day', 'night', 'sky', 'cloud', 'realistic', 'photorealistic',
  'highres', 'absurdres', 'jewelry', 'earrings', 'necklace', 'shirt', 'dress', 'skirt',
  'underwear', 'panties', 'bra', 'swimsuit', 'bikini', 'uncensored', 'censored',
  'sweat', 'shiny skin', 'hetero', 'sex', 'pov', 'from behind', 'head out of frame',
])

/**
 * Tags excluded from `concepts` as well, because they say nothing about WHICH
 * LoRA a prompt wants. Every set has `1girl` and `blonde hair` in it, so matching
 * on them ranks all 39 equally and adds pure noise to the score.
 *
 * This is deliberately much smaller than GENERIC_TAGS. `pussy`, `nipples` and
 * `close-up` are bad triggers, because they are ordinary vocabulary rather than
 * an author's token, and at the same time they are excellent evidence about which
 * LoRA a prompt is reaching for. They stay in the concept list. The two filters
 * do different jobs and must not be collapsed into one.
 */
const CONCEPT_NOISE: ReadonlySet<string> = new Set([
  '1girl', '1boy', '2girls', 'multiple girls', 'multiple boys', 'solo', 'solo focus',
  'looking at viewer', 'looking_at_viewer', 'looking back', 'looking up', 'looking down',
  'long hair', 'short hair', 'medium hair', 'very long hair', 'black hair', 'blonde hair',
  'brown hair', 'blue hair', 'pink hair', 'purple hair', 'red hair', 'white hair',
  'silver hair', 'green hair', 'grey hair', 'multicolored hair', 'twintails', 'ponytail',
  'braid', 'ahoge', 'bangs', 'hair between eyes', 'hair ornament', 'blue eyes', 'brown eyes',
  'green eyes', 'red eyes', 'purple eyes', 'yellow eyes', 'grey eyes', 'pink eyes',
  'smile', 'grin', 'open mouth', 'closed mouth', 'closed eyes', 'parted lips', 'teeth',
  'simple background', 'white background', 'grey background', 'black background',
  'blurry background', 'blurry', 'depth of field', 'highres', 'absurdres', 'lowres',
  'day', 'night', 'sky', 'blue sky', 'cloud', 'cloudy sky', 'indoors', 'outdoors',
  'standing', 'sitting', 'lying', 'upper body', 'lower body', 'full body', 'portrait',
  'collarbone', 'navel', 'jewelry', 'earrings', 'necklace', 'bracelet', 'ring', 'choker',
  'shirt', 'white shirt', 'dress', 'white dress', 'skirt', 'pleated skirt', 'jacket',
  'shorts', 'pants', 'long sleeves', 'short sleeves', 'bare shoulders', 'shoes', 'boots',
  'gloves', 'hat', 'bow', 'flower', 'tree', 'grass', 'water', 'window', 'holding',
  'nail polish', 'fingernails', 'makeup', 'mole', 'heart', 'animal ears', 'pointy ears',
  'original', 'artist name', 'signature', 'watermark', 'english text',
])

/**
 * Pony-lineage quality prefixes. These are near universal in any Pony dataset
 * because the base model demands them, so they pass the universality test without
 * being this LoRA's own token. They are kept, because good-hands-for-pony really
 * was trained with them and dropping them weakens it, but they are labelled
 * `quality` so a prompt builder can deduplicate them against its own preamble
 * instead of stacking `score_9` three times.
 */
const QUALITY_TAGS: ReadonlySet<string> = new Set([
  'score_9', 'score_8_up', 'score_8', 'score_7_up', 'score_7', 'score_6_up',
  'source_anime', 'source_pony', 'source_furry', 'source_cartoon',
  'masterpiece', 'best quality', 'high quality', 'very awa', 'newest', 'absurdres',
])

// ---------------------------------------------------------------------------
// safetensors header
// ---------------------------------------------------------------------------

type RawMeta = Record<string, string>

/**
 * Reads `__metadata__` out of a safetensors file without touching the tensors.
 * Returns null when the file has no metadata block, which is a normal state, not
 * a failure: several sliders here ship a bare header.
 */
function readSafetensorsMetadata(path: string): RawMeta | null {
  const fd = openSync(path, 'r')
  try {
    const lenBuf = Buffer.alloc(8)
    if (readSync(fd, lenBuf, 0, 8, 0) !== 8) return null
    const headerLen = Number(lenBuf.readBigUInt64LE(0))
    // A sane header is kilobytes to a few megabytes. Anything else means this is
    // not a safetensors file, or is corrupt, and we refuse rather than allocate.
    if (!Number.isSafeInteger(headerLen) || headerLen <= 0 || headerLen > 64 * 1024 * 1024) return null
    const header = Buffer.alloc(headerLen)
    if (readSync(fd, header, 0, headerLen, 8) !== headerLen) return null
    const parsed: unknown = JSON.parse(header.toString('utf8'))
    if (typeof parsed !== 'object' || parsed === null) return null
    const meta = (parsed as Record<string, unknown>).__metadata__
    if (typeof meta !== 'object' || meta === null) return null
    const out: RawMeta = {}
    for (const [k, v] of Object.entries(meta as Record<string, unknown>)) {
      if (typeof v === 'string') out[k] = v
    }
    return Object.keys(out).length > 0 ? out : null
  } catch {
    return null
  } finally {
    closeSync(fd)
  }
}

/** kohya writes ss_tag_frequency as a JSON string: { bucketName: { tag: count } }. */
type TagFrequency = Record<string, Record<string, number>>

function parseTagFrequency(meta: RawMeta): TagFrequency | null {
  const src = meta.ss_tag_frequency
  if (!src) return null
  try {
    const parsed: unknown = JSON.parse(src)
    if (typeof parsed !== 'object' || parsed === null) return null
    const out: TagFrequency = {}
    for (const [bucket, tags] of Object.entries(parsed as Record<string, unknown>)) {
      if (typeof tags !== 'object' || tags === null) continue
      const inner: Record<string, number> = {}
      for (const [tag, count] of Object.entries(tags as Record<string, unknown>)) {
        const n = typeof count === 'number' ? count : Number(count)
        const clean = tag.trim()
        if (clean && Number.isFinite(n) && n > 0) inner[clean] = n
      }
      if (Object.keys(inner).length > 0) out[bucket] = inner
    }
    return Object.keys(out).length > 0 ? out : null
  } catch {
    return null
  }
}

// ---------------------------------------------------------------------------
// Base architecture
// ---------------------------------------------------------------------------

/** Mirrors LoraArch in src/lib/loras.ts so the two indexes join without a map. */
type IndexedBase = 'pony' | 'illustrious' | 'sdxl' | 'flux1d' | 'sd15' | 'wan' | 'unknown'

/**
 * Checkpoint name fragments, longest and most specific first. These are matched
 * against ss_sd_model_name, which is the checkpoint the LoRA was actually trained
 * on, and is therefore worth more than any title or filename.
 */
const CHECKPOINT_HINTS: ReadonlyArray<readonly [RegExp, IndexedBase]> = [
  [/pony[\s_-]*diffusion|ponydiffusionv6|ponyrealism|ponymagine|_pony|pony_|pdxl|ponyxl/i, 'pony'],
  [/noobai|noobxl|anynoobai|illustrious|illustriousxl|ilxl|animagine|_ill\b|ill_/i, 'illustrious'],
  [/sd_xl_base|sdxl_base|stable-diffusion-xl/i, 'sdxl'],
]

/** Weaker fragments, used on titles and filenames where marketing words creep in. */
const NAME_HINTS: ReadonlyArray<readonly [RegExp, IndexedBase]> = [
  [/\bwan2?2?[\s_-]|wan22|wan2\.2/i, 'wan'],
  [/\bflux\b|flux\.?1|flux1/i, 'flux1d'],
  [/pony|pdxl|ponyxl/i, 'pony'],
  [/illustrious|illustriousxl|ilxl|noobai|noobxl|\bnoob\b|\bil\b|\bill\b/i, 'illustrious'],
  [/\bsdxl\b|xl[\s_-]?1\.0/i, 'sdxl'],
]

function hintFrom(text: string, table: ReadonlyArray<readonly [RegExp, IndexedBase]>): IndexedBase | null {
  for (const [re, base] of table) if (re.test(text)) return base
  return null
}

/**
 * Every family a name mentions, not just the first. `detail-enhancer-il-pony`
 * names two, and a reader deserves to see both rather than whichever the regex
 * table happened to reach first.
 */
function allHints(text: string, table: ReadonlyArray<readonly [RegExp, IndexedBase]>): IndexedBase[] {
  const out: IndexedBase[] = []
  for (const [re, base] of table) if (re.test(text) && !out.includes(base)) out.push(base)
  return out
}

type BaseVerdict = { base: IndexedBase; evidence: string; claims: IndexedBase[]; note?: string }

/**
 * ss_base_model_version settles the architecture family. It cannot settle Pony
 * against Illustrious, because both report themselves as plain SDXL, so inside
 * the SDXL lineage the training checkpoint name decides, then the author's output
 * name, then the filename. The filename is last because it is the marketing
 * title: "penis-real-diffusion-ILLUSTRIOUS-lora" was trained on Pony, and the
 * file says so.
 */
function inferBase(file: string, meta: RawMeta | null): BaseVerdict {
  const claims = new Set<IndexedBase>(allHints(file, NAME_HINTS))
  const fromFile = hintFrom(file, NAME_HINTS)

  if (!meta) {
    return fromFile
      ? { base: fromFile, evidence: 'filename only, the file carries no metadata', claims: [...claims] }
      : { base: 'unknown', evidence: 'no metadata and no recognisable name', claims: [] }
  }

  const version = meta.ss_base_model_version ?? ''
  const arch = meta['modelspec.architecture'] ?? ''
  const trainedOn = meta.ss_sd_model_name ?? ''
  const title = meta['modelspec.title'] ?? meta.ss_output_name ?? ''
  const notes: string[] = []

  if (/flux/i.test(version) || /flux/i.test(arch)) {
    return { base: 'flux1d', evidence: `ss_base_model_version ${version || arch}`, claims: [...claims] }
  }
  if (/^sd_1\.5|stable-diffusion-v1\b/i.test(version)) {
    return { base: 'sd15', evidence: `ss_base_model_version ${version}`, claims: [...claims] }
  }
  if (/wan/i.test(title) || /wan/i.test(file)) {
    return { base: 'wan', evidence: 'name identifies Wan 2.2', claims: [...claims] }
  }

  const isSdxl = /sdxl_base/i.test(version) || /stable-diffusion-xl/i.test(arch)
  if (/stable-diffusion-v1/i.test(arch) && isSdxl) {
    notes.push(
      'modelspec.architecture says stable-diffusion-v1 while ss_base_model_version says SDXL. ' +
        'The trainer wrote one of the two wrong. SDXL is taken as the truth because the tensor ' +
        'shapes in this file are SDXL shapes.',
    )
  }
  if (!isSdxl && !version && !arch) {
    return fromFile
      ? { base: fromFile, evidence: 'filename, the metadata names no base', claims: [...claims], note: notes[0] }
      : { base: 'unknown', evidence: 'the metadata names no base', claims: [], note: notes[0] }
  }

  // Inside the SDXL lineage, decide which finetune the weights actually saw.
  const fromCheckpoint = trainedOn ? hintFrom(trainedOn, CHECKPOINT_HINTS) : null
  const fromTitle = title ? hintFrom(title, NAME_HINTS) : null
  for (const c of allHints(title, NAME_HINTS)) claims.add(c)

  let base: IndexedBase
  let evidence: string
  if (fromCheckpoint) {
    base = fromCheckpoint
    evidence = `trained on ${trainedOn}`
  } else if (fromTitle && fromTitle !== 'wan' && fromTitle !== 'flux1d') {
    base = fromTitle
    evidence = `author's output name ${title}`
  } else if (fromFile && fromFile !== 'wan' && fromFile !== 'flux1d') {
    base = fromFile
    evidence = `filename, the training checkpoint ${trainedOn || 'is not named'} says nothing`
  } else {
    base = 'sdxl'
    evidence = `ss_base_model_version ${version || arch}, no finetune identified`
  }

  const other = [...claims].filter((c) => c !== base && c !== 'unknown')
  if (other.length > 0 && fromCheckpoint) {
    notes.push(
      `The name claims ${other.join(' and ')} but it was trained on ${trainedOn}. ` +
        'Crossing inside the SDXL lineage loads and does something, it is just untested.',
    )
  }
  return { base, evidence, claims: [...claims], note: notes.join(' ') || undefined }
}

// ---------------------------------------------------------------------------
// Trigger extraction
// ---------------------------------------------------------------------------

type TriggerKind = 'invented' | 'phrase' | 'quality' | 'folder'
type TriggerConfidence = 'strong' | 'likely' | 'weak' | 'none' | 'no-data'

type Trigger = {
  tag: string
  count: number
  share: number
  kind: TriggerKind
  uniqueToThisLora: boolean
}

type Concept = { tag: string; count: number; share: number }

/** Summed tag counts across every dataset bucket, plus the per-bucket detail. */
type Tally = {
  totals: Map<string, number>
  imageCount: number
  folders: string[]
  /** repeats parsed off the `<n>_<concept>` bucket names, 0 when they carry none. */
  repeats: number[]
  bucketPeaks: number[]
}

function tally(tf: TagFrequency): Tally {
  const totals = new Map<string, number>()
  const folders: string[] = []
  const repeats: number[] = []
  const bucketPeaks: number[] = []
  for (const [bucket, tags] of Object.entries(tf)) {
    folders.push(bucket)
    const m = /^(\d+)_(.*)$/.exec(bucket)
    repeats.push(m ? Number(m[1]) : 0)
    let peak = 0
    for (const [tag, count] of Object.entries(tags)) {
      totals.set(tag, (totals.get(tag) ?? 0) + count)
      if (count > peak) peak = count
    }
    bucketPeaks.push(peak)
  }
  let imageCount = 0
  for (const n of totals.values()) if (n > imageCount) imageCount = n
  return { totals, imageCount, folders, repeats, bucketPeaks }
}

const wordCount = (tag: string) => tag.trim().split(/\s+/).length

/** Digits substituted into letters: p3p05y, v4n1lla, carto4on, 5lutty, 90sanime. */
const hasDigitSubstitution = (t: string) => /[0-9][a-z]|[a-z][0-9]/i.test(t)

/**
 * A word shaped like a minted token rather than like English. Three shapes pass:
 * digits substituted into letters, an underscore join the way sd-scripts writes
 * `good_hands`, and a long unbroken concatenation such as `addmicrodetails`. A
 * short gibberish acronym passes too, because `rnct` and `pfbk` are exactly that.
 *
 * The seven and eight letter band is deliberately excluded: that is where the
 * ordinary English words live, and letting it through starts calling `realism`
 * and `detailed` triggers. Hyphens are excluded for the same reason, since real
 * compounds use them and `3-point-lighting` is not a trigger.
 */
function mintedShape(t: string): boolean {
  if (/\s/.test(t) || t.includes('-')) return false
  if (t.length < 4) return false
  return hasDigitSubstitution(t) || t.includes('_') || t.length >= 9 || t.length <= 6
}

/** Words used by any other indexed LoRA, so the token is not this author's invention. */
type Vocabulary = { tagDf: Map<string, number>; wordDf: Map<string, number> }

/**
 * A token this LoRA minted for itself. The test is entirely local: the tag is
 * used by no other indexed LoRA, no other indexed LoRA even uses it as a word
 * inside a longer tag, and it is shaped like a token rather than like English.
 *
 * Verified against this folder, it accepts addmicrodetails, rnct, penisquiron,
 * handslora, puffynips, pfbk, p3p05y, v4n1lla, carto4on, amateurquiron,
 * aidmahyperrealism, illustriousanime, hentai_studio_quality, high_detail,
 * good_hands and 90sanimeaesthetic, and it rejects realism, advertising,
 * photography, detailed, vagina, cervix and 3-point-lighting.
 */
function isMinted(tag: string, vocab: Vocabulary): boolean {
  const t = tag.trim().toLowerCase()
  if (!mintedShape(t)) return false
  if ((vocab.tagDf.get(t) ?? 1) !== 1) return false
  // An underscore join is already unambiguous, and its parts are ordinary words
  // by design, so the word level test would reject every one of them.
  if (t.includes('_')) return true
  return (vocab.wordDf.get(t) ?? 1) <= 1
}

/**
 * Some trainers write `looking_at_viewer`, others `looking at viewer`. Both
 * wordlists above are spelled with spaces, so a tag is tested in both spellings
 * or half the underscore-style datasets slip their filler through.
 */
function inList(list: ReadonlySet<string>, tag: string): boolean {
  const lower = tag.toLowerCase()
  return list.has(lower) || list.has(lower.replace(/_/g, ' '))
}

function isGeneric(tag: string, df: number): boolean {
  return inList(GENERIC_TAGS, tag) || df >= COMMON_DF
}

/**
 * Strips the kohya `<repeats>_` prefix and normalises underscores, so
 * `7_5lutty_pussy` becomes `5lutty pussy`. Returns null for buckets that carry no
 * concept at all: `img`, `dataset`, and the junk folder names like `10_asd`.
 */
function folderConcept(bucket: string): string | null {
  const m = /^(\d+)_(.*)$/.exec(bucket)
  const body = (m ? m[2] : bucket).replace(/_/g, ' ').trim()
  if (!body) return null
  if (/^(img|images?|dataset|train|data|detailer|concept)$/i.test(body)) return null
  // `asd` and `bdf` are placeholders somebody typed to make the trainer run.
  if (!body.includes(' ') && body.length < 5 && !hasDigitSubstitution(body)) return null
  return body
}

type Extraction = {
  triggers: Trigger[]
  concepts: Concept[]
  imageCount: number
  folders: string[]
  confidence: TriggerConfidence
  notes: string[]
  /** True when a human should look at this one before trusting the trigger. */
  shaky: boolean
}

function extract(tf: TagFrequency, vocab: Vocabulary, meta: RawMeta | null): Extraction {
  const df = vocab.tagDf
  const t = tally(tf)
  const notes: string[] = []
  let shaky = false
  if (t.imageCount === 0) {
    return {
      triggers: [], concepts: [], imageCount: 0, folders: t.folders,
      confidence: 'none', notes: ['ss_tag_frequency is present but empty.'], shaky: true,
    }
  }

  const rows = [...t.totals.entries()]
    .map(([tag, count]) => ({ tag, count, share: count / t.imageCount, df: df.get(tag.toLowerCase()) ?? 1 }))
    .sort((a, b) => b.count - a.count)

  // How much of the training set the captions actually cover. kohya's
  // ss_num_train_images counts repeats, so dividing by the bucket repeats gives
  // the real image count to compare the top tag against. A large shortfall means
  // per-image natural-language captions with almost no shared vocabulary, and
  // then "the most frequent tag" is a sentence that happened to recur, not a
  // trigger. sdxl-film-photography-style and big-natural-breasts are both this.
  const declared = Number(meta?.ss_num_train_images ?? 0)
  const reps = t.repeats.filter((r) => r > 0)
  let lowCoverage = false
  if (declared > 0 && reps.length > 0 && reps.length === t.repeats.length) {
    const covered = t.bucketPeaks.reduce((sum, peak, i) => sum + peak * t.repeats[i], 0)
    if (covered < declared * 0.5) {
      lowCoverage = true
      shaky = true
      notes.push(
        `Captions cover only about ${Math.round((covered / declared) * 100)} percent of the ` +
          `${declared} training samples, so the tag counts are a thin sample of the set and the ` +
          'frequency ranking is not reliable. Treat any trigger below as a suggestion.',
      )
    }
  }

  const captionish = rows.filter((r) => wordCount(r.tag) > MAX_TRIGGER_WORDS).length
  if (captionish > rows.length * 0.5) {
    shaky = true
    notes.push(
      'Most of this dataset was captioned in natural language rather than booru tags, so the ' +
        'frequency table describes sentences. Short tokens in it are still trustworthy, long ones are not.',
    )
  }

  const triggers: Trigger[] = []
  const taken = new Set<string>()
  const push = (tag: string, count: number, share: number, kind: TriggerKind, unique: boolean) => {
    const key = tag.toLowerCase()
    if (taken.has(key)) return
    taken.add(key)
    triggers.push({ tag, count, share: Math.min(1, Number(share.toFixed(4))), kind, uniqueToThisLora: unique })
  }

  for (const r of rows) {
    if (r.share < TRIGGER_SHARE) break
    if (wordCount(r.tag) > MAX_TRIGGER_WORDS) continue
    const quality = inList(QUALITY_TAGS, r.tag)
    if (!quality && isGeneric(r.tag, r.df)) continue
    const kind: TriggerKind = quality ? 'quality' : isMinted(r.tag, vocab) ? 'invented' : 'phrase'
    push(r.tag, r.count, r.share, kind, r.df === 1)
  }

  // The rescue. Only for a token with digits substituted into it, which is as
  // unambiguous as an author's invention gets, and which no other LoRA uses.
  for (const r of rows) {
    if (r.share >= TRIGGER_SHARE || r.share < RESCUE_SHARE) continue
    if (!hasDigitSubstitution(r.tag) || !isMinted(r.tag, vocab)) continue
    if (inList(GENERIC_TAGS, r.tag)) continue
    push(r.tag, r.count, r.share, 'invented', true)
    shaky = true
    notes.push(
      `${r.tag} appears in only ${Math.round(r.share * 100)} percent of the captions, below the ` +
        'usual bar, but it is a minted token that no other LoRA here uses, so it was kept.',
    )
  }

  // Fallback: the author's dataset folder, when the captions gave nothing.
  if (triggers.length === 0) {
    for (const bucket of t.folders) {
      const concept = folderConcept(bucket)
      if (!concept) continue
      // `5_pussy` is a class folder, not an instance token. Same test as for tags.
      if (isGeneric(concept, df.get(concept.toLowerCase()) ?? 1)) continue
      const inTags = t.totals.get(concept) ?? 0
      push(concept, inTags, inTags / t.imageCount, 'folder', (df.get(concept.toLowerCase()) ?? 1) === 1)
    }
    if (triggers.length > 0) {
      shaky = true
      notes.push(
        'No caption tag was near universal, so the trigger above comes from the training folder ' +
          `name (${t.folders.join(', ')}) rather than from the tag counts. That is how the author ` +
          'labelled the concept, but it is weaker evidence than a tag in every caption.',
      )
    }
  }

  const concepts: Concept[] = []
  for (const r of rows) {
    if (concepts.length >= MAX_CONCEPTS) break
    if (taken.has(r.tag.toLowerCase())) continue
    if (r.share < MIN_CONCEPT_SHARE) break
    if (wordCount(r.tag) > MAX_TRIGGER_WORDS) continue
    // Filler that every LoRA carries ranks them all equally, which is noise.
    if (inList(CONCEPT_NOISE, r.tag)) continue
    concepts.push({ tag: r.tag, count: r.count, share: Number(r.share.toFixed(4)) })
  }

  // Five near universal phrases is a caption template, not a trigger. The
  // polyhedron skin LoRA captions every image "advertising, photography,
  // photostudio lighting", and all three pass the universality test while none of
  // them is the token that addresses the LoRA.
  const plain = triggers.filter((x) => x.kind === 'phrase').length
  if (plain > 2 && !triggers.some((x) => x.kind === 'invented')) {
    shaky = true
    notes.push(
      `${plain} different tags are in nearly every caption here. That is the shape of a caption ` +
        'template rather than of a single trigger word, so the first one is probably the one that ' +
        'matters and the rest are the author being thorough.',
    )
  }

  let confidence: TriggerConfidence
  if (triggers.length === 0) confidence = 'none'
  else if (lowCoverage) confidence = 'weak'
  else if (triggers.some((x) => x.kind === 'invented' && x.share >= TRIGGER_SHARE)) confidence = 'strong'
  else if (triggers.some((x) => x.kind === 'folder')) confidence = 'weak'
  else if (triggers.some((x) => x.kind === 'invented')) confidence = 'likely'
  else if (triggers.some((x) => x.share >= TRIGGER_SHARE)) confidence = 'likely'
  else confidence = 'weak'

  if (triggers.length === 0) {
    const named = t.folders.map(folderConcept).filter((x): x is string => x !== null)
    notes.push(
      'Every near universal tag in this set is ordinary booru vocabulary, so this LoRA most likely ' +
        'has no trigger word and works on weight alone. That is a reading of the data, not a promise.' +
        (named.length > 0
          ? ` Its training folders are named ${named.join(' and ')}, which is what the author was ` +
            'aiming at even though no caption token carries it.'
          : ''),
    )
  }

  return { triggers, concepts, imageCount: t.imageCount, folders: t.folders, confidence, notes, shaky }
}

/**
 * What to actually paste into a prompt. When the LoRA minted a token, that token
 * alone is the trigger and the rest of the near universal tags are the author's
 * caption template, which the prompt does not need and which would dilute it.
 * When there is no minted token, everything near universal goes in, because then
 * the template is all the evidence there is.
 */
function promptTagsOf(triggers: Trigger[]): string[] {
  const minted = triggers.filter((x) => x.kind === 'invented' || x.kind === 'quality')
  const chosen = minted.some((x) => x.kind === 'invented') ? minted : triggers
  return chosen.map((x) => x.tag)
}

// ---------------------------------------------------------------------------
// Walk, index, emit
// ---------------------------------------------------------------------------

type Entry = {
  file: string
  stem: string
  bytes: number
  base: IndexedBase
  baseEvidence: string
  nameClaims: IndexedBase[]
  trainedOn: string
  authorName: string
  networkDim: number
  hasMetadata: boolean
  hasTagFrequency: boolean
  imageCount: number
  datasetFolders: string[]
  triggers: Trigger[]
  /** The subset of `triggers` worth pasting into a prompt. See promptTagsOf. */
  promptTags: string[]
  triggerPhrase: string
  confidence: TriggerConfidence
  concepts: Concept[]
  notes: string[]
  /** Report only, not emitted: a human should eyeball this extraction. */
  shaky: boolean
}

function stemOf(file: string): string {
  return file.replace(/\.safetensors$/i, '')
}

function build(): { entries: Entry[]; dirFingerprint: string } {
  const files = readdirSync(LORA_DIR)
    .filter((f) => f.toLowerCase().endsWith('.safetensors'))
    .sort((a, b) => a.localeCompare(b))

  type Loaded = { file: string; bytes: number; meta: RawMeta | null; tf: TagFrequency | null }
  const loaded: Loaded[] = files.map((file) => {
    const path = join(LORA_DIR, file)
    const meta = readSafetensorsMetadata(path)
    return { file, bytes: statSync(path).size, meta, tf: meta ? parseTagFrequency(meta) : null }
  })

  // Pass one: how many LoRAs use each tag at all, and how many use each word
  // inside any tag. This is the local stand-in for a booru vocabulary list, and
  // it is better than one because it is measured against the folder being
  // indexed, so it adapts as LoRAs arrive. Measured on the current 39 taggable
  // files: `1girl` is in 30, `blush` in 22, `spread pussy` in 12, and every
  // minted token is in exactly 1.
  const tagDf = new Map<string, number>()
  const wordDf = new Map<string, number>()
  for (const l of loaded) {
    if (!l.tf) continue
    const tags = new Set<string>()
    const words = new Set<string>()
    for (const bucket of Object.values(l.tf)) {
      for (const tag of Object.keys(bucket)) {
        const norm = tag.trim().toLowerCase()
        tags.add(norm)
        for (const w of norm.replace(/_/g, ' ').split(/[^a-z0-9'-]+/)) if (w) words.add(w)
      }
    }
    for (const tag of tags) tagDf.set(tag, (tagDf.get(tag) ?? 0) + 1)
    for (const w of words) wordDf.set(w, (wordDf.get(w) ?? 0) + 1)
  }
  const vocab: Vocabulary = { tagDf, wordDf }

  const entries: Entry[] = loaded.map((l) => {
    const verdict = inferBase(l.file, l.meta)
    const notes: string[] = []
    if (verdict.note) notes.push(verdict.note)

    let triggers: Trigger[] = []
    let concepts: Concept[] = []
    let imageCount = 0
    let folders: string[] = []
    let confidence: TriggerConfidence = 'no-data'
    let shaky = false

    if (l.tf) {
      const ex = extract(l.tf, vocab, l.meta)
      triggers = ex.triggers
      concepts = ex.concepts
      imageCount = ex.imageCount
      folders = ex.folders
      confidence = ex.confidence
      shaky = ex.shaky
      notes.push(...ex.notes)
    } else if (l.meta) {
      notes.push(
        'The file carries training metadata but no ss_tag_frequency, so there is no trigger data in ' +
          'it. Whether it needs a trigger word is unknown from the file alone.',
      )
    } else {
      notes.push(
        'The file carries no metadata block at all, so there is no trigger data in it. Sliders are ' +
          'usually shipped this way and are steered by weight rather than by a word, but that is a ' +
          'guess from the filename, not something this file says.',
      )
    }

    return {
      file: l.file,
      stem: stemOf(l.file),
      bytes: l.bytes,
      base: verdict.base,
      baseEvidence: verdict.evidence,
      nameClaims: verdict.claims.filter((c) => c !== verdict.base),
      trainedOn: l.meta?.ss_sd_model_name ?? '',
      authorName: l.meta?.['modelspec.title'] ?? l.meta?.ss_output_name ?? '',
      networkDim: Number(l.meta?.ss_network_dim ?? 0) || 0,
      hasMetadata: l.meta !== null,
      hasTagFrequency: l.tf !== null,
      imageCount,
      datasetFolders: folders,
      triggers,
      promptTags: promptTagsOf(triggers),
      triggerPhrase: promptTagsOf(triggers).join(', '),
      confidence,
      concepts,
      notes,
      shaky,
    }
  })

  const fingerprint = createHash('sha256')
    .update(entries.map((e) => `${e.file}:${e.bytes}`).join('\n'))
    .digest('hex')
    .slice(0, 16)

  return { entries, dirFingerprint: fingerprint }
}

// ---------------------------------------------------------------------------
// Emission
// ---------------------------------------------------------------------------

const HERE = dirname(fileURLToPath(import.meta.url))
const TARGET = resolve(HERE, '..', 'src', 'lib', 'loraIndex.ts')
const BEGIN = '// #region generated by scripts/index-loras.ts'
const END = '// #endregion generated by scripts/index-loras.ts'

const q = (s: string) => JSON.stringify(s)

function emitEntry(e: Entry): string {
  const lines: string[] = ['  {']
  lines.push(`    file: ${q(e.file)},`)
  lines.push(`    stem: ${q(e.stem)},`)
  lines.push(`    bytes: ${e.bytes},`)
  lines.push(`    base: ${q(e.base)},`)
  lines.push(`    baseEvidence: ${q(e.baseEvidence)},`)
  lines.push(`    nameClaims: [${e.nameClaims.map(q).join(', ')}],`)
  lines.push(`    trainedOn: ${q(e.trainedOn)},`)
  lines.push(`    authorName: ${q(e.authorName)},`)
  lines.push(`    networkDim: ${e.networkDim},`)
  lines.push(`    hasMetadata: ${e.hasMetadata},`)
  lines.push(`    hasTagFrequency: ${e.hasTagFrequency},`)
  lines.push(`    imageCount: ${e.imageCount},`)
  lines.push(`    datasetFolders: [${e.datasetFolders.map(q).join(', ')}],`)
  lines.push(`    confidence: ${q(e.confidence)},`)
  lines.push(`    promptTags: [${e.promptTags.map(q).join(', ')}],`)
  lines.push(`    triggerPhrase: ${q(e.triggerPhrase)},`)
  if (e.triggers.length === 0) lines.push('    triggers: [],')
  else {
    lines.push('    triggers: [')
    for (const t of e.triggers) {
      lines.push(
        `      { tag: ${q(t.tag)}, count: ${t.count}, share: ${t.share}, kind: ${q(t.kind)}, uniqueToThisLora: ${t.uniqueToThisLora} },`,
      )
    }
    lines.push('    ],')
  }
  if (e.concepts.length === 0) lines.push('    concepts: [],')
  else {
    lines.push('    concepts: [')
    for (const c of e.concepts) lines.push(`      { tag: ${q(c.tag)}, count: ${c.count}, share: ${c.share} },`)
    lines.push('    ],')
  }
  if (e.notes.length === 0) lines.push('    notes: [],')
  else {
    lines.push('    notes: [')
    for (const n of e.notes) lines.push(`      ${q(n)},`)
    lines.push('    ],')
  }
  lines.push('  },')
  return lines.join('\n')
}

function emit(entries: Entry[], fingerprint: string): string {
  const withTriggers = entries.filter((e) => e.triggers.length > 0).length
  const noMeta = entries.filter((e) => !e.hasMetadata).length
  const noTags = entries.filter((e) => e.hasMetadata && !e.hasTagFrequency).length
  const body = entries.map(emitEntry).join('\n')
  return [
    BEGIN,
    '// Do not edit by hand. Run `npm run index-loras` after adding or removing LoRAs.',
    `// Source folder: ${LORA_DIR}`,
    `// ${entries.length} files, ${withTriggers} yielded a trigger, ${noMeta} carry no metadata block,`,
    `// ${noTags} carry metadata but no ss_tag_frequency.`,
    '',
    `export const LORA_DIR = ${q(LORA_DIR)}`,
    '',
    '/** Changes whenever a file is added, removed or replaced. Cheap staleness check. */',
    `export const LORA_INDEX_FINGERPRINT = ${q(fingerprint)}`,
    '',
    'export const LORA_INDEX: readonly LoraIndexEntry[] = [',
    body,
    ']',
    END,
    '',
  ].join('\n')
}

function report(entries: Entry[]): void {
  const withTriggers = entries.filter((e) => e.triggers.length > 0)
  const noMeta = entries.filter((e) => !e.hasMetadata)
  const noTags = entries.filter((e) => e.hasMetadata && !e.hasTagFrequency)
  const noTrigger = entries.filter((e) => e.hasTagFrequency && e.triggers.length === 0)
  const shaky = entries.filter((e) => e.shaky && e.hasTagFrequency)

  const line = (s = '') => process.stdout.write(s + '\n')
  line(`Indexed ${entries.length} LoRAs from ${LORA_DIR}`)
  line(`  ${withTriggers.length} yielded at least one trigger`)
  line(`    ${entries.filter((e) => e.confidence === 'strong').length} strong   a minted token in nearly every caption`)
  line(`    ${entries.filter((e) => e.confidence === 'likely').length} likely   a near universal tag unique to this LoRA`)
  line(`    ${entries.filter((e) => e.confidence === 'weak').length} weak     thin captions, or the folder name only`)
  line(`  ${noTrigger.length} have tag data and no trigger, so they run on weight alone`)
  line(`  ${noMeta.length} carry no metadata block at all, nothing can be said about them`)
  line(`  ${noTags.length} carry metadata but no ss_tag_frequency`)
  line()
  line('Triggers recovered:')
  for (const e of withTriggers) {
    const extra = e.triggers.length > e.promptTags.length ? `   (also captioned: ${e.triggers.map((t) => t.tag).join(', ')})` : ''
    line(`  ${e.confidence.padEnd(7)} ${e.base.padEnd(12)} ${e.triggerPhrase.padEnd(34)} ${e.stem}${extra}`)
  }
  line()
  line(`Shaky extractions, ${shaky.length} of ${entries.filter((e) => e.hasTagFrequency).length} taggable files. Check these before trusting the trigger:`)
  if (shaky.length === 0) line('  none')
  for (const e of shaky) {
    line(`  ${e.stem}`)
    line(`    trigger: ${e.triggerPhrase || '(none)'}`)
    for (const n of e.notes) line(`    ${n}`)
  }
  line()
  line('No trigger data in the file. The UI must say that, not "no trigger needed":')
  for (const e of [...noMeta, ...noTags]) line(`  ${e.hasMetadata ? 'no ss_tag_frequency' : 'no metadata block  '}  ${e.stem}`)
}

function main(): void {
  // Without this the first thing a wrong folder produced was readdirSync's
  // ENOENT stack trace, which names the folder but not the setting behind it.
  if (!existsSync(LORA_DIR)) {
    process.stderr.write(
      `No LoRA folder at ${LORA_DIR}. Set SWITCHGEN_MODELS to the models root, or SWITCHGEN_LORA_DIR to the folder ` +
        'itself. npm scripts do not read .env, so export them in the shell first. Nothing was written.\n',
    )
    process.exit(1)
  }
  const { entries, dirFingerprint } = build()
  const current = readFileSync(TARGET, 'utf8')
  const begin = current.indexOf(BEGIN)
  const end = current.indexOf(END)
  if (begin === -1 || end === -1 || end < begin) {
    process.stderr.write(
      `${TARGET} has no generated region. It must contain the marker lines\n  ${BEGIN}\n  ${END}\n` +
        'around the block this script owns. Nothing was written.\n',
    )
    process.exit(1)
  }
  const next = current.slice(0, begin) + emit(entries, dirFingerprint) + current.slice(end + END.length + 1)
  writeFileSync(TARGET, next)
  report(entries)
  process.stdout.write(`\nWrote the generated region of ${TARGET}\n`)
}

main()
