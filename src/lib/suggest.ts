/**
 * suggest.ts
 *
 * Which LoRAs this prompt wants, why, and what has to be added to the prompt
 * for them to work at full strength.
 *
 *
 * THE MEASUREMENT THIS FILE EXISTS FOR.
 *
 * Measured on Pony V6 XL, same seed, add-micro-details at strength 0.6,
 * whole-frame Laplacian variance:
 *
 *     base, no LoRA                    165.9   1.000x
 *     add-micro-details, no trigger    269.5   1.625x
 *     add-micro-details, with trigger  353.8   2.132x
 *
 * Loading the weights gets 1.625x. Saying the word on top of that is worth a
 * further 1.313x and changes 67 percent of the pixels. So a suggestion that
 * names a LoRA and stops there has thrown away about a third of the effect.
 * Every suggestion this module produces carries the tokens that have to go into
 * the prompt, and {@link composePrompt} is the call that puts them there.
 *
 * Any other per LoRA figure quoted in this project, including the stack ratios
 * repeated in MEASURED_STRENGTH below, was measured WITHOUT triggers. Those are
 * floors, not results: they understate their LoRA by roughly a third. Every
 * sentence this file generates from one says so.
 *
 *
 * WHAT THIS IS AND IS NOT.
 *
 * It is a lookup over ./loraIndex, which was read out of the safetensors headers
 * of the installed files: the real training tag distribution, with counts. So
 * "your prompt says spread pussy, and 95 percent of this LoRA's training images
 * were captioned spread pussy" is a fact about the file, not a guess about the
 * picture.
 *
 * It is NOT a model of what the picture will look like. Nothing here has seen an
 * image. Two halves of the user's question are answered differently:
 *
 *   detect from the prompt    this module, exactly, with no download and no
 *                             server, because the tags are the LoRAs' own words
 *   detect from the image     not here. It needs a captioner. models/clip_vision
 *                             is empty and no WD14 node is installed, so that
 *                             half is a separate job, not a flag on this one.
 *
 * There is one soft middle: a prompt written as prose ("a woman kneeling on a
 * bed in evening light") uses none of the booru vocabulary these LoRAs were
 * trained on, and plain string matching is at its weakest there. A local LLM
 * used to be probed to rewrite such prose as tags; nothing ever listened on
 * the ports it tried, so that path is gone. The picture reader in vision.ts
 * covers the case from the other side: it tags the picture, not the prose.
 *
 *
 * THE BLIND SPOT, SAID OUT LOUD.
 *
 * 22 of the 61 installed files ship no tag data at all, so they can never be
 * ranked here, however well they would fit. They are not bad LoRAs and they are
 * not incompatible. They are invisible to this method, which is a property of
 * this method. {@link SuggestResult.blindSpot} carries the count so the UI can
 * say so instead of implying the other 39 are all there is.
 */
import { wantsExplicitAnatomy } from './intent'
import type { AnatomyLevel } from './recipe'
import type { FamilyDef } from './registry'
import {
  LORA_INDEX,
  byFilename,
  matchPrompt,
  missingTriggers,
  triggerNote,
  withoutTriggerData,
  type IndexedBase,
  type LoraIndexEntry,
  type TriggerConfidence,
} from './loraIndex'
import {
  ARCH_LABEL,
  STRENGTH,
  archFor,
  defaultStrength,
  type LoraArch,
  type LoraInfo,
  type LoraLibrary,
} from './loras'

// ---------------------------------------------------------------------------
// The ceiling
// ---------------------------------------------------------------------------

/**
 * How many LoRAs a suggestion set is allowed to propose, and why.
 *
 * This is the one number users would otherwise have to discover by ruining
 * pictures. Measured on Pony V6, same seed, same prompt, whole-frame Laplacian
 * variance against a 165.9 baseline, all three runs WITHOUT trigger words and
 * therefore floors:
 *
 *     anatomy-helper 0.4 + add-micro-details 0.6                    1.14x
 *     the same two plus real-nipples 0.5, micro-details up to 0.7   0.92x
 *
 * The third LoRA took the stack from above base sharpness to below it. That is
 * one measurement of one stack on one checkpoint, not a law, so the third slot
 * is allowed and flagged rather than forbidden.
 */
const STACK_CAP = {
  /** Hard ceiling on what {@link suggest} will put in `stack`. */
  max: 3,
  /** Past this, every further suggestion carries a caution. */
  adviseAt: 2,
  why:
    'Two measured LoRAs came out at 1.14x base sharpness. Adding a third took the same stack to 0.92x, which is below no LoRA at all. Both runs were made without trigger words, so both are floors.',
} as const

/**
 * The trigger measurement, in one place, so every sentence that cites it cites
 * the same numbers. Transcribed from the run described at the top of this file
 * and of ./loraIndex, not retyped from memory.
 */
const MEASURED_TRIGGER = {
  file: 'add-micro-details-concept-illustrious-pony-noobai.safetensors',
  measuredOn: 'ponyDiffusionV6XL.safetensors',
  strength: 0.6,
  baselineLaplacian: 165.9,
  withoutTrigger: 1.625,
  withTrigger: 2.132,
  pixelsChanged: 0.67,
  sentence:
    'On the one LoRA measured both ways, leaving its trigger out of the prompt dropped it from 2.132x base sharpness to 1.625x.',
} as const

// ---------------------------------------------------------------------------
// Strengths that came from a run rather than from a README
// ---------------------------------------------------------------------------

type MeasuredStrength = { strength: number; note: string }

/**
 * The only strengths in this project that were measured rather than read off an
 * author's model card. Duplicated here rather than imported from ./recipe on
 * purpose: recipe.ts imports this module, and a value import back the other way
 * would close a cycle that Vite resolves by handing one side an undefined
 * binding at module init.
 *
 * All of these were measured with no trigger word in the prompt, so each note
 * labels itself a floor.
 */
const MEASURED_STRENGTH: Record<string, MeasuredStrength> = {
  'add-micro-details-concept-illustrious-pony-noobai.safetensors': {
    strength: 0.6,
    note:
      'Measured on Pony V6 at 0.6: 1.625x base sharpness without its trigger, 2.132x with it. The 1.625x is a floor.',
  },
  'anatomy-helper.safetensors': {
    strength: 0.4,
    note:
      'Capped at 0.4 because it degrades sharpness as it rises: measured 0.812x base at 0.3, 0.718x at 0.5, 0.437x at 0.8. Those runs carried no trigger, so they are floors, and this LoRA has no trigger to carry.',
  },
  'real-nipples-and-areola-textures-gmr.safetensors': {
    strength: 0.5,
    note:
      'The strength used in the measured emphasised stack, which came out at 0.92x base sharpness. That run carried no trigger, so it is a floor, and this LoRA does have one.',
  },
}

/**
 * LoRAs worth proposing on evidence rather than on prompt overlap.
 *
 * add-micro-details is the single best evidenced LoRA installed, and prompt
 * matching would never find it: nobody types "addmicrodetails" and its training
 * concepts are texture words that do not appear in a description of a scene. A
 * ranking that can only surface what the prompt already names would bury the one
 * thing that was actually measured to help.
 */
const STAPLES: { file: string; bases: IndexedBase[]; why: string }[] = [
  {
    file: MEASURED_TRIGGER.file,
    bases: ['pony', 'illustrious', 'sdxl'],
    why:
      'Measured on Pony V6 at strength 0.6: 2.132x base sharpness with its word in the prompt, 1.625x without it. It is the only add-on here measured both ways.',
  },
]

/**
 * A staple enters the ranking at this score. A tuning constant, not a
 * measurement: it sits low enough that a prompt naming two of a LoRA's training
 * tags outranks it, and high enough that it survives the cut on a prompt that
 * names nothing.
 */
const STAPLE_SCORE = 1

// ---------------------------------------------------------------------------
// Input and output
// ---------------------------------------------------------------------------

/**
 * What counts as installed. A {@link LoraLibrary} from loadLoraLibrary() is the
 * normal case; a bare list of filenames is accepted for callers that only have
 * /api/models. Omitted, every indexed file is assumed present, which is true the
 * moment the index was built and slowly stops being true after that.
 */
export type InstalledLoras = LoraLibrary | Iterable<string>

export type SuggestInput = {
  /** The user's own words, exactly as typed. Never modified. */
  prompt: string
  /** The registry family chosen. Null or omitted means nothing is chosen yet. */
  family?: FamilyDef | null
  /**
   * Checkpoint filename. This, not the family, decides the architecture: one
   * family can carry weights of several lineages and a LoRA is patched onto the
   * weights, not onto the tab they were found under.
   */
  model?: string
  /** Skips family and model when the caller already knows the architecture. */
  arch?: LoraArch
  installed?: InstalledLoras
  /** The one anatomy control. Omitted, treated as 'natural'. */
  anatomy?: AnatomyLevel
  /** Files already in the user's stack. Not re-suggested, and counted against the cap. */
  already?: string[]
  /** Length of `ranked`. The cap on `stack` is {@link STACK_CAP}, separately. */
  limit?: number
}

export type SuggestReason =
  /** The prompt uses words this LoRA was trained on. */
  | 'prompt'
  /** A local model rewrote the prompt as tags, and those matched. */
  /** Proposed on a measurement rather than on prompt overlap. */
  | 'measured'
  /** The anatomy control asked for it. */
  | 'anatomy-level'

export type StrengthSource =
  /** From a run logged in this project. */
  | 'measured'
  /** The author's own recommendation, from the catalogue. */
  | 'author'
  /** Neither was available, so the project default. */
  | 'default'

export type SuggestHit = {
  /** The indexed tag that matched, in the LoRA's own spelling. */
  tag: string
  /** Share of this LoRA's training images captioned with it, 0 to 1. */
  share: number
  via: 'trigger' | 'concept'
  /** Where the tag came from. Only the reader's own text, now. */
  source: 'prompt'
}

export type Suggestion = {
  file: string
  label: string
  /** The index row, so a caller can show counts, notes and the full tag list. */
  entry: LoraIndexEntry
  /** The catalogue row, when a library was supplied. Null for unlisted files. */
  info: LoraInfo | null
  reason: SuggestReason
  /** Higher is a better fit. Ordering only: not a probability and not calibrated. */
  score: number
  /** The tags that earned the score, strongest first. Empty for staples. */
  hits: SuggestHit[]
  /**
   * The tokens that must be added to the prompt when this LoRA is applied.
   * Empty when the file says it has no trigger, and also empty when the file
   * says nothing at all. `triggerConfidence` tells those two apart.
   */
  trigger: string[]
  triggerConfidence: TriggerConfidence
  strength: number
  strengthSource: StrengthSource
  /** Load order slot. Lower is patched first. See {@link orderSlot}. */
  slot: number
  /** One or two plain sentences saying why this was suggested. Ready for the UI. */
  why: string
  /** Things that are true and unwelcome. Never folded into `why`. */
  cautions: string[]
}

export type Rejection = { file: string; label: string; why: string }

export type SuggestResult = {
  /** The architecture everything was checked against. */
  arch: LoraArch
  /** Every candidate above the cut, best first. */
  ranked: Suggestion[]
  /**
   * The subset to actually apply: at most {@link STACK_CAP}.max, in load order,
   * with concept duplicates dropped. This is what a one click Apply should use.
   */
  stack: Suggestion[]
  /** The tokens `stack` needs that the prompt does not already contain. */
  triggers: string[]
  /** Ranked candidates that were thrown out, with the reason. Show these. */
  rejected: Rejection[]
  /** Sentences about the method itself, not about any one LoRA. */
  notes: string[]
  /** How many installed files carry no tag data and so can never be ranked here. */
  blindSpot: number
}

// ---------------------------------------------------------------------------
// Base compatibility
//
// The hard gate, and the reason it is hard: LoraLoader does not refuse a LoRA
// trained for another architecture. It loads what keys match, which on a Flux
// UNet is roughly none, and renders a quietly poisoned picture with no error
// anywhere. Nothing crosses this except where the UNet key layout really is
// shared.
// ---------------------------------------------------------------------------

type Compat = { exact: IndexedBase[]; lineage: IndexedBase[] }

/**
 * Pony, Illustrious and plain SDXL share the SDXL UNet, so a crossing inside
 * that set loads and does part of its job. The conditioning differs, so it is a
 * discount and a caution, not a match. Everything else has no entry, which means
 * nothing indexed can be applied to it at all.
 *
 * A Wan model is known by its size, and the index knows Wan only as `wan`, so
 * every size reads that one base. Whether a given file suits the size is
 * `fitFor`'s question, which the add-on rack asks of every file it offers.
 */
const COMPAT: Partial<Record<LoraArch, Compat>> = {
  pony: { exact: ['pony'], lineage: ['illustrious', 'sdxl'] },
  illustrious: { exact: ['illustrious'], lineage: ['pony', 'sdxl'] },
  sdxl: { exact: ['sdxl'], lineage: ['pony', 'illustrious'] },
  flux1d: { exact: ['flux1d'], lineage: [] },
  wan: { exact: ['wan'], lineage: [] },
  'wan-14b': { exact: ['wan'], lineage: [] },
  'wan-5b': { exact: ['wan'], lineage: [] },
  'wan-1.3b': { exact: ['wan'], lineage: [] },
}

/** Weight on a lineage crossing. Half the effect is the honest guess; 0.75 is the score discount. */
const LINEAGE_FACTOR = 0.75

/**
 * Below this, a candidate is one incidental tag that under a quarter of its
 * training set carried, which is coincidence rather than evidence.
 */
const MIN_SCORE = 0.25

// ---------------------------------------------------------------------------
// Anatomy
// ---------------------------------------------------------------------------

/**
 * Explicit anatomy vocabulary, used to honour the anatomy control for files the
 * catalogue does not list. Named plainly because the index tags are named
 * plainly and a euphemism here would simply fail to match.
 */
const ANATOMY_TAGS = new Set([
  'pussy',
  'vulva',
  'vagina',
  'clitoris',
  'labia',
  'cervix',
  'nipples',
  'nipple',
  'areola',
  'areolae',
  'penis',
  'testicles',
  'anus',
  'anal',
  'pubic hair',
  'cum',
  'gaping',
  'spread pussy',
  'erection',
  'breasts',
])

/**
 * Words that put a person in the frame.
 *
 * Deliberately broad and deliberately neutral: an add-on trained on human skin
 * is equally wrong for a snow leopard whether the prompt is a clean portrait or
 * an explicit one, so this asks only "is there a person here", never "what kind
 * of picture is this". Explicit wording counts as a person and nothing more.
 */
const PERSON_WORDS = [
  'person', 'people', 'man', 'men', 'woman', 'women', 'girl', 'boy', 'lady', 'guy',
  'child', 'teen', 'adult', 'couple', 'human', 'figure', 'model', 'portrait',
  'face', 'facial', 'eyes', 'eye', 'skin', 'hand', 'hands', 'finger', 'fingers',
  'body', 'torso', 'chest', 'legs', 'arm', 'arms', 'shoulder', 'hair', 'lips',
  'mouth', 'smile', 'smiling', 'freckles', 'nude', 'naked', 'she', 'he', 'her',
  'his', 'him', 'herself', 'himself', 'selfie', 'headshot', 'bust',
]

/** True when the prompt puts a person in the picture. */
function describesPerson(prompt: string): boolean {
  const padded = ` ${prompt.toLowerCase().replace(/[^a-z0-9]+/g, ' ')} `
  if (PERSON_WORDS.some(w => padded.includes(` ${w} `))) return true
  // An explicit brief is a person brief, whatever else it names.
  return wantsExplicitAnatomy(prompt)
}

/**
 * True when an add-on only earns its keep on a human subject. Catalogued
 * 'anatomy' and 'hands' add-ons are trained on bodies, skin, faces and fingers;
 * applied to a landscape or an animal they spend capacity on features the
 * picture does not contain.
 */
function isPersonSpecific(entry: LoraIndexEntry, info: LoraInfo | null): boolean {
  if (info) return info.category === 'anatomy' || info.category === 'hands'
  // No catalogue row. Fall back to what the file calls itself: these add-ons
  // are named after the body parts they were trained on.
  const name = `${entry.stem} ${entry.triggerPhrase}`.toLowerCase()
  if (/\b(skin|hand|hands|finger|eyes|face|facial|anatomy|body|breast|nipple|areola|genital|pussy|penis|nude)\b/.test(name)) {
    return true
  }
  return isAnatomyLora(entry, info)
}

function isAnatomyLora(entry: LoraIndexEntry, info: LoraInfo | null): boolean {
  if (info) return info.category === 'anatomy'
  const top = [...entry.promptTags, ...entry.concepts.slice(0, 8).map(c => c.tag)]
  return top.some(t => ANATOMY_TAGS.has(t.toLowerCase()))
}

/**
 * True when the add-on's own vocabulary is explicit: its name, its trigger,
 * its top training tags, or the catalogue's description of it. Judged with the
 * same word list a prompt is judged with, so the two verdicts cannot drift
 * apart: what counts as explicit in a brief counts as explicit in a file.
 */
function isExplicitLora(entry: LoraIndexEntry, info: LoraInfo | null): boolean {
  const text = [
    entry.stem,
    entry.triggerPhrase,
    ...entry.promptTags,
    ...entry.concepts.slice(0, 8).map(c => c.tag),
    info?.does ?? '',
  ].join(' ')
  return wantsExplicitAnatomy(text)
}

/**
 * Load order. Each LoraLoader patches the model the previous one produced, so a
 * detail LoRA placed before an anatomy LoRA gets painted over. The measured
 * natural stack ran anatomy-helper then add-micro-details, and this reproduces
 * that order rather than inventing one.
 */
function orderSlot(entry: LoraIndexEntry, info: LoraInfo | null): number {
  const name = `${entry.stem} ${entry.triggerPhrase}`.toLowerCase()
  if (/detail|quality|sharp|skin|hands|eyes|face|realism/.test(name)) return 2
  if (info && (info.category === 'anime' || info.category === 'photoreal')) return 1
  if (isAnatomyLora(entry, info)) return 0
  return 1
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function stemOf(file: string): string {
  return file.replace(/\.safetensors$/i, '')
}

/** The base as a reader's phrase. `sd15` has no LoraArch label, so it falls through to itself. */
function baseLabel(entry: LoraIndexEntry): string {
  return ARCH_LABEL[entry.base as LoraArch] ?? entry.base
}

function labelOf(entry: LoraIndexEntry, info: LoraInfo | null): string {
  if (info) return info.label
  const words = entry.stem.replace(/[-_]+/g, ' ').replace(/\s+/g, ' ').trim()
  return words.charAt(0).toUpperCase() + words.slice(1)
}

function pct(share: number): number {
  return Math.round(share * 100)
}

/** "a, b and c", because a bare comma list reads like machine output in a sentence. */
function listOf(items: string[]): string {
  if (items.length <= 1) return items[0] ?? ''
  return `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`
}

function isLibrary(x: InstalledLoras): x is LoraLibrary {
  return typeof (x as LoraLibrary).byFile?.get === 'function'
}

/**
 * The set of filenames that really are on disk, or null for "assume all of them".
 * Null is the honest answer when the caller told us nothing: the index was built
 * by scanning the folder, so every entry was present at build time.
 */
function installedSet(installed?: InstalledLoras): Set<string> | null {
  if (!installed) return null
  const out = new Set<string>()
  const add = (file: string) => {
    out.add(file.toLowerCase())
    out.add(stemOf(file).toLowerCase())
  }
  if (isLibrary(installed)) {
    for (const info of installed.all) if (info.installed) add(info.file)
  } else {
    for (const file of installed) if (typeof file === 'string' && file) add(file)
  }
  return out
}

function infoFor(installed: InstalledLoras | undefined, file: string): LoraInfo | null {
  if (!installed || !isLibrary(installed)) return null
  return installed.byFile.get(file) ?? null
}

function strengthFor(entry: LoraIndexEntry, info: LoraInfo | null): {
  strength: number
  source: StrengthSource
  note: string
} {
  const measured = MEASURED_STRENGTH[entry.file]
  if (measured) return { strength: measured.strength, source: 'measured', note: measured.note }
  if (info) {
    return {
      strength: defaultStrength(info),
      source: 'author',
      note: `Strength ${defaultStrength(info)} is the author's own recommendation, not a figure measured here.`,
    }
  }
  return {
    strength: STRENGTH.default,
    source: 'default',
    note: `No measured strength and no catalogue entry for this file, so it starts at the project default of ${STRENGTH.default}.`,
  }
}

/**
 * "Its word, X, is added to the prompt for you." The one sentence for that
 * fact, on the desk's offers and on the region bench alike. Some add-ons
 * answer to several captions at once, and those are called words.
 */
export function addedSentence(phrase: string): string {
  const words = phrase.trim()
  if (!words) return ''
  return words.includes(',')
    ? `Its words, ${words}, are added to the prompt for you.`
    : `Its word, ${words}, is added to the prompt for you.`
}

/**
 * The trigger sentence for one LoRA. The five confidence levels say genuinely
 * different things and collapsing them is the mistake the index was built to
 * prevent: "no trigger data" is not "no trigger needed", and treating it as one
 * silently drops a third of the effect.
 *
 * This is read on the desk, word for word, under each add-on offered, so it
 * speaks of the add-on and its word. It is the same sentence the region bench
 * prints for the same fact (recipe.ts, authorNote).
 */
function triggerSentence(entry: LoraIndexEntry): string {
  switch (entry.confidence) {
    case 'strong':
    case 'likely':
      return addedSentence(entry.triggerPhrase)
    case 'weak':
    case 'none':
    case 'no-data':
      return triggerNote(entry)
  }
}

// ---------------------------------------------------------------------------
// suggest
// ---------------------------------------------------------------------------

/**
 * Rank the installed LoRAs against a prompt.
 *
 * Synchronous and pure: no network, no probe, no await. Everything it needs was
 * read out of the LoRA files when the index was built. {@link suggestAsync}
 * wraps this with the optional local LLM path and changes nothing else.
 *
 * The score is
 *
 *     prompt hits x base factor x confidence factor x level factor
 *
 * and every one of those four is visible in the returned object: `hits` carries
 * the tags and their training shares, `cautions` carries the base crossing and
 * the weak trigger, and `why` is the same information as a sentence. It orders
 * candidates. It is not a probability and it is not calibrated against anything.
 */
export function suggest(input: SuggestInput): SuggestResult {
  const prompt = input.prompt.trim()
  const anatomy = input.anatomy ?? 'natural'
  const arch = input.arch ?? archFor(input.family ?? null, input.model ?? '')
  const onDisk = installedSet(input.installed)
  const already = new Set((input.already ?? []).map(f => f.toLowerCase()))
  const notes: string[] = []
  const rejected: Rejection[] = []

  const compat = COMPAT[arch]
  const unchecked = arch === 'unknown' && !input.model

  if (!compat && !unchecked) {
    // Nothing indexed was trained for this architecture, and crossing into it is
    // not a partial effect, it is noise with no error message.
    notes.push(
      `Nothing installed was trained for ${ARCH_LABEL[arch]}, so no LoRA is suggested. A LoRA from another architecture loads without complaining and renders a poisoned picture rather than failing, so none is offered.`,
    )
    return {
      arch,
      ranked: [],
      stack: [],
      triggers: [],
      rejected,
      notes,
      blindSpot: withoutTriggerData().length,
    }
  }

  if (unchecked) {
    notes.push('No checkpoint chosen yet, so nothing here has been checked for base compatibility.')
  }

  const bases = compat ? [...compat.exact, ...compat.lineage] : undefined
  const exact = new Set(compat?.exact ?? [])

  // One pass over the index: the reader's own words against each file's
  // training vocabulary.
  const fromPrompt = matchPrompt(prompt, { bases, minScore: 0 })

  type Bucket = { entry: LoraIndexEntry; relevance: number; hits: SuggestHit[] }
  const buckets = new Map<string, Bucket>()

  for (const m of fromPrompt) {
    let bucket = buckets.get(m.entry.file)
    if (!bucket) {
      bucket = { entry: m.entry, relevance: 0, hits: [] }
      buckets.set(m.entry.file, bucket)
    }
    bucket.relevance += m.score
    for (const h of m.hits) {
      if (bucket.hits.some(existing => existing.tag === h.tag)) continue
      bucket.hits.push({ tag: h.tag, share: h.share, via: h.via, source: 'prompt' })
    }
  }

  // The staples and the anatomy level: candidates justified by a run rather than
  // by prompt overlap, which is how the one LoRA that was actually measured gets
  // onto a ranking built out of word matching.
  const forced = new Map<string, { reason: SuggestReason; why: string; score: number }>()

  for (const staple of STAPLES) {
    if (!compat) continue
    if (!staple.bases.includes(arch as IndexedBase)) continue
    forced.set(staple.file, { reason: 'measured', why: staple.why, score: STAPLE_SCORE })
  }

  if (anatomy !== 'off' && compat) {
    forced.set('anatomy-helper.safetensors', {
      reason: 'anatomy-level',
      why:
        'Asked for by the anatomy setting. In the measured pair of add-ons it held 1.14x base sharpness alongside add-micro-details, and on its own it costs sharpness at every strength tried, which is why it is capped at 0.4. Both runs were made without the add-ons’ words in the prompt, so both are floors.',
      score: STAPLE_SCORE * 0.9,
    })
  }

  const out: Suggestion[] = []

  const consider = (
    entry: LoraIndexEntry,
    relevance: number,
    hits: SuggestHit[],
    reason: SuggestReason,
    forcedWhy?: string,
  ) => {
    const info = infoFor(input.installed, entry.file)
    const label = labelOf(entry, info)

    if (already.has(entry.file.toLowerCase())) return
    if (onDisk && !onDisk.has(entry.file.toLowerCase())) {
      rejected.push({ file: entry.file, label, why: 'Indexed but not in the add-ons folder now, so ComfyUI cannot load it.' })
      return
    }

    // The hard gate, applied to staples too. A LoRA trained for another
    // architecture is not a weaker LoRA: LoraLoader finds almost no matching
    // keys, reports nothing, and the picture comes back quietly poisoned.
    if (compat && !compat.exact.includes(entry.base) && !compat.lineage.includes(entry.base)) {
      rejected.push({
        file: entry.file,
        label,
        why: `Trained on ${baseLabel(entry)}, and ${ARCH_LABEL[arch]} is a different architecture. It would load without an error and render noise rather than failing, so it is not offered.`,
      })
      return
    }

    const cautions: string[] = []
    const anatomical = isAnatomyLora(entry, info)

    // THE SUBJECT GATE.
    //
    // This used to read `anatomy === 'off' && anatomical`, which hid every
    // body-related add-on behind a global setting while happily recommending a
    // human skin-hands-eyes add-on for a snow leopard, on the strength of both
    // prompts containing the word 'photography'. Whether the picture has a
    // person in it is a question about the subject, and it is asked the same
    // way for every subject: a hands add-on is wrong for a landscape for
    // exactly the reason a landscape add-on would be wrong for a portrait.
    if (isPersonSpecific(entry, info) && !describesPerson(input.prompt)) {
      rejected.push({
        file: entry.file,
        label,
        why: `${label} is trained on people, and nothing in this prompt names a person. It would spend its capacity on skin, hands and faces the picture does not contain.`,
      })
      return
    }

    // THE CONTENT GATE.
    //
    // Whether there is a person in the picture and whether the reader wants
    // explicit content are two different questions, and the gate above answers
    // only the first. Replacing the old setting check with it alone let an
    // explicit add-on reach the offers for any portrait. An explicit add-on is
    // offered when the brief asks for it, by the anatomy setting at its top
    // level or by explicit words in the prompt itself. At Standard and at
    // "Sharper faces and hands" a portrait is a portrait. This is judged on
    // the file's own vocabulary and not on its catalogue shelf: an explicit
    // style add-on is as explicit as an explicit anatomy one.
    if (isExplicitLora(entry, info) && anatomy !== 'emphasised' && !wantsExplicitAnatomy(prompt)) {
      rejected.push({
        file: entry.file,
        label,
        why: `${label} is an explicit add-on. Nothing in this prompt asks for that and the anatomy setting is not "Also explicit anatomy", so it is not offered.`,
      })
      return
    }

    // Base. The index base comes from the file's own header; the catalogue arch
    // comes from a README. Where they disagree the file wins, and the
    // disagreement is worth saying rather than resolving in silence.
    let baseFactor = 1
    if (compat && !exact.has(entry.base)) {
      baseFactor = LINEAGE_FACTOR
      cautions.push(
        `Trained on ${baseLabel(entry)} and applied to ${ARCH_LABEL[arch]}. Both are SDXL lineage, so the UNet keys match and it loads and does part of its job, but the conditioning differs and the effect is partial.`,
      )
    }
    if (info && info.arch !== entry.base && info.arch !== 'unknown') {
      cautions.push(
        `The catalogue calls this ${ARCH_LABEL[info.arch]} and the file's own header says ${baseLabel(entry)}. The header was read off the weights, so it is the one trusted here.`,
      )
    }

    // Two kinds of trigger do two different things to a prompt. A minted token
    // such as p3p05y or rnct means nothing to the base model, so adding it
    // addresses the LoRA and nothing else. An ordinary tag such as cervix or
    // gaping pussy is vocabulary the base model already knows, so adding it
    // changes what gets drawn as well as how. The reader should be told which
    // one is about to be put in front of their sentence.
    if (entry.promptTags.some(tag => entry.triggers.find(t => t.tag === tag)?.kind === 'phrase')) {
      cautions.push(
        `Its trigger is ordinary vocabulary rather than a minted token, so adding ${entry.triggerPhrase} to the prompt changes what gets drawn and not only how it is drawn.`,
      )
    }

    let confidenceFactor = 1
    if (entry.confidence === 'weak') {
      // A wrong trigger is worse than no trigger: it spends prompt on a token
      // the model has no meaning for.
      confidenceFactor = 0.9
      cautions.push(`Its trigger rests on thin evidence. ${entry.notes[0] ?? ''}`.trim())
    }

    let levelFactor = 1
    if (anatomical && anatomy === 'emphasised') levelFactor = 1.2

    const score = relevance * baseFactor * confidenceFactor * levelFactor
    if (score < MIN_SCORE) return

    const s = strengthFor(entry, info)
    if (s.source !== 'measured') cautions.push(s.note)
    if (info?.slider) {
      cautions.push('A slider LoRA: the sign picks a direction rather than an amount, and it has no trigger word.')
    }
    if (info?.caution) cautions.push(info.caution)

    hits.sort((a, b) => b.share - a.share || a.tag.localeCompare(b.tag))

    const why = forcedWhy
      ? `${forcedWhy} ${triggerSentence(entry)}`
      : `${matchSentence(hits, entry)} ${triggerSentence(entry)}`

    out.push({
      file: entry.file,
      label,
      entry,
      info,
      reason,
      score: Number(score.toFixed(4)),
      hits,
      trigger: entry.promptTags.slice(),
      triggerConfidence: entry.confidence,
      strength: s.strength,
      strengthSource: s.source,
      slot: orderSlot(entry, info),
      why: why.trim(),
      cautions: cautions.filter(Boolean),
    })
  }

  for (const bucket of buckets.values()) {
    if (forced.has(bucket.entry.file)) continue
    consider(bucket.entry, bucket.relevance, bucket.hits, 'prompt')
  }

  for (const [file, f] of forced) {
    const entry = byFilename(file)
    if (!entry) continue
    const bucket = buckets.get(file)
    // A staple that the prompt also happens to name keeps both justifications,
    // and the prompt overlap only raises it.
    consider(entry, f.score + (bucket?.relevance ?? 0), bucket?.hits ?? [], f.reason, f.why)
  }

  out.sort((a, b) => b.score - a.score || a.file.localeCompare(b.file))
  const ranked = input.limit ? out.slice(0, input.limit) : out

  // The stack. Capped, deduplicated by concept, and put in load order.
  const stack: Suggestion[] = []
  const claimed = new Set<string>()
  const room = Math.max(0, STACK_CAP.max - already.size)
  for (const s of ranked) {
    if (stack.length >= room) break
    // A weak trigger is the one case where applying a LoRA automatically could
    // make the picture worse than not applying it: the token goes into the
    // prompt, the model has no meaning for it, and it spends attention on
    // nothing. Ranked, so it can be chosen on purpose. Never chosen for anyone.
    if (s.triggerConfidence === 'weak') {
      rejected.push({
        file: s.file,
        label: s.label,
        why: `Ranked but not applied automatically: its trigger rests on weak evidence, and a wrong trigger poisons a prompt with a token the model cannot read. ${triggerNote(s.entry)}`,
      })
      continue
    }
    const tags = s.hits.map(h => h.tag)
    // Two LoRAs trained on the same handful of tags fight over the same weights
    // and the second one mostly cancels the first.
    if (tags.length && tags.every(t => claimed.has(t))) {
      rejected.push({
        file: s.file,
        label: s.label,
        why: `Covers the same tags as a higher ranked LoRA already in the stack: ${listOf(tags.slice(0, 3))}.`,
      })
      continue
    }
    for (const t of tags) claimed.add(t)
    if (stack.length >= STACK_CAP.adviseAt) {
      s.cautions.push(`Number ${stack.length + 1} in the stack. ${STACK_CAP.why}`)
    }
    stack.push(s)
  }
  stack.sort((a, b) => a.slot - b.slot || b.score - a.score)

  const triggers = missingTriggers(stack.map(s => s.file), prompt)

  if (stack.length) {
    const wanted = stack.some(s => s.trigger.length > 0)
    if (triggers.length) {
      notes.push(
        `${triggers.length === 1 ? 'One token gets' : `${triggers.length} tokens get`} added to the prompt for this stack: ${listOf(triggers)}. ${MEASURED_TRIGGER.sentence}`,
      )
    } else if (wanted) {
      notes.push('Every trigger this stack needs is already in your prompt, so nothing is added to it.')
    } else {
      notes.push('Nothing in this stack has a trigger word, so your prompt is sent exactly as you wrote it.')
    }
  }
  const blind = withoutTriggerData().length
  if (blind) {
    notes.push(
      `${blind} of the ${LORA_INDEX.length} installed files carry no training tags in their headers, so they can never be ranked here however well they would fit. Pick those by hand.`,
    )
  }

  return { arch, ranked, stack, triggers, rejected, notes, blindSpot: blind }
}

/**
 * The sentence that says why a prompt match happened, built from the tags
 * themselves so it can be checked against the LoRA's own file.
 */
function matchSentence(hits: SuggestHit[], entry: LoraIndexEntry): string {
  const top = hits.slice(0, 3)
  if (!top.length) return `Ranked against ${entry.imageCount || 'its'} training images.`
  const words = listOf(top.map(h => h.tag))
  const shares = listOf(top.map(h => `${pct(h.share)} percent`))
  return `Your prompt uses ${words}, which its training captions carried in ${shares} of ${entry.imageCount} images.`
}
