import type { LoraIndexEntry } from '../src/lib/loraIndex'
import type { LoraInfo, LoraLibrary } from '../src/lib/loras'

export const SCREENCAP = 'fine-anime-screencap-xl-anime-screencap-style-lora-illustrious-and-ponyxl.safetensors'
export const EXPLICIT = 'pony-nsfw-explicit-realistic-photography.safetensors'
export const NOOBAI = 'NoobAI-XL-v1.1.safetensors'

/** The two files the measured "Sharper faces and hands" set names, spelled as recipe.ts spells them. */
export const ANATOMY_HELPER = 'anatomy-helper.safetensors'
export const MICRO_DETAILS = 'add-micro-details-concept-illustrious-pony-noobai.safetensors'

/**
 * A library of add-ons, each marked installed when it is in `installed`. The
 * default is the two whose training tags are in FIXTURE_INDEX, which is what
 * the prompt-matched offers need; pass the helpers too for a measured set.
 */
export function library(installed: string[] = [SCREENCAP, EXPLICIT]): LoraLibrary {
  const rows: LoraInfo[] = [
    info(SCREENCAP, 'Fine anime screencap', 'anime', 'fine anime screencap, anime coloring, anime screencap'),
    info(EXPLICIT, 'Pony explicit photography', 'photoreal', 'explicit photography, v4n1lla, realistic'),
    info(ANATOMY_HELPER, 'Anatomy helper', 'anatomy', ''),
    info(MICRO_DETAILS, 'Add micro details', 'anatomy', 'addmicrodetails'),
  ].map((r) => ({ ...r, installed: installed.includes(r.file) }))
  return {
    all: rows,
    byFile: new Map(rows.map((r) => [r.file, r])),
    folder: 'Lora',
    installed: rows.filter((r) => r.installed).length,
    unlisted: 0,
    available: rows.filter((r) => !r.installed).length,
  }
}

function info(file: string, label: string, category: LoraInfo['category'], trigger: string): LoraInfo {
  return {
    file,
    label,
    installed: true,
    bytes: 200_000_000,
    approxBytes: false,
    arch: 'pony',
    category,
    priority: 5,
    bases: ['ponyDiffusionV6XL.safetensors', NOOBAI],
    claims: [],
    trigger,
    recommended: 0.8,
    slider: false,
    usage: 'both',
    does: `${label}, for the test suite.`,
  }
}

/**
 * The caption index rows the recipe reads for SCREENCAP and EXPLICIT: their
 * trigger words and the training vocabulary a prompt is matched against.
 *
 * The recipe tests used to read these from the checked-in src/lib/loraIndex.ts,
 * which `npm run index-loras` rewrites from whatever LoRA folder the machine
 * has. On any folder without these two files the offers came back empty and
 * the tests failed, or passed for nothing. The rows are trimmed copies of what
 * the indexer wrote for the real files, so the tests keep their meaning.
 */
export const FIXTURE_INDEX: LoraIndexEntry[] = [
  entry(SCREENCAP, {
    base: 'pony',
    imageCount: 104,
    confidence: 'likely',
    promptTags: ['fine anime screencap_xl', 'anime coloring', 'anime screencap'],
    triggers: [
      { tag: 'fine anime screencap_xl', count: 104, share: 1, kind: 'phrase', uniqueToThisLora: true },
      { tag: 'anime coloring', count: 104, share: 1, kind: 'phrase', uniqueToThisLora: false },
      { tag: 'anime screencap', count: 104, share: 1, kind: 'phrase', uniqueToThisLora: false },
    ],
    concepts: [
      { tag: 'school uniform', count: 21, share: 0.2019 },
      { tag: 'close-up', count: 20, share: 0.1923 },
      { tag: 'sunset', count: 3, share: 0.0288 },
    ],
  }),
  entry(EXPLICIT, {
    base: 'pony',
    imageCount: 1339,
    confidence: 'likely',
    promptTags: ['v4n1lla'],
    triggers: [{ tag: 'v4n1lla', count: 698, share: 0.5213, kind: 'invented', uniqueToThisLora: true }],
    concepts: [
      { tag: 'realistic', count: 1339, share: 1 },
      { tag: 'explicit photography', count: 698, share: 0.5213 },
      { tag: 'a woman', count: 581, share: 0.4339 },
      { tag: 'nude', count: 579, share: 0.4324 },
      { tag: 'breasts', count: 509, share: 0.3801 },
      { tag: 'lips', count: 506, share: 0.3779 },
      { tag: 'uncensored', count: 489, share: 0.3652 },
    ],
  }),
]

function entry(
  file: string,
  e: Pick<LoraIndexEntry, 'base' | 'imageCount' | 'confidence' | 'promptTags' | 'triggers' | 'concepts'>,
): LoraIndexEntry {
  const stem = file.replace(/\.safetensors$/, '')
  return {
    file,
    stem,
    bytes: 200_000_000,
    baseEvidence: 'the test suite',
    nameClaims: [],
    trainedOn: '',
    authorName: stem,
    networkDim: 8,
    hasMetadata: true,
    hasTagFrequency: true,
    datasetFolders: ['dataset'],
    triggerPhrase: e.promptTags.join(', '),
    notes: [],
    ...e,
  }
}
