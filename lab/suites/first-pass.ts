/**
 * Night 1, the core: every installed picture model on the same thirteen
 * prompts, four seeds each (run core-1).
 *
 * All thirteen picture models are in the first pass, by the user's decision:
 * the four SDXL checkpoints, the three Anima mixes, Krea2, Chroma, Klein,
 * Qwen-Image 2.1 and both Z-Images. 13 models x 13 prompts x 4 seeds = 676
 * pictures. The editing model (Qwen-Image-Edit 2511) joins the editing,
 * reference and chain tests of ext-edit.
 *
 * The shared settings live here and the other three suites import them, so
 * a picture made tonight is the same picture (the same cell) wherever another
 * night asks for it.
 */
import { defineSuite } from '../core/suite.ts'
import type { ModelSpec, ShapeName, Slot, Suite } from '../core/types.ts'

export const STUDY = 'first-pass'

/** The seeds, in grid order: 1001 top left, 4004 bottom right. */
export const SEEDS = [1001, 2002, 3003, 4004]
export const STEPS = 28
/** The common sampler. Klein keeps its own Flux2Scheduler; euler is its only sampler choice. */
export const SAMPLER = { name: 'euler', scheduler: 'simple' }
export const SHAPES: Record<ShapeName, [number, number]> = {
  square: [1024, 1024],
  wide: [1344, 768],
  tall: [768, 1344],
}

/** Every contestant, by key. The keys never reach the phone. */
export const MODELS: Record<string, ModelSpec> = {
  noobai: { file: 'NoobAI-XL-v1.1.safetensors', role: 'generate' },
  semireal: { file: 'semiRealIllustrious_v40.safetensors', role: 'generate' },
  pony: { file: 'ponyDiffusionV6XL.safetensors', role: 'generate' },
  wai: { file: 'waiMatureIllustrious_v30.safetensors', role: 'generate' },
  oneObsession: { file: 'oneObsession_anima29BV1.safetensors', role: 'generate' },
  miaomiaoHarem: { file: 'miaomiaoHarem_29BBETA10.safetensors', role: 'generate' },
  miaomiaoRealskin: { file: 'miaomiaoRealskin_anima13.safetensors', role: 'generate' },
  krea2: { file: 'moodyCutieMixKrea2_v50_int8.safetensors', role: 'generate' },
  chroma: { file: 'Chroma1-HD-fp8mixed.safetensors', role: 'generate' },
  klein: { file: 'flux-2-klein-4b-fp8.safetensors', role: 'generate' },
  qwen21: { file: 'qwen_image_2.1_int8_convrot.safetensors', role: 'generate' },
  zbase: { file: 'Z-Image-Base-bf16.safetensors', role: 'generate' },
  zturbo: { file: 'Z-Image-Turbo-fp8mix.safetensors', role: 'generate' },
  qwenEdit: { file: 'qwen-image-edit-2511-Q4_K_M.gguf', role: 'edit' },
}

/** The thirteen picture models, in the order their groups are sent. */
export const CORE = [
  'noobai',
  'semireal',
  'pony',
  'wai',
  'oneObsession',
  'miaomiaoHarem',
  'miaomiaoRealskin',
  'krea2',
  'chroma',
  'klein',
  'qwen21',
  'zbase',
  'zturbo',
]

/** The models trained on booru tags, for the prompt-style pairs. */
export const TAG_MODELS = ['noobai', 'semireal', 'pony', 'wai', 'oneObsession', 'miaomiaoHarem', 'miaomiaoRealskin']

/**
 * The reference photos. The cat's description is the user's starting one,
 * which they can change on the lab page. The scene (an everyday room or
 * table) has not arrived yet; the nights that need it say so and refuse to
 * start until it is there.
 */
export const REFS: Suite['refs'] = {
  cat: { describe: 'a tortoiseshell tabby cat with a white belly and white paws, wearing a dark collar' },
  scene: { describe: 'an everyday room or table', needsMask: true },
}

/** The core's slots, exported so the calibration and the prompt-style pairs use the exact same words. */
export const FRUIT: Slot = {
  id: 'following.fruit',
  block: 'following',
  second: 'variation',
  shape: 'square',
  op: 't2i',
  text: 'Three red apples and one green pear on a wooden table, a blue mug to the left of the fruit, a folded yellow napkin in front of the fruit, soft morning light.',
  checklist: [
    'exactly three apples',
    'the apples are red',
    'exactly one green pear',
    'a blue mug left of the fruit',
    'a folded yellow napkin in front',
    'a wooden table',
  ],
  models: 'core',
}

export const KITCHEN: Slot = {
  id: 'photo.kitchen',
  block: 'photo',
  shape: 'square',
  op: 't2i',
  text: 'A candid photograph of a woman in her sixties laughing in a sunlit kitchen, natural skin texture, soft window light, 50mm lens.',
  humans: true,
  models: 'core',
}

export const BAKERY: Slot = {
  id: 'text.bakery',
  block: 'text',
  shape: 'square',
  op: 't2i',
  text: 'A vintage bakery storefront with a hand-painted sign above the window that reads "FRESH BREAD DAILY".',
  expect: ['FRESH BREAD DAILY'],
  models: 'core',
}

export const THUMBNAIL: Slot = {
  id: 'layout.thumbnail',
  block: 'layout',
  second: 'shapes',
  shape: 'wide',
  op: 't2i',
  text: 'A video thumbnail: a surprised man in his thirties on the right third of the frame pointing to the left, the left half an empty bright yellow background kept clear for a title, no text.',
  layout: 'The man on the right third, pointing left; the left half empty and yellow; no text.',
  humans: true,
  models: 'core',
}

const CHARACTER =
  'A woman in her twenties with short silver hair, round tortoiseshell glasses and a yellow raincoat'

export const CORE_SLOTS: Slot[] = [
  FRUIT,
  {
    id: 'style.ghibli',
    block: 'style',
    shape: 'square',
    op: 't2i',
    text: 'A red fox resting under a cherry tree beside a small wooden shrine, as a Studio Ghibli film background painting.',
    checklist: ['hand-painted gouache look', 'soft natural palette', 'painterly clouds and foliage', 'gentle light'],
    models: 'core',
  },
  KITCHEN,
  BAKERY,
  THUMBNAIL,
  {
    id: 'anatomy.hands',
    block: 'anatomy',
    shape: 'square',
    op: 't2i',
    text: "Close-up of an adult pianist's two hands on the piano keys, all ten fingers visible, natural window light.",
    humans: true,
    models: 'core',
  },
  {
    id: 'anatomy.dancer',
    block: 'anatomy',
    second: 'shapes',
    shape: 'tall',
    op: 't2i',
    text: 'A full-length photograph of a flamenco dancer, a woman in her thirties in a red dress, mid-turn with both arms raised, on a wooden stage.',
    humans: true,
    models: 'core',
  },
  {
    id: 'character.cafe',
    set: 'character',
    block: 'character',
    shape: 'square',
    op: 't2i',
    // The whole description again in both scenes: a model has no memory, so
    // "the same woman" would test nothing.
    text: `${CHARACTER}, reading a book in a café.`,
    condition: 'scene 1',
    humans: true,
    models: 'core',
  },
  {
    id: 'character.bike',
    set: 'character',
    block: 'character',
    shape: 'square',
    op: 't2i',
    text: `${CHARACTER}, cycling through a rainy city street.`,
    condition: 'scene 2',
    humans: true,
    models: 'core',
  },
  {
    id: 'sensitivity.red',
    set: 'sensitivity',
    block: 'sensitivity',
    shape: 'square',
    op: 't2i',
    text: 'A red vintage car parked beside a lake in autumn.',
    condition: 'red',
    models: 'core',
  },
  {
    id: 'sensitivity.blue',
    set: 'sensitivity',
    block: 'sensitivity',
    shape: 'square',
    op: 't2i',
    text: 'A blue vintage car parked beside a lake in autumn.',
    condition: 'blue',
    models: 'core',
  },
  {
    id: 'defaults.person',
    block: 'defaults',
    second: 'variation',
    shape: 'square',
    op: 't2i',
    // The one prompt without an adult named: it asks what a model draws
    // unprompted. The picture reader's quarantine rule backs it up.
    text: 'a person',
    defaultsProbe: true,
    models: 'core',
  },
  {
    id: 'content.beach',
    block: 'content',
    shape: 'square',
    op: 't2i',
    text: 'A woman in her thirties relaxing on a beach towel, reading a book, on a summer afternoon.',
    humans: true,
    measuredOnly: true,
    models: 'core',
  },
]

export default defineSuite({
  id: 'first-pass-core',
  version: 1,
  study: STUDY,
  seeds: SEEDS,
  steps: STEPS,
  sampler: SAMPLER,
  shapes: SHAPES,
  models: MODELS,
  core: CORE,
  refs: REFS,
  slots: CORE_SLOTS,
  chains: [],
})
