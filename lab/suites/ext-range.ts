/**
 * Night 3, range (run extb-1; it resumes the next night if --until cuts it).
 *
 * Six styles on one subject, one subject in three frame shapes, two more
 * photographs, two harder prompt-following tests, a book cover layout, a
 * second defaults probe, a second content measure, and the prompt-style
 * pairs: booru tags against sentences for the seven tag-trained models. The
 * sentence halves of those pairs are the core's own words, so they are the
 * core's pictures, reused.
 */
import { defineSuite } from '../core/suite.ts'
import type { Slot } from '../core/types.ts'
import { CORE, FRUIT, KITCHEN, MODELS, REFS, SAMPLER, SEEDS, SHAPES, STEPS, STUDY, TAG_MODELS } from './first-pass.ts'

const FISHERMAN = 'An old fisherman in his seventies mending a net on a harbour wall, boats behind him'

const STYLES: { key: string; words: string; marks: string[] }[] = [
  {
    key: 'disney',
    words: 'as a still from a Disney-style 3D animated film',
    marks: ['smooth 3D rendering', 'appealing stylised proportions', 'soft cinematic light', 'saturated, friendly colour'],
  },
  {
    key: 'rembrandt',
    words: 'as an oil painting in the style of Rembrandt, dramatic chiaroscuro',
    marks: ['visible oil brushwork', 'deep dark ground', 'one warm light falling on the face', 'earthy, muted palette'],
  },
  {
    key: 'davinci',
    words: 'as a Leonardo da Vinci ink and sepia study on aged paper',
    marks: ['ink and sepia line only', 'hatching for shade', 'aged, stained paper', 'a study, not a finished painting'],
  },
  {
    key: 'ukiyoe',
    words: 'as a Japanese ukiyo-e woodblock print',
    marks: ['flat areas of colour', 'bold outlines', 'stylised waves and sky', 'woodblock texture'],
  },
  {
    key: 'watercolour',
    words: 'as a loose watercolour on textured paper',
    marks: ['transparent washes', 'soft bleeding edges', 'paper texture showing through', 'loose, few strokes'],
  },
  {
    key: 'pixel',
    words: 'as 16-bit pixel art',
    marks: ['visible square pixels', 'a small limited palette', 'no smooth gradients or blur', 'crisp pixel outlines'],
  },
]

const LIGHTHOUSE = 'A lighthouse on a rocky coast at dusk, waves breaking.'

const slots: Slot[] = [
  ...STYLES.map(
    (st): Slot => ({
      id: `style.range.${st.key}`,
      block: 'style',
      shape: 'square',
      op: 't2i',
      text: `${FISHERMAN}, ${st.words}.`,
      checklist: st.marks,
      humans: true,
      models: 'core',
    }),
  ),
  ...(['square', 'wide', 'tall'] as const).map(
    (shape): Slot => ({
      id: `shapes.lighthouse.${shape}`,
      set: 'shapes.lighthouse',
      block: 'shapes',
      shape,
      op: 't2i',
      text: LIGHTHOUSE,
      condition: shape,
      models: 'core',
    }),
  ),
  {
    id: 'photo.materials',
    block: 'photo',
    shape: 'square',
    op: 't2i',
    text: 'A product photograph of a glass perfume bottle on wet black slate, water droplets, a brushed steel lid, studio lighting.',
    models: 'core',
  },
  {
    id: 'photo.night',
    block: 'photo',
    shape: 'square',
    op: 't2i',
    text: 'A street photograph of a man in his forties in a wool coat waiting at a tram stop at night in the rain, neon reflections on wet cobblestones.',
    humans: true,
    models: 'core',
  },
  {
    id: 'following.binding',
    block: 'following',
    shape: 'square',
    op: 't2i',
    text: 'A man in a red hat and a green coat standing next to a woman in a blue scarf and a white coat, on a snowy street.',
    checklist: [
      'the man has the red hat',
      'the man has the green coat',
      'the woman has the blue scarf',
      'the woman has the white coat',
      'nothing swapped',
    ],
    humans: true,
    models: 'core',
  },
  {
    id: 'following.spatial',
    block: 'following',
    shape: 'square',
    op: 't2i',
    text: 'A black cat sitting on top of a stack of three books, a lit candle behind the books, a small potted cactus on the right.',
    checklist: ['a black cat', 'on top of the books', 'exactly three books', 'a lit candle behind the books', 'a small potted cactus on the right'],
    models: 'core',
  },
  {
    id: 'layout.cover',
    block: 'layout',
    shape: 'tall',
    op: 't2i',
    text: 'A book cover: a lone lighthouse at the bottom of the frame under a vast starry sky, the top third empty dark sky for a title, no text.',
    layout: 'The lighthouse at the bottom; the top third empty dark sky; no text.',
    models: 'core',
  },
  {
    id: 'defaults.house',
    block: 'defaults',
    shape: 'square',
    op: 't2i',
    text: 'a house',
    models: 'core',
  },
  {
    id: 'content.gym',
    block: 'content',
    shape: 'square',
    op: 't2i',
    text: 'An adult fitness instructor stretching in a bright studio.',
    humans: true,
    measuredOnly: true,
    models: 'core',
  },
  // Prompt style: the core's sentence against booru tags, for the models
  // trained on tags. The sentence member is the core's own words and model,
  // so its cells are the core's cells and are reused.
  {
    ...FRUIT,
    id: 'promptstyle.fruit.sentence',
    set: 'promptstyle.fruit',
    second: undefined,
    condition: 'sentence',
    models: TAG_MODELS,
  },
  {
    ...FRUIT,
    id: 'promptstyle.fruit.tags',
    set: 'promptstyle.fruit',
    second: undefined,
    text: '3 apples, red apple, 1 pear, green pear, blue mug, mug on left, yellow napkin, folded napkin, wooden table, morning light, still life',
    condition: 'tags',
    models: TAG_MODELS,
  },
  {
    ...KITCHEN,
    id: 'promptstyle.kitchen.sentence',
    set: 'promptstyle.kitchen',
    condition: 'sentence',
    models: TAG_MODELS,
  },
  {
    ...KITCHEN,
    id: 'promptstyle.kitchen.tags',
    set: 'promptstyle.kitchen',
    text: '1woman, mature female, 60s, laughing, kitchen, sunlight, window light, realistic, photo, skin texture',
    condition: 'tags',
    models: TAG_MODELS,
  },
]

export default defineSuite({
  id: 'ext-range',
  version: 1,
  study: STUDY,
  seeds: SEEDS,
  steps: STEPS,
  sampler: SAMPLER,
  shapes: SHAPES,
  models: MODELS,
  core: CORE,
  refs: REFS,
  slots,
  chains: [],
})
