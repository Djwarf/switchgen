/**
 * Night 2, editing and references (run exta-1).
 *
 * References come from the user's own photos: 'cat' (their cat, with a
 * description they can edit on the lab page) and 'scene' (an everyday room or
 * table, with a rectangle drawn on the lab page). The scene photo has not
 * arrived yet; until it is there, and a rectangle drawn on it, the plan names
 * the tests that wait for it and the run refuses to start.
 *
 * The picture models get a description, the editing model an instruction, so
 * the phone shows each set's neutral task instead of either wording.
 *
 * Chains are scored on their block's card as one more contestant, with an
 * automatic pair against the picture they started from. Five of them start
 * from a core picture, which is reused, not drawn again.
 */
import { defineSuite } from '../core/suite.ts'
import type { Slot } from '../core/types.ts'
import { CORE, MODELS, REFS, SAMPLER, SEEDS, SHAPES, STEPS, STUDY } from './first-pass.ts'

const WITH_EDITOR = [...CORE, 'qwenEdit']

const GROUP =
  'Four adult friends in their thirties sitting around a café table, all smiling at the camera, medium-wide shot.'

const slots: Slot[] = [
  {
    id: 'ref.cat.ghibli',
    block: 'reference',
    shape: 'square',
    op: 'i2i',
    source: { ref: 'cat' },
    denoise: 0.65,
    text: 'A Studio Ghibli style illustration of {describe}.',
    perModelText: {
      qwenEdit: "Turn this photo into a Studio Ghibli style illustration. Keep the cat's markings, pose and the setting.",
    },
    task: 'Restyle the reference as a Studio Ghibli illustration, keeping this cat.',
    models: WITH_EDITOR,
  },
  {
    id: 'ref.cat.oil',
    block: 'reference',
    shape: 'square',
    op: 'i2i',
    source: { ref: 'cat' },
    denoise: 0.65,
    text: 'An oil painting of {describe} in the style of Rembrandt, dramatic chiaroscuro.',
    perModelText: {
      qwenEdit: 'Make this an oil painting in the style of Rembrandt; keep the cat exactly as it is.',
    },
    task: 'Repaint the reference as a Rembrandt oil painting, keeping this cat.',
    models: WITH_EDITOR,
  },
  {
    id: 'character.cat.sofa',
    set: 'character.cat',
    block: 'character',
    shape: 'square',
    op: 'edit',
    source: { ref: 'cat' },
    text: 'Place this cat on a green velvet sofa by a sunny window.',
    condition: 'scene 1',
    models: 'edit',
  },
  {
    id: 'character.cat.library',
    set: 'character.cat',
    block: 'character',
    shape: 'square',
    op: 'edit',
    source: { ref: 'cat' },
    text: 'Show this cat sitting on a stack of old books in a library.',
    condition: 'scene 2',
    models: 'edit',
  },
  {
    id: 'edit.winter',
    block: 'edit',
    shape: 'square',
    op: 'edit',
    source: { ref: 'scene' },
    text: 'Make it look like a snowy winter day; change nothing else.',
    models: 'edit',
  },
  {
    id: 'edit.mono',
    block: 'edit',
    shape: 'square',
    op: 'edit',
    source: { ref: 'scene' },
    text: 'Turn it into a black-and-white photograph; change nothing else.',
    models: 'edit',
  },
  {
    id: 'region.scene',
    block: 'region',
    shape: 'square',
    op: 'region',
    source: { ref: 'scene' },
    denoise: 0.55,
    text: 'a small vase of yellow tulips',
    perModelText: {
      qwenEdit: 'Place a small vase of yellow tulips in the {position} of the picture; change nothing else.',
    },
    task: 'Put a small vase of yellow tulips in the outlined area and change nothing else.',
    models: WITH_EDITOR,
  },
  {
    id: 'detail.group',
    block: 'anatomy',
    shape: 'square',
    op: 't2i',
    text: GROUP,
    humans: true,
    models: 'core',
  },
  {
    id: 'detail.group.face',
    block: 'detail',
    shape: 'square',
    op: 'face',
    target: 'face',
    after: 'detail.group',
    text: GROUP,
    condition: 'with the face pass',
    humans: true,
    models: 'core',
  },
  {
    id: 'detail.group.hires',
    block: 'detail',
    shape: 'square',
    op: 'hires',
    after: 'detail.group',
    text: GROUP,
    condition: 'with the hires pass',
    humans: true,
    models: 'core',
  },
  {
    id: 'negative.party.without',
    set: 'negative.party',
    block: 'negative',
    shape: 'square',
    op: 't2i',
    text: 'A birthday party table with a cake and wrapped presents.',
    condition: 'without',
    models: 'core',
  },
  {
    id: 'negative.party.with',
    set: 'negative.party',
    block: 'negative',
    shape: 'square',
    op: 't2i',
    text: 'A birthday party table with a cake and wrapped presents.',
    negativeAdd: 'balloons',
    condition: 'with "balloons" in the negative',
    models: 'core',
  },
  {
    id: 'negative.beach.without',
    set: 'negative.beach',
    block: 'negative',
    shape: 'square',
    op: 't2i',
    text: 'A tropical beach at sunset.',
    condition: 'without',
    models: 'core',
  },
  {
    id: 'negative.beach.with',
    set: 'negative.beach',
    block: 'negative',
    shape: 'square',
    op: 't2i',
    text: 'A tropical beach at sunset.',
    negativeAdd: 'people, boats',
    condition: 'with "people, boats" in the negative',
    models: 'core',
  },
  {
    id: 'negative.words',
    block: 'following',
    shape: 'square',
    op: 't2i',
    // Asks for no one, so it is marked as showing no person.
    text: 'A tropical beach at sunset with no people and no boats.',
    checklist: ['no people', 'no boats'],
    humans: false,
    models: 'core',
  },
  {
    id: 'text.magazine',
    block: 'text',
    shape: 'tall',
    op: 't2i',
    // "an adult astronaut": every prompt that shows a person names an adult.
    text: 'A magazine cover titled "ORBIT" with the headline "Life on Mars?" and a photo of an adult astronaut.',
    expect: ['ORBIT', 'Life on Mars?'],
    humans: true,
    models: 'core',
  },
  {
    id: 'text.thumbnail',
    block: 'text',
    second: 'layout',
    shape: 'wide',
    op: 't2i',
    text: 'A video thumbnail of a man in his thirties looking shocked at a giant pizza, with bold yellow text that reads "I ATE 10 PIZZAS".',
    expect: ['I ATE 10 PIZZAS'],
    humans: true,
    models: 'core',
  },
]

export default defineSuite({
  id: 'ext-edit',
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
  chains: [
    {
      id: 'chain.text-fix',
      name: 'NoobAI, then Qwen-Edit fixes the sign',
      block: 'text',
      compose: { slot: 'text.bakery', model: 'noobai' },
      steps: [
        {
          model: 'qwenEdit',
          op: 'edit',
          text: 'Make the sign read exactly "FRESH BREAD DAILY" in the same painted style. Change nothing else.',
        },
      ],
    },
    {
      id: 'chain.photo-polish',
      name: 'Qwen-Image 2.1, then a light Chroma pass',
      block: 'photo',
      compose: { slot: 'photo.kitchen', model: 'qwen21' },
      steps: [{ model: 'chroma', op: 'i2i', denoise: 0.35, text: '{text}' }],
    },
    {
      id: 'chain.draft-upscale',
      name: 'Z-Image Turbo draft, then Z-Image Base at 2.25 megapixels',
      block: 'following',
      compose: { slot: 'following.fruit', model: 'zturbo' },
      steps: [{ model: 'zbase', op: 'i2i', denoise: 0.4, megapixels: 2.25, text: '{text}' }],
    },
    {
      id: 'chain.klein-faces',
      name: 'Klein, then Z-Image Base redraws the faces (lab-only graph)',
      block: 'anatomy',
      also: ['detail'],
      compose: { slot: 'detail.group', model: 'klein' },
      steps: [{ model: 'zbase', op: 'detailOnPicture', target: 'face', text: '{text}' }],
    },
    {
      id: 'chain.thumbnail-3',
      name: 'Qwen-Image 2.1 layout, NoobAI restyle, Qwen-Edit title',
      block: 'layout',
      compose: { slot: 'layout.thumbnail', model: 'qwen21' },
      steps: [
        { model: 'noobai', op: 'i2i', denoise: 0.5, text: '{text}, anime style' },
        {
          model: 'qwenEdit',
          op: 'edit',
          text: 'Add the title "TOP 10 SECRETS" in bold white letters on the yellow space.',
        },
      ],
    },
    {
      id: 'chain.cat-twice',
      name: 'Qwen-Edit places the cat, then moves it again',
      block: 'character',
      compose: { slot: 'character.cat.sofa', model: 'qwenEdit' },
      steps: [{ model: 'qwenEdit', op: 'edit', text: 'Show this cat sitting on a stack of old books in a library.' }],
    },
  ],
})
