/**
 * The recipe: three answers in, every decision out.
 *
 * The compose screen used to ask for forty two things. It now asks for three:
 * a prompt, a look, and how much anatomy. This module is what makes that
 * possible. It takes those three answers and derives everything the old form
 * collected by hand: which family, which weight file, which LoRAs at which
 * strengths, the prompt prefix that file was trained with, the sampler, the
 * size, and which quality passes are worth offering afterwards.
 *
 * Nothing here is new capability. Every decision is delegated:
 *
 *   intent.ts    ranks the installed files against the brief
 *   hardware.ts  removes anything this machine cannot hold, before ranking
 *   loras.ts     says which LoRA fits which checkpoint, and resolves the stack
 *   refine.ts    inserts the LoRA chain and reports what passes a graph allows
 *   workflows.ts holds the graphs and the per model defaults
 *
 * Two rules govern the numbers in this file.
 *
 * First, a strength is either MEASURED or it is the author's own recommendation
 * from the catalogue, and every LoRA the recipe returns says which of the two
 * it is. Nothing is picked because it felt right.
 *
 * Second, the measurement is Laplacian variance, which is SHARPNESS. It is not
 * anatomical correctness. A LoRA can draw a hand correctly and soften the skin
 * around it, and this metric will call that a loss. The ratios below justify
 * default strengths. They are not a claim that the picture is more correct, and
 * no copy produced here says that they are.
 */
import { feasibility, type Hardware, type ModelFile, type Verdict } from './hardware'
import type { PassBlocks } from './availability'
import {
  intentReport,
  wantsExplicitAnatomy,
  type Brief,
  type Intent,
  type IntentReport,
  type Recommendation,
} from './intent'
import {
  EMPTY_LIBRARY,
  archFor,
  defaultStrength,
  fitFor,
  resolveStack,
  targetFor,
  triggersFor,
  type LoraInfo,
  type LoraLibrary,
  type LoraStack,
  type LoraTarget,
} from './loras'
import { addedSentence, suggest as suggestLoras } from './suggest'
import { byFilename as indexedLora } from './loraIndex'

/**
 * The token a suggested LoRA answers to, read from its own training captions.
 * Only 'strong' and 'likely' are used: a guessed trigger is worse than none,
 * because it puts a word in the prompt the model was never taught.
 */
function indexedTrigger(file: string): string | null {
  const e = indexedLora(file)
  if (!e || !e.triggerPhrase) return null
  return e.confidence === 'strong' || e.confidence === 'likely' ? e.triggerPhrase : null
}
import { canTakeLoras, capabilitiesOf, detailSentence, withLoras, type Capabilities, type DerivedDef } from './refine'
import { FAMILIES, defaultsFor, deriveImg2Img, IMG2IMG, instantiate, type FamilyDef, type Params } from './workflows'

// ---------------------------------------------------------------------------
// The three questions
// ---------------------------------------------------------------------------

/**
 * The looks the default screen offers. `Intent` carries a fourth, cartoon,
 * which stays reachable through the full panel: it is a small slice of the work
 * done here and it does not earn a chip on a screen with three of them.
 */
export type Look = 'photoreal' | 'anime' | 'illustration'

export const LOOKS: readonly { id: Look; label: string; blurb: string }[] = [
  { id: 'photoreal', label: 'Photoreal', blurb: 'Skin, light and lens.' },
  { id: 'anime', label: 'Anime', blurb: 'Cel shading and booru tags.' },
  { id: 'illustration', label: 'Illustration', blurb: 'Painted, drawn, semi real.' },
] as const

/** The one anatomy control. Everything the old rack did, in three words. */
export type AnatomyLevel = 'off' | 'natural' | 'emphasised'

export const ANATOMY_LEVELS: readonly { id: AnatomyLevel; label: string; blurb: string }[] = [
  { id: 'off', label: 'Standard', blurb: 'Nothing extra.' },
  {
    id: 'natural',
    label: 'Sharper faces and hands',
    blurb: 'Adds two helpers for body structure and skin texture. Measured 1.14x sharper.',
  },
  {
    id: 'emphasised',
    label: 'Also explicit anatomy',
    blurb: 'Adds a nude-detail helper on top. Measured 0.92x, softer than no helper at all.',
  },
] as const

// ---------------------------------------------------------------------------
// The measurements
//
// Transcribed from the handoff run, not retyped from memory. Every field that
// reaches the screen comes from here, so there is exactly one place to check a
// number against the log that produced it.
// ---------------------------------------------------------------------------

const ANATOMY_HELPER = 'anatomy-helper.safetensors'
const MICRO_DETAILS = 'add-micro-details-concept-illustrious-pony-noobai.safetensors'
const REAL_NIPPLES = 'real-nipples-and-areola-textures-gmr.safetensors'

/** The checkpoint every ratio below was measured on. */
export const MEASURED_ON = 'ponyDiffusionV6XL.safetensors'

export type MeasuredStack = {
  entries: readonly (readonly [file: string, strength: number])[]
  laplacian: number
  /** Laplacian variance divided by the no LoRA baseline on the same seed. */
  ratio: number
  verdict: string
}

/**
 * Declared separately so the two element rows keep their tuple type. Inlining
 * them in the object below widens each row to (string | number)[] and the
 * strength stops being a number the compiler can see.
 */
const NATURAL_STACK: MeasuredStack = {
  entries: [
    [ANATOMY_HELPER, 0.4],
    [MICRO_DETAILS, 0.6],
  ],
  laplacian: 189.2,
  ratio: 1.14,
  verdict: 'Sharper than using none, while still helping with bodies and hands.',
}

const EMPHASISED_STACK: MeasuredStack = {
  entries: [
    [ANATOMY_HELPER, 0.4],
    [REAL_NIPPLES, 0.5],
    [MICRO_DETAILS, 0.7],
  ],
  laplacian: 152.4,
  ratio: 0.919,
  verdict:
    'Worse than using none. Three body add-ons together overrun what the detail one repays, so more is not better here.',
}

export const MEASURED = {
  method:
    'Laplacian variance over the whole frame. Pony V6 XL at 832x1216, 28 steps, CFG 7, dpmpp_2m with karras, seed 99 held constant and the prompt unchanged.',
  metricCaveat:
    'This measures sharpness, not whether a body came out right. A drop is a reason to look, never proof that an add-on is wrong.',
  baselineLaplacian: 165.9,
  /** anatomy-helper degrades monotonically. This is why it is capped at 0.4. */
  helperCurve: [
    { strength: 0.3, ratio: 0.812 },
    { strength: 0.5, ratio: 0.718 },
    { strength: 0.8, ratio: 0.437 },
  ] as const,
  /** The only measured restorer: it pays sharpness back rather than spending it. */
  microDetailsAlone: { strength: 0.6, ratio: 1.625 },
  stacks: { natural: NATURAL_STACK, emphasised: EMPHASISED_STACK },
  /**
   * The automatic face pass, measured on the same metric and the same seed.
   * It is the reason no detail pass runs on its own in this module.
   */
  faceDetailer: { ratio: 0.714 },
  /** Rule 5 of the handoff: these figures are Pony's, not Illustrious's. */
  transferNote:
    'Tested on Pony V6. Illustrious and NoobAI take the same add-ons but were not tested, so treat those strengths as a starting point.',
} as const

/** Denoise and pixel budget the picture desk already uses for image to image. */
const I2I = { denoise: 0.65, megapixels: 1 } as const

// ---------------------------------------------------------------------------
// What comes back
// ---------------------------------------------------------------------------

export type NoteKind = 'model' | 'anatomy' | 'prompt' | 'passes' | 'hardware' | 'source'

/** One decision, said in one sentence, with its provenance attached. */
export type RecipeNote = {
  kind: NoteKind
  text: string
  /** True only when the sentence rests on a number from {@link MEASURED}. */
  measured: boolean
}

export type RecipeLora = {
  file: string
  label: string
  strength: number
  /** True when this strength comes from a measured run, false when it is the author's. */
  measured: boolean
  why: string
}

/** A pass the result can offer once the picture exists. */
export type PassOffer = {
  /** The family's graph supports it and ComfyUI has what it loads. False means the button is not drawn. */
  available: boolean
  /** Whether {@link decide} thinks it should run without being asked. */
  auto: boolean
  why: string
  /**
   * The graph could carry it, but a file or node pack it loads is missing
   * here, said in one sentence that names it. Absent when nothing is missing.
   */
  blocked?: string
}

export type RecipePasses = {
  face: PassOffer
  hand: PassOffer
  /** Masked region re render, the only thing that fixes a region with no detector. */
  refine: PassOffer
  /** Render it bigger. */
  hires: PassOffer
}

export type Plan = {
  look: Look
  anatomy: AnatomyLevel
  prompt: string

  /** The family and weight file chosen, and the reader facing names for both. */
  familyId: string
  model: string
  label: string
  familyLabel: string
  /** The registry family, before anything is derived from it. */
  base: FamilyDef
  /** What to instantiate: image to image and the LoRA chain already applied. */
  def: FamilyDef | DerivedDef
  /** Ready for instantiate(def, params). */
  params: Params

  /** The LoRA chain in load order, each with its provenance. */
  loras: RecipeLora[]
  /** The predicted sharpness ratio, present only when a measured stack was used. */
  sharpness: { ratio: number; verdict: string; onMeasuredModel: boolean } | null
  /** Wanted by the chosen level but not on disk. Shown, never silently skipped. */
  missingLoras: { file: string; label: string; why: string }[]
  /** Region LoRAs held for the refine pass, where close framing is what they need. */
  refineLoras: RecipeLora[]
  /**
   * Matched from your wording and NOT applied. These are offers: the reader
   * accepts one and it joins {@link loras}, or ignores it and nothing happens.
   * Previously these were pushed straight into the chain, which is the
   * "auto parameters" complaint: the app decided, printed what it had decided,
   * and gave the reader no moment to disagree before the press.
   */
  offers: RecipeLora[]

  passes: RecipePasses
  capabilities: Capabilities
  /**
   * The memory verdict for the graph this plan builds: the chosen file, not
   * the family's default, with every add-on in the chain. Null when the
   * machine has not been measured.
   */
  verdict: Verdict | null

  /** Every decision, in the order it was made. */
  notes: RecipeNote[]
  /** Things that are true and unwelcome. Never hidden behind More. */
  warnings: string[]
  /** The full ranking, for the More panel. Rank 1 is what was chosen. */
  report: IntentReport
}

/** Nothing installed can do this here. Carries the reason and the ranking. */
export type NoPlan = {
  ok: false
  look: Look
  anatomy: AnatomyLevel
  prompt: string
  reason: string
  report: IntentReport
  notes: RecipeNote[]
  warnings: string[]
}

/**
 * Discriminated on `ok`, so a caller writes
 * `if (!recipe.ok) return <p>{recipe.reason}</p>` and the compiler keeps it
 * honest from there.
 */
export type Recipe = ({ ok: true } & Plan) | NoPlan

export type RecipeInput = {
  prompt: string
  look: Look
  anatomy: AnatomyLevel
  /** Uploaded filename. Present means the desk is working from a picture. */
  sourceImage?: string
  /** From probeHardware(). Null before the probe lands: ranking then skips feasibility. */
  hardware?: Hardware | null
  /** From modelFiles(). Needed alongside hardware for feasibility. */
  sizes?: Map<string, ModelFile>
  /** Checkpoint filenames ComfyUI reports present. Omitted, every family is considered. */
  installed?: Iterable<string>
  /** From loadLoraLibrary(). Omitted, no LoRA can be resolved and the recipe says so. */
  loras?: LoraLibrary
  /**
   * The reader's own decisions about offered add-ons, by filename. `accepted`
   * are applied as if measured-for-this-prompt; `declined` are never offered
   * again for this composition. Anything in neither list is still an open offer.
   * decide() is pure, so these must be threaded in rather than remembered here:
   * that is what makes a choice survive the next recompute instead of being
   * quietly overwritten by the next suggestion.
   */
  addOns?: { accepted?: readonly string[]; declined?: readonly string[] }
  /** Omitted, one is rolled. */
  seed?: number
  /**
   * What stops a quality pass from running on this ComfyUI, from
   * availability.passBlocks. A pass with a sentence here is not offered, and
   * the sentence says what to install.
   */
  passBlocks?: Partial<PassBlocks>
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

const note = (kind: NoteKind, text: string, measured = false): RecipeNote => ({ kind, text, measured })

/** The booru trained SDXL lineage. The only weights the anatomy LoRAs patch. */
const BOORU = new Set(['pony', 'illustrious', 'sdxl'])

const pct = (r: number) => `${r.toFixed(2)}x`

function rollSeed(): number {
  return Math.floor(Math.random() * 0xffffffff)
}

function labelOf(lib: LoraLibrary, file: string): string {
  return lib.byFile.get(file)?.label ?? file.replace(/\.safetensors$/i, '').replace(/[-_]+/g, ' ')
}

/**
 * The prompt prefix a weight file was trained with, read out of the registry.
 *
 * Every file that needs one declares it as `positivePrefix` in perModel. This
 * used to fall back to scraping `score_\d+` tags out of Pony's prose notes,
 * which worked by accident and would have broken the moment the note was
 * reworded; Pony and NoobAI now declare theirs like their siblings, so the
 * scrape is gone. A file with no declared prefix gets no prefix.
 */
function prefixFor(def: FamilyDef, model: string): { tokens: string[]; why: string } {
  const per = (def.perModel?.[model] ?? {}) as Record<string, unknown>
  const declared = per.positivePrefix
  if (typeof declared === 'string' && declared.trim()) {
    const tokens = declared
      .split(',')
      .map(t => t.trim())
      .filter(Boolean)
    return { tokens, why: 'Quality prefix this file was trained with, from the registry.' }
  }
  return { tokens: [], why: '' }
}

/**
 * Words fit to go into a prompt, out of the words the add-ons answer to.
 *
 * The catalogue's hand-written words were scraped from model cards, and a few
 * rows carry the card's markdown instead, such as "## 🧠 Usage (Python)". Sent
 * as a prompt word it conditions the picture on a heading, and printed on the
 * bench it tells the reader that heading is the add-on's word. A word that
 * opens like a heading, runs over a line, or is the card's usage title is
 * left out, and the add-on runs without a word, as it does when it has none.
 */
export function plainWords(words: readonly string[]): string[] {
  return words.filter((w) => {
    const t = w.trim()
    return !!t && !t.startsWith('#') && !/[\r\n]/.test(t) && !/\busage\s*\(/i.test(t)
  })
}

/**
 * The four passes, as the result will offer them. A pass the graph can carry
 * is still withheld when ComfyUI lacks a file it loads, and the sentence that
 * names the file goes with it, so the panel can say why the row is missing.
 */
export function passesFor(
  capabilities: Capabilities,
  why: Record<keyof RecipePasses, string>,
  blocks: Partial<PassBlocks> = {},
): RecipePasses {
  const one = (can: boolean, text: string, blocked: string | null | undefined): PassOffer =>
    can && blocked ? { available: false, auto: false, why: text, blocked } : { available: can, auto: false, why: text }
  return {
    face: one(capabilities.faceDetail, why.face, blocks.face),
    hand: one(capabilities.handDetail, why.hand, blocks.hand),
    refine: one(capabilities.refine, why.refine, blocks.refine),
    hires: one(capabilities.hires, why.hires, null),
  }
}

/** The LoRA stack a level asks for, before anything is checked against disk. */
function wantedFor(level: AnatomyLevel): MeasuredStack | null {
  if (level === 'natural') return MEASURED.stacks.natural
  if (level === 'emphasised') return MEASURED.stacks.emphasised
  return null
}

/**
 * The architecture the measured stack was measured on. MEASURED_ON names the
 * checkpoint; this is its arch, used to break ties between booru carriers so
 * the printed figure describes the stack that actually runs.
 */
const MEASURED_ARCH = 'pony' as const

/** True when this checkpoint can carry the anatomy LoRAs at all. */
function carriesAnatomy(def: FamilyDef, model: string): boolean {
  return canTakeLoras(def) && BOORU.has(archFor(def, model))
}

/** The image to image derivation of a family, or null when it has none. */
function img2imgOf(def: FamilyDef): FamilyDef | null {
  return IMG2IMG[def.id] ?? deriveImg2Img(def)
}

// ---------------------------------------------------------------------------
// Reading the questions out of the prompt
// ---------------------------------------------------------------------------

/**
 * The anatomy level a prompt implies, for prefilling the control.
 *
 * Explicit wording gets `natural`, which is the measured stack that comes out
 * sharper than no LoRA at all. It never proposes `emphasised`: that stack
 * measured below base, so it is a choice the reader makes, not one made for
 * them.
 */
export function suggestAnatomy(prompt: string): AnatomyLevel {
  return wantsExplicitAnatomy(prompt) ? 'natural' : 'off'
}

// ---------------------------------------------------------------------------
// decide
// ---------------------------------------------------------------------------

/**
 * Turn three answers into every decision the old form collected by hand.
 *
 * Order matters and is deliberate:
 *
 *   1. build the brief, reading explicitness from the anatomy control and from
 *      the prompt, so a nude asked for in words is ranked as a nude
 *   2. rank what is installed and will fit in this machine's memory
 *   3. narrow to families that can do image to image, when there is a source
 *   4. prefer a checkpoint that can carry the anatomy LoRAs, when any were asked
 *      for, and say out loud what that displaced
 *   5. resolve the measured stack against what is on disk
 *   6. shape the prompt with the file's own prefix and the LoRAs' triggers
 *   7. decide nothing about quality passes, and report what the result may offer
 */
export function decide(input: RecipeInput): Recipe {
  const prompt = input.prompt.trim()
  const anatomy = input.anatomy
  const look = input.look
  const lib = input.loras ?? EMPTY_LIBRARY
  const notes: RecipeNote[] = []
  const warnings: string[] = []

  // 1. The brief. Detail and explicitness are SEPARATE questions and must not
  // share a switch. This line once read `anatomy !== 'off' || wantsExplicit...`,
  // which made "I want better hands and faces" the identical input as "I want
  // porn": intent.ts weightsFor() swings anatomy 0.05 -> 0.40 and craft 0.35 ->
  // 0.15 on this boolean, so asking for detail demoted the craft models
  // (Chroma, Z-Image, Flux, Qwen) in favour of booru bases. So 'natural', the
  // "Sharper faces and hands" level, never sets it. 'emphasised' is labelled
  // "Also explicit anatomy" and MUST set it, or the reader's clearest possible
  // statement of intent is discarded because they did not also type an explicit
  // word. The prompt sets it too, which is what keeps a nude described in words
  // off a base that was trained without any.
  const explicit = anatomy === 'emphasised' || wantsExplicitAnatomy(prompt)
  const brief: Brief = { intent: look as Intent, explicit, mode: 'image' }

  // 2. Rank. feasibility() runs inside intentReport when hardware and sizes are
  // both present, so a model that cannot run here is never offered, not merely
  // ranked low.
  const report = intentReport(brief, {
    families: FAMILIES,
    installed: input.installed,
    sizes: input.sizes,
    hardware: input.hardware ?? null,
  })

  if (!input.hardware || !input.sizes) {
    notes.push(
      note('hardware', 'Machine not measured yet, so this ranking has not been checked against free memory.'),
    )
  }

  // 3. A source picture removes every family whose graph cannot be rewired to
  // sample from an encoded image. Falling back to text to image here would
  // silently throw the reader's picture away.
  let pool = report.ranked
  if (input.sourceImage) {
    const able = pool.filter(r => img2imgOf(r.def) !== null)
    if (able.length < pool.length) {
      notes.push(
        note('source', 'Working from your picture, so only families that can sample from an encoded image were considered.'),
      )
    }
    pool = able
  }

  let pick: Recommendation | undefined = pool[0]
  if (!pick) {
    return {
      ok: false,
      look,
      anatomy,
      prompt,
      reason: input.sourceImage
        ? 'Nothing installed can work from a picture here. Open More: it says what is missing or too large, and its catalogue can fetch a model.'
        : report.note,
      report,
      notes,
      warnings,
    }
  }

  // 4. Anatomy routing. The anatomy LoRAs are SDXL booru files; they do not
  // attach to Chroma, Z-Image, Flux or Qwen. When the reader asked for anatomy
  // help, a base that can carry it beats one that cannot, and the base that was
  // displaced is named rather than quietly dropped.
  if (anatomy !== 'off' && !carriesAnatomy(pick.def, pick.model)) {
    // THE LOOK PICKS THE MODEL. THE DETAIL SETTING NEVER DOES.
    //
    // This block used to REPLACE `pick` with a booru base whenever the detail
    // setting was on, so asking for sharper faces on a photoreal brief quietly
    // swapped Flux or Chroma for Pony and returned a painterly anime picture.
    // The reader had chosen a look; a detail slider outranked it.
    //
    // Now the model the look earned is kept, the add-ons that cannot attach are
    // simply not applied, and the trade is stated. A reader who actually wants
    // the booru base can pick it themselves - that is a choice, not a silent
    // substitution.
    const carriers = pool.filter(r => carriesAnatomy(r.def, r.model))
    const carrier =
      carriers.find(r => archFor(r.def, r.model) === MEASURED_ARCH) ?? carriers[0]
    if (carrier) {
      notes.push(
        note(
          'model',
          `${pick.label} suits the look you asked for, so it is what renders. The face and hand helpers are SDXL booru files and will not attach to it, so they are not applied. ${carrier.label} would take them, if you would rather have the helpers than this look.`,
        ),
      )
    } else {
      // A warning, not a note: the reader asked for anatomy help and none of
      // it will be applied. That is the kind of thing a quiet line under the
      // fold is for hiding, and it is exactly what must not be hidden.
      warnings.push(
        `${pick.label} renders this on its own. The face and hand helpers are SDXL booru files and nothing installed here can carry them.`,
      )
    }
  }

  notes.push(note('model', `${pick.label}: ${pick.why}`))
  if (pick.caveat) notes.push(note('model', pick.caveat))
  if (pick.warning) warnings.push(pick.warning)
  // The memory warning waits for step 7b, where the graph that will run is
  // priced with its add-ons on it rather than the file on its own.

  const baseDef = pick.def
  const model = pick.model
  const target: LoraTarget = targetFor(baseDef, model)

  // 5. The stack. Resolved against what is actually on disk and against this
  // checkpoint's architecture, so a stack saved before a model change cannot
  // reach the queue with a LoRA that patches nothing.
  const wanted = wantedFor(anatomy)
  const loras: RecipeLora[] = []
  const missingLoras: Plan['missingLoras'] = []
  let stack: LoraStack = []
  let sharpness: Plan['sharpness'] = null
  /** The sentence quoting the figure, withdrawn in step 7 if the reader adds to the set. */
  let figureNote: RecipeNote | null = null

  if (wanted && carriesAnatomy(baseDef, model)) {
    const entries: { file: string; strength: number; enabled: boolean }[] = []
    for (const [file, strength] of wanted.entries) {
      const info = lib.byFile.get(file)
      if (!info) {
        missingLoras.push({ file, label: labelOf(lib, file), why: 'Not in your add-ons folder, and no details known for it.' })
        continue
      }
      if (!info.installed) {
        missingLoras.push({ file, label: info.label, why: 'In the catalogue, not downloaded yet.' })
        continue
      }
      const fit = fitFor(info, target)
      if (fit.level === 'mismatch') {
        missingLoras.push({ file, label: info.label, why: fit.why })
        continue
      }
      entries.push({ file, strength, enabled: true })
    }
    stack = entries

    // No `pass` argument. The catalogue files add-micro-details and
    // real-nipples as refine pass LoRAs, and in general that is right: they
    // were trained on close framing. The measured run that produced these
    // ratios applied them in the first render, and the measurement is what the
    // numbers on screen refer to, so this resolve matches the run.
    const resolved = resolveStack(stack, lib, target)
    for (const w of resolved.warnings) warnings.push(`${w.label}: ${w.why}`)
    for (const d of resolved.dropped) missingLoras.push({ file: d.file, label: d.label, why: d.why })

    for (const spec of resolved.specs) {
      loras.push({
        file: spec.name,
        label: labelOf(lib, spec.name),
        strength: spec.strength,
        measured: true,
        why:
          spec.name === MICRO_DETAILS
            ? `The only measured restorer: ${pct(MEASURED.microDetailsAlone.ratio)} as sharp as no add-ons, on its own at ${MEASURED.microDetailsAlone.strength}.`
            : spec.name === ANATOMY_HELPER
              ? `Capped at 0.4. It degrades monotonically above that: ${pct(0.718)} at 0.5 and ${pct(0.437)} at 0.8.`
              : 'Part of the measured stack for this level.',
      })
    }

    const complete = resolved.specs.length === wanted.entries.length
    if (complete) {
      sharpness = {
        ratio: wanted.ratio,
        verdict: wanted.verdict,
        onMeasuredModel: model === MEASURED_ON,
      }
      figureNote = note(
        'anatomy',
        `These add-ons measured ${pct(wanted.ratio)} as sharp as the same picture with none of them. ${wanted.verdict}`,
        true,
      )
      notes.push(figureNote)
      if (model !== MEASURED_ON) notes.push(note('anatomy', MEASURED.transferNote, true))
      if (anatomy === 'emphasised') {
        warnings.push(
          `"Also explicit anatomy" measured ${pct(MEASURED.stacks.emphasised.ratio)} as sharp as no add-ons, below the ${pct(MEASURED.stacks.natural.ratio)} of "Sharper faces and hands". Three body add-ons cost more sharpness than the detail add-on pays back. More is not better here.`,
        )
      }
    } else if (loras.length) {
      notes.push(
        note(
          'anatomy',
          'Part of the measured stack is missing, so no sharpness figure applies to what will actually run.',
        ),
      )
    }
  } else if (wanted) {
    notes.push(
      note(
        'anatomy',
        `${pick.label} cannot take those add-ons: they were made for a different kind of model. Your detail setting changes nothing for this render.`,
      ),
    )
  } else {
    notes.push(note('anatomy', 'No body add-ons. The model is drawing this on its own.'))
  }

  if (missingLoras.length) {
    warnings.push(
      `Not applied: ${missingLoras.map(m => m.label).join(', ')}. Download them under Add-ons and they start being used.`,
    )
  }

  // Region LoRAs are held back rather than stacked. They were trained on close
  // framing and do almost nothing at whole body scale, which is exactly the
  // crop the refine pass renders. Offering them on the result puts the choice
  // where the reader can see whether the region needs it.
  const refineLoras = heldForRefine(lib, target, new Set(loras.map(l => l.file)))

  // 6. The graph. Image to image first; the add-on chain is inserted in 6b
  // below, once every add-on is known, so the chain goes into the graph that
  // will actually run and carries everything the record will say it did.
  let def: FamilyDef | DerivedDef = baseDef
  if (input.sourceImage) {
    const i2i = img2imgOf(baseDef)
    if (i2i) def = i2i
    else warnings.push(`${pick.label} cannot sample from a picture, so this will render from the prompt alone.`)
  }

  // 7. The prompt. The prefix and the trigger tokens are the two pieces of
  // trivia the old form made the reader carry.
  const d = defaultsFor(baseDef, model)
  const per = (baseDef.perModel?.[model] ?? {}) as Record<string, unknown>
  const prefix = prefixFor(baseDef, model)
  const lower = prompt.toLowerCase()
  const resolvedFiles = loras.map(l => l.file)
  // THE PROMPT GETS A VOTE.
  //
  // The anatomy level alone cannot know that 'anime screencap, 90s retro'
  // wants the 90s aesthetic LoRA, or that 'freckles, natural skin texture'
  // wants the skin one. suggest() scores the prompt against each LoRA's real
  // training vocabulary and returns picks with their triggers.
  //
  // It is additive and capped: the measured anatomy stack stays exactly as it
  // was. Anything the prompt suggests on top is marked measured:false,
  // because no run measured it, and accepting one withdraws the figure below.
  let promptPicks: { file: string; strength: number; why: string }[] = []
  try {
    // The LIBRARY, not a list of filenames. infoFor() returns null for a bare
    // iterable, so every catalogue fact would be thrown away. The call is typed
    // as written: a cast used to sit here, and that cast is how that bug hid.
    const sug = suggestLoras({ prompt, model, installed: lib, anatomy, already: resolvedFiles })
    promptPicks = sug.stack
      .filter(s => !resolvedFiles.includes(s.file))
      .slice(0, 2)
      .map(s => ({ file: s.file, strength: s.strength, why: s.why || 'Suggested by your prompt.' }))
  } catch {
    // suggestion is a bonus, never a dependency: a failure here must not stop a render
    promptPicks = []
  }
  // OFFERS, NOT DECISIONS.
  //
  // These used to be pushed straight into `loras`. The reader asked for a
  // picture and silently got two extra add-ons and their trigger words in the
  // prompt. Now a pick is applied only if the reader has accepted it, hidden
  // only if they have declined it, and otherwise carried out as an open offer
  // the desk can print beside the button.
  //
  // The reader's decisions arrive as input, not as state kept here, so an
  // accepted add-on survives every recompute. That is the whole point: a
  // control that silently reverts is worse than no control.
  const accepted = new Set(input.addOns?.accepted ?? [])
  const declined = new Set(input.addOns?.declined ?? [])
  const offers: RecipeLora[] = []
  const appliedPicks: typeof promptPicks = []
  for (const pick of promptPicks) {
    if (declined.has(pick.file)) continue
    const row: RecipeLora = {
      file: pick.file,
      label: labelOf(lib, pick.file),
      strength: pick.strength,
      measured: false,
      why: pick.why,
    }
    if (accepted.has(pick.file)) {
      loras.push(row)
      appliedPicks.push(pick)
    } else {
      offers.push(row)
    }
  }
  // KEPT BECAUSE YOU ADDED IT.
  //
  // An accepted add-on is a decision, and a decision must not evaporate the
  // moment the wording drifts away from the vocabulary that first suggested
  // it. A "no" already survived every recompute; a "yes" only survived while
  // the prompt kept matching. Anything accepted that the prompt no longer
  // names is carried on the reader's say-so, subject to the two checks that
  // are not a matter of opinion: the file is on disk, and it fits the model.
  for (const file of accepted) {
    if (declined.has(file)) continue
    if (appliedPicks.some(p => p.file === file) || resolvedFiles.includes(file)) continue
    const info = lib.byFile.get(file)
    if (!info || !info.installed) {
      notes.push(note('prompt', `${labelOf(lib, file)} was added earlier but is not in the add-ons folder now, so it is left out.`))
      continue
    }
    const fit = fitFor(info, target)
    if (fit.level === 'mismatch') {
      notes.push(note('prompt', `${info.label} was added earlier but does not fit ${pick.label}, so it is left out. ${fit.why}`))
      continue
    }
    const row: RecipeLora = {
      file,
      label: info.label,
      strength: defaultStrength(info),
      measured: false,
      why: 'Kept because you added it. Your wording no longer matches it, so this is your choice rather than a suggestion.',
    }
    loras.push(row)
    appliedPicks.push({ file, strength: row.strength, why: row.why })
  }
  if (appliedPicks.length) {
    notes.push(note('prompt', `You added ${appliedPicks.length} add-on` +
      `${appliedPicks.length === 1 ? '' : 's'} matched to your wording: ` +
      `${appliedPicks.map(p => labelOf(lib, p.file)).join(', ')}. ` +
      `Matched from training vocabulary, not measured.`))
  }
  // The figure describes the measured set and nothing else. Once the reader
  // adds to it, the chain that runs is one nobody measured, so the number is
  // withdrawn rather than left standing over a stack it does not describe.
  if (appliedPicks.length && sharpness) {
    sharpness = null
    if (figureNote) notes.splice(notes.indexOf(figureNote), 1)
    notes.push(
      note('anatomy', 'You added add-ons on top of the measured set, so no sharpness figure applies to what will actually run.'),
    )
  }

  // 6b. The chain. Only now is every add-on known: the measured stack, the
  // offers the reader accepted, and the ones carried on their say-so. This
  // used to run before the offers were resolved, so an accepted add-on got its
  // trigger word in the prompt and its name in the record while the queued
  // graph carried no loader for it. The button did nothing to the picture.
  if (loras.length) {
    const chained = withLoras(def, loras.map(l => ({ name: l.file, strength: l.strength })))
    if (chained) def = chained
    else warnings.push(`${pick.label} cannot take add-ons, so none were applied.`)
  }

  // Suggested LoRAs need their triggers as much as measured ones do. An
  // untriggered stack measured 0.786x of base, ie worse than using none, so
  // adding a LoRA without its token actively harms the picture.
  const suggestedTriggers = appliedPicks.flatMap(pick => {
    const e = indexedTrigger(pick.file)
    return e ? [e] : []
  })
  const triggers = plainWords([...triggersFor(stack, lib, target), ...suggestedTriggers])
    .filter((t, n, a) => a.indexOf(t) === n)
    .filter(t => !lower.includes(t.toLowerCase()))
  if (prefix.tokens.length) notes.push(note('prompt', prefix.why))
  if (triggers.length) {
    notes.push(note('prompt', `These words were added to your prompt so the add-ons work: ${triggers.join(', ')}.`))
  }

  const positive = [...prefix.tokens, prompt, ...triggers].filter(Boolean).join(', ')

  const params: Params = {
    model,
    positive,
    negative: d.negative ?? '',
    seed: input.seed ?? rollSeed(),
    steps: d.steps,
    cfg: d.cfg,
    width: d.width,
    height: d.height,
    sampler: d.sampler,
    scheduler: d.scheduler,
  }
  if (typeof per.clipSkip === 'number') params.clipSkip = per.clipSkip
  if (input.sourceImage) {
    params.image = input.sourceImage
    params.denoise = I2I.denoise
    params.megapixels = I2I.megapixels
  }

  // 7b. Memory, priced on the graph that will be queued. The ranking priced
  // this file on its own; the add-on chain loads on top of it, and a stack of
  // them is gigabytes the ranking never saw.
  const verdict =
    input.hardware && input.sizes ? feasibility(baseDef, input.sizes, input.hardware, instantiate(def, params)) : null
  if (verdict && verdict.level !== 'ok') warnings.push(verdict.reason)

  // 8. Passes. Nothing here runs on its own. The face pass measured 0.714 on
  // the same metric as the stacks above, which is not an endorsement, and the
  // hand pass was not measured at all. Both are real fixes for the two regions
  // that fail first, so they are offered on the finished picture, where the
  // reader can see whether the hands came out wrong before spending a pass.
  const capabilities = capabilitiesOf(def)
  const passes = passesFor(
    capabilities,
    {
      face: `${detailSentence('face')} Measured ${pct(MEASURED.faceDetailer.ratio)} whole frame sharpness, so it is offered rather than run blind.`,
      hand: `${detailSentence('hand')} It is given more freedom than a face, because hands come out wrong rather than merely soft. Not measured here.`,
      refine: 'Draw a mask over a region and it is cropped, upscaled to full working resolution and rendered alone. The only thing that adds real detail to a region with no detector.',
      hires: 'Renders the same picture larger, at low denoise, from the finished latent.',
    },
    input.passBlocks,
  )
  notes.push(
    note('passes', 'Face, hands, a masked region and a larger render are offered on the finished picture, not before it.'),
  )

  return {
    ok: true,
    look,
    anatomy,
    prompt,
    familyId: baseDef.id,
    model,
    label: pick.label,
    familyLabel: pick.familyLabel,
    base: baseDef,
    def,
    params,
    loras,
    sharpness,
    missingLoras,
    refineLoras,
    offers,
    passes,
    capabilities,
    verdict,
    notes,
    warnings,
    report,
  }
}

/**
 * Region LoRAs that fit this checkpoint and are not already in the stack.
 *
 * Their strengths are the authors' own, out of the catalogue, and they are
 * flagged unmeasured because nothing in the handoff run covers them.
 */
function heldForRefine(lib: LoraLibrary, target: LoraTarget, inStack: Set<string>): RecipeLora[] {
  const out: RecipeLora[] = []
  for (const info of lib.all) {
    if (info.category !== 'anatomy') continue
    if (info.usage === 'base') continue
    if (!info.installed) continue
    if (inStack.has(info.file)) continue
    if (fitFor(info, target).level === 'mismatch') continue
    // The word the prompt actually gets, which is the caption index's when it
    // has a confident one and the catalogue's otherwise: the same choice
    // triggersFor makes when the add-on is chained.
    const word = plainWords(triggersFor([{ file: info.file, strength: 1, enabled: true }], lib))[0] ?? ''
    out.push({
      file: info.file,
      label: info.label,
      strength: info.recommended,
      measured: false,
      why: authorNote(info, word),
    })
  }
  return out.sort((a, b) => a.label.localeCompare(b.label))
}

function authorNote(info: LoraInfo, word: string): string {
  const added = addedSentence(word)
  return `${info.does}${added ? ` ${added}` : ''} Strength ${info.recommended} is the author's recommendation, not a measurement taken here.`
}

/**
 * The region add-ons that fit a weight file, for the refine bench to offer.
 *
 * The same list decide() holds back as `refineLoras`, worked out for whichever
 * model draws the region rather than the one the desk picked: the bench often
 * draws with a different one, and an add-on made for another kind of model
 * loads without error and changes nothing. Offered, never applied unasked.
 */
export function regionAddOns(
  lib: LoraLibrary,
  def: FamilyDef,
  model: string,
  exclude: Iterable<string> = [],
): RecipeLora[] {
  return heldForRefine(lib, targetFor(def, model), new Set(exclude))
}

/** Every file a measured set names. decide() reads their prompt words from the catalogue. */
const MEASURED_FILES = new Set<string>(
  [...NATURAL_STACK.entries, ...EMPHASISED_STACK.entries].map(([file]) => file),
)

/**
 * The positive prompt the desk sends for these words on this file with this
 * add-on chain: the prefix the file was trained with, the words, then the word
 * each add-on answers to when the words do not already carry it.
 *
 * For rebuilding a finished picture's prompt from its record. The record files
 * the reader's words and the chain, not the prompt that was sent, and a pass
 * queued from the bare words draws a different picture at the same seed: on
 * Pony a picture without its score tags looks like plain SDXL, and an add-on
 * without its word loads and does little. A hand edited prompt is the one
 * thing this cannot rebuild, because nothing on the record keeps it.
 *
 * The words follow whichever builder made the original. decide() takes the
 * measured sets' words from the catalogue and every other add-on's from its
 * own training captions; the edit desk takes all of them from the catalogue.
 * A record recovered from ComfyUI's history files the whole sent prompt as its
 * words, so a prefix the words already open with is not added a second time.
 */
export function positiveFor(input: {
  def: FamilyDef
  model: string
  prompt: string
  loras: readonly { name: string; strength: number }[]
  lib: LoraLibrary
  /** Words already worked out for add-ons chained on top, added after the rest. */
  extra?: readonly string[]
}): string {
  const words = input.prompt.trim()
  const lower = words.toLowerCase()
  const target = targetFor(input.def, input.model)
  const fromCatalogue = (name: string) => input.def.mode === 'edit' || MEASURED_FILES.has(name)
  const stack = input.loras
    .filter(l => fromCatalogue(l.name))
    .map(l => ({ file: l.name, strength: l.strength, enabled: true }))
  const captioned = input.loras
    .filter(l => !fromCatalogue(l.name) && l.strength !== 0)
    .flatMap(l => {
      const t = indexedTrigger(l.name)
      return t ? [t] : []
    })
  const triggers = plainWords([...triggersFor(stack, input.lib, target), ...captioned, ...(input.extra ?? [])])
    .filter((t, n, a) => a.indexOf(t) === n)
    .filter(t => !lower.includes(t.toLowerCase()))
  const prefix = prefixFor(input.def, input.model).tokens
  const opened = prefix.length > 0 && lower.startsWith(prefix.join(', ').toLowerCase())
  return [...(opened ? [] : prefix), words, ...triggers].filter(Boolean).join(', ')
}

// ---------------------------------------------------------------------------
// explain
// ---------------------------------------------------------------------------

/**
 * Two or three sentences for under the compose button.
 *
 * Plain, printed, and numbered only where a number was measured. No em dashes.
 */
export function explain(recipe: Recipe): string {
  if (!recipe.ok) return recipe.reason

  const out: string[] = []

  // An instruction model follows the change, not a look. The sentence about
  // "photoreal work" would be wrong for it, and the size sentence matters more.
  if (recipe.base.mode === 'edit') {
    out.push(`${recipe.label} follows the instruction. The size and the shape come from your picture, not from a size control.`)
    if (recipe.loras.length) {
      out.push(
        `${recipe.loras.length} add-on${recipe.loras.length === 1 ? '' : 's'} applied at the author's strength: ${recipe.loras.map(l => l.label).join(', ')}. Not measured here.`,
      )
    }
    out.push('Faces, hands, a masked region and a larger render are offered once the change exists.')
    return out.join(' ')
  }

  // The look picks the model; the detail setting never does (see decide(),
  // step 4). So the first sentence names the look and nothing else. It used to
  // add "because it carries explicit anatomy and takes the anatomy add-ons"
  // whenever the setting was on, which was false for every model that cannot.
  out.push(`${recipe.label} for ${lookWord(recipe.look)} work.`)

  // The measured helpers and the reader's own add-ons are different claims and
  // are counted apart: only the first rests on a measured run.
  const helpers = recipe.loras.filter(l => l.measured)
  const chosen = recipe.loras.filter(l => !l.measured)
  const wanted = wantedFor(recipe.anatomy)
  const whole = !!wanted && helpers.length === wanted.entries.length
  const counted = `${helpers.length} helper${helpers.length === 1 ? '' : 's'} for this detail setting`

  if (helpers.length && recipe.sharpness) {
    const where = recipe.sharpness.onMeasuredModel ? '' : ' That figure was measured on Pony V6, so treat it as a starting point here.'
    out.push(
      `${counted}, at the measured strengths. Together they came out ${pct(recipe.sharpness.ratio)} as sharp as the same picture without them.${where}`,
    )
    if (recipe.anatomy === 'emphasised') {
      out.push(
        `"Sharper faces and hands" measured ${pct(MEASURED.stacks.natural.ratio)} on the same test, so the explicit set trades sharpness for anatomy.`,
      )
    }
  } else if (helpers.length && whole) {
    out.push(`${counted}, at the measured strengths. You added more on top, so no sharpness figure applies to what will run.`)
  } else if (helpers.length) {
    out.push(`${counted}. Part of the measured set is missing, so no sharpness figure applies.`)
  } else if (recipe.anatomy !== 'off') {
    out.push(
      carriesAnatomy(recipe.base, recipe.model)
        ? `None of the helpers for this detail setting could be applied here.${recipe.warnings.length ? ' The notes below say why.' : ''}`
        : `The helpers for this detail setting do not attach to ${recipe.label}, so none are applied.`,
    )
  }

  if (chosen.length) {
    out.push(
      `Also on because you added ${chosen.length === 1 ? 'it' : 'them'}: ${chosen.map(l => l.label).join(', ')}. Not measured here.`,
    )
  }

  out.push('Faces, hands, a masked region and a larger render are offered once the picture exists.')

  return out.join(' ')
}

function lookWord(look: Look): string {
  return look === 'photoreal' ? 'photoreal' : look === 'anime' ? 'anime' : 'illustration'
}
