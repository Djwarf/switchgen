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
import type { Hardware, ModelFile } from './hardware'
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
  fitFor,
  resolveStack,
  targetFor,
  triggersFor,
  type LoraInfo,
  type LoraLibrary,
  type LoraStack,
  type LoraTarget,
} from './loras'
import { suggest as suggestLoras } from './suggest'
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
import { canTakeLoras, capabilitiesOf, withLoras, type Capabilities, type DerivedDef } from './refine'
import { FAMILIES, defaultsFor, deriveImg2Img, IMG2IMG, type FamilyDef, type Params } from './workflows'

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
  /** The family's graph supports it. False means the button is not drawn. */
  available: boolean
  /** Whether {@link decide} thinks it should run without being asked. */
  auto: boolean
  why: string
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
 * Two shapes live in perModel. The Illustrious entries carry an explicit
 * `positivePrefix`. Pony V6 carries no prefix field, only a note saying it
 * needs the score tags and renders like base SDXL without them, so the tags are
 * lifted out of that note rather than typed in here: when the registry is
 * regenerated the prefix follows it instead of rotting.
 */
function prefixFor(def: FamilyDef, model: string): { tokens: string[]; why: string } {
  const per = (def.perModel?.[model] ?? {}) as Record<string, unknown>

  const explicit = per.positivePrefix
  if (typeof explicit === 'string' && explicit.trim()) {
    const tokens = explicit
      .split(',')
      .map(t => t.trim())
      .filter(Boolean)
    return { tokens, why: 'Quality prefix this file was trained with, from the registry.' }
  }

  const prose = [per.notes, per.note].filter(v => typeof v === 'string').join(' ')
  const found = prose.match(/score_\d+(?:_up)?/g)
  if (found?.length) {
    const seen: string[] = []
    for (const t of found) if (!seen.includes(t)) seen.push(t)
    return {
      tokens: seen,
      why: 'Score prefix this file needs. Without it the output collapses towards base SDXL.',
    }
  }

  return { tokens: [], why: '' }
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

  // 1. The brief. The anatomy control is a statement about the picture, so it
  // sets `explicit` on its own; the prompt can set it too, which is what keeps
  // a nude described in words off a base that was trained without any.
  // Detail and explicitness are SEPARATE questions and must not share a switch.
  // This line used to read `anatomy !== 'off' || wantsExplicitAnatomy(prompt)`,
  // which made "I want better hands and faces" the identical input as "I want
  // porn": intent.ts weightsFor() swings anatomy 0.05 -> 0.40 and craft 0.35 ->
  // 0.15 on this boolean, so asking for detail actively demoted the craft models
  // (Chroma, Z-Image, Flux, Qwen) in favour of booru bases. Wanting good anatomy
  // is a universal quality need; wanting explicit content is a content choice.
  // Only the brief itself decides the latter now.
  const explicit = wantsExplicitAnatomy(prompt)
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
        ? 'Nothing installed can work from a picture here. The blocked list says what is missing or too large.'
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
      notes.push(
        note(
          'model',
          `${pick.label} renders this on its own. The face and hand helpers are SDXL booru files and nothing installed here can carry them.`,
        ),
      )
    }
  }

  notes.push(note('model', `${pick.label}: ${pick.why}`))
  if (pick.caveat) notes.push(note('model', pick.caveat))
  if (pick.warning) warnings.push(pick.warning)
  if (pick.verdict && pick.verdict.level !== 'ok') warnings.push(pick.verdict.reason)

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
            ? `The only measured restorer: ${pct(MEASURED.microDetailsAlone.ratio)} base sharpness on its own at ${MEASURED.microDetailsAlone.strength}.`
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
      notes.push(
        note(
          'anatomy',
          `This stack measured ${pct(wanted.ratio)} base sharpness. ${wanted.verdict}`,
          true,
        ),
      )
      if (model !== MEASURED_ON) notes.push(note('anatomy', MEASURED.transferNote, true))
      if (anatomy === 'emphasised') {
        warnings.push(
          `Emphasised measured ${pct(MEASURED.stacks.emphasised.ratio)} against base, below the ${pct(MEASURED.stacks.natural.ratio)} that Natural measured. Three anatomy LoRAs cost more sharpness than micro details repays. More is not better here.`,
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

  // 6. The graph. Image to image first, LoRA chain second, so the chain is
  // inserted into the graph that will actually run.
  let def: FamilyDef | DerivedDef = baseDef
  if (input.sourceImage) {
    const i2i = img2imgOf(baseDef)
    if (i2i) def = i2i
    else warnings.push(`${pick.label} cannot sample from a picture, so this will render from the prompt alone.`)
  }
  if (loras.length) {
    const chained = withLoras(def, loras.map(l => ({ name: l.file, strength: l.strength })))
    if (chained) def = chained
    else warnings.push(`${pick.label} cannot take add-ons, so none were applied.`)
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
  // was, so the figure printed on screen still describes what runs. Anything
  // the prompt suggests on top is marked measured:false, because no run
  // measured it.
  let promptPicks: { file: string; strength: number; why: string }[] = []
  try {
    const sug = suggestLoras({
      // Pass the LIBRARY, not a list of filenames. infoFor() returns null for a
      // bare iterable, so every catalogue fact - category, label, recommended
      // strength - was being thrown away, and the subject gate had nothing to
      // read. InstalledLoras accepts either; only one of them works.
      prompt, model, installed: lib, anatomy,
      already: resolvedFiles,
    } as never) as { stack?: { file: string; strength: number; why?: string }[] }
    promptPicks = (sug.stack ?? [])
      .filter(s => !resolvedFiles.includes(s.file))
      .slice(0, 2)
      .map(s => ({ file: s.file, strength: s.strength, why: s.why ?? 'Suggested by your prompt.' }))
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
  if (appliedPicks.length) {
    notes.push(note('prompt', `You added ${appliedPicks.length} add-on` +
      `${appliedPicks.length === 1 ? '' : 's'} matched to your wording: ` +
      `${appliedPicks.map(p => labelOf(lib, p.file)).join(', ')}. ` +
      `Matched from training vocabulary, not measured.`))
  }

  // Suggested LoRAs need their triggers as much as measured ones do. An
  // untriggered stack measured 0.786x of base, ie worse than using none, so
  // adding a LoRA without its token actively harms the picture.
  const suggestedTriggers = appliedPicks.flatMap(pick => {
    const e = indexedTrigger(pick.file)
    return e ? [e] : []
  })
  const triggers = [...triggersFor(stack, lib, target), ...suggestedTriggers]
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

  // 8. Passes. Nothing here runs on its own. The face pass measured 0.714 on
  // the same metric as the stacks above, which is not an endorsement, and the
  // hand pass was not measured at all. Both are real fixes for the two regions
  // that fail first, so they are offered on the finished picture, where the
  // reader can see whether the hands came out wrong before spending a pass.
  const capabilities = capabilitiesOf(def)
  const passes: RecipePasses = {
    face: {
      available: capabilities.faceDetail,
      auto: false,
      why: `Re renders every detected face at 768 and pastes it back. Measured ${pct(MEASURED.faceDetailer.ratio)} whole frame sharpness, so it is offered rather than run blind.`,
    },
    hand: {
      available: capabilities.handDetail,
      auto: false,
      why: 'Re renders every detected hand with more freedom than a face, because hands come out wrong rather than merely soft. Not measured here.',
    },
    refine: {
      available: capabilities.refine,
      auto: false,
      why: 'Draw a mask over a region and it is cropped, upscaled to full working resolution and rendered alone. The only thing that adds real detail to a region with no detector.',
    },
    hires: {
      available: capabilities.hires,
      auto: false,
      why: 'Renders the same picture larger, at low denoise, from the finished latent.',
    },
  }
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
    out.push({
      file: info.file,
      label: info.label,
      strength: info.recommended,
      measured: false,
      why: authorNote(info),
    })
  }
  return out.sort((a, b) => a.label.localeCompare(b.label))
}

function authorNote(info: LoraInfo): string {
  const trigger = info.trigger.trim() ? ` Needs the trigger ${info.trigger}.` : ''
  return `${info.does}${trigger} Strength ${info.recommended} is the author's recommendation, not a measurement taken here.`
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

  const pickedFor =
    recipe.anatomy === 'off'
      ? `${recipe.label} for ${lookWord(recipe.look)} work.`
      : `${recipe.label} for ${lookWord(recipe.look)} work, because it carries explicit anatomy in its training data and takes the anatomy LoRAs.`
  out.push(pickedFor)

  if (recipe.loras.length && recipe.sharpness) {
    const where = recipe.sharpness.onMeasuredModel ? '' : ' That figure was measured on Pony V6, so treat it as a starting point here.'
    out.push(
      `${recipe.loras.length} LoRAs at the measured strengths: the stack came out at ${pct(recipe.sharpness.ratio)} base sharpness.${where}`,
    )
    if (recipe.anatomy === 'emphasised') {
      out.push('Natural measured 1.14x on the same test, so emphasised buys anatomy weight at the cost of sharpness.')
    }
  } else if (recipe.loras.length) {
    out.push(`${recipe.loras.length} anatomy LoRAs applied. Part of the measured stack is missing, so no sharpness figure applies.`)
  } else if (recipe.anatomy !== 'off') {
    out.push('No anatomy LoRAs could be applied here. The warnings say why.')
  }

  out.push('Faces, hands, a masked region and a larger render are offered once the picture exists.')

  return out.join(' ')
}

function lookWord(look: Look): string {
  return look === 'photoreal' ? 'photoreal' : look === 'anime' ? 'anime' : 'illustration'
}
