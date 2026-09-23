/**
 * Intent routing: which base to use for the look you actually want.
 *
 * THE PROBLEM THIS SOLVES.
 *
 * The picker in Pictures lists weight files. A weight file is not a choice a
 * person can make. "Photoreal, and the anatomy has to be right" is a choice a
 * person can make, and it has exactly one correct answer on this machine at
 * any moment, which depends on what is installed and what will fit in RAM.
 * This module turns the second question into the first, and shows its working
 * so the reader can overrule it.
 *
 * WHAT THE RANKING IS ACTUALLY MEASURING.
 *
 * Four numbers per weight file, all of them 0 to 10, all of them about the
 * training data rather than about the architecture:
 *
 *   style    how well this base renders the requested look
 *   anatomy  how well it renders explicit human anatomy from the prompt alone
 *   craft    prompt adherence, hands, small structures, text
 *   speed    derived from the registry's step count, not hand written
 *
 * The anatomy number is the one that matters most here and it is the one
 * people get wrong. It is not a measure of how good the model is. It is a
 * measure of WHAT WAS IN THE TRAINING SET. The booru trained SDXL bases
 * (Pony, NoobAI, the Illustrious finetunes) render nipples, vulvas and
 * penises far more reliably than Flux.2 Klein or Z-Image, and it is not
 * because they are better models: they are older, smaller and worse at hands.
 * It is because Danbooru and e621 are tagged explicit image sets and those
 * bases were trained on them with those tags intact, while the modern
 * photoreal bases were trained on filtered data and have simply never seen
 * the structures in question. A model cannot reconstruct what it was never
 * shown, and no amount of prompt wording fixes that.
 *
 * WHAT RANKING FIRST STILL DOES NOT FIX.
 *
 * Picking the right base improves anatomy at LARGE scale. It does nothing for
 * anatomy at small scale, because small scale is an arithmetic problem, not a
 * knowledge problem: a region covering 5% of a 1024px frame is about 29x29
 * latent cells, and a nipple or a set of labia inside it is 6x6 cells. See the
 * header of lib/refine.ts. So every recommendation here carries the same
 * second half: choose the base for the look, then mask the region and run a
 * refine pass for the detail. One without the other disappoints.
 *
 * HONESTY RULES FOR THIS FILE.
 *
 *  - Never recommend a family that will not fit. feasibility() from
 *    lib/hardware.ts decides that, from real file sizes and real free RAM.
 *  - Never recommend a file that has no verified graph. Some weights on disk
 *    have no family in the registry yet; they are reported as a gap, by name,
 *    not quietly dropped.
 *  - Every score carries a sentence saying why. If a reason cannot be written
 *    in plain words it is not a reason, it is a preference.
 *
 * Dependency direction: this module imports workflows.ts, hardware.ts and
 * refine.ts. Nothing imports it except the UI.
 */
import { FAMILIES, defaultsFor, type FamilyDef } from './workflows'
import { feasibility, modelGraph, type Hardware, type Level, type ModelFile, type Verdict } from './hardware'
import { MASK_ONLY_REGIONS } from './refine'

// ---------------------------------------------------------------------------
// Vocabulary
// ---------------------------------------------------------------------------

/** The four looks a person actually asks for. */
export type Intent = 'photoreal' | 'anime' | 'cartoon' | 'illustration'

export type IntentOption = {
  id: Intent
  /** Chip label. */
  label: string
  /** One line under the chip. No dashes joining clauses. */
  blurb: string
}

export const INTENTS: readonly IntentOption[] = [
  {
    id: 'photoreal',
    label: 'Photoreal',
    blurb: 'Skin, light and lens. It should look like a photograph, not like a render of one.',
  },
  {
    id: 'anime',
    label: 'Anime',
    blurb: 'Cel shading, line weight, booru tag vocabulary. Japanese animation and manga styling.',
  },
  {
    id: 'cartoon',
    label: 'Cartoon',
    blurb: 'Western toon and comic styling: flat fills, exaggerated shapes, bold outlines.',
  },
  {
    id: 'illustration',
    label: 'Illustration',
    blurb: 'Painted and drawn work: concept art, watercolour, ink, semi real character art.',
  },
] as const

/** Which weights a LoRA can possibly attach to. Names mismatch silently. */
export type Arch = 'sdxl' | 'flux' | 'chroma' | 'zimage' | 'qwen' | 'anima' | 'wan' | 'unknown'

/** How this base expects a prompt to be written. */
export type TagStyle = 'booru' | 'score-tags' | 'natural' | 'instruction'

export const TAG_STYLE_NOTE: Record<TagStyle, string> = {
  booru:
    'Write comma separated booru tags, not sentences. Start with the quality tags this file expects, then subject, then pose, then setting.',
  'score-tags':
    'Write comma separated booru tags and keep the score prefix this file was trained with. Without it the output collapses towards the low quality end of its training set.',
  natural:
    'Write a plain sentence or two describing the picture. Tag soup underperforms here: these models were trained on pictures captioned in full sentences.',
  instruction:
    'Write the change you want, not a description of the finished picture. "Remove her shirt" beats a full scene description.',
}

// ---------------------------------------------------------------------------
// Profiles
//
// One entry per weight file. Keyed by the exact filename ComfyUI reports, so a
// file gains its profile the moment a family for it lands in the registry.
// Numbers are 0 to 10. They are judgements, and the sentence beside each one
// is the justification, which is the point: a score with no sentence is not
// allowed in this table.
// ---------------------------------------------------------------------------

export type ModelProfile = {
  arch: Arch
  tags: TagStyle
  /** What the weights are, in one clause. Architecture and training set. */
  lineage: string
  /** What it is for, in one clause. */
  best: string
  style: Record<Intent, number>
  /** Explicit human anatomy from the prompt alone, at large scale. */
  anatomy: number
  /** Prompt adherence, hands, small structures, legible text. */
  craft: number
  /** Why the anatomy number is what it is. Training data, stated plainly. */
  anatomyReason: string
  /** The honest downside. Shown whether or not it affects the ranking. */
  caveat?: string
  /** Overrides the generic per intent sentence where there is something to add. */
  styleNotes?: Partial<Record<Intent, string>>
}

const PROFILES: Record<string, ModelProfile> = {
  // --- Photoreal, uncensored ------------------------------------------------
  'Chroma1-HD-fp8mixed.safetensors': {
    arch: 'chroma',
    tags: 'natural',
    lineage: 'Flux derivative, de distilled and retrained with anatomical content left in the dataset.',
    best: 'the photoreal pick when the picture contains a body',
    style: { photoreal: 9, anime: 6, cartoon: 6, illustration: 8 },
    anatomy: 9,
    craft: 7,
    anatomyReason:
      'It has seen nipples, vulvas and penises in training, which the other photoreal models here have not, so it renders them instead of smoothing them into a mannequin.',
    caveat:
      'It runs with real CFG rather than a distilled shortcut, so it wants more steps than the 8 step models and costs more per picture.',
    styleNotes: {
      illustration:
        'Painterly and editorial styles come out well, though a booru trained model will beat it on anime line work.',
    },
  },

  // --- Photoreal, modern, filtered -----------------------------------------
  'Z-Image-Base-bf16.safetensors': {
    arch: 'zimage',
    tags: 'natural',
    lineage: 'Modern 6B single stream DiT, full precision weights.',
    best: 'clean photoreal scenes and portraits',
    style: { photoreal: 9, anime: 5, cartoon: 5, illustration: 6 },
    anatomy: 3,
    craft: 8,
    anatomyReason:
      'Its training data was filtered, so explicit anatomy is underrepresented: expect blurred or invented structures no matter how the prompt is worded.',
  },
  'Z-Image-Turbo-fp8mix.safetensors': {
    arch: 'zimage',
    tags: 'natural',
    lineage: 'The distilled sibling of Z-Image Base, quantised, 8 steps at CFG 1.',
    best: 'fast photoreal drafts and composition tests',
    style: { photoreal: 8, anime: 5, cartoon: 5, illustration: 6 },
    anatomy: 3,
    craft: 7,
    anatomyReason:
      'Same filtered training set as the Base weights, and distillation removes the CFG headroom you would otherwise use to push a difficult region.',
    caveat: 'Distilled speed costs a little fine texture against the Base weights.',
  },
  'flux-2-klein-4b-fp8.safetensors': {
    arch: 'flux',
    tags: 'natural',
    lineage: 'Distilled 4B Flux.2, guidance baked in, quantised to fp8.',
    best: 'crisp photoreal and product style images in very few steps',
    style: { photoreal: 9, anime: 5, cartoon: 6, illustration: 7 },
    anatomy: 2,
    craft: 8,
    anatomyReason:
      'Heavily filtered training data. Nudity resolves into smooth featureless forms, and genitalia are effectively absent from what it learned.',
    caveat: 'Distilled, so CFG is fixed near 1 and negative prompts do very little.',
  },
  'qwen_image_2.1_int8_convrot.safetensors': {
    arch: 'qwen',
    tags: 'natural',
    lineage: 'Large MMDiT with a language model text encoder, int8 quantised.',
    best: 'complex instructions, many subjects, and legible text inside the picture',
    style: { photoreal: 8, anime: 5, cartoon: 6, illustration: 7 },
    anatomy: 2,
    craft: 9,
    anatomyReason:
      'The strongest prompt follower installed and the weakest at explicit anatomy: it will obey a nudity instruction and then render something anatomically absent, because the structures were filtered out of its training data.',
  },
  'moodyCutieMixKrea2_v50_int8.safetensors': {
    arch: 'flux',
    tags: 'natural',
    lineage: 'Community mix on the Krea lineage, quantised, distilled to few steps.',
    best: 'a softer, warmer photoreal aesthetic than the stock models',
    style: { photoreal: 8, anime: 4, cartoon: 4, illustration: 6 },
    anatomy: 5,
    craft: 7,
    anatomyReason:
      'A community mix, so its dataset is partly unfiltered: better at nudity than stock Flux, still well short of Chroma or the booru models at genital detail.',
    caveat: 'Mix provenance is not documented, so treat the anatomy score as an observation rather than a specification.',
  },

  // --- Anima: uncensored DiT, anime through to realistic skin ---------------
  'miaomiaoRealskin_anima13.safetensors': {
    arch: 'anima',
    tags: 'natural',
    lineage: 'Anima DiT finetune aimed at realistic skin rather than cel shading.',
    best: 'realistic skin from an uncensored model, the middle ground between Chroma and the anime models',
    style: { photoreal: 7, anime: 7, cartoon: 5, illustration: 6 },
    anatomy: 9,
    craft: 6,
    anatomyReason: 'Uncensored training set, with explicit anatomy present and tagged.',
    caveat: 'Skin realism is good, scene realism is not: backgrounds and hands lag the dedicated photoreal models.',
  },
  'miaomiaoHarem_29BBETA10.safetensors': {
    arch: 'anima',
    tags: 'natural',
    lineage: '2.9B Anima DiT, anime trained, uncensored.',
    best: 'anime characters with reliable anatomy and a modern DiT prompt understanding',
    style: { photoreal: 3, anime: 9, cartoon: 6, illustration: 7 },
    anatomy: 9,
    craft: 6,
    anatomyReason: 'Uncensored training set. Explicit anatomy renders without fighting the model.',
    caveat: 'Beta weights. Expect more run to run variance than the SDXL anime models.',
  },
  'oneObsession_anima29BV1.safetensors': {
    arch: 'anima',
    tags: 'natural',
    lineage: '2.9B Anima DiT, anime trained, uncensored.',
    best: 'explicit anime scenes described in sentences rather than tags',
    style: { photoreal: 3, anime: 9, cartoon: 6, illustration: 7 },
    anatomy: 9,
    craft: 6,
    anatomyReason: 'Uncensored training set, weighted towards explicit material.',
  },

  // --- Booru trained SDXL: the anatomy specialists --------------------------
  'ponyDiffusionV6XL.safetensors': {
    arch: 'sdxl',
    tags: 'score-tags',
    lineage: 'SDXL, trained on a large tagged set spanning furry, western cartoon and anime, explicit material included.',
    best: 'cartoon and stylised bodies, and the most reliable explicit anatomy installed',
    style: { photoreal: 2, anime: 8, cartoon: 9, illustration: 7 },
    anatomy: 10,
    craft: 5,
    anatomyReason:
      'Its training set is explicitly tagged adult art across several traditions, so vulvas, penises, nipples and sex acts are things it has actually learned rather than things it is guessing at.',
    caveat:
      'Weak at hands, faces at distance and text, and it cannot do photoreal at all. Pair it with a face pass and a hand pass.',
    styleNotes: {
      cartoon: 'The best cartoon model here by a wide margin: western toon and furry art are a large part of what it was trained on.',
      photoreal: 'Do not use this for photoreal. It pulls every subject towards illustration.',
    },
  },
  'NoobAI-XL-v1.1.safetensors': {
    arch: 'sdxl',
    tags: 'booru',
    lineage: 'SDXL on the Illustrious lineage, trained on full Danbooru and e621 tag sets.',
    best: 'anime, with the deepest tag vocabulary of anything installed',
    style: { photoreal: 3, anime: 10, cartoon: 7, illustration: 8 },
    anatomy: 10,
    craft: 5,
    anatomyReason:
      'Trained on complete booru sets with explicit ratings intact, so anatomy and named sex acts respond to their tags directly.',
    caveat: 'SDXL era hands and faces. Artist tags change the output drastically, which is a feature until it is a surprise.',
  },
  'semiRealIllustrious_v40.safetensors': {
    arch: 'sdxl',
    tags: 'booru',
    lineage: 'SDXL Illustrious finetune pulled towards semi realistic rendering.',
    best: 'semi real character art: anime structure with photographic shading',
    style: { photoreal: 5, anime: 8, cartoon: 6, illustration: 9 },
    anatomy: 8,
    craft: 6,
    anatomyReason: 'Booru derived training, so explicit anatomy is present, softened slightly by the semi real finetune.',
    caveat: 'CFG above 6 turns the skin plastic. The registry card keeps it at 5 for a reason.',
  },
  'waiMatureIllustrious_v30.safetensors': {
    arch: 'sdxl',
    tags: 'booru',
    lineage: 'SDXL Illustrious finetune on mature material.',
    best: 'adult anime characters, mature proportions, explicit scenes',
    style: { photoreal: 4, anime: 9, cartoon: 6, illustration: 8 },
    anatomy: 9,
    craft: 6,
    anatomyReason: 'Finetuned specifically on mature and explicit booru material, which is exactly the vocabulary in question.',
    caveat: 'VAE is baked in. Do not attach an external VAE to this file.',
  },

  // --- Editing --------------------------------------------------------------
  'qwen-image-edit-2511-Q4_K_M.gguf': {
    arch: 'qwen',
    tags: 'instruction',
    lineage: 'Instruction edit model, GGUF Q4_K_M quantised.',
    best: 'changing one thing in an existing picture while the rest stays put',
    style: { photoreal: 8, anime: 6, cartoon: 6, illustration: 7 },
    anatomy: 2,
    craft: 9,
    anatomyReason:
      'It follows the instruction and then renders anatomy it was never trained on, which is why undressing a subject produces a smooth featureless body. The fix is not a better instruction: mask the region and run a refine pass on a model that can draw anatomy.',
    caveat:
      'Q4 quantisation costs fine texture. Small structures degrade first, and small structures are the ones people complain about.',
  },
}

/** Plain names for files the registry has no label for yet. */
const PLAIN_NAMES: Record<string, string> = {
  'Chroma1-HD-fp8mixed.safetensors': 'Chroma1 HD',
  'ponyDiffusionV6XL.safetensors': 'Pony Diffusion V6 XL',
  'NoobAI-XL-v1.1.safetensors': 'NoobAI XL v1.1',
  'Z-Image-Base-bf16.safetensors': 'Z-Image Base',
  'Z-Image-Turbo-fp8mix.safetensors': 'Z-Image Turbo',
  'flux-2-klein-4b-fp8.safetensors': 'Flux.2 Klein 4B',
  'qwen_image_2.1_int8_convrot.safetensors': 'Qwen Image 2.1',
  'moodyCutieMixKrea2_v50_int8.safetensors': 'Moody Cutie Mix, Krea 2',
  'miaomiaoRealskin_anima13.safetensors': 'Miaomiao Realskin (Anima)',
  'miaomiaoHarem_29BBETA10.safetensors': 'Miaomiao Harem 2.9B (Anima)',
  'oneObsession_anima29BV1.safetensors': 'One Obsession 2.9B (Anima)',
  'qwen-image-edit-2511-Q4_K_M.gguf': 'Qwen Image Edit 2511',
}

/** Last resort when neither the registry nor the table names a file. */
function titleFromFilename(model: string): string {
  return model
    .replace(/\.(safetensors|gguf|ckpt|pt|sft)$/i, '')
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

/** The reader facing name of a weight file. Registry label wins. */
function plainName(model: string, def?: FamilyDef): string {
  const per = def?.perModel?.[model] as { label?: unknown } | undefined
  if (per && typeof per.label === 'string' && per.label) return per.label
  return PLAIN_NAMES[model] ?? titleFromFilename(model)
}

/**
 * Architecture guess for a file with no profile, from its family graph.
 * Used only for LoRA compatibility copy, never for scoring.
 */
function archFromFamily(def: FamilyDef | undefined): Arch {
  if (!def) return 'unknown'
  const id = def.id.toLowerCase()
  if (id.includes('sdxl') || id.includes('illustrious') || id.includes('pony') || id.includes('noob')) return 'sdxl'
  if (id.includes('chroma')) return 'chroma'
  if (id.includes('flux') || id.includes('krea')) return 'flux'
  if (id.includes('z-image') || id.includes('zimage')) return 'zimage'
  if (id.includes('qwen')) return 'qwen'
  if (id.includes('anima')) return 'anima'
  if (id.includes('wan')) return 'wan'
  return 'unknown'
}

/** The profile for a weight file, or null when the file is not in the table. */
function profileFor(model: string, def?: FamilyDef): ModelProfile | null {
  return PROFILES[model] ?? (def ? fallbackProfile(model, def) : null)
}

/**
 * How well a weight file serves a look, 0 to 10, from the same table the
 * ranking reads. The region bench uses it to choose a stand-in when the model
 * that made a picture cannot redraw part of it: the stand-in should at least
 * be good at the kind of picture it is patching.
 */
export function lookScore(model: string, def: FamilyDef, intent: Intent): number {
  return profileFor(model, def)?.style[intent] ?? 0
}

/** The look a weight file is strongest at, by the same table. Ties go to the earlier look. */
export function strongestLook(model: string, def: FamilyDef): Intent {
  const style = profileFor(model, def)?.style
  if (!style) return 'photoreal'
  let best: Intent = 'photoreal'
  for (const opt of INTENTS) if (style[opt.id] > style[best]) best = opt.id
  return best
}

/**
 * A neutral profile for an installed file nobody has rated yet. It scores
 * mid on everything and says so, which keeps a new model visible in the list
 * without pretending anybody has tested it.
 */
function fallbackProfile(model: string, def: FamilyDef): ModelProfile {
  return {
    arch: archFromFamily(def),
    tags: def.mode === 'edit' ? 'instruction' : 'natural',
    lineage: `${def.label}. No profile recorded for this file yet.`,
    best: 'unrated',
    style: { photoreal: 5, anime: 5, cartoon: 5, illustration: 5 },
    anatomy: 5,
    craft: 5,
    anatomyReason: `Nobody has rated ${plainName(model, def)} for explicit anatomy here, so this ranking is a placeholder. Run it and find out.`,
  }
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

/** What the reader wants. `explicit` is the axis that reorders everything. */
export type Brief = {
  intent: Intent
  /** True when the picture contains explicit human anatomy. */
  explicit: boolean
  /** Which desk is asking. Edit families never appear in an image ranking. */
  mode?: 'image' | 'edit'
}

export type RankOptions = {
  /** Defaults to every registry family. */
  families?: readonly FamilyDef[]
  /** Filenames ComfyUI reports as present. When given, anything else is skipped. */
  installed?: Iterable<string>
  /** Real file sizes, from modelFiles(). Needed for feasibility. */
  sizes?: Map<string, ModelFile>
  /** Real memory, from probeHardware(). Needed for feasibility. */
  hardware?: Hardware | null
  /** Trim the ranked list. Blocked and unrouted entries are never trimmed. */
  limit?: number
}

export type Recommendation = {
  familyId: string
  model: string
  /** Reader facing name of the weight file. */
  label: string
  /** Reader facing name of the family it belongs to. */
  familyLabel: string
  def: FamilyDef
  profile: ModelProfile
  /** 0 to 100. Comparable within one ranking only. */
  score: number
  /** 1 is the top of the list. */
  rank: number
  /** The component scores, so the UI can show the working. */
  parts: { style: number; anatomy: number; craft: number; speed: number; hardware: number }
  /** One line saying why this sits where it sits. */
  why: string
  /** The honest downside, or null when there is nothing to warn about. */
  caveat: string | null
  /** Present when the brief asks for explicit anatomy and this base is poor at it. */
  warning: string | null
  verdict: Verdict | null
  fit: Level | null
  /** How to phrase a prompt for this base. */
  tagStyle: TagStyle
}

/** A file that will not run here, and the measured reason. */
export type BlockedModel = { model: string; label: string; familyId: string; why: string }

/** A file on disk with a profile but no verified graph, so it cannot be offered. */
export type UnroutedModel = { model: string; label: string; why: string }

export type IntentReport = {
  brief: Required<Brief>
  ranked: Recommendation[]
  blocked: BlockedModel[]
  unrouted: UnroutedModel[]
  /** The headline the UI can show above the list. */
  note: string
  /** What still has to happen after the base is chosen. Always present. */
  anatomyNote: string
}

/**
 * Component weights.
 *
 * With `explicit` off, the look is most of the decision and craft carries the
 * rest. With `explicit` on, anatomy is weighted as heavily as style, which is
 * what pushes the booru trained SDXL bases above the newer photoreal ones for
 * this kind of work even though they are worse models in every other respect.
 */
function weightsFor(brief: Brief) {
  return brief.explicit
    ? { style: 0.4, anatomy: 0.4, craft: 0.15, speed: 0.05 }
    : { style: 0.5, anatomy: 0.05, craft: 0.35, speed: 0.1 }
}

/** Speed from the registry's own step count, so it tracks the recipe. */
function speedScore(def: FamilyDef, model: string): number {
  const steps = defaultsFor(def, model).steps || def.defaults.steps || 20
  // 4 steps scores 10, 30 steps scores about 2. Linear in between.
  const s = 10 - (steps - 4) / 3
  return Math.max(0, Math.min(10, s))
}

/** Penalty in final points for a fit that is not comfortable. */
function hardwarePenalty(v: Verdict | null): number {
  if (!v) return 0
  let p = 0
  if (v.level === 'tight') p -= 6
  if (v.level === 'risky') p -= 18
  if (v.offloads) p -= 4
  return p
}

const BAND = (n: number, intent: Intent): string => {
  const what = intent === 'photoreal' ? 'photoreal work' : `${intent} work`
  if (n >= 9) return `The strongest option installed for ${what}.`
  if (n >= 7) return `Strong at ${what}.`
  if (n >= 5) return `Workable for ${what} without being its strength.`
  return `Not what this model is for: ${what} fights it.`
}

function justify(p: ModelProfile, brief: Brief): string {
  const parts: string[] = [p.lineage]
  const styled = p.styleNotes?.[brief.intent] ?? BAND(p.style[brief.intent], brief.intent)
  parts.push(styled)
  if (brief.explicit) parts.push(p.anatomyReason)
  else if (p.best !== 'unrated') parts.push(`Best used for ${p.best}.`)
  return parts.join(' ')
}

function warnAbout(p: ModelProfile, brief: Brief, label: string): string | null {
  if (!brief.explicit) return null
  if (p.anatomy >= 6) return null
  return `${label} was not trained on explicit anatomy in any quantity. Nipples, vulvas and penises will come out smooth, merged or invented, and prompt wording will not change that. A masked refine pass on one of the booru trained models is the fix.`
}

// ---------------------------------------------------------------------------
// Ranking
// ---------------------------------------------------------------------------

/** The full picture: what to use, what will not fit, and what has no graph. */
export function intentReport(brief: Brief, opts: RankOptions = {}): IntentReport {
  const mode = brief.mode ?? 'image'
  const full: Required<Brief> = { intent: brief.intent, explicit: brief.explicit, mode }
  const families = opts.families ?? FAMILIES
  const installed = opts.installed ? new Set(opts.installed) : null
  const sizes = opts.sizes
  const hw = opts.hardware ?? null
  const w = weightsFor(full)

  const ranked: Recommendation[] = []
  const blocked: BlockedModel[] = []
  const routed = new Set<string>()

  for (const def of families) {
    if (def.mode !== mode) continue

    // A dual-model family carries a matched pair in its graph and is picked as
    // one thing. Its first file stands for the family in the list.
    const models = def.dualModel ? def.models.slice(0, 1) : def.models

    for (const model of models) {
      routed.add(model)
      if (installed && !installed.has(model)) continue

      const profile = profileFor(model, def)
      if (!profile) continue

      const label = plainName(model, def)
      // Priced on this file, not the family's default: a family lists several
      // checkpoints and quants, and they are not the same size.
      const verdict = sizes && hw ? feasibility(def, sizes, hw, modelGraph(def, model)) : null
      if (verdict && !verdict.selectable) {
        blocked.push({ model, label, familyId: def.id, why: verdict.reason })
        continue
      }

      const style = profile.style[full.intent]
      const speed = speedScore(def, model)
      const raw =
        style * w.style + profile.anatomy * w.anatomy + profile.craft * w.craft + speed * w.speed
      const penalty = hardwarePenalty(verdict)
      const score = Math.max(0, Math.min(100, Math.round(raw * 10 + penalty)))

      ranked.push({
        familyId: def.id,
        model,
        label,
        familyLabel: def.label,
        def,
        profile,
        score,
        rank: 0,
        parts: {
          style: Math.round(style * w.style * 10),
          anatomy: Math.round(profile.anatomy * w.anatomy * 10),
          craft: Math.round(profile.craft * w.craft * 10),
          speed: Math.round(speed * w.speed * 10),
          hardware: penalty,
        },
        why: justify(profile, full),
        caveat: profile.caveat ?? null,
        warning: warnAbout(profile, full, label),
        verdict,
        fit: verdict ? verdict.level : null,
        tagStyle: profile.tags,
      })
    }
  }

  ranked.sort((a, b) => b.score - a.score || a.label.localeCompare(b.label))
  ranked.forEach((r, i) => {
    r.rank = i + 1
  })

  const unrouted: UnroutedModel[] = []
  if (installed) {
    for (const model of installed) {
      if (routed.has(model)) continue
      const p = PROFILES[model]
      if (!p) continue
      unrouted.push({
        model,
        label: plainName(model),
        why: `${plainName(model)} is on disk but no verified graph exists for it yet, so it cannot be queued. Worth adding: it is ${p.best}.`,
      })
    }
  }

  const trimmed = opts.limit && opts.limit > 0 ? ranked.slice(0, opts.limit) : ranked

  return {
    brief: full,
    ranked: trimmed,
    blocked,
    unrouted,
    note: headline(full, trimmed, unrouted),
    anatomyNote: anatomyNote(full),
  }
}

/** The line above the list. Names the winner and the reason in one breath. */
function headline(brief: Required<Brief>, ranked: Recommendation[], unrouted: UnroutedModel[]): string {
  const top = ranked[0]
  if (!top) {
    return 'Nothing installed can run this here. Check the blocked list for what is missing or too large.'
  }
  const look = brief.intent === 'photoreal' ? 'photoreal' : brief.intent
  const lead = brief.explicit
    ? `For explicit ${look} work, ${top.label} ranks first.`
    : `For ${look} work, ${top.label} ranks first.`
  const gap = unrouted.length
    ? ` ${unrouted.map((u) => u.label).join(' and ')} would rank here too, but ${unrouted.length === 1 ? 'it has' : 'they have'} no verified graph yet.`
    : ''
  return lead + gap
}

/**
 * The half of the answer that is not about model choice. Always shown, because
 * the base decides whether anatomy is plausible and the refine pass decides
 * whether it is correct.
 */
function anatomyNote(brief: Brief): string {
  const regions = MASK_ONLY_REGIONS.filter((r) => r !== 'any other region').join(', ')
  if (!brief.explicit) {
    return 'Faces and hands have detectors, so they can be detailed automatically. Everything else needs a drawn mask and a refine pass.'
  }
  return `Choosing the model fixes anatomy at large scale only. At small scale the region simply does not have the latent cells to be correct, whichever model renders it. Draw a mask over ${regions} and run a refine pass: the crop is upscaled to full working resolution and re rendered alone, which is the only thing that adds real detail. There is no detector for these regions, so the mask has to be drawn. Budget one full generation per pass.`
}

// ---------------------------------------------------------------------------
// Reading the brief out of a prompt
//
// Convenience only. The chips are the real control; this is for prefilling
// them and for the "you asked for a photograph but you are on an anime base"
// hint. Clinical vocabulary, because that is what people type.
// ---------------------------------------------------------------------------

const EXPLICIT_WORDS: readonly string[] = [
  'nude', 'nudity', 'naked', 'topless', 'bottomless', 'undressed', 'unclothed',
  'breast', 'breasts', 'nipple', 'nipples', 'areola', 'areolae', 'cleavage',
  'vulva', 'vagina', 'labia', 'clitoris', 'pussy',
  'penis', 'erection', 'erect', 'testicles', 'scrotum', 'cock', 'dick',
  'genital', 'genitals', 'genitalia', 'anus', 'anal',
  'sex', 'intercourse', 'penetration', 'fellatio', 'cunnilingus', 'masturbation', 'masturbating',
  'orgasm', 'cum', 'semen', 'explicit', 'nsfw', 'porn', 'pornographic', 'uncensored',
  'spread legs', 'rating:explicit', 'rating explicit',
]

function countHits(haystack: string, words: readonly string[]): number {
  let n = 0
  for (const w of words) {
    const escaped = w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    const re = new RegExp(`(^|[^a-z0-9])${escaped}([^a-z0-9]|$)`, 'i')
    if (re.test(haystack)) n++
  }
  return n
}

/** True when the prompt describes explicit human anatomy. */
export function wantsExplicitAnatomy(prompt: string): boolean {
  return countHits(` ${prompt.toLowerCase()} `, EXPLICIT_WORDS) > 0
}

// ---------------------------------------------------------------------------
// LoRA fit
//
// A LoRA is a set of weight deltas keyed to the layer names of one
// architecture. Attach an SDXL LoRA to Chroma and ComfyUI does not error: the
// keys match nothing, the deltas apply to nothing, and the picture comes out
// exactly as it would have without it. So compatibility has to be stated
// before the run, not discovered after it.
// ---------------------------------------------------------------------------

export type LoraFit = {
  name: string
  arch: Arch
  /** true fits, false cannot fit, null means the filename does not say. */
  fits: boolean | null
  note: string
}

