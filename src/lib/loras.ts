/**
 * LoRA library, compatibility and stacks.
 *
 * WHY A LORA IS THE OTHER HALF OF THE ANATOMY ANSWER.
 *
 * refine.ts solves a capacity problem: a region that occupies 36 latent cells
 * cannot hold a correct nipple, a correct vulva or a correct set of fingers, so
 * the region is cropped, upscaled and re rendered until it has seven hundred.
 * That gives the model room to draw. It does not change what the model believes
 * the thing looks like. A LoRA does exactly that, and nothing else.
 *
 * So the two stack, and they stack in one direction:
 *
 *   LoRA decides WHAT is drawn. Refine decides HOW MUCH DETAIL fits.
 *
 * An anatomy LoRA on a 5% region at base resolution still has 36 cells to work
 * with and still fails. A refine pass on a base that has never seen explicit
 * anatomy renders a smooth, confident, wrong shape at high resolution. Both
 * together is the configuration that works, which is why the catalogue below
 * carries a `usage` field: `base` means apply it to the first pass, `refine`
 * means it earns its keep in the crop pass where the pixels exist.
 *
 * THE MISMATCH TRAP, WHICH IS THE POINT OF HALF THIS FILE.
 *
 * ComfyUI will happily load a Pony LoRA onto a Flux model. LoraLoader matches
 * weights by key name, patches whatever keys happen to match, and reports
 * nothing when almost none of them do. There is no error, no warning and no
 * refusal: there is a picture that looks like static, or worse, a picture that
 * looks nearly right and is subtly poisoned. The user then blames the prompt,
 * the sampler or the base model, because those are the three things a UI
 * usually lets you doubt.
 *
 * Therefore every LoRA in this module carries the base it was actually trained
 * against, and fitFor() is consulted before anything reaches a graph:
 *
 *   match     the catalogue verified this LoRA against this exact checkpoint,
 *             or the architectures are the same
 *   untested  it will load and do something; nobody has checked that the
 *             something is good. Same-lineage SDXL crossings live here
 *   mismatch  different architecture. Never sent to ComfyUI, whatever the
 *             stack says, because the failure is silent
 *
 * resolveStack() drops mismatches rather than trusting the UI to have greyed
 * them out, because a stack is persisted per family and the user can change
 * checkpoint under a saved stack at any time.
 *
 * WHERE THE DATA COMES FROM.
 *
 * Two sources, joined on filename:
 *
 *   /api/models   what is on disk right now, with real byte sizes. This is the
 *                 only authority on whether a LoRA can actually be loaded.
 *   LORA_CATALOGUE  a verified metadata table: base architecture, trigger
 *                 words, recommended strength, what the thing does, and the
 *                 caveats. Every row was checked against the file's README
 *                 base_model tag rather than its marketing title, and where the
 *                 two disagree the caveat says so.
 *
 * The catalogue is embedded rather than fetched. It is static data produced
 * once by the research pass, it is needed before the first render, and a
 * missing fetch would silently degrade every compatibility check to "unknown",
 * which is the one failure mode this file exists to prevent.
 *
 * A file on disk with no catalogue row is still offered. It is simply marked as
 * having no verified base, which is the truth.
 */
import { downloadFile, type FetchProgress } from './downloads'
import { byFilename as indexedLora } from './loraIndex'

/**
 * The trigger a LoRA actually answers to.
 *
 * loras.ts carries a hand-written catalogue. loraIndex.ts carries what each
 * file's own training captions say. Where the file speaks, the file wins: the
 * catalogue had `rnat` for a LoRA whose 26 training images all say `rnct`, and
 * a wrong trigger is not a small error. A LoRA running without its trigger
 * measured BELOW using no LoRA at all (0.786x mean over three seeds), so a
 * wrong token is worse than an empty one.
 *
 * The catalogue still answers for the 22 files that carry no tag data, and for
 * anything the index rates below `likely`.
 */
function bestTrigger(file: string, handWritten: string): string {
  const e = indexedLora(file)
  if (e && (e.confidence === 'strong' || e.confidence === 'likely') && e.triggerPhrase) {
    return e.triggerPhrase
  }
  return handWritten
}

import type { LoraSpec } from './refine'
import type { FamilyDef } from './registry'

// ---------------------------------------------------------------------------
// Architecture vocabulary
// ---------------------------------------------------------------------------

/**
 * The weight families a LoRA can belong to. These are architectures, not
 * products: `illustrious` covers NoobAI-XL and every Illustrious finetune,
 * because they share a text encoder and a UNet key layout, which is the only
 * thing LoraLoader cares about.
 */
export type LoraArch =
  | 'pony'
  | 'illustrious'
  | 'sdxl'
  | 'flux1d'
  | 'flux2'
  | 'chroma'
  | 'z-image'
  | 'qwen-image'
  | 'qwen-image-edit'
  | 'anima'
  | 'krea2'
  | 'wan'
  | 'hunyuan'
  | 'ltxv'
  | 'unknown'

export const ARCH_LABEL: Record<LoraArch, string> = {
  pony: 'Pony Diffusion XL',
  illustrious: 'Illustrious and NoobAI XL',
  sdxl: 'SDXL 1.0',
  flux1d: 'Flux.1 dev',
  flux2: 'Flux.2 Klein',
  chroma: 'Chroma1',
  'z-image': 'Z-Image',
  'qwen-image': 'Qwen Image',
  'qwen-image-edit': 'Qwen Image Edit',
  anima: 'Anima',
  krea2: 'Krea 2',
  wan: 'Wan 2.2',
  hunyuan: 'HunyuanVideo',
  ltxv: 'LTX-Video',
  unknown: 'no verified base',
}

/**
 * Architectures close enough that a crossing loads and does something partial.
 * Pony, Illustrious and plain SDXL share the SDXL UNet, so the keys match even
 * though the conditioning was trained differently. Crossing inside this set is
 * `untested`, not `mismatch`.
 */
const SDXL_LINEAGE: ReadonlySet<LoraArch> = new Set<LoraArch>(['pony', 'illustrious', 'sdxl'])

/** Names the catalogue's `claims` field uses, mapped to our architectures. */
const ARCH_ALIAS: Record<string, LoraArch> = {
  pony: 'pony',
  ponyxl: 'pony',
  illustrious: 'illustrious',
  illustriousxl: 'illustrious',
  noobai: 'illustrious',
  sdxl: 'sdxl',
  sd15: 'unknown',
  flux: 'flux1d',
  flux1d: 'flux1d',
  chroma: 'chroma',
  'z-image': 'z-image',
  zimage: 'z-image',
}

export type LoraCategory =
  | 'anatomy'
  | 'photoreal'
  | 'anime'
  | 'hands'
  | 'edit'
  | 'speed'
  | 'flux1d-no-base'
  | 'other'

export const CATEGORY_LABEL: Record<LoraCategory, string> = {
  photoreal: 'Realistic skin and texture',
  anime: 'Anime and cartoon',
  hands: 'Hands, eyes and faces',
  edit: 'Changing a picture',
  speed: 'Faster, slightly rougher',
  anatomy: 'Bodies and explicit detail',
  'flux1d-no-base': 'Needs Flux.1 dev, which is not downloaded',
  other: 'In your folder, no details known',
}

/** One line per category, for the picker's section heads. */
export const CATEGORY_NOTE: Record<LoraCategory, string> = {
  anatomy:
    'Explicit anatomy. Most were made from close-up photographs, so they do far more when you paint over a specific area than across a whole picture at once.',
  photoreal: 'Skin texture, pores, film grain and the absence of plastic sheen.',
  anime: 'Line quality, eye detail and screencap flatness on the booru bases.',
  hands: 'The regions the automatic detailers can find on their own.',
  edit: 'Applied to the edit model, which is the one that struggles with changed anatomy.',
  speed: 'Fewer steps for the same picture. A quality trade, not a quality gain.',
  'flux1d-no-base':
    'Trained for Flux.1 dev, which is not installed here. Listed so they are not downloaded twice by mistake.',
  other:
    'Files somebody put in the LoRA folder by hand. No verified base, so no compatibility claim is made about them.',
}

export type LoraUsage = 'base' | 'refine' | 'both'

export const USAGE_LABEL: Record<LoraUsage, string> = {
  base: 'whole picture',
  refine: 'touch-ups only',
  both: 'either',
}

// ---------------------------------------------------------------------------
// The catalogue
// ---------------------------------------------------------------------------

export type CatalogueEntry = {
  /** Filename as it lands in the LoRA folder, and as LoraLoader.lora_name lists it. */
  file: string
  /** The base the weights were verified against, from the README base_model tag. */
  arch: LoraArch
  category: LoraCategory
  /** 1 is a first install, 9 is "only if the base ever arrives". */
  priority: number
  /** Installed checkpoints this was verified against, filename or stem. */
  bases: string[]
  /** Extra families the marketing title claims. Unverified by definition. */
  claims?: string[]
  /** Tokens the LoRA needs in the prompt. Empty when it needs none. */
  trigger: string
  /** The author's recommended strength. Negative means apply it as a negative. */
  recommended: number
  /** A slider LoRA: the sign selects direction and there is no trigger word. */
  slider: boolean
  usage: LoraUsage
  /** Size on the wire, megabytes, for the download estimate. */
  mb: number
  url: string
  does: string
  caution?: string
  caveat?: string
}

/**
 * Verified LoRA metadata. Generated by the research pass from each file's
 * README, not from its Civitai title: where the two disagree, `caveat` records
 * the disagreement and `arch` follows the README.
 */
const LORA_CATALOGUE: readonly CatalogueEntry[] = [
  {
    file: "add-micro-details-concept-illustrious-pony-noobai.safetensors",
    arch: "pony",
    category: "anatomy",
    priority: 1,
    bases: ["ponyDiffusionV6XL.safetensors"],
    claims: ["illustrious", "noobai"],
    trigger: "addmicrodetails",
    recommended: 0.6,
    slider: false,
    usage: "refine",
    mb: 217.9,
    url: "https://huggingface.co/Muapi/add-micro-details-concept-illustrious-pony-noobai/resolve/main/add-micro-details-concept-illustrious-pony-noobai.safetensors",
    does: "Micro-detail across skin, hair and anatomy on all three booru bases. Trigger addmicrodetails. Best applied in the refine pass where the pixels exist to hold the detail.",
    caveat: "The Civitai-derived title also claims illustrious, noobai support, but the README base_model tag says Pony. The tag is what was verified. Treat the extra families as plausible but untested.",
  },
  {
    file: "detailed-pussy.safetensors",
    arch: "illustrious",
    category: "anatomy",
    priority: 1,
    bases: ["semiRealIllustrious_v40", "waiMatureIllustrious_v30", "NoobAI-XL-v1.1.safetensors"],
    trigger: "Detailed Pussy",
    recommended: 0.7,
    slider: false,
    usage: "refine",
    mb: 217.9,
    url: "https://huggingface.co/Muapi/detailed-pussy/resolve/main/detailed-pussy.safetensors",
    does: "Sharpens vulval detail on Illustrious and NoobAI. Built for close-up framing, which is exactly the crop-and-rerender refine pass.",
  },
  {
    file: "penis-real-diffusion-illustrious-lora-realistic.safetensors",
    arch: "pony",
    category: "anatomy",
    priority: 1,
    bases: ["ponyDiffusionV6XL.safetensors"],
    claims: ["illustrious"],
    trigger: "penisQuiron",
    recommended: 0.75,
    slider: false,
    usage: "refine",
    mb: 217.9,
    url: "https://huggingface.co/Muapi/penis-real-diffusion-illustrious-lora-realistic/resolve/main/penis-real-diffusion-illustrious-lora-realistic.safetensors",
    does: "The strongest penis-structure LoRA found for the booru bases: glans, corona and shaft proportion. Trigger penisQuiron. Quiron is a consistently well-trained author in this ecosystem.",
    caveat: "The Civitai-derived title also claims illustrious support, but the README base_model tag says Pony. The tag is what was verified. Treat the extra families as plausible but untested.",
  },
  {
    file: "perfect-pussy-pony-illustrious.safetensors",
    arch: "pony",
    category: "anatomy",
    priority: 1,
    bases: ["ponyDiffusionV6XL.safetensors"],
    claims: ["illustrious"],
    trigger: "P3P05Y, Pussy",
    recommended: 0.7,
    slider: false,
    usage: "refine",
    mb: 217.9,
    url: "https://huggingface.co/Muapi/perfect-pussy-pony-illustrious/resolve/main/perfect-pussy-pony-illustrious.safetensors",
    does: "Coherent labia and clitoral hood geometry for Pony V6. Trigger token P3P05Y is a hard requirement, it does almost nothing untriggered.",
    caveat: "The Civitai-derived title also claims illustrious support, but the README base_model tag says Pony. The tag is what was verified. Treat the extra families as plausible but untested.",
  },
  {
    file: "pornmaster-noobxl-illustrious-add-details.safetensors",
    arch: "illustrious",
    category: "anatomy",
    priority: 1,
    bases: ["semiRealIllustrious_v40", "waiMatureIllustrious_v30", "NoobAI-XL-v1.1.safetensors"],
    trigger: "",
    recommended: 0.5,
    slider: false,
    usage: "both",
    mb: 971.5,
    url: "https://huggingface.co/Muapi/pornmaster-noobxl-illustrious-add-details/resolve/main/pornmaster-noobxl-illustrious-add-details.safetensors",
    does: "Broad explicit-detail lift tuned on NoobAI and Illustrious. 1.0 GB, the largest LoRA in this set, so it carries real capacity. No trigger word, it acts as a general detail bias.",
  },
  {
    file: "real-nipples-and-areola-textures-gmr.safetensors",
    arch: "pony",
    category: "anatomy",
    priority: 1,
    bases: ["ponyDiffusionV6XL.safetensors"],
    trigger: "rnct", // read from ss_tag_frequency: rnct appears in 26 of 26 training images. "rnat" was a transcription error and would have injected a token the LoRA was never trained on.
    recommended: 0.7,
    slider: false,
    usage: "refine",
    mb: 217.9,
    url: "https://huggingface.co/Muapi/real-nipples-and-areola-textures-gmr/resolve/main/real-nipples-and-areola-textures-gmr.safetensors",
    does: "Areolar texture, Montgomery glands and nipple shape on Pony V6. Nipples fail for the same latent-cell reason as genitals, and this is the best fix for them.",
  },
  {
    file: "uncensored-ponyxl.safetensors",
    arch: "pony",
    category: "anatomy",
    priority: 1,
    bases: ["ponyDiffusionV6XL.safetensors"],
    trigger: "spread pussy, cervix, clitoris, urethra, close-up",
    recommended: 0.8,
    slider: false,
    usage: "both",
    mb: 109.2,
    url: "https://huggingface.co/Muapi/uncensored-ponyxl/resolve/main/uncensored-ponyxl.safetensors",
    does: "Restores explicit vulval structure on Pony V6: labia, clitoris, urethra, cervix. Trained on close-up explicit reference, so it holds together when a crop is re-rendered at full resolution.",
  },
  {
    file: "anatomy-helper.safetensors",
    arch: "illustrious",
    category: "anatomy",
    priority: 2,
    bases: ["semiRealIllustrious_v40", "waiMatureIllustrious_v30", "NoobAI-XL-v1.1.safetensors"],
    trigger: "",
    recommended: 0.6,
    slider: false,
    usage: "base",
    mb: 217.9,
    url: "https://huggingface.co/Muapi/anatomy-helper/resolve/main/anatomy-helper.safetensors",
    does: "General limb and torso proportion correction on Illustrious. No trigger word. Works in the base pass, unlike the region LoRAs.",
  },
  {
    file: "bad-anatomy-sd-xl-negative-lora-improved-anatomy-for-xl-models.safetensors",
    arch: "sdxl",
    category: "anatomy",
    priority: 2,
    bases: ["ponyDiffusionV6XL.safetensors", "NoobAI-XL-v1.1.safetensors", "semiRealIllustrious_v40", "waiMatureIllustrious_v30"],
    trigger: "bad anatomy, bad hands, bad feet",
    recommended: -1.0,
    slider: false,
    usage: "base",
    mb: 650.4,
    url: "https://huggingface.co/Muapi/bad-anatomy-sd-xl-negative-lora-improved-anatomy-for-xl-models/resolve/main/bad-anatomy-sd-xl-negative-lora-improved-anatomy-for-xl-models.safetensors",
    does: "A negative LoRA: trained ON malformed anatomy so you apply it at NEGATIVE weight to push away from it. Alternative usage is positive weight with its trigger words in the negative prompt. Trained on SDXL 1.0 base, so it is weaker on Pony and Illustrious than a native-base LoRA.",
    caution: "SDXL 1.0 base training. Loads on Pony and Illustrious but conditioning differs, so expect a partial effect.",
  },
  {
    file: "breast-size-slider-pony-illustrious.safetensors",
    arch: "illustrious",
    category: "anatomy",
    priority: 2,
    bases: ["semiRealIllustrious_v40", "waiMatureIllustrious_v30", "NoobAI-XL-v1.1.safetensors"],
    claims: ["pony"],
    trigger: "",
    recommended: 1.0,
    slider: true,
    usage: "base",
    mb: 8.4,
    url: "https://huggingface.co/Muapi/breast-size-slider-pony-illustrious/resolve/main/breast-size-slider-pony-illustrious.safetensors",
    does: "Breast size as a continuous axis. Negative shrinks, positive enlarges. No trigger word, the weight IS the control.",
    caveat: "The Civitai-derived title also claims pony support, but the README base_model tag says Illustrious. The tag is what was verified. Treat the extra families as plausible but untested.",
  },
  {
    file: "clitoris.safetensors",
    arch: "pony",
    category: "anatomy",
    priority: 2,
    bases: ["ponyDiffusionV6XL.safetensors"],
    trigger: "big clitoris, clitoris, erection",
    recommended: 0.6,
    slider: false,
    usage: "refine",
    mb: 54.8,
    url: "https://huggingface.co/Muapi/clitoris/resolve/main/clitoris.safetensors",
    does: "Clitoral glans and hood specifically, the single smallest structure in the region and the first thing the base model dissolves.",
  },
  {
    file: "innie-pussy-aka-puffy-pussy-pony.safetensors",
    arch: "pony",
    category: "anatomy",
    priority: 2,
    bases: ["ponyDiffusionV6XL.safetensors"],
    trigger: "pfbk",
    recommended: 0.7,
    slider: false,
    usage: "refine",
    mb: 772.2,
    url: "https://huggingface.co/Muapi/innie-pussy-aka-puffy-pussy-pony/resolve/main/innie-pussy-aka-puffy-pussy-pony.safetensors",
    does: "The opposite morphology to the meaty-labia LoRA: closed, puffy vulva on Pony V6. Pair with it, do not stack both.",
  },
  {
    file: "lora-illustriousxl-improved-spread-pussy.safetensors",
    arch: "illustrious",
    category: "anatomy",
    priority: 2,
    bases: ["semiRealIllustrious_v40", "waiMatureIllustrious_v30", "NoobAI-XL-v1.1.safetensors"],
    trigger: "spread pussy, urethra, clitoris, clitoral hood, uterus, cervix, uncensored,",
    recommended: 0.7,
    slider: false,
    usage: "refine",
    mb: 217.9,
    url: "https://huggingface.co/Muapi/lora-illustriousxl-improved-spread-pussy/resolve/main/lora-illustriousxl-improved-spread-pussy.safetensors",
    does: "Open-vulva poses on Illustrious, where the base model reliably smears internal structure. Trigger list names urethra and clitoral hood explicitly.",
  },
  {
    file: "meaty-labia-slutty-pussy-anima-il.safetensors",
    arch: "illustrious",
    category: "anatomy",
    priority: 2,
    bases: ["semiRealIllustrious_v40", "waiMatureIllustrious_v30", "NoobAI-XL-v1.1.safetensors"],
    claims: ["anima"],
    trigger: "5lutty pussy, long labia, dark labia, urethra,, pink nipples,",
    recommended: 0.65,
    slider: false,
    usage: "refine",
    mb: 217.9,
    url: "https://huggingface.co/Muapi/meaty-labia-slutty-pussy-anima-il/resolve/main/meaty-labia-slutty-pussy-anima-il.safetensors",
    does: "Labia minora prominence and urethral detail on Illustrious. Also listed for the Anima family, which covers miaomiaoHarem and oneObsession.",
    caveat: "The Civitai-derived title also claims anima support, but the README base_model tag says Illustrious. The tag is what was verified. Treat the extra families as plausible but untested.",
  },
  {
    file: "ponyxl-lora-concept-unshaven-pubic-hair.safetensors",
    arch: "pony",
    category: "anatomy",
    priority: 2,
    bases: ["ponyDiffusionV6XL.safetensors"],
    trigger: "hairy pussy",
    recommended: 0.6,
    slider: false,
    usage: "both",
    mb: 217.9,
    url: "https://huggingface.co/Muapi/ponyxl-lora-concept-unshaven-pubic-hair/resolve/main/ponyxl-lora-concept-unshaven-pubic-hair.safetensors",
    does: "Pubic hair as real hair rather than an airbrushed smudge. Most-downloaded pubic LoRA in the set, and hair at the mons is a common giveaway of low regional resolution.",
  },
  {
    file: "taretiti-sagging-breasts.safetensors",
    arch: "illustrious",
    category: "anatomy",
    priority: 2,
    bases: ["semiRealIllustrious_v40", "waiMatureIllustrious_v30", "NoobAI-XL-v1.1.safetensors"],
    trigger: "1girl,sagging breasts,hanging breasts,breasts,solo,cleavage,huge breasts,blush,large breasts,looking at viewer,white background,sweat",
    recommended: 0.7,
    slider: false,
    usage: "both",
    mb: 979.0,
    url: "https://huggingface.co/Muapi/taretiti-sagging-breasts/resolve/main/taretiti-sagging-breasts.safetensors",
    does: "Natural breast ptosis and volume on Illustrious. 1.0 GB. Counters the default floating-sphere breast shape, which is a proportion failure rather than a resolution one.",
  },
  {
    file: "areolae-size-slider.safetensors",
    arch: "illustrious",
    category: "anatomy",
    priority: 3,
    bases: ["semiRealIllustrious_v40", "waiMatureIllustrious_v30", "NoobAI-XL-v1.1.safetensors"],
    trigger: "large areolae",
    recommended: 1.0,
    slider: true,
    usage: "refine",
    mb: 24.4,
    url: "https://huggingface.co/Muapi/areolae-size-slider/resolve/main/areolae-size-slider.safetensors",
    does: "Areola diameter axis on Illustrious, independent of nipple size. Has a trigger, large areolae.",
  },
  {
    file: "big-natural-breasts-for-pony-diffusion-saggy-breasts-big-naturals.safetensors",
    arch: "pony",
    category: "anatomy",
    priority: 3,
    bases: ["ponyDiffusionV6XL.safetensors"],
    trigger: "## 🧠 Usage (Python)",
    recommended: 0.65,
    slider: false,
    usage: "both",
    mb: 217.9,
    url: "https://huggingface.co/Muapi/big-natural-breasts-for-pony-diffusion-saggy-breasts-big-naturals/resolve/main/big-natural-breasts-for-pony-diffusion-saggy-breasts-big-naturals.safetensors",
    does: "Natural breast shape and weight on Pony V6.",
  },
  {
    file: "breast-size-slider-lora-illustriousxl.safetensors",
    arch: "illustrious",
    category: "anatomy",
    priority: 3,
    bases: ["semiRealIllustrious_v40", "waiMatureIllustrious_v30", "NoobAI-XL-v1.1.safetensors"],
    trigger: "",
    recommended: 1.0,
    slider: true,
    usage: "base",
    mb: 8.4,
    url: "https://huggingface.co/Muapi/breast-size-slider-lora-illustriousxl/resolve/main/breast-size-slider-lora-illustriousxl.safetensors",
    does: "Illustrious-specific breast size slider. Distinct file from the Pony one, confirmed by sha256.",
  },
  {
    file: "fucked-silly-lewd-details-enchancer-v2.0-extreme-sex-illustriousxl-noobai-pony-xl.safetensors",
    arch: "pony",
    category: "anatomy",
    priority: 3,
    bases: ["ponyDiffusionV6XL.safetensors"],
    claims: ["noobai"],
    trigger: "extreme sex",
    recommended: 0.5,
    slider: false,
    usage: "refine",
    mb: 435.3,
    url: "https://huggingface.co/Muapi/fucked-silly-lewd-details-enchancer-v2.0-extreme-sex-illustriousxl-noobai-pony-xl/resolve/main/fucked-silly-lewd-details-enchancer-v2.0-extreme-sex-illustriousxl-noobai-pony-xl.safetensors",
    does: "Explicit-scene detail enhancer spanning Illustrious, NoobAI and Pony. Also pushes facial expression hard, so keep strength low if you do not want that.",
    caveat: "The Civitai-derived title also claims noobai support, but the README base_model tag says Pony. The tag is what was verified. Treat the extra families as plausible but untested.",
  },
  {
    file: "gaping-pussy-illustrious-xl.safetensors",
    arch: "illustrious",
    category: "anatomy",
    priority: 3,
    bases: ["semiRealIllustrious_v40", "waiMatureIllustrious_v30", "NoobAI-XL-v1.1.safetensors"],
    trigger: "gaping pussy, pussy, spread pussy, 1girl, male hands, pov, hetero, gaping pussy, pussy, spread pussy, 1girl, own hands, solo, legs up",
    recommended: 0.6,
    slider: false,
    usage: "refine",
    mb: 453.6,
    url: "https://huggingface.co/Muapi/gaping-pussy-illustrious-xl/resolve/main/gaping-pussy-illustrious-xl.safetensors",
    does: "Wide-open vaginal opening with interior depth on Illustrious. Narrow use, high strength will distort at anything but close framing.",
  },
  {
    file: "nipple-size-slider-pony-sdxl.safetensors",
    arch: "pony",
    category: "anatomy",
    priority: 3,
    bases: ["ponyDiffusionV6XL.safetensors"],
    claims: ["sdxl"],
    trigger: "",
    recommended: 1.0,
    slider: true,
    usage: "refine",
    mb: 8.4,
    url: "https://huggingface.co/Muapi/nipple-size-slider-pony-sdxl/resolve/main/nipple-size-slider-pony-sdxl.safetensors",
    does: "Nipple size axis for Pony.",
    caveat: "The Civitai-derived title also claims sdxl support, but the README base_model tag says Pony. The tag is what was verified. Treat the extra families as plausible but untested.",
  },
  {
    file: "penis-size-slider-pony-illustrious.safetensors",
    arch: "illustrious",
    category: "anatomy",
    priority: 3,
    bases: ["semiRealIllustrious_v40", "waiMatureIllustrious_v30", "NoobAI-XL-v1.1.safetensors"],
    claims: ["pony"],
    trigger: "",
    recommended: 1.0,
    slider: true,
    usage: "refine",
    mb: 8.4,
    url: "https://huggingface.co/Muapi/penis-size-slider-pony-illustrious/resolve/main/penis-size-slider-pony-illustrious.safetensors",
    does: "Penis length axis across Pony and Illustrious.",
    caveat: "The Civitai-derived title also claims pony support, but the README base_model tag says Illustrious. The tag is what was verified. Treat the extra families as plausible but untested.",
  },
  {
    file: "pubic-hair-slider-pony-illustrious.safetensors",
    arch: "pony",
    category: "anatomy",
    priority: 3,
    bases: ["ponyDiffusionV6XL.safetensors"],
    claims: ["illustrious"],
    trigger: "## 🧠 Usage (Python)",
    recommended: 1.0,
    slider: true,
    usage: "both",
    mb: 8.4,
    url: "https://huggingface.co/Muapi/pubic-hair-slider-pony-illustrious/resolve/main/pubic-hair-slider-pony-illustrious.safetensors",
    does: "Pubic hair density axis, bare through full.",
    caveat: "The Civitai-derived title also claims illustrious support, but the README base_model tag says Pony. The tag is what was verified. Treat the extra families as plausible but untested.",
  },
  {
    file: "puffy-nipples-ponyxl.safetensors",
    arch: "pony",
    category: "anatomy",
    priority: 3,
    bases: ["ponyDiffusionV6XL.safetensors"],
    trigger: "PuffyNips",
    recommended: 0.65,
    slider: false,
    usage: "refine",
    mb: 217.9,
    url: "https://huggingface.co/Muapi/puffy-nipples-ponyxl/resolve/main/puffy-nipples-ponyxl.safetensors",
    does: "Puffy areola morphology on Pony V6. Morphology-specific, not a general quality lift.",
  },
  {
    file: "pussy-of-queens-realistic-tight-pussy.safetensors",
    arch: "pony",
    category: "anatomy",
    priority: 3,
    bases: ["ponyDiffusionV6XL.safetensors"],
    trigger: "",
    recommended: 0.7,
    slider: false,
    usage: "refine",
    mb: 40.9,
    url: "https://huggingface.co/Muapi/pussy-of-queens-realistic-tight-pussy/resolve/main/pussy-of-queens-realistic-tight-pussy.safetensors",
    does: "Photoreal-leaning vulva on Pony V6. Small file, light touch, stacks cleanly under a stronger structural LoRA.",
  },
  {
    file: "vagina-size-opening-slider-pony-sdxl.safetensors",
    arch: "pony",
    category: "anatomy",
    priority: 3,
    bases: ["ponyDiffusionV6XL.safetensors"],
    claims: ["sdxl"],
    trigger: "",
    recommended: 1.0,
    slider: true,
    usage: "refine",
    mb: 8.4,
    url: "https://huggingface.co/Muapi/vagina-size-opening-slider-pony-sdxl/resolve/main/vagina-size-opening-slider-pony-sdxl.safetensors",
    does: "Vaginal opening aperture axis on Pony.",
    caveat: "The Civitai-derived title also claims sdxl support, but the README base_model tag says Pony. The tag is what was verified. Treat the extra families as plausible but untested.",
  },
  {
    file: "anal-hair-hairy-anus-pubic-hair-illustrious.safetensors",
    arch: "illustrious",
    category: "anatomy",
    priority: 4,
    bases: ["semiRealIllustrious_v40", "waiMatureIllustrious_v30", "NoobAI-XL-v1.1.safetensors"],
    trigger: "anal hair, hairy anus,, pubic hair, excessive pubic hair",
    recommended: 0.55,
    slider: false,
    usage: "refine",
    mb: 218.0,
    url: "https://huggingface.co/Muapi/anal-hair-hairy-anus-pubic-hair-illustrious/resolve/main/anal-hair-hairy-anus-pubic-hair-illustrious.safetensors",
    does: "Perianal and pubic hair detail on Illustrious.",
  },
  {
    file: "huge-nipples.safetensors",
    arch: "illustrious",
    category: "anatomy",
    priority: 4,
    bases: ["semiRealIllustrious_v40", "waiMatureIllustrious_v30", "NoobAI-XL-v1.1.safetensors"],
    trigger: "huge nipples",
    recommended: 0.6,
    slider: false,
    usage: "refine",
    mb: 39.7,
    url: "https://huggingface.co/Muapi/huge-nipples/resolve/main/huge-nipples.safetensors",
    does: "Enlarged nipple morphology on Illustrious.",
  },
  {
    file: "nipple-size-slider-ilxl-pdxl-goofy-ai.safetensors",
    arch: "pony",
    category: "anatomy",
    priority: 4,
    bases: ["ponyDiffusionV6XL.safetensors"],
    trigger: "nipples",
    recommended: 1.0,
    slider: true,
    usage: "refine",
    mb: 8.4,
    url: "https://huggingface.co/Muapi/nipple-size-slider-ilxl-pdxl-goofy-ai/resolve/main/nipple-size-slider-ilxl-pdxl-goofy-ai.safetensors",
    does: "Nipple size axis covering both Illustrious and Pony.",
  },
  {
    file: "penis-thickness-slider-pony.safetensors",
    arch: "pony",
    category: "anatomy",
    priority: 4,
    bases: ["ponyDiffusionV6XL.safetensors"],
    trigger: "",
    recommended: 1.0,
    slider: true,
    usage: "refine",
    mb: 8.4,
    url: "https://huggingface.co/Muapi/penis-thickness-slider-pony/resolve/main/penis-thickness-slider-pony.safetensors",
    does: "Penis girth axis, independent of length.",
  },
  {
    file: "pony-nsfw-explicit-realistic-photography.safetensors",
    arch: "pony",
    category: "photoreal",
    priority: 1,
    bases: ["ponyDiffusionV6XL.safetensors"],
    trigger: "explicit photography, v4n1lla, realistic",
    recommended: 0.7,
    slider: false,
    usage: "base",
    mb: 217.9,
    url: "https://huggingface.co/Muapi/pony-nsfw-explicit-realistic-photography/resolve/main/pony-nsfw-explicit-realistic-photography.safetensors",
    does: "Pushes Pony V6 from illustration toward explicit photography. Trigger explicit photography, v4n1lla, realistic. This is how you get photoreal output from a booru base that already knows anatomy.",
  },
  {
    file: "pytorch_lora_weights.safetensors",
    arch: "z-image",
    category: "photoreal",
    priority: 1,
    bases: ["Z-Image-Turbo-fp8mix.safetensors", "Z-Image-Base-bf16"],
    trigger: "",
    recommended: 0.8,
    slider: false,
    usage: "both",
    mb: 81.2,
    url: "https://huggingface.co/suayptalha/Z-Image-Turbo-Realism-LoRA/resolve/main/pytorch_lora_weights.safetensors",
    does: "Native Z-Image-Turbo realism LoRA. Z-Image is the other modern photoreal base installed, and unlike Chroma it has a real native LoRA ecosystem.",
  },
  {
    file: "realistic-skin-for-pony.safetensors",
    arch: "pony",
    category: "photoreal",
    priority: 1,
    bases: ["ponyDiffusionV6XL.safetensors"],
    trigger: "real skin, body freckles, mole, , <lora:RealisticSkinv2_ponyv6_loraplus:1:start=XX>",
    recommended: 0.65,
    slider: false,
    usage: "refine",
    mb: 109.1,
    url: "https://huggingface.co/Muapi/realistic-skin-for-pony/resolve/main/realistic-skin-for-pony.safetensors",
    does: "Skin texture, freckles and moles native to Pony V6. The right choice when the base is Pony and you want photoreal skin rather than cel shading.",
  },
  {
    file: "skin-realism-acne-skin-details-imperfections-sdxl.safetensors",
    arch: "sdxl",
    category: "photoreal",
    priority: 1,
    bases: ["ponyDiffusionV6XL.safetensors", "NoobAI-XL-v1.1.safetensors", "semiRealIllustrious_v40", "waiMatureIllustrious_v30"],
    trigger: "Detailed natural skin and blemishes without-makeup and acne",
    recommended: 0.6,
    slider: false,
    usage: "refine",
    mb: 217.9,
    url: "https://huggingface.co/Muapi/skin-realism-acne-skin-details-imperfections-sdxl/resolve/main/skin-realism-acne-skin-details-imperfections-sdxl.safetensors",
    does: "Pores, blemishes and skin imperfection on SDXL. The most-downloaded skin LoRA that actually targets an installed family rather than Flux.1-dev. Use it to make the booru bases read as photographic.",
  },
  {
    file: "uncannyPhotorealism_chroma_v13_r128.safetensors",
    arch: "chroma",
    category: "photoreal",
    priority: 1,
    bases: ["Chroma1-HD-fp8mixed.safetensors"],
    trigger: "",
    recommended: 0.9,
    slider: false,
    usage: "both",
    mb: 855.1,
    url: "https://huggingface.co/Omnico/Chroma1_diff_loras/resolve/main/uncannyPhotorealism_chroma_v13_r128.safetensors",
    does: "Rank-128 extracted difference between the uncannyPhotorealism Chroma finetune and Chroma base. Applying it to Chroma1-HD pulls the installed base toward that photoreal finetune without downloading a second 9 GB checkpoint. This is the single best photorealism lever for Chroma, because it is native to Chroma rather than a Flux.1-dev LoRA hoping to load.",
  },
  {
    file: "amateur-style-diffusion-illustrious-pony-lora.safetensors",
    arch: "pony",
    category: "photoreal",
    priority: 2,
    bases: ["ponyDiffusionV6XL.safetensors"],
    claims: ["illustrious"],
    trigger: "amateurQuironStyle, amateurQuiron, flash photo, webcam photo",
    recommended: 0.7,
    slider: false,
    usage: "base",
    mb: 217.9,
    url: "https://huggingface.co/Muapi/amateur-style-diffusion-illustrious-pony-lora/resolve/main/amateur-style-diffusion-illustrious-pony-lora.safetensors",
    does: "Snapshot and flash-photo look on Pony and Illustrious. Amateur framing and lighting is the most reliable route to photoreal, because polished studio lighting reads as rendered.",
    caveat: "The Civitai-derived title also claims illustrious support, but the README base_model tag says Pony. The tag is what was verified. Treat the extra families as plausible but untested.",
  },
  {
    file: "dejpeg_strong.safetensors",
    arch: "z-image",
    category: "photoreal",
    priority: 2,
    bases: ["Z-Image-Turbo-fp8mix.safetensors", "Z-Image-Base-bf16"],
    trigger: "",
    recommended: 0.8,
    slider: false,
    usage: "refine",
    mb: 162.2,
    url: "https://huggingface.co/wcde/Z-Image-Turbo-DeJPEG-Lora/resolve/main/dejpeg_strong.safetensors",
    does: "Removes the JPEG-artifact texture Z-Image inherits from its training data. Directly relevant to a composite pipeline, since blocking artifacts at a crop seam are very visible. Repo also ships dejpeg_lite and dejpeg_detailed at the same size if strong is too aggressive.",
  },
  {
    file: "photonicFusionChroma_debloated_r128.safetensors",
    arch: "chroma",
    category: "photoreal",
    priority: 2,
    bases: ["Chroma1-HD-fp8mixed.safetensors"],
    trigger: "",
    recommended: 0.85,
    slider: false,
    usage: "both",
    mb: 855.1,
    url: "https://huggingface.co/Omnico/Chroma1_diff_loras/resolve/main/photonicFusionChroma_debloated_r128.safetensors",
    does: "Same extraction method against the photonicFusion Chroma finetune. Different photoreal character: cleaner, less filmic. Try against the uncanny one, do not stack at full weight.",
  },
  {
    file: "sdxl-film-photography-style.safetensors",
    arch: "sdxl",
    category: "photoreal",
    priority: 2,
    bases: ["ponyDiffusionV6XL.safetensors", "NoobAI-XL-v1.1.safetensors", "semiRealIllustrious_v40", "waiMatureIllustrious_v30"],
    trigger: "film photography style, light grain, medium grain, heavy grain",
    recommended: 0.6,
    slider: false,
    usage: "base",
    mb: 870.3,
    url: "https://huggingface.co/Muapi/sdxl-film-photography-style/resolve/main/sdxl-film-photography-style.safetensors",
    does: "Film grain and photographic tonality on SDXL. 912 MB. Grain also masks the resolution seam where a refined crop composites back.",
  },
  {
    file: "Y3HY6ZP921WTG1DRKG2R6H4FT0.safetensors",
    arch: "z-image",
    category: "photoreal",
    priority: 2,
    bases: ["Z-Image-Turbo-fp8mix.safetensors", "Z-Image-Base-bf16"],
    trigger: "",
    recommended: 0.7,
    slider: false,
    usage: "refine",
    mb: 162.2,
    url: "https://huggingface.co/thutes-gbr25/fast-high-detail-Z-Image-Turbo-LoRA/resolve/main/Y3HY6ZP921WTG1DRKG2R6H4FT0.safetensors",
    does: "High-frequency detail for Z-Image-Turbo. Aimed at the refine pass, where added detail has pixels to live in.",
  },
  {
    file: "gonzalomoChroma_v30_r128.safetensors",
    arch: "chroma",
    category: "photoreal",
    priority: 3,
    bases: ["Chroma1-HD-fp8mixed.safetensors"],
    trigger: "",
    recommended: 0.85,
    slider: false,
    usage: "both",
    mb: 855.1,
    url: "https://huggingface.co/Omnico/Chroma1_diff_loras/resolve/main/gonzalomoChroma_v30_r128.safetensors",
    does: "Extraction of the gonzalomo Chroma finetune. Third option in the same family, useful as an A/B.",
  },
  {
    file: "detail-slider-lora-illustrious-xl.safetensors",
    arch: "illustrious",
    category: "anime",
    priority: 1,
    bases: ["semiRealIllustrious_v40", "waiMatureIllustrious_v30", "NoobAI-XL-v1.1.safetensors"],
    trigger: "",
    recommended: 1.0,
    slider: true,
    usage: "both",
    mb: 8.4,
    url: "https://huggingface.co/Muapi/detail-slider-lora-illustrious-xl/resolve/main/detail-slider-lora-illustrious-xl.safetensors",
    does: "Detail density axis for Illustrious. Positive adds linework and texture, negative simplifies. The standard first LoRA on any Illustrious stack.",
  },
  {
    file: "detail-slider-lora-ponyxl-sdxl.safetensors",
    arch: "pony",
    category: "anime",
    priority: 1,
    bases: ["ponyDiffusionV6XL.safetensors"],
    claims: ["sdxl"],
    trigger: "",
    recommended: 1.0,
    slider: true,
    usage: "both",
    mb: 16.6,
    url: "https://huggingface.co/Muapi/detail-slider-lora-ponyxl-sdxl/resolve/main/detail-slider-lora-ponyxl-sdxl.safetensors",
    does: "The Pony V6 counterpart of the same slider. Distinct file, confirmed by sha256.",
    caveat: "The Civitai-derived title also claims sdxl support, but the README base_model tag says Pony. The tag is what was verified. Treat the extra families as plausible but untested.",
  },
  {
    file: "eyes-for-illustrious-lora-perfect-anime-eyes.safetensors",
    arch: "illustrious",
    category: "anime",
    priority: 1,
    bases: ["semiRealIllustrious_v40", "waiMatureIllustrious_v30", "NoobAI-XL-v1.1.safetensors"],
    trigger: "perfect eyes",
    recommended: 0.7,
    slider: false,
    usage: "refine",
    mb: 217.9,
    url: "https://huggingface.co/Muapi/eyes-for-illustrious-lora-perfect-anime-eyes/resolve/main/eyes-for-illustrious-lora-perfect-anime-eyes.safetensors",
    does: "Anime eye structure: iris gradient, highlight placement, lash line. Eyes occupy a tiny fraction of the frame, so they fail for the same latent-resolution reason as genitals and benefit most from a refine pass.",
  },
  {
    file: "fine-anime-screencap-xl-anime-screencap-style-lora-illustrious-and-ponyxl.safetensors",
    arch: "pony",
    category: "anime",
    priority: 1,
    bases: ["ponyDiffusionV6XL.safetensors"],
    claims: ["illustrious"],
    trigger: "fine anime screencap_xl, anime screencap",
    recommended: 0.7,
    slider: false,
    usage: "base",
    mb: 40.9,
    url: "https://huggingface.co/Muapi/fine-anime-screencap-xl-anime-screencap-style-lora-illustrious-and-ponyxl/resolve/main/fine-anime-screencap-xl-anime-screencap-style-lora-illustrious-and-ponyxl.safetensors",
    does: "Anime TV screencap look on Illustrious and Pony: flat cel shading, limited palette, hard shadow edges. This is what makes anime output read as anime rather than as a painted render.",
    caveat: "The Civitai-derived title also claims illustrious support, but the README base_model tag says Pony. The tag is what was verified. Treat the extra families as plausible but untested.",
  },
  {
    file: "cartoon-styles.safetensors",
    arch: "illustrious",
    category: "anime",
    priority: 2,
    bases: ["semiRealIllustrious_v40", "waiMatureIllustrious_v30", "NoobAI-XL-v1.1.safetensors"],
    trigger: "Carto4on",
    recommended: 0.7,
    slider: false,
    usage: "base",
    mb: 217.9,
    url: "https://huggingface.co/Muapi/cartoon-styles/resolve/main/cartoon-styles.safetensors",
    does: "Cartoon rendering on Illustrious, trigger Carto4on. The cartoon half of the user's request, distinct from anime.",
  },
  {
    file: "detailer-tool-concept-lora-illustriousxl.safetensors",
    arch: "illustrious",
    category: "anime",
    priority: 2,
    bases: ["semiRealIllustrious_v40", "waiMatureIllustrious_v30", "NoobAI-XL-v1.1.safetensors"],
    trigger: "detailed",
    recommended: 0.6,
    slider: false,
    usage: "both",
    mb: 217.9,
    url: "https://huggingface.co/Muapi/detailer-tool-concept-lora-illustriousxl/resolve/main/detailer-tool-concept-lora-illustriousxl.safetensors",
    does: "General Illustrious detail lift, trigger detailed. Blunter than the slider but has a token you can weight in the prompt.",
  },
  {
    file: "hd-eyes-for-illustrious.safetensors",
    arch: "illustrious",
    category: "anime",
    priority: 2,
    bases: ["semiRealIllustrious_v40", "waiMatureIllustrious_v30", "NoobAI-XL-v1.1.safetensors"],
    trigger: "HDeyes, detailed eyes",
    recommended: 0.65,
    slider: false,
    usage: "refine",
    mb: 54.8,
    url: "https://huggingface.co/Muapi/hd-eyes-for-illustrious/resolve/main/hd-eyes-for-illustrious.safetensors",
    does: "Higher-frequency alternative to the above for Illustrious. Triggers HDeyes, detailed eyes.",
  },
  {
    file: "hentai-studio-quality-anima-illustriousxl-zimageturbo.safetensors",
    arch: "illustrious",
    category: "anime",
    priority: 2,
    bases: ["semiRealIllustrious_v40", "waiMatureIllustrious_v30", "NoobAI-XL-v1.1.safetensors"],
    claims: ["anima", "z-image"],
    trigger: "hentai_studio_quality, shiny skin",
    recommended: 0.65,
    slider: false,
    usage: "base",
    mb: 217.9,
    url: "https://huggingface.co/Muapi/hentai-studio-quality-anima-illustriousxl-zimageturbo/resolve/main/hentai-studio-quality-anima-illustriousxl-zimageturbo.safetensors",
    does: "Studio-production hentai look on Illustrious, and per its own listing on Z-Image-Turbo and the Anima family, which covers the installed miaomiaoHarem and oneObsession models. Trigger hentai_studio_quality.",
    caveat: "The Civitai-derived title also claims anima, z-image support, but the README base_model tag says Illustrious. The tag is what was verified. Treat the extra families as plausible but untested.",
  },
  {
    file: "memaxl-flat-anime-style-noob-illustrious-pony-xl.safetensors",
    arch: "sdxl",
    category: "anime",
    priority: 2,
    bases: ["ponyDiffusionV6XL.safetensors", "NoobAI-XL-v1.1.safetensors", "semiRealIllustrious_v40", "waiMatureIllustrious_v30"],
    claims: ["illustrious", "noobai", "pony"],
    trigger: "",
    recommended: 0.7,
    slider: false,
    usage: "base",
    mb: 217.9,
    url: "https://huggingface.co/Muapi/memaxl-flat-anime-style-noob-illustrious-pony-xl/resolve/main/memaxl-flat-anime-style-noob-illustrious-pony-xl.safetensors",
    does: "Flat anime shading across NoobAI, Illustrious and Pony. Stronger flattening than the screencap LoRA, good when the base drifts toward semi-real.",
    caveat: "The Civitai-derived title also claims illustrious, noobai, pony support, but the README base_model tag says SDXL 1.0. The tag is what was verified. Treat the extra families as plausible but untested.",
  },
  {
    file: "realistic-anime-style-illustrious-pony-flux.safetensors",
    arch: "illustrious",
    category: "anime",
    priority: 2,
    bases: ["semiRealIllustrious_v40", "waiMatureIllustrious_v30", "NoobAI-XL-v1.1.safetensors"],
    claims: ["flux1d", "pony"],
    trigger: "illustriousanime",
    recommended: 0.65,
    slider: false,
    usage: "base",
    mb: 217.9,
    url: "https://huggingface.co/Muapi/realistic-anime-style-illustrious-pony-flux/resolve/main/realistic-anime-style-illustrious-pony-flux.safetensors",
    does: "Semi-real anime, the register semiRealIllustrious and waiMatureIllustrious already sit in. Trigger illustriousanime.",
    caveat: "The Civitai-derived title also claims flux1d, pony support, but the README base_model tag says Illustrious. The tag is what was verified. Treat the extra families as plausible but untested.",
  },
  {
    file: "smooth-detailer-booster-noobai-illustrious-pony.safetensors",
    arch: "pony",
    category: "anime",
    priority: 2,
    bases: ["ponyDiffusionV6XL.safetensors"],
    claims: ["illustrious", "noobai"],
    trigger: "",
    recommended: 0.5,
    slider: false,
    usage: "both",
    mb: 217.9,
    url: "https://huggingface.co/Muapi/smooth-detailer-booster-noobai-illustrious-pony/resolve/main/smooth-detailer-booster-noobai-illustrious-pony.safetensors",
    does: "Detail boost that avoids the crunchy over-sharpened look the harder detail LoRAs cause on NoobAI. No trigger.",
    caveat: "The Civitai-derived title also claims illustrious, noobai support, but the README base_model tag says Pony. The tag is what was verified. Treat the extra families as plausible but untested.",
  },
  {
    file: "90s-anime-aesthetic-anima-illustriousxl-zimageturbo-flux-chroma.safetensors",
    arch: "illustrious",
    category: "anime",
    priority: 3,
    bases: ["semiRealIllustrious_v40", "waiMatureIllustrious_v30", "NoobAI-XL-v1.1.safetensors"],
    claims: ["anima", "chroma", "flux1d", "z-image"],
    trigger: "90s_anime_aesthetic, 1990s (style), retro, retro artstyle",
    recommended: 0.7,
    slider: false,
    usage: "base",
    mb: 243.9,
    url: "https://huggingface.co/Muapi/90s-anime-aesthetic-anima-illustriousxl-zimageturbo-flux-chroma/resolve/main/90s-anime-aesthetic-anima-illustriousxl-zimageturbo-flux-chroma.safetensors",
    does: "Retro 90s cel look. Listed for Illustrious, the Anima family, Z-Image-Turbo and Chroma, so it is one of the few style LoRAs that spans both installed halves.",
    caveat: "The Civitai-derived title also claims anima, chroma, flux1d, z-image support, but the README base_model tag says Illustrious. The tag is what was verified. Treat the extra families as plausible but untested.",
  },
  {
    file: "anime-screencap-xl.safetensors",
    arch: "sdxl",
    category: "anime",
    priority: 3,
    bases: ["ponyDiffusionV6XL.safetensors", "NoobAI-XL-v1.1.safetensors", "semiRealIllustrious_v40", "waiMatureIllustrious_v30"],
    trigger: "anime screencap",
    recommended: 0.65,
    slider: false,
    usage: "base",
    mb: 217.9,
    url: "https://huggingface.co/Muapi/anime-screencap-xl/resolve/main/anime-screencap-xl.safetensors",
    does: "Plain SDXL anime screencap. Useful on NoobAI where the Illustrious-tuned screencap LoRA can overcook.",
  },
  {
    file: "bss-detail-enhancer-slider-xl-il-pn.safetensors",
    arch: "illustrious",
    category: "anime",
    priority: 3,
    bases: ["semiRealIllustrious_v40", "waiMatureIllustrious_v30", "NoobAI-XL-v1.1.safetensors"],
    trigger: "## 🧠 Usage (Python)",
    recommended: 1.0,
    slider: true,
    usage: "both",
    mb: 16.6,
    url: "https://huggingface.co/Muapi/bss-detail-enhancer-slider-xl-il-pn/resolve/main/bss-detail-enhancer-slider-xl-il-pn.safetensors",
    does: "Detail slider covering SDXL, Illustrious and Pony in one file.",
  },
  {
    file: "detail-enhancer-il-pony.safetensors",
    arch: "illustrious",
    category: "anime",
    priority: 3,
    bases: ["semiRealIllustrious_v40", "waiMatureIllustrious_v30", "NoobAI-XL-v1.1.safetensors"],
    claims: ["pony"],
    trigger: "## 🧠 Usage (Python)",
    recommended: 0.6,
    slider: false,
    usage: "both",
    mb: 217.9,
    url: "https://huggingface.co/Muapi/detail-enhancer-il-pony/resolve/main/detail-enhancer-il-pony.safetensors",
    does: "Detail enhancer for Illustrious and Pony, no trigger word.",
    caveat: "The Civitai-derived title also claims pony support, but the README base_model tag says Illustrious. The tag is what was verified. Treat the extra families as plausible but untested.",
  },
  {
    file: "detail-tweaker-illustrious-by-stable-yogi.safetensors",
    arch: "illustrious",
    category: "anime",
    priority: 3,
    bases: ["semiRealIllustrious_v40", "waiMatureIllustrious_v30", "NoobAI-XL-v1.1.safetensors"],
    trigger: "detailed, realistic, photorealstic,",
    recommended: 0.6,
    slider: false,
    usage: "both",
    mb: 217.9,
    url: "https://huggingface.co/Muapi/detail-tweaker-illustrious-by-stable-yogi/resolve/main/detail-tweaker-illustrious-by-stable-yogi.safetensors",
    does: "Detail tweaker for Illustrious with triggers spanning detailed, realistic and photorealistic, so it can bias style as well as density.",
  },
  {
    file: "vixon-s-illustrious-styles-high-detail.safetensors",
    arch: "illustrious",
    category: "anime",
    priority: 3,
    bases: ["semiRealIllustrious_v40", "waiMatureIllustrious_v30", "NoobAI-XL-v1.1.safetensors"],
    trigger: "high_detail",
    recommended: 0.6,
    slider: false,
    usage: "both",
    mb: 217.9,
    url: "https://huggingface.co/Muapi/vixon-s-illustrious-styles-high-detail/resolve/main/vixon-s-illustrious-styles-high-detail.safetensors",
    does: "High-detail Illustrious style, trigger high_detail.",
  },
  {
    file: "good-hands-for-pony.safetensors",
    arch: "pony",
    category: "hands",
    priority: 1,
    bases: ["ponyDiffusionV6XL.safetensors"],
    trigger: "good_hands",
    recommended: 0.7,
    slider: false,
    usage: "refine",
    mb: 217.9,
    url: "https://huggingface.co/Muapi/good-hands-for-pony/resolve/main/good-hands-for-pony.safetensors",
    does: "Finger count and joint structure native to Pony V6. Trigger good_hands. The right LoRA inside a hand_yolov8s detailer pass on a Pony base.",
  },
  {
    file: "nice hands.safetensors",
    arch: "unknown",
    category: "hands",
    priority: 1,
    bases: [],
    claims: ["sdxl"],
    trigger: "",
    recommended: 1.5,
    slider: true,
    usage: "refine",
    mb: 8.4,
    url: "https://huggingface.co/ntc-ai/SDXL-LoRA-slider.nice-hands/resolve/main/nice%20hands.safetensors",
    does: "Hand quality as a slider on SDXL. Pair it with the Impact Pack hand_yolov8s.pt detector: detect hands, crop, re-render the crop with this LoRA at high weight, composite back. That is the whole fix for hands.",
    caveat: "The Civitai-derived title also claims sdxl support, but the README base_model tag says \"stabilityai/stable-diffusion-xl-base-1.0\". The tag is what was verified. Treat the extra families as plausible but untested.",
  },
  {
    file: "sdxl-pdxl-tuning-face-detailer-lora.safetensors",
    arch: "sdxl",
    category: "hands",
    priority: 1,
    bases: ["ponyDiffusionV6XL.safetensors", "NoobAI-XL-v1.1.safetensors", "semiRealIllustrious_v40", "waiMatureIllustrious_v30"],
    trigger: "",
    recommended: 0.6,
    slider: false,
    usage: "refine",
    mb: 38.3,
    url: "https://huggingface.co/Muapi/sdxl-pdxl-tuning-face-detailer-lora/resolve/main/sdxl-pdxl-tuning-face-detailer-lora.safetensors",
    does: "Built specifically to be loaded inside a face-detailer pass on SDXL and Pony, which is exactly how FaceDetailer with face_yolov8m.pt runs. Small file, no trigger.",
  },
  {
    file: "hands-sdxl-beta.safetensors",
    arch: "sdxl",
    category: "hands",
    priority: 2,
    bases: ["ponyDiffusionV6XL.safetensors", "NoobAI-XL-v1.1.safetensors", "semiRealIllustrious_v40", "waiMatureIllustrious_v30"],
    trigger: "handslora",
    recommended: 0.65,
    slider: false,
    usage: "refine",
    mb: 217.9,
    url: "https://huggingface.co/Muapi/hands-sdxl-beta/resolve/main/hands-sdxl-beta.safetensors",
    does: "Hand structure on SDXL, trigger handslora. Fallback when the Pony-native one is too strong.",
  },
  {
    file: "perfect-eyes-xl.safetensors",
    arch: "sdxl",
    category: "hands",
    priority: 2,
    bases: ["ponyDiffusionV6XL.safetensors", "NoobAI-XL-v1.1.safetensors", "semiRealIllustrious_v40", "waiMatureIllustrious_v30"],
    trigger: "green eyes, blue eyes, brown eyes, perfecteyes",
    recommended: 0.6,
    slider: false,
    usage: "refine",
    mb: 217.9,
    url: "https://huggingface.co/Muapi/perfect-eyes-xl/resolve/main/perfect-eyes-xl.safetensors",
    does: "Photoreal eye structure on SDXL, trigger perfecteyes. The counterpart to the anime eye LoRAs, for the photoreal side of a FaceDetailer pass.",
  },
  {
    file: "polyhedron_all-sdxl-1.0-skin-hands-eyes-m-f.safetensors",
    arch: "sdxl",
    category: "hands",
    priority: 2,
    bases: ["ponyDiffusionV6XL.safetensors", "NoobAI-XL-v1.1.safetensors", "semiRealIllustrious_v40", "waiMatureIllustrious_v30"],
    trigger: "perfect eyes, skin blemish, detailed skin",
    recommended: 0.6,
    slider: false,
    usage: "refine",
    mb: 1740.2,
    url: "https://huggingface.co/Muapi/polyhedron_all-sdxl-1.0-skin-hands-eyes-m-f/resolve/main/polyhedron_all-sdxl-1.0-skin-hands-eyes-m-f.safetensors",
    does: "Skin, hands and eyes in one 1.8 GB SDXL LoRA, male and female. Largest-capacity option in this category. Triggers perfect eyes, skin blemish, detailed skin.",
  },
  {
    file: "qwen-edit-enhance_64-v3_000001000.safetensors",
    arch: "qwen-image-edit",
    category: "edit",
    priority: 1,
    bases: ["qwen-image-edit-2511-Q4_K_M.gguf"],
    trigger: "",
    recommended: 0.9,
    slider: false,
    usage: "refine",
    mb: 1125.2,
    url: "https://huggingface.co/vafipas663/Qwen-Edit-2509-Upscale-LoRA/resolve/main/qwen-edit-enhance_64-v3_000001000.safetensors",
    does: "Detail-restoring upscale LoRA for Qwen-Image-Edit. Turns the edit model into a refiner: feed it the upscaled crop and it reconstructs detail instead of merely resampling. Complements 4x-UltraSharp, which only resamples. The rank-64 build is the 1.2 GB one, chosen over the 2.4 GB and 4.7 GB builds to leave VRAM headroom on a 16 GB card.",
    caution: "Trained against Qwen-Image-Edit-2509, while the installed edit model is 2511. Same architecture family and it is widely reported to carry over, but this pairing is unverified here. Test before relying on it.",
  },
  {
    file: "qwen-image-edit-plus-nsfw-lora.safetensors",
    arch: "qwen-image-edit",
    category: "edit",
    priority: 1,
    bases: ["qwen-image-edit-2511-Q4_K_M.gguf"],
    trigger: "",
    recommended: 1.0,
    slider: false,
    usage: "base",
    mb: 562.7,
    url: "https://huggingface.co/ScottzillaSystems/qwen-image-edit-plus-nsfw-lora/resolve/main/qwen-image-edit-plus-nsfw-lora.safetensors",
    does: "MCNL v1, a multi-concept NSFW LoRA whose README declares base_model Qwen/Qwen-Image-Edit-2511, matching the installed qwen-image-edit-2511 GGUF exactly. Qwen-Image-Edit is alignment-trained away from explicit anatomy, so on a nude edit it resolves toward smooth featureless forms no matter how many pixels it has. That is a knowledge problem, not a resolution problem, and this LoRA is the direct fix for the user's edit complaint. By far the highest-download item in this whole set.",
    caution: "The LoRA is fp16 safetensors while the installed base is a Q4_K_M GGUF. ComfyUI applies LoRA patches to GGUF-quantised models, but the effective strength on a 4-bit base is lower than on bf16. Expect to raise strength, and verify it loads before trusting it.",
  },
  {
    file: "flymy_qwen_image_edit_inscene_lora.safetensors",
    arch: "qwen-image-edit",
    category: "edit",
    priority: 2,
    bases: ["qwen-image-edit-2511-Q4_K_M.gguf"],
    trigger: "",
    recommended: 0.8,
    slider: false,
    usage: "base",
    mb: 45.1,
    url: "https://huggingface.co/flymy-ai/qwen-image-edit-inscene-lora/resolve/main/flymy_qwen_image_edit_inscene_lora.safetensors",
    does: "Holds scene, lighting and identity consistent across an edit. 47 MB, the cheapest item in the set. Relevant because the composite step fails visibly when the re-rendered region does not match the surrounding lighting.",
    caution: "Trained against 2509, installed base is 2511. Unverified here.",
  },
  {
    file: "Hyper-Chroma-low-step-LoRA.safetensors",
    arch: "chroma",
    category: "speed",
    priority: 2,
    bases: ["Chroma1-HD-fp8mixed.safetensors"],
    trigger: "",
    recommended: 1.0,
    slider: false,
    usage: "base",
    mb: 107.0,
    url: "https://huggingface.co/clover-supply/Chroma-loras/resolve/main/Hyper-Chroma-low-step-LoRA.safetensors",
    does: "Low-step LoRA for Chroma, cutting a Chroma render to roughly 8 to 12 steps. Listed because the refine pass costs about one full generation per region, so on a 16 GB card step count is the budget. Trades some fidelity for that speed.",
  },
  {
    file: "detailed-perfection-style-hands-feet-face-body-all-in-one-xl-f1d-sd1.5-pony-illu.safetensors",
    arch: "flux1d",
    category: "flux1d-no-base",
    priority: 9,
    bases: [],
    claims: ["pony", "sd15"],
    trigger: "perfection style",
    recommended: 0.7,
    slider: false,
    usage: "base",
    mb: 641.5,
    url: "https://huggingface.co/Muapi/detailed-perfection-style-hands-feet-face-body-all-in-one-xl-f1d-sd1.5-pony-illu/resolve/main/detailed-perfection-style-hands-feet-face-body-all-in-one-xl-f1d-sd1.5-pony-illu.safetensors",
    does: "Hands, feet, face and body in one. Highest-download hands LoRA found, 984. Title claims XL, SD1.5, Pony and Illustrious coverage but the README base_model tag is Flux.1 D.",
    caution: "No installed base accepts Flux.1-dev LoRAs. Chroma1-HD is an architecturally modified FLUX.1-schnell derivative, flux-2-klein-4b is Flux.2, and Z-Image, qwen_image_2.1 and moodyCutieMixKrea2 are separate architectures entirely. Recorded for completeness and in case a Flux.1-dev base is installed later. Where the Civitai title also claims SDXL, Pony or Illustrious support, that claim is not supported by the README metadata and was not verified.",
    caveat: "The Civitai-derived title also claims pony, sd15 support, but the README base_model tag says Flux.1 D. The tag is what was verified. Treat the extra families as plausible but untested.",
  },
  {
    file: "Hands-XL-SD-1.5-FLUX.1-dev-Pony-Illustrious.safetensors",
    arch: "flux1d",
    category: "flux1d-no-base",
    priority: 9,
    bases: [],
    claims: ["illustrious", "pony", "sd15"],
    trigger: "Detailed hand, Hand, Perfect hand",
    recommended: 0.7,
    slider: false,
    usage: "base",
    mb: 327.9,
    url: "https://huggingface.co/Muapi/hands-xl-sd-1.5-flux.1-dev-pony-illustrious/resolve/main/Hands-XL-SD-1.5-FLUX.1-dev-Pony-Illustrious.safetensors",
    does: "Hand detail. Same multi-base title claim, same Flux.1 D tag.",
    caution: "No installed base accepts Flux.1-dev LoRAs. Chroma1-HD is an architecturally modified FLUX.1-schnell derivative, flux-2-klein-4b is Flux.2, and Z-Image, qwen_image_2.1 and moodyCutieMixKrea2 are separate architectures entirely. Recorded for completeness and in case a Flux.1-dev base is installed later. Where the Civitai title also claims SDXL, Pony or Illustrious support, that claim is not supported by the README metadata and was not verified.",
    caveat: "The Civitai-derived title also claims illustrious, pony, sd15 support, but the README base_model tag says Flux.1 D. The tag is what was verified. Treat the extra families as plausible but untested.",
  },
  {
    file: "hyper-realism-lora-by-aidma-flux-illustrious.safetensors",
    arch: "flux1d",
    category: "flux1d-no-base",
    priority: 9,
    bases: [],
    claims: ["illustrious"],
    trigger: "aidmaHyperrealism",
    recommended: 0.7,
    slider: false,
    usage: "base",
    mb: 73.1,
    url: "https://huggingface.co/Muapi/hyper-realism-lora-by-aidma-flux-illustrious/resolve/main/hyper-realism-lora-by-aidma-flux-illustrious.safetensors",
    does: "aidmaHyperrealism. Listing claims Illustrious as well as Flux, but the README base_model tag is Flux.1 D, so the Illustrious claim is unverified.",
    caution: "No installed base accepts Flux.1-dev LoRAs. Chroma1-HD is an architecturally modified FLUX.1-schnell derivative, flux-2-klein-4b is Flux.2, and Z-Image, qwen_image_2.1 and moodyCutieMixKrea2 are separate architectures entirely. Recorded for completeness and in case a Flux.1-dev base is installed later. Where the Civitai title also claims SDXL, Pony or Illustrious support, that claim is not supported by the README metadata and was not verified.",
    caveat: "The Civitai-derived title also claims illustrious support, but the README base_model tag says Flux.1 D. The tag is what was verified. Treat the extra families as plausible but untested.",
  },
  {
    file: "photorealistic-skin-no-plastic-flux.safetensors",
    arch: "flux1d",
    category: "flux1d-no-base",
    priority: 9,
    bases: [],
    trigger: "aidmarealisticskin",
    recommended: 0.7,
    slider: false,
    usage: "base",
    mb: 73.1,
    url: "https://huggingface.co/Muapi/photorealistic-skin-no-plastic-flux/resolve/main/photorealistic-skin-no-plastic-flux.safetensors",
    does: "aidmarealisticskin, the best-known anti-plastic skin LoRA for Flux.1-dev.",
    caution: "No installed base accepts Flux.1-dev LoRAs. Chroma1-HD is an architecturally modified FLUX.1-schnell derivative, flux-2-klein-4b is Flux.2, and Z-Image, qwen_image_2.1 and moodyCutieMixKrea2 are separate architectures entirely. Recorded for completeness and in case a Flux.1-dev base is installed later. Where the Civitai title also claims SDXL, Pony or Illustrious support, that claim is not supported by the README metadata and was not verified.",
  },
  {
    file: "realistic-photos-detailed-skin-textures-flux-v3.safetensors",
    arch: "flux1d",
    category: "flux1d-no-base",
    priority: 9,
    bases: [],
    trigger: "dsv4",
    recommended: 0.7,
    slider: false,
    usage: "base",
    mb: 1168.7,
    url: "https://huggingface.co/Muapi/realistic-photos-detailed-skin-textures-flux-v3/resolve/main/realistic-photos-detailed-skin-textures-flux-v3.safetensors",
    does: "Detailed skin textures v3 for Flux.1-dev, trigger dsv4.",
    caution: "No installed base accepts Flux.1-dev LoRAs. Chroma1-HD is an architecturally modified FLUX.1-schnell derivative, flux-2-klein-4b is Flux.2, and Z-Image, qwen_image_2.1 and moodyCutieMixKrea2 are separate architectures entirely. Recorded for completeness and in case a Flux.1-dev base is installed later. Where the Civitai title also claims SDXL, Pony or Illustrious support, that claim is not supported by the README metadata and was not verified.",
  },
]

// ---------------------------------------------------------------------------
// What is on disk
// ---------------------------------------------------------------------------

/** One row of /api/models. Same shape hardware.ts reads. */
type ModelFile = { name: string; rel: string; folder: string; size: number; mtime: number }

/**
 * The folder ComfyUI loads LoRAs from, as it appears under the models root.
 * It is `Lora` on this machine and `loras` on a default install, so it is
 * detected from the files themselves rather than assumed, and only used as a
 * fallback for the download destination when nothing is installed yet.
 */
const LORA_FOLDER = /^(lora|loras)$/i
const DEFAULT_LORA_FOLDER = 'Lora'

const isLoraFile = (f: ModelFile) => LORA_FOLDER.test(f.folder.split(/[/\\]/)[0] ?? '')

/**
 * The name LoraLoader wants. Its enum is relative to the LoRA folder root, so
 * `Lora/sub/thing.safetensors` on disk is `sub/thing.safetensors` in the graph.
 * Getting this wrong produces a queue time enum error rather than a bad
 * picture, which at least is loud.
 */
function loraNameOf(f: ModelFile): string {
  const parts = f.rel.replace(/\\/g, '/').split('/')
  return parts.length > 1 ? parts.slice(1).join('/') : parts[0]
}

const stem = (file: string) =>
  file
    .replace(/\\/g, '/')
    .split('/')
    .pop()!
    .replace(/\.(safetensors|ckpt|pt|pth|sft|bin|gguf)$/i, '')
    .toLowerCase()

const CATALOGUE_BY_STEM: ReadonlyMap<string, CatalogueEntry> = new Map(
  LORA_CATALOGUE.map(e => [stem(e.file), e]),
)

/** `detailed-pussy.safetensors` becomes `Detailed pussy`. */
function titleOf(file: string): string {
  const base = stem(file).replace(/[-_]+/g, ' ').replace(/\s+/g, ' ').trim()
  return base.charAt(0).toUpperCase() + base.slice(1)
}

/** A LoRA the user can pick: catalogue metadata joined to what is on disk. */
export type LoraInfo = {
  /** lora_name as LoraLoader expects it. The stack's key. */
  file: string
  label: string
  installed: boolean
  /** Real bytes when installed, else the catalogue's download size. */
  bytes: number
  /** True when `bytes` is the wire size rather than a stat of a real file. */
  approxBytes: boolean
  arch: LoraArch
  category: LoraCategory
  priority: number
  bases: string[]
  claims: string[]
  trigger: string
  recommended: number
  slider: boolean
  usage: LoraUsage
  does: string
  caution?: string
  caveat?: string
  url?: string
}

export type LoraLibrary = {
  all: readonly LoraInfo[]
  byFile: ReadonlyMap<string, LoraInfo>
  /** Folder LoRAs live in, for the download destination. */
  folder: string
  installed: number
  /** On disk but absent from the catalogue, so with no verified base. */
  unlisted: number
  /** In the catalogue but not on disk yet. */
  available: number
}

export const EMPTY_LIBRARY: LoraLibrary = {
  all: [],
  byFile: new Map(),
  folder: DEFAULT_LORA_FOLDER,
  installed: 0,
  unlisted: 0,
  available: 0,
}

function infoFromCatalogue(e: CatalogueEntry, onDisk: ModelFile | null, file: string): LoraInfo {
  return {
    file,
    label: titleOf(e.file),
    installed: !!onDisk,
    bytes: onDisk ? onDisk.size : Math.round(e.mb * 1024 * 1024),
    approxBytes: !onDisk,
    arch: e.arch,
    category: e.category,
    priority: e.priority,
    bases: e.bases,
    claims: e.claims ?? [],
    trigger: e.trigger,
    recommended: e.recommended,
    slider: e.slider,
    usage: e.usage,
    does: e.does,
    caution: e.caution,
    caveat: e.caveat,
    url: e.url,
  }
}

/**
 * A file in the folder that the catalogue has never heard of.
 *
 * Its architecture is guessed from the filename, and only from the filename,
 * because that is all there is. A guess is better than nothing here: the two
 * Wan video LoRAs shipped with this machine are named for their model, and
 * without the guess they would read as plausible on an SDXL checkpoint. Where
 * the name says nothing the architecture stays unknown, which the picker shows
 * as a caution rather than as a fit.
 */
function infoFromDisk(f: ModelFile): LoraInfo {
  const guess = archFor(null, f.name)
  return {
    file: loraNameOf(f),
    label: titleOf(f.name),
    installed: true,
    bytes: f.size,
    approxBytes: false,
    arch: guess,
    category: 'other',
    priority: 5,
    bases: [],
    claims: [],
    trigger: '',
    recommended: 0.7,
    slider: false,
    usage: 'both',
    does:
      guess === 'unknown'
        ? 'Not in the verified catalogue, and the filename says nothing about its base. It is offered without a compatibility claim.'
        : `Not in the verified catalogue. The filename suggests ${ARCH_LABEL[guess]}, which is a guess rather than a read of the file.`,
  }
}

/**
 * Join the LoRA folder to the catalogue.
 *
 * Installed files win on every measurable field: a real stat beats a published
 * size, and a file that is present is loadable whatever the catalogue thinks.
 * Catalogue rows with no file are still returned, marked not installed, so the
 * picker can offer to fetch them. Sorted by category, then by priority, then by
 * name, which puts the anatomy LoRAs the user asked about at the top.
 */
export async function loadLoraLibrary(): Promise<LoraLibrary> {
  let files: ModelFile[] = []
  try {
    const r = await fetch('/api/models')
    if (!r.ok) throw new Error(`HTTP ${r.status}`)
    const body = (await r.json()) as { files?: ModelFile[] }
    files = (body.files ?? []).filter(isLoraFile)
  } catch {
    // No server, or the endpoint is down. The catalogue still describes what
    // exists; everything simply reads as not installed.
    files = []
  }

  const folder = files[0]?.folder.split(/[/\\]/)[0] ?? DEFAULT_LORA_FOLDER
  const byStem = new Map(files.map(f => [stem(f.name), f]))
  const out: LoraInfo[] = []
  let unlisted = 0

  for (const e of LORA_CATALOGUE) {
    const hit = byStem.get(stem(e.file)) ?? null
    out.push(infoFromCatalogue(e, hit, hit ? loraNameOf(hit) : e.file))
  }
  for (const f of files) {
    if (CATALOGUE_BY_STEM.has(stem(f.name))) continue
    // Two files with the same stem in different subfolders: keep the one the
    // stem index points at, so the picker does not list the same LoRA twice.
    if (byStem.get(stem(f.name)) !== f) continue
    out.push(infoFromDisk(f))
    unlisted += 1
  }

  // Section order, which is also what the picker opens on. 'anatomy' used to
  // lead, so the first thing anyone saw when browsing add-ons was the explicit
  // section, whatever they were making. It is still here and still complete -
  // it just no longer greets everyone at the door.
  const rank: LoraCategory[] = [
    'photoreal',
    'anime',
    'hands',
    'edit',
    'speed',
    'anatomy',
    'other',
    'flux1d-no-base',
  ]
  out.sort(
    (a, b) =>
      rank.indexOf(a.category) - rank.indexOf(b.category) ||
      a.priority - b.priority ||
      a.label.localeCompare(b.label),
  )

  return {
    all: out,
    byFile: new Map(out.map(i => [i.file, i])),
    folder,
    installed: out.filter(i => i.installed).length,
    unlisted,
    available: out.filter(i => !i.installed).length,
  }
}

// ---------------------------------------------------------------------------
// Compatibility
// ---------------------------------------------------------------------------

export type FitLevel = 'match' | 'untested' | 'mismatch'

export type Fit = { level: FitLevel; why: string }

/** The checkpoint a stack is being built against. */
export type LoraTarget = {
  familyId: string
  /** Checkpoint filename, which decides the architecture inside a mixed family. */
  model: string
  arch: LoraArch
}

/**
 * Architecture of the selected checkpoint.
 *
 * The filename is read first and the family id second. One registry family can
 * carry several checkpoints of different lineages, and a LoRA cares about the
 * weights it is patched onto, not about which tab the user found them under.
 */
export function archFor(def: FamilyDef | null, model: string): LoraArch {
  const name = (model || '').toLowerCase()
  if (/pony/.test(name)) return 'pony'
  // NoobAI-XL is Illustrious lineage: same UNet key layout, same LoRAs.
  if (/noobai|noob[-_]?xl|illustrious|waimature|wai[-_]?mature|semireal/.test(name)) {
    return 'illustrious'
  }
  if (/chroma/.test(name)) return 'chroma'
  if (/z[-_]?image/.test(name)) return 'z-image'
  if (/qwen.*edit|edit.*qwen/.test(name)) return 'qwen-image-edit'
  if (/qwen/.test(name)) return 'qwen-image'
  if (/klein|flux[-_]?2/.test(name)) return 'flux2'
  if (/flux/.test(name)) return 'flux1d'
  if (/anima|miaomiao|obsession/.test(name)) return 'anima'
  if (/krea/.test(name)) return 'krea2'
  if (/hunyuan/.test(name)) return 'hunyuan'
  if (/ltxv?[-_]/.test(name)) return 'ltxv'
  if (/wan/.test(name)) return 'wan'

  const id = (def?.id ?? '').toLowerCase()
  if (id.includes('pony')) return 'pony'
  if (id.includes('illustrious') || id.includes('noobai')) return 'illustrious'
  if (id.includes('chroma')) return 'chroma'
  if (id.includes('z-image')) return 'z-image'
  if (id.includes('qwen-image-edit')) return 'qwen-image-edit'
  if (id.includes('qwen-image')) return 'qwen-image'
  if (id.includes('klein') || id.includes('flux2')) return 'flux2'
  if (id.includes('anima')) return 'anima'
  if (id.includes('krea')) return 'krea2'
  if (id.includes('hunyuan')) return 'hunyuan'
  if (id.includes('ltxv')) return 'ltxv'
  if (id.startsWith('wan')) return 'wan'
  if (id.includes('sdxl')) return 'sdxl'
  return 'unknown'
}

export function targetFor(def: FamilyDef | null, model: string): LoraTarget {
  return { familyId: def?.id ?? '', model, arch: archFor(def, model) }
}

/**
 * Whether this LoRA belongs on this checkpoint.
 *
 * Four tests in descending order of evidence: a verified pairing, a matching
 * architecture, a claim on the model card, and shared SDXL lineage. Anything
 * else is a mismatch, and a mismatch is never queued.
 */
export function fitFor(info: LoraInfo, target: LoraTarget): Fit {
  const model = stem(target.model)
  if (model && info.bases.some(b => stem(b) === model)) {
    return { level: 'match', why: `Verified against ${titleOf(target.model)}.` }
  }
  if (info.arch !== 'unknown' && info.arch === target.arch) {
    return { level: 'match', why: `Trained on ${ARCH_LABEL[info.arch]}, which is what is loaded.` }
  }
  if (info.claims.some(c => ARCH_ALIAS[c.toLowerCase()] === target.arch)) {
    return {
      level: 'untested',
      why: `The model card also claims ${ARCH_LABEL[target.arch]}, but the file's own base tag says ${ARCH_LABEL[info.arch]}. It will load. Whether it helps here has not been checked.`,
    }
  }
  if (SDXL_LINEAGE.has(info.arch) && SDXL_LINEAGE.has(target.arch)) {
    return {
      level: 'untested',
      why: `Made for ${ARCH_LABEL[info.arch]}, being used on ${ARCH_LABEL[target.arch]}. They are close relatives, so it works, but only partly. Expect a weaker version of what it promises.`,
    }
  }
  if (info.arch === 'unknown') {
    return {
      level: 'untested',
      why: 'No verified base for this file. It may do nothing, or it may do damage. Try it at low strength and watch one picture.',
    }
  }
  return {
    level: 'mismatch',
    why: `Made for ${ARCH_LABEL[info.arch]}. ${ARCH_LABEL[target.arch]} is a different kind of model entirely. It would not fail, it would quietly spoil the picture with no error anywhere, so it is never sent.`,
  }
}

// ---------------------------------------------------------------------------
// The stack
// ---------------------------------------------------------------------------

export type StackEntry = {
  /** lora_name, the key into LoraLibrary.byFile. */
  file: string
  strength: number
  /**
   * Text encoder strength. Left undefined it follows the model strength, which
   * is right for almost every LoRA. Only checkpoint families patch CLIP at all.
   */
  clipStrength?: number
  /** Off keeps the entry and its strength, and leaves it out of the graph. */
  enabled: boolean
}

export type LoraStack = readonly StackEntry[]

/**
 * The ordinary strength band. LoraLoader itself accepts minus one hundred to
 * one hundred; nothing useful lives out there. Above 1.5 a LoRA stops adding
 * its subject and starts burning contrast and melting hands.
 */
export const STRENGTH = { min: 0, max: 1.5, step: 0.05, default: 0.7 } as const

/**
 * Slider LoRAs are the exception, and they are the reason this is not just
 * clamped to zero at the bottom. A breast size or areola size slider encodes a
 * direction: positive enlarges, negative shrinks, and the useful range the
 * authors publish is minus three to three. Clamping those to zero would remove
 * half of what the user actually asked for.
 */
const SLIDER_STRENGTH = { min: -3, max: 3, step: 0.1, default: 1 } as const

/** Negative LoRAs, applied below zero to subtract what they encode. */
const NEGATIVE_STRENGTH = { min: -1.5, max: 1.5, step: 0.05, default: -1 } as const

export function boundsFor(info: LoraInfo | undefined): {
  min: number
  max: number
  step: number
  default: number
} {
  if (!info) return STRENGTH
  if (info.slider) return SLIDER_STRENGTH
  if (info.recommended < 0) return NEGATIVE_STRENGTH
  return STRENGTH
}

export function defaultStrength(info: LoraInfo | undefined): number {
  const b = boundsFor(info)
  const want = info?.recommended ?? b.default
  return Math.min(b.max, Math.max(b.min, Math.round(want * 100) / 100))
}

/**
 * Past four LoRAs the patches start fighting: each one rewrites the same
 * attention weights, and the sixth is usually the reason the fifth stopped
 * working. It is a soft limit, said out loud, not enforced.
 */
export const STACK_ADVICE_AT = 4

export function addToStack(stack: LoraStack, info: LoraInfo): LoraStack {
  if (stack.some(e => e.file === info.file)) return stack
  return [...stack, { file: info.file, strength: defaultStrength(info), enabled: true }]
}

export function removeFromStack(stack: LoraStack, file: string): LoraStack {
  return stack.filter(e => e.file !== file)
}

export function patchStack(stack: LoraStack, file: string, patch: Partial<StackEntry>): LoraStack {
  return stack.map(e => (e.file === file ? { ...e, ...patch } : e))
}

/**
 * Move one entry. Order is load order, and it matters: each LoraLoader patches
 * the model the previous one produced, so a style LoRA after an anatomy LoRA
 * paints over it rather than beside it.
 */
export function moveInStack(stack: LoraStack, from: number, to: number): LoraStack {
  if (from === to || from < 0 || from >= stack.length) return stack
  const next = [...stack]
  const [moved] = next.splice(from, 1)
  next.splice(Math.min(Math.max(0, to), next.length), 0, moved)
  return next
}

/**
 * Trigger tokens the enabled entries need in the prompt, deduplicated.
 *
 * Only entries that will actually be sent are counted. Asking the user to type
 * `addmicrodetails` for a LoRA that is not downloaded, or that is about to be
 * dropped for the wrong architecture, is asking them to poison a prompt on
 * behalf of a LoRA that never loads.
 */
export function triggersFor(stack: LoraStack, lib: LoraLibrary, target?: LoraTarget): string[] {
  const seen = new Set<string>()
  for (const e of stack) {
    if (!e.enabled || e.strength === 0) continue
    const info = lib.byFile.get(e.file)
    if (!info || !info.installed) continue
    if (target && fitFor(info, target).level === 'mismatch') continue
    const t = bestTrigger(e.file, info.trigger).trim()
    if (t) seen.add(t)
  }
  return [...seen]
}

/** Trigger tokens that are needed and are not in the prompt yet. */
export function missingTriggers(
  stack: LoraStack,
  lib: LoraLibrary,
  prompt: string,
  target?: LoraTarget,
): string[] {
  const text = prompt.toLowerCase()
  return triggersFor(stack, lib, target).filter(t => !text.includes(t.toLowerCase()))
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

const KEY = (familyId: string) => `switchgen.loras.v1.${familyId}`

/**
 * localStorage throws on the getter itself in a locked down window, so every
 * access is wrapped. A stack that cannot be saved is not an error worth
 * interrupting a generation for: it lasts the session and says nothing.
 */
function readRaw(key: string): string | null {
  try {
    return localStorage.getItem(key)
  } catch {
    return null
  }
}

function writeRaw(key: string, value: string): boolean {
  try {
    localStorage.setItem(key, value)
    return true
  } catch {
    return false
  }
}

/** Anything a previous version, a hand edit or a quota failure may have left. */
function sanitise(value: unknown): LoraStack {
  if (!Array.isArray(value)) return []
  const out: StackEntry[] = []
  for (const raw of value) {
    if (!raw || typeof raw !== 'object') continue
    const e = raw as Record<string, unknown>
    if (typeof e.file !== 'string' || !e.file) continue
    if (out.some(x => x.file === e.file)) continue
    const strength = typeof e.strength === 'number' && Number.isFinite(e.strength) ? e.strength : 0.7
    const clip = typeof e.clipStrength === 'number' && Number.isFinite(e.clipStrength) ? e.clipStrength : undefined
    out.push({
      file: e.file,
      strength: Math.min(3, Math.max(-3, Math.round(strength * 100) / 100)),
      clipStrength: clip === undefined ? undefined : Math.min(3, Math.max(-3, clip)),
      enabled: e.enabled !== false,
    })
  }
  return out
}

export function loadStack(familyId: string): LoraStack {
  if (!familyId) return []
  const raw = readRaw(KEY(familyId))
  if (!raw) return []
  try {
    return sanitise(JSON.parse(raw))
  } catch {
    return []
  }
}

/** @returns false when the stack could only be held in memory. */
export function saveStack(familyId: string, stack: LoraStack): boolean {
  if (!familyId) return false
  if (!stack.length) {
    try {
      localStorage.removeItem(KEY(familyId))
    } catch {
      /* nothing useful to do */
    }
    return true
  }
  return writeRaw(KEY(familyId), JSON.stringify(stack))
}

// ---------------------------------------------------------------------------
// Handing the stack to refine.ts
// ---------------------------------------------------------------------------

export type ResolvedStack = {
  /** Exactly what withLoras() takes, in load order. */
  specs: LoraSpec[]
  /** Entries left out, and why. Show these: silence here is the bug. */
  dropped: { file: string; label: string; why: string }[]
  /** Untested crossings that were sent anyway. */
  warnings: { file: string; label: string; why: string }[]
  /** Held back because this pass is not the one they belong in. Not an error. */
  deferred: { file: string; label: string; why: string }[]
  bytes: number
}

/** Which pass a resolve is for. Omitted, every enabled LoRA is applied. */
export type Pass = 'base' | 'refine'

/**
 * Turn a saved stack into LoraSpecs for withLoras().
 *
 * Three reasons an entry is dropped, all of them the result of state changing
 * underneath a saved stack rather than of user error:
 *
 *   the file is gone            deleted from disk since the stack was saved
 *   the file is not installed   added from the catalogue and never fetched
 *   the architecture is wrong   the checkpoint changed under the stack
 *
 * A fourth group is held rather than dropped. Pass `pass` and a LoRA whose
 * catalogue row says it belongs in the other pass is reported in `deferred`:
 * a vulva LoRA trained on close framing does almost nothing at whole body
 * scale, and does its whole job in the crop that refine.ts renders.
 *
 * The last one is why this function exists at all. A mismatch is invisible at
 * queue time and nearly invisible in the picture, so it is caught here, on the
 * way out, rather than trusted to a greyed out row the user may never have
 * looked at.
 */
export function resolveStack(
  stack: LoraStack,
  lib: LoraLibrary,
  target: LoraTarget,
  pass?: Pass,
): ResolvedStack {
  const specs: LoraSpec[] = []
  const dropped: ResolvedStack['dropped'] = []
  const warnings: ResolvedStack['warnings'] = []
  const deferred: ResolvedStack['deferred'] = []
  let bytes = 0

  for (const e of stack) {
    if (!e.enabled) continue
    const info = lib.byFile.get(e.file)
    const label = info?.label ?? titleOf(e.file)
    if (!info) {
      dropped.push({ file: e.file, label, why: 'No longer in the LoRA folder.' })
      continue
    }
    if (!info.installed) {
      dropped.push({ file: e.file, label, why: 'Not downloaded yet, so ComfyUI cannot load it.' })
      continue
    }
    const fit = fitFor(info, target)
    if (fit.level === 'mismatch') {
      dropped.push({ file: e.file, label, why: fit.why })
      continue
    }
    if (pass && info.usage !== 'both' && info.usage !== pass) {
      deferred.push({
        file: e.file,
        label,
        why:
          info.usage === 'refine'
            ? 'Trained on close framing, so it is applied in the refine pass rather than the first render.'
            : 'Meant for the first render, where it shapes the whole frame.',
      })
      continue
    }
    if (fit.level === 'untested') warnings.push({ file: e.file, label, why: fit.why })
    if (e.strength === 0) continue
    specs.push({
      name: info.file,
      strength: e.strength,
      clipStrength: e.clipStrength,
    })
    bytes += info.bytes
  }

  return { specs, dropped, warnings, deferred, bytes }
}

// ---------------------------------------------------------------------------
// Fetching one that is not installed yet
// ---------------------------------------------------------------------------

export type { FetchProgress } from './downloads'

/**
 * Download one catalogue LoRA into the LoRA folder. The reader lives in
 * downloads.ts now, shared with the tagger and the catalogue; this names the
 * destination.
 */
export async function fetchLora(
  info: LoraInfo,
  folder: string,
  onProgress: (p: FetchProgress) => void,
  signal?: AbortSignal,
): Promise<void> {
  if (!info.url) throw new Error(`${info.label} has no verified download URL.`)
  await downloadFile(
    { url: info.url, filename: info.file, dest: `${folder || DEFAULT_LORA_FOLDER}/${info.file}` },
    onProgress,
    signal,
  )
}

/** `217.9 MB`, `1.4 GB`. Sizes, never durations. */
export function size(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`
  if (bytes >= 1024 ** 2) return `${Math.round(bytes / 1024 ** 2)} MB`
  return `${Math.max(1, Math.round(bytes / 1024))} kB`
}
