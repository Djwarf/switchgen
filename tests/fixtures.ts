import type { LoraInfo, LoraLibrary } from '../src/lib/loras'

/** A library of two installed add-ons whose training tags are in the index. */
export function library(installed: string[] = [SCREENCAP, EXPLICIT]): LoraLibrary {
  const rows: LoraInfo[] = [
    info(SCREENCAP, 'Fine anime screencap', 'anime', 'fine anime screencap, anime coloring, anime screencap'),
    info(EXPLICIT, 'Pony explicit photography', 'photoreal', 'explicit photography, v4n1lla, realistic'),
  ].map((r) => ({ ...r, installed: installed.includes(r.file) }))
  return {
    all: rows,
    byFile: new Map(rows.map((r) => [r.file, r])),
    folder: 'Lora',
    installed: rows.filter((r) => r.installed).length,
    unlisted: 0,
    available: 0,
  }
}

export const SCREENCAP = 'fine-anime-screencap-xl-anime-screencap-style-lora-illustrious-and-ponyxl.safetensors'
export const EXPLICIT = 'pony-nsfw-explicit-realistic-photography.safetensors'
export const NOOBAI = 'NoobAI-XL-v1.1.safetensors'

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
