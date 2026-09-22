/**
 * Add-ons on a clip.
 *
 * Two things make video different from pictures here. The Wan 2.2 14B
 * families run in two halves, a high-noise model and a low-noise one, and an
 * add-on trained for them ships as a pair of files, one per half: a
 * `_HIGH.safetensors` and a `_LOW.safetensors` with the same stem. The rack
 * shows such a pair as one row and applies each file to its own half. A file
 * with no partner on a two-half family is applied to both halves at the same
 * strength, and the rack says so, because nobody has measured that.
 *
 * Nothing on a clip is measured the way the picture stacks were. Every
 * strength here is the author's, and the copy says so.
 */
import { withVideoLoras, type LoraSpec, type VideoLoraSpec } from './refine'
import type { FamilyDef } from './workflows'

const PAIR = /^(.*?)([_-])(HIGH|LOW)(\.[^.]+)$/i

/** `Wan22_I2V_NSFW_General_HIGH.safetensors` → its stem and which half it is for. */
export function pairedHalf(file: string): { stem: string; half: 'high' | 'low' } | null {
  const m = PAIR.exec(file)
  if (!m) return null
  return { stem: `${m[1]}${m[4]}`, half: m[3].toLowerCase() as 'high' | 'low' }
}

/** The other half's filename, whether or not it exists. */
export function partnerOf(file: string): string | null {
  const m = PAIR.exec(file)
  if (!m) return null
  const other = m[3].toUpperCase() === 'HIGH' ? 'LOW' : 'HIGH'
  return `${m[1]}${m[2]}${other}${m[4]}`
}

/**
 * Turn a resolved stack into the specs the graph takes. On a two-half family
 * a row whose partner is installed becomes two specs, one per half; a row
 * with no partner applies to both. On a one-model family the list passes
 * through, halves and all.
 */
export function expandVideoStack(
  specs: readonly LoraSpec[],
  def: FamilyDef,
  installed: ReadonlySet<string>,
): VideoLoraSpec[] {
  if (!def.dualModel) return specs.map((s) => ({ ...s }))
  const out: VideoLoraSpec[] = []
  const seen = new Set<string>()
  for (const s of specs) {
    if (seen.has(s.name)) continue
    const half = pairedHalf(s.name)
    const partner = half ? partnerOf(s.name) : null
    if (half && partner && installed.has(partner)) {
      const high = half.half === 'high' ? s.name : partner
      const low = half.half === 'low' ? s.name : partner
      out.push({ name: high, strength: s.strength, half: 'high' }, { name: low, strength: s.strength, half: 'low' })
      seen.add(high)
      seen.add(low)
    } else {
      out.push({ ...s, half: 'both' })
      seen.add(s.name)
    }
  }
  return out
}

/** One row per pair for a rack: the LOW half is folded into its HIGH partner when both are present. */
export function collapsePairs<T extends { name: string }>(entries: readonly T[]): T[] {
  const files = new Set(entries.map((e) => e.name))
  return entries.filter((e) => {
    const half = pairedHalf(e.name)
    if (!half || half.half !== 'low') return true
    const partner = partnerOf(e.name)
    return !(partner && files.has(partner))
  })
}

/** Chain a resolved stack into a family's graph, or hand the graph back untouched when there is nothing to chain. */
export function chainVideoStack(
  def: FamilyDef,
  specs: readonly LoraSpec[],
  installed: ReadonlySet<string>,
): ReturnType<typeof withVideoLoras> {
  const expanded = expandVideoStack(specs, def, installed)
  if (!expanded.length) return null
  return withVideoLoras(def, expanded)
}
