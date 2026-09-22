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
import type { StackEntry } from './loras'
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
 * The add-ons a family's own graph already loads, by file and strength. The
 * Wan 2.2 I2V pair ships with its NSFW General pair built in, one file on each
 * half at 0.85.
 */
export function familyLoras(def: FamilyDef): { name: string; strength: number }[] {
  const out: { name: string; strength: number }[] = []
  for (const node of Object.values(def.graph)) {
    if (node.class_type !== 'LoraLoaderModelOnly' && node.class_type !== 'LoraLoader') continue
    const name = node.inputs.lora_name
    if (typeof name !== 'string' || !name) continue
    const strength = node.inputs.strength_model
    out.push({ name, strength: typeof strength === 'number' ? strength : 1 })
  }
  return out
}

/**
 * Every file a family already loads, and the other half of each pair. Adding
 * one of these to the rack would chain the same weights in front of the
 * family's own copy, so the file would patch each half twice.
 */
export function builtInLoras(def: FamilyDef): Set<string> {
  const out = new Set<string>()
  for (const { name } of familyLoras(def)) {
    out.add(name)
    const partner = partnerOf(name)
    if (partner) out.add(partner)
  }
  return out
}

/**
 * Turn a resolved stack into the specs the graph takes.
 *
 * On a two-half family a HIGH or LOW row becomes one spec per half. Each half
 * runs at its own row's strength when both halves have a row, and at this
 * row's strength when the partner is merely installed. A partner that is on
 * the rack but switched off, or otherwise left out, is `excluded`: it is
 * never pulled back in, and the row is treated as having no partner, which
 * sends it to both halves. On a one-model family the list passes through,
 * halves and all.
 */
export function expandVideoStack(
  specs: readonly LoraSpec[],
  def: FamilyDef,
  installed: ReadonlySet<string>,
  excluded: ReadonlySet<string> = new Set(),
): VideoLoraSpec[] {
  if (!def.dualModel) return specs.map((s) => ({ ...s }))
  const byName = new Map(specs.map((s) => [s.name, s]))
  const out: VideoLoraSpec[] = []
  const seen = new Set<string>()
  for (const s of specs) {
    if (seen.has(s.name)) continue
    const half = pairedHalf(s.name)
    const partner = half ? partnerOf(s.name) : null
    const own = partner ? byName.get(partner) : undefined
    if (half && partner && (own || (installed.has(partner) && !excluded.has(partner)))) {
      const other: LoraSpec = own ?? { ...s, name: partner }
      const high = half.half === 'high' ? s : other
      const low = half.half === 'low' ? s : other
      out.push({ ...high, half: 'high' }, { ...low, half: 'low' })
      seen.add(high.name)
      seen.add(low.name)
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

/**
 * What a family will actually run from a resolved stack: the family's own
 * add-ons taken out, since the graph already loads them, and pairs expanded.
 * This is what goes into the graph and what the record should say ran.
 */
export function videoLorasToRun(
  def: FamilyDef,
  specs: readonly LoraSpec[],
  installed: ReadonlySet<string>,
  excluded: ReadonlySet<string> = new Set(),
): VideoLoraSpec[] {
  const own = builtInLoras(def)
  return expandVideoStack(
    specs.filter((s) => !own.has(s.name)),
    def,
    installed,
    excluded,
  )
}

/** Chain a resolved stack into a family's graph, or hand the graph back untouched when there is nothing to chain. */
export function chainVideoStack(
  def: FamilyDef,
  specs: readonly LoraSpec[],
  installed: ReadonlySet<string>,
  excluded: ReadonlySet<string> = new Set(),
): ReturnType<typeof withVideoLoras> {
  const expanded = videoLorasToRun(def, specs, installed, excluded)
  if (!expanded.length) return null
  return withVideoLoras(def, expanded)
}

/** One add-on as a record lists it. `half` says where it ran, when the record says. */
export type RecordedLora = {
  name: string
  strength: number
  clipStrength?: number
  half?: 'high' | 'low' | 'both'
}

/**
 * Rebuild a rack from the add-ons a record says ran, so the next clip runs
 * them the same way.
 *
 * On a two-half family the halves of a pair recorded at one strength fold to
 * one row, and at two strengths keep a row each, since each half runs at its
 * own row's strength. A half recorded alone comes back as one row, which
 * pulls its partner in when the partner is installed: that is how a lone row
 * ran before the record listed both halves. Only a half the record marks as
 * having run on both halves had its partner left out on purpose, and it gets
 * the partner back as a switched-off row so the partner stays out. The
 * family's own add-ons are never put on the rack, because its graph loads
 * them already.
 */
export function rackFromRecord(
  loras: readonly RecordedLora[],
  def: FamilyDef,
  installed: ReadonlySet<string>,
): StackEntry[] {
  const own = builtInLoras(def)
  const kept = loras.filter((l) => !own.has(l.name))
  const row = (l: RecordedLora, enabled = true): StackEntry => ({
    file: l.name,
    strength: l.strength,
    clipStrength: l.clipStrength,
    enabled,
  })
  if (!def.dualModel) return kept.map((l) => row(l))

  const byName = new Map(kept.map((l) => [l.name, l]))
  const out: StackEntry[] = []
  const done = new Set<string>()
  for (const l of kept) {
    if (done.has(l.name)) continue
    done.add(l.name)
    const half = pairedHalf(l.name)
    const partner = half ? partnerOf(l.name) : null
    if (!half || !partner) {
      out.push(row(l))
      continue
    }
    const twin = byName.get(partner)
    if (twin) {
      done.add(partner)
      const high = half.half === 'high' ? l : twin
      const low = high === l ? twin : l
      out.push(row(high))
      if (low.strength !== high.strength) out.push(row(low))
    } else {
      out.push(row(l))
      if (l.half === 'both' && installed.has(partner)) out.push(row({ ...l, name: partner }, false))
    }
  }
  return out
}
