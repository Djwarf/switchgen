/**
 * Why a model cannot take a test, read from the app's registry and never
 * guessed. A cell that is not applicable is left out of its set and shown
 * hatched with its reason; it is never scored, and never averaged as a 1.
 */
import { IMG2IMG, defaultsFor, familyOwning, type FamilyDef } from '../../src/lib/workflows.ts'
import { capabilitiesOf, deriveAutoDetail } from '../../src/lib/refine.ts'
import type { Op, Slot } from './types.ts'

export const REASONS = {
  unknown: "not a model in the app's registry",
  editOnly: 'an editing model: it changes a picture it is given and does not make one from words',
  notEditor: 'not an editing model: instruction edits are the editing model\'s test',
  noI2i: 'no image-to-image graph in the app',
  noRegion: 'no region pass in the app',
  noFace: 'no face pass in the app',
  noHand: 'no hand pass in the app',
  noHires: 'no hires pass in the app',
  noDetailOnPicture: 'no image-to-image graph with a face pass in the app, so the lab cannot build a pass over a picture',
  noNegative: 'no negative prompt input',
  guidanceOff: 'guidance off by design (CFG 1), so a negative prompt does nothing',
} as const

/** Why the operation itself cannot be built for this family, or null. */
export function opReason(def: FamilyDef, op: Op, target?: 'face' | 'hand'): string | null {
  if (def.mode === 'edit') return op === 'edit' ? null : REASONS.editOnly
  if (def.mode !== 'image') return REASONS.unknown
  switch (op) {
    case 't2i':
      return null
    case 'edit':
      return REASONS.notEditor
    case 'i2i':
      return IMG2IMG[def.id] ? null : REASONS.noI2i
    case 'region':
      return capabilitiesOf(def).refine ? null : REASONS.noRegion
    case 'face':
      return capabilitiesOf(def).faceDetail ? null : REASONS.noFace
    case 'hand':
      return capabilitiesOf(def).handDetail ? null : REASONS.noHand
    case 'hires':
      return capabilitiesOf(def).hires ? null : REASONS.noHires
    case 'detailOnPicture': {
      const i2i = IMG2IMG[def.id]
      return i2i && deriveAutoDetail(i2i, target ?? 'face') ? null : REASONS.noDetailOnPicture
    }
  }
}

/** Whether a negative prompt can do anything for this file: null when it can, else why not. */
export function negativeReason(def: FamilyDef, file: string): string | null {
  if (!def.bindings.negative?.length) return REASONS.noNegative
  if (defaultsFor(def, file).cfg <= 1) return REASONS.guidanceOff
  return null
}

/** True when a slot tests the negative prompt, so a model without one cannot take it. */
export function testsNegative(slot: Pick<Slot, 'block' | 'negativeAdd'>): boolean {
  return slot.block === 'negative' || !!slot.negativeAdd
}

/**
 * Why `file` cannot take `slot` with operation `op`, or null when it can.
 * `op` is the operation the model would run, which for the editing model on
 * a reference or region test is 'edit'.
 */
export function naReason(file: string, op: Op, slot: Pick<Slot, 'block' | 'negativeAdd' | 'target'>): string | null {
  const def = familyOwning(file)
  if (!def) return REASONS.unknown
  const why = opReason(def, op, slot.target)
  if (why) return why
  if (testsNegative(slot)) return negativeReason(def, file)
  return null
}
