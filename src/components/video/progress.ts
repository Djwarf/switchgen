/**
 * How far along a clip is, and how long its drawing has left, read from what
 * ComfyUI reports about it.
 *
 * Kept apart from the desk so it can be checked without loading the desk,
 * which takes up saved work from the tab the moment it loads.
 *
 * Wan 2.2 14B draws in two passes, a KSamplerAdvanced for each expert, the
 * first handing its leftover noise to the second. ComfyUI slices the steps
 * for each pass and reports each as its own step 1 of N. Read as one count,
 * the bar ran to the end of the first pass, fell back to a tenth under
 * "step 1 of 10", and the time left, divided by the second pass's steps, grew
 * several times over. Each pass is placed on one rule here, and timed at its
 * own pace.
 */
import { clamp } from '../../lib/num'
import type { ApiNode, ApiWorkflow } from '../../lib/comfy'

/** One of the sampling passes of a family that samples in more than one, counted from 1. */
export type SamplingPass = { index: number; count: number }

/** What the readouts below need of a clip. */
export type DrawingState = {
  /** Steps as ComfyUI reports them for the node running now; 0 of 1 for a node that counts none. */
  value: number
  max: number
  /** The desk's name for the node running now; 'Drawing' for a sampler. */
  stage: string
  pass: SamplingPass | null
  /** When the drawing began, so a bar past it holds at its end. */
  samplingAt: number | null
  /** When the current pass's steps were first read, and at which step. */
  pace: { at: number; step: number } | null
  graph: ApiWorkflow
}

/** A KSamplerAdvanced that hands its leftover noise on: the first of two passes. */
const handsOn = (n: ApiNode | undefined) =>
  n?.class_type === 'KSamplerAdvanced' && n.inputs.return_with_leftover_noise === 'enable'

/**
 * The sampling pass a node is, on a family that samples in two. Null for any
 * other node, and for a graph whose sampler hands nothing on, which samples
 * once.
 */
export function passOf(graph: ApiWorkflow, nodeId: string | null): SamplingPass | null {
  if (!nodeId) return null
  const node = graph[nodeId]
  if (node?.class_type !== 'KSamplerAdvanced') return null
  if (!Object.values(graph).some(handsOn)) return null
  return handsOn(node) ? { index: 1, count: 2 } : { index: 2, count: 2 }
}

/**
 * Steps in the passes after this one, read off the graph: a KSamplerAdvanced
 * draws from its start step to its end step, or to its step count when the
 * end is past it. Null when the graph does not say.
 */
export function stepsAfter(graph: ApiWorkflow, pass: SamplingPass): number | null {
  if (pass.index >= pass.count) return 0
  let total = 0
  let found = false
  for (const node of Object.values(graph)) {
    if (node.class_type !== 'KSamplerAdvanced' || handsOn(node)) continue
    const { steps, start_at_step: start, end_at_step: end } = node.inputs
    if (typeof steps !== 'number' || typeof start !== 'number' || typeof end !== 'number') return null
    total += Math.max(0, Math.min(end, steps) - start)
    found = true
  }
  return found ? total : null
}

/**
 * How far along a clip is, from 0.03 to 0.97, never going back: two passes
 * read as one rule, and once the drawing is done the bar holds at its end
 * through developing and encoding, where it used to drop to the start.
 */
export function drawnFraction(j: Pick<DrawingState, 'value' | 'max' | 'pass' | 'stage' | 'samplingAt'>): number {
  if (j.stage === 'Drawing') {
    const within = j.max > 1 ? clamp(j.value / j.max, 0, 1) : 0
    const whole = j.pass ? (j.pass.index - 1 + within) / j.pass.count : within
    return clamp(whole, 0.03, 0.97)
  }
  return j.samplingAt !== null ? 0.97 : 0.03
}

/**
 * What is left of the drawing at the pace of the pass drawing now, in ms,
 * with the passes still to come added at that pace. Null until two steps of
 * this pass have been read. The change of model between passes is not in
 * it: nothing here has measured one.
 */
export function drawingLeft(
  j: Pick<DrawingState, 'value' | 'max' | 'pass' | 'pace' | 'stage' | 'graph'>,
  now: number,
): number | null {
  if (j.stage !== 'Drawing' || j.max <= 1 || !j.pace || j.value < 2) return null
  const steps = j.value - j.pace.step
  if (steps < 1) return null
  const perStep = (now - j.pace.at) / steps
  const later = j.pass ? (stepsAfter(j.graph, j.pass) ?? (j.pass.count - j.pass.index) * j.max) : 0
  const left = perStep * (j.max - j.value + later)
  return left > 0 ? left : null
}

/**
 * The pace to keep after one more report: timed afresh from the first step
 * of a new pass, since each pass counts from 1, and from the first step of
 * the drawing otherwise.
 */
export function nextPace(
  current: Pick<DrawingState, 'pass' | 'pace'>,
  report: { pass: SamplingPass | null; drawing: boolean; value: number },
  now: number,
): DrawingState['pace'] {
  const newPass = report.pass !== null && report.pass.index !== current.pass?.index
  if (newPass) return report.drawing ? { at: now, step: report.value } : null
  if (report.drawing && !current.pace) return { at: now, step: report.value }
  return current.pace
}

/**
 * Each chip's refusal, from the memory verdict a clip with that value would
 * get, or none at all when every chip in the row is refused. Then it is not
 * this row's choice that decides but the other row's, or the add-ons, and
 * greying the whole row would hide that and could leave no chip to press on
 * either row.
 */
export function refusalsFor<T>(values: readonly T[], reasonOf: (v: T) => string | null): (string | null)[] {
  const reasons = values.map(reasonOf)
  return reasons.length && reasons.every((r) => r !== null) ? values.map(() => null) : reasons
}
