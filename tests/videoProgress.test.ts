import { describe, expect, it } from 'vitest'
import type { ApiWorkflow } from '../src/lib/comfy'
import { drawingLeft, drawnFraction, nextPace, passOf, refusalsFor, stepsAfter } from '../src/components/video/progress'

/**
 * The Video desk's bar and time left for a clip that samples in two passes,
 * as Wan 2.2 14B does: a KSamplerAdvanced for each expert, the first handing
 * its leftover noise on, each reported by ComfyUI as its own step 1 of N.
 * Pure: no desk is loaded, so nothing saved in the tab is taken up.
 */
const wan14b = {
  '12': { class_type: 'KSamplerAdvanced', inputs: { steps: 20, start_at_step: 0, end_at_step: 10, return_with_leftover_noise: 'enable' } },
  '13': { class_type: 'KSamplerAdvanced', inputs: { steps: 20, start_at_step: 10, end_at_step: 10000, return_with_leftover_noise: 'disable' } },
  '8': { class_type: 'VAEDecode', inputs: {} },
} as unknown as ApiWorkflow
const oneSampler = {
  '3': { class_type: 'KSamplerAdvanced', inputs: { steps: 20, start_at_step: 0, end_at_step: 10000, return_with_leftover_noise: 'disable' } },
} as unknown as ApiWorkflow
const first = { index: 1, count: 2 }
const second = { index: 2, count: 2 }

describe('the pass a node draws', () => {
  it('is the first for the sampler that hands its noise on, and the second for the other', () => {
    expect(passOf(wan14b, '12')).toEqual(first)
    expect(passOf(wan14b, '13')).toEqual(second)
    expect(passOf(wan14b, '8')).toBeNull()
    expect(passOf(wan14b, null)).toBeNull()
  })

  it('is none on a graph that samples once', () => {
    expect(passOf(oneSampler, '3')).toBeNull()
  })

  it('knows how many steps the passes after it draw', () => {
    expect(stepsAfter(wan14b, first)).toBe(10)
    expect(stepsAfter(wan14b, second)).toBe(0)
  })
})

describe('the bar', () => {
  const drawing = (value: number, max: number, pass: typeof first | null) => ({ stage: 'Drawing', value, max, pass, samplingAt: 1 })

  it('never goes back when the second pass starts counting from one', () => {
    expect(drawnFraction(drawing(10, 10, first))).toBe(0.5)
    expect(drawnFraction(drawing(0, 1, second))).toBe(0.5)
    expect(drawnFraction(drawing(1, 10, second))).toBeCloseTo(0.55)
  })

  it('holds at its end while the frames develop, and starts from its foot', () => {
    expect(drawnFraction({ stage: 'Developing the frames', value: 0, max: 1, pass: second, samplingAt: 1 })).toBe(0.97)
    expect(drawnFraction({ stage: 'Loading the model', value: 0, max: 1, pass: null, samplingAt: null })).toBe(0.03)
  })
})

describe('the time left', () => {
  it('times the first pass at its own pace and adds the second pass\'s steps', () => {
    // Step 1 at 0 s and step 5 at 100 s: 25 s a step, 5 left and 10 to come.
    const left = drawingLeft({ stage: 'Drawing', value: 5, max: 10, pass: first, pace: { at: 0, step: 1 }, graph: wan14b }, 100_000)
    expect(left).toBe(375_000)
  })

  it('times the second pass from its own first step, not from the clip\'s start', () => {
    // Step 1 of this pass at 500 s and step 3 at 550 s: 25 s a step, 7 left.
    const left = drawingLeft({ stage: 'Drawing', value: 3, max: 10, pass: second, pace: { at: 500_000, step: 1 }, graph: wan14b }, 550_000)
    expect(left).toBe(175_000)
  })

  it('starts the pace again with each pass, at its first drawing step', () => {
    // The second pass's model loads before it draws: no pace until it does.
    expect(nextPace({ pass: first, pace: { at: 0, step: 1 } }, { pass: second, drawing: false, value: 0 }, 400)).toBeNull()
    expect(nextPace({ pass: second, pace: null }, { pass: second, drawing: true, value: 1 }, 500)).toEqual({ at: 500, step: 1 })
    expect(nextPace({ pass: second, pace: { at: 500, step: 1 } }, { pass: second, drawing: true, value: 4 }, 600)).toEqual({ at: 500, step: 1 })
  })
})

describe('the chips a memory verdict greys', () => {
  it('greys the ones refused', () => {
    expect(refusalsFor([33, 49, 81], (f) => (f > 49 ? 'no' : null))).toEqual([null, null, 'no'])
  })

  it('greys none when every one is refused, since the other row or the add-ons decide it', () => {
    expect(refusalsFor([33, 49], () => 'no')).toEqual([null, null])
  })
})
