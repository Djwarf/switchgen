/**
 * Night 0, the calibration (run cal-1). It gates the core run: the core
 * refuses to start until these pairs are judged, unless the user skips it.
 *
 * Step sweep: the four guidance-off models whose home step count is far from
 * 28 (Z-Image Turbo, Klein and Krea2 at 8; Qwen-Image 2.1 at 20), at 8, 16,
 * 28 and 40 steps on three core prompts. 4 x 3 x 4 x 4 = 192 pictures; the 48
 * at 28 steps are core cells and are reused, not drawn again.
 *
 * Sampler check: each model whose home sampler differs from the common
 * euler/simple, one per home sampler, at its home pair against euler/simple on
 * two core prompts. NoobAI (euler_ancestral/normal), semiReal (dpmpp_2m/karras,
 * which Pony and WAI share), oneObsession (er_sde/simple, which the other two
 * Anima mixes share), Z-Image Base (res_multistep/simple, which Turbo shares)
 * and Chroma (euler/beta). 5 x 2 x 2 x 4 = 80 pictures; the 40 euler/simple
 * ones are core cells.
 *
 * Total 272 pictures, 88 of them reused by the core. Judged only as blind
 * "which is better" pairs; the results show numbers, never pictures, so no
 * model's look is taught before the blind core run.
 */
import { familyOwning } from '../../src/lib/workflows.ts'
import { homeSteps } from '../core/params.ts'
import { defineSuite } from '../core/suite.ts'
import { BAKERY, CORE, FRUIT, KITCHEN, MODELS, REFS, SAMPLER, SEEDS, SHAPES, STEPS, STUDY } from './first-pass.ts'

const SWEEP_MODELS = ['zturbo', 'klein', 'krea2', 'qwen21']

/** Each swept model's home step count, read from the registry so it cannot drift. */
function homeFor(keys: string[]): Record<string, number> {
  const out: Record<string, number> = {}
  for (const k of keys) {
    const file = MODELS[k].file
    const def = familyOwning(file)
    if (def) out[k] = homeSteps(def, file)
  }
  return out
}

export default defineSuite({
  id: 'calibration',
  version: 1,
  study: STUDY,
  seeds: SEEDS,
  steps: STEPS,
  sampler: SAMPLER,
  shapes: SHAPES,
  models: MODELS,
  core: CORE,
  refs: REFS,
  // The core's exact words. `models: []` because these slots are only drawn
  // through the sweep and the sampler check below, never for every model.
  slots: [
    { ...FRUIT, models: [] },
    { ...KITCHEN, models: [] },
    { ...BAKERY, models: [] },
  ],
  chains: [],
  sweep: {
    models: SWEEP_MODELS,
    slots: ['following.fruit', 'photo.kitchen', 'text.bakery'],
    steps: [8, 16, 28, 40],
    homeFor: homeFor(SWEEP_MODELS),
  },
  samplerCheck: {
    models: ['noobai', 'semireal', 'oneObsession', 'zbase', 'chroma'],
    slots: ['photo.kitchen', 'following.fruit'],
  },
})
