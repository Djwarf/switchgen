/**
 * A family's own settings, as the reel's bench takes them.
 *
 * Kept apart from the desk so the change of style can be checked without
 * mounting it: moving between the two Wan 2.2 14B pairs used to keep the old
 * pair's length, 81 frames on the image-to-video pair, which the registry
 * records being killed for memory at that length.
 */
import { snapLength } from '../../lib/continuation'
import { defaultsFor } from '../../lib/workflows'
import type { ReelFamily } from './Bench'
import type { ReelDraft } from './store'

/**
 * A family's own verified recipe, as bench settings. A reel is one continuous
 * piece, so the whole bench follows the family rather than carrying settings
 * across from a different model.
 */
export function recipeFor(f: ReelFamily): Partial<ReelDraft> {
  const d = defaultsFor(f.def, f.model)
  return {
    familyId: f.def.id,
    model: f.model,
    width: d.width,
    height: d.height,
    fps: d.fps || 24,
    length: snapLength(d.length || 81),
    steps: d.steps,
    cfg: d.cfg,
    sampler: d.sampler,
    scheduler: d.scheduler,
    negative: null,
  }
}
