/**
 * A stand-in for ComfyUI's GET /object_info, built from the lab's own graphs:
 * every node class they use, each input they set (links as '*', numbers as
 * FLOAT, text as STRING), and for each choice input the values the graphs
 * pick. LoadImage and LoadImageMask list one example file, since the checker
 * accepts annotated output paths for them. No real ComfyUI is read.
 */
import type { ApiWorkflow } from '../../../src/lib/comfy.ts'
import type { ObjectInfo } from '../../core/validate.ts'

const COMBOS = new Set([
  'sampler_name', 'scheduler', 'ckpt_name', 'unet_name', 'clip_name', 'vae_name', 'model_name', 'type', 'weight_dtype', 'upscale_method',
  'channel', 'crop', 'sampling', 'device', 'reference_latents_method', 'sam_detection_hint', 'sam_mask_hint_use_negative',
])

export function objectInfoFrom(graphs: Iterable<ApiWorkflow>): ObjectInfo {
  const info: Record<string, { input: { required: Record<string, unknown> }; output: string[] }> = {}
  const combos = new Map<string, Set<unknown>>()
  for (const g of graphs) {
    for (const node of Object.values(g)) {
      const nd = (info[node.class_type] ??= { input: { required: {} }, output: ['*', '*', '*'] })
      for (const [k, v] of Object.entries(node.inputs)) {
        if (Array.isArray(v)) nd.input.required[k] = ['*']
        else if ((node.class_type === 'LoadImage' || node.class_type === 'LoadImageMask') && k === 'image') nd.input.required[k] = [['example.png']]
        else if (COMBOS.has(k)) {
          const key = `${node.class_type}.${k}`
          const set = combos.get(key) ?? new Set()
          set.add(v)
          combos.set(key, set)
        } else if (typeof v === 'number') nd.input.required[k] = ['FLOAT', {}]
        else if (typeof v === 'boolean') nd.input.required[k] = ['BOOLEAN', {}]
        else nd.input.required[k] = ['STRING', {}]
      }
    }
  }
  for (const [key, set] of combos) {
    const dot = key.indexOf('.')
    info[key.slice(0, dot)].input.required[key.slice(dot + 1)] = [[...set]]
  }
  return info as ObjectInfo
}
