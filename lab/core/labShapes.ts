/**
 * The one graph shape the lab builds that the app does not have.
 *
 * detailOnPicture runs a face (or hand) pass over a picture that already
 * exists, made by another model: Klein composes, Z-Image Base redraws the
 * faces. The app's face pass only runs inside a fresh generation, so this is
 * marked lab-only wherever it appears, and a finding about it is a finding
 * about a graph the app would first have to gain.
 *
 * It is still built from the app's own pieces, not by hand: the image-to-image
 * graph of model B (IMG2IMG), with the app's face pass added to it
 * (deriveAutoDetail). Only two things change. The pass is pointed at the
 * loaded picture instead of B's own render, and the nodes that no longer feed
 * the saved picture (B's sampler, its decode and the source's re-encode) are
 * dropped, since ComfyUI would otherwise load them for nothing.
 */
import { IMG2IMG, type FamilyDef } from '../../src/lib/workflows.ts'
import { deriveAutoDetail, type DerivedDef, type DetailTarget } from '../../src/lib/refine.ts'

export const DETAIL_ON_PICTURE_NOTE =
  "Lab only: the app has no face or hand pass over a picture it did not make. This one is built from the app's own image-to-image graph and face pass, with the pass pointed at the loaded picture."

type Graph = FamilyDef['graph']

const isLink = (v: unknown): v is [string, number] =>
  Array.isArray(v) && v.length === 2 && typeof v[0] === 'string' && typeof v[1] === 'number'

/** The ids of every node the given sinks read from, the sinks included. */
export function reachableFrom(graph: Graph, sinks: string[]): Set<string> {
  const seen = new Set<string>()
  const stack = [...sinks]
  while (stack.length) {
    const id = stack.pop() as string
    if (seen.has(id) || !graph[id]) continue
    seen.add(id)
    for (const v of Object.values(graph[id].inputs)) {
      if (isLink(v)) stack.push(v[0])
      else if (Array.isArray(v)) for (const el of v) if (isLink(el)) stack.push(el[0])
    }
  }
  return seen
}

export type LabShape = { def: DerivedDef; note: string; labOnly: true }

/**
 * A face or hand pass of family B over a loaded picture. Null when B has no
 * image-to-image graph or no face pass in the app.
 */
export function detailOnPicture(defB: FamilyDef, target: DetailTarget): LabShape | null {
  const i2i = IMG2IMG[defB.id]
  if (!i2i) return null
  const detailed = deriveAutoDetail(i2i, target)
  if (!detailed) return null

  const graph: Graph = JSON.parse(JSON.stringify(detailed.graph))
  const load = i2i.bindings.image?.[0]?.[0]
  const passes = Object.entries(graph).filter(([, n]) => n.class_type === 'FaceDetailer')
  const sinks = Object.entries(graph).filter(([, n]) => n.class_type === 'SaveImage')
  if (!load || !graph[load] || passes.length !== 1 || sinks.length !== 1) return null
  const [passId, pass] = passes[0]
  pass.inputs.image = [load, 0]

  const keep = reachableFrom(graph, [sinks[0][0]])
  if (!keep.has(passId)) return null
  for (const id of Object.keys(graph)) if (!keep.has(id)) delete graph[id]

  const live = (binds: [string, string][] | undefined) => (binds ?? []).filter(([id]) => keep.has(id))
  const bindings: FamilyDef['bindings'] = {}
  for (const [k, v] of Object.entries(detailed.bindings) as [keyof FamilyDef['bindings'], [string, string][]][]) {
    const kept = live(v)
    if (kept.length) bindings[k] = kept
  }
  const extra: DerivedDef['derived']['extra'] = {}
  for (const [k, v] of Object.entries(detailed.derived.extra) as [keyof DerivedDef['derived']['extra'], [string, string][]][]) {
    const kept = live(v)
    if (kept.length) extra[k] = kept
  }

  return {
    def: {
      ...detailed,
      id: `${defB.id}__detail_on_picture_${target}`,
      label: `${defB.label}: ${target} pass over a loaded picture (lab only)`,
      graph,
      bindings,
      derived: { ...detailed.derived, extra, note: DETAIL_ON_PICTURE_NOTE },
    },
    note: DETAIL_ON_PICTURE_NOTE,
    labOnly: true,
  }
}
