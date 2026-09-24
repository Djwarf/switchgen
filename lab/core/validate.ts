/**
 * The lab's own check of a graph against ComfyUI's /object_info.
 *
 * The app has one (scripts/validate-workflows.ts), but it fetches /object_info
 * at the top level as soon as it is imported, so it cannot be imported. This
 * is the same idea in a function: node classes, required inputs, unexpected
 * inputs, combo values, number ranges and links. It reads ComfyUI with a GET
 * and nothing else; the lab never sends ComfyUI any work.
 */
import { CHAIN_TOKEN } from '../../server/runner/comfyRecord.mjs'
import type { ApiWorkflow } from '../../src/lib/comfy.ts'
import { PLACEHOLDER } from './graphs.ts'

/** ComfyUI's description of one node class, as much of it as the check reads. */
export type NodeInfo = {
  input?: { required?: Record<string, unknown>; optional?: Record<string, unknown>; hidden?: Record<string, unknown> }
  output?: unknown[]
}
export type ObjectInfo = Record<string, NodeInfo>

type FetchLike = (url: string, init?: { method?: string; headers?: Record<string, string>; signal?: AbortSignal }) => Promise<{
  ok: boolean
  status: number
  json(): Promise<unknown>
}>

/** GET {comfyUrl}/object_info. A read, and the only request the lab makes to ComfyUI. */
export async function fetchObjectInfo(comfyUrl: string, fetchImpl: FetchLike = fetch as unknown as FetchLike): Promise<ObjectInfo> {
  const url = `${comfyUrl.replace(/\/+$/, '')}/object_info`
  let res: Awaited<ReturnType<FetchLike>>
  try {
    res = await fetchImpl(url, { method: 'GET', headers: { accept: 'application/json' }, signal: AbortSignal.timeout(30_000) })
  } catch (e) {
    const cause = (e as { cause?: { code?: string } }).cause?.code
    throw new Error(`Cannot reach ComfyUI at ${comfyUrl} (${cause ?? (e as Error).message}).`)
  }
  if (!res.ok) throw new Error(`ComfyUI answered ${res.status} for /object_info.`)
  const info = await res.json()
  if (!info || typeof info !== 'object' || Array.isArray(info)) throw new Error('ComfyUI answered /object_info with something that is not a list of nodes.')
  return info as ObjectInfo
}

const isLink = (v: unknown): v is [string, number] =>
  Array.isArray(v) && v.length === 2 && typeof v[0] === 'string' && typeof v[1] === 'number'

/** The allowed values of a combo input, or null when the input is not a combo. */
function optionsOf(spec: unknown): unknown[] | null {
  if (!Array.isArray(spec)) return null
  if (Array.isArray(spec[0])) return spec[0] as unknown[]
  if (spec[0] === 'COMBO') {
    const o = (spec[1] as { options?: unknown })?.options
    return Array.isArray(o) ? o : null
  }
  return null
}

/** An input's type name ('MODEL', 'INT', ...), or null for a combo or an unknown shape. */
function typeOf(spec: unknown): string | null {
  return Array.isArray(spec) && typeof spec[0] === 'string' && spec[0] !== 'COMBO' ? spec[0] : null
}

function limits(spec: unknown): { min?: number; max?: number } {
  const o = Array.isArray(spec) ? (spec[1] as { min?: unknown; max?: unknown } | undefined) : undefined
  return {
    min: typeof o?.min === 'number' ? o.min : undefined,
    max: typeof o?.max === 'number' ? o.max : undefined,
  }
}

/** A file under outputs, input or temp in the form LoadImage reads: "sub/name.png [output]". */
const ANNOTATED = /^[^\0\n]+ \[(output|input|temp)\]$/

const FILE_INPUT = new Set(['LoadImage.image', 'LoadImageMask.image'])

/**
 * Every problem with a graph against ComfyUI's node list; empty when it would
 * be accepted. A picture input of LoadImage or LoadImageMask may hold an
 * annotated path, the chain token or a lab placeholder, none of which is in
 * the node's own list of files.
 */
export function checkGraph(graph: ApiWorkflow, info: ObjectInfo): string[] {
  const out: string[] = []
  for (const [id, node] of Object.entries(graph)) {
    const nd = info[node.class_type]
    if (!nd) {
      out.push(`node ${id}: ComfyUI has no node class "${node.class_type}"`)
      continue
    }
    const req = nd.input?.required ?? {}
    const known: Record<string, unknown> = { ...req, ...(nd.input?.optional ?? {}) }
    for (const k of Object.keys(req)) if (!(k in node.inputs)) out.push(`node ${id} (${node.class_type}): missing required input "${k}"`)
    for (const [k, v] of Object.entries(node.inputs)) {
      if (!(k in known)) {
        out.push(`node ${id} (${node.class_type}): unexpected input "${k}"`)
        continue
      }
      const spec = known[k]
      if (isLink(v)) {
        const from = graph[v[0]]
        if (!from) {
          out.push(`node ${id} (${node.class_type}): "${k}" links to node ${v[0]}, which is not in the graph`)
          continue
        }
        const outs = info[from.class_type]?.output
        if (Array.isArray(outs)) {
          if (v[1] < 0 || v[1] >= outs.length) {
            out.push(`node ${id} (${node.class_type}): "${k}" reads output ${v[1]} of node ${v[0]} (${from.class_type}), which has ${outs.length}`)
            continue
          }
          const have = outs[v[1]]
          const want = typeOf(spec)
          if (typeof have === 'string' && want && want !== '*' && have !== '*' && !want.includes(',') && !have.includes(',') && want !== have) {
            out.push(`node ${id} (${node.class_type}): "${k}" wants ${want} but node ${v[0]} gives ${have}`)
          }
        }
        continue
      }
      if (Array.isArray(v)) {
        for (const el of v) if (isLink(el) && !graph[el[0]]) out.push(`node ${id} (${node.class_type}): "${k}" links to node ${el[0]}, which is not in the graph`)
        continue
      }
      const opts = optionsOf(spec)
      if (FILE_INPUT.has(`${node.class_type}.${k}`)) {
        const ok =
          typeof v === 'string' &&
          (opts?.includes(v) || v === CHAIN_TOKEN || PLACEHOLDER.test(v) || (ANNOTATED.test(v) && !v.split(/[\\/]/).includes('..')))
        if (!ok) out.push(`node ${id} (${node.class_type}): "${k}" = ${JSON.stringify(v)} is not a file ComfyUI can load`)
        continue
      }
      if (opts) {
        if (!opts.includes(v)) {
          const shown = opts.length > 6 ? `${opts.slice(0, 6).join(', ')}, …` : opts.join(', ')
          out.push(`node ${id} (${node.class_type}): "${k}" = ${JSON.stringify(v)} is not one of [${shown}]`)
        }
        continue
      }
      const type = typeOf(spec)
      if (type === 'INT' || type === 'FLOAT') {
        if (typeof v !== 'number' || !Number.isFinite(v)) {
          out.push(`node ${id} (${node.class_type}): "${k}" should be a number, not ${JSON.stringify(v)}`)
          continue
        }
        if (type === 'INT' && !Number.isInteger(v)) out.push(`node ${id} (${node.class_type}): "${k}" should be a whole number, not ${v}`)
        const { min, max } = limits(spec)
        if (min !== undefined && v < min) out.push(`node ${id} (${node.class_type}): "${k}" = ${v} is below its minimum ${min}`)
        if (max !== undefined && v > max) out.push(`node ${id} (${node.class_type}): "${k}" = ${v} is above its maximum ${max}`)
      } else if (type === 'STRING' && typeof v !== 'string') {
        out.push(`node ${id} (${node.class_type}): "${k}" should be text, not ${JSON.stringify(v)}`)
      } else if (type === 'BOOLEAN' && typeof v !== 'boolean') {
        out.push(`node ${id} (${node.class_type}): "${k}" should be true or false, not ${JSON.stringify(v)}`)
      }
    }
  }
  return out
}
