/**
 * What the runner reads out of ComfyUI's answers, and the graph edits it makes.
 *
 * Pure: nothing here reads a file, the network or the clock, so the runner's
 * decisions about a finished run can be tested against recorded /history
 * entries alone. The one module that talks to ComfyUI is ./comfy.mjs.
 *
 * Most of this is a port of the page's own readers (src/lib/comfy.ts and
 * src/lib/continuation.ts), because the runner files what the page used to
 * file and must see a run's files the same way: which are clips, which are
 * stills, and which ComfyUI answered from its cache. tests/comfyReadParity
 * runs the same /history entries through both. Two things differ on purpose:
 *
 *   - startedAt comes from the `execution_start` message only. The page falls
 *     back to the time the prompt was queued, which counts the wait in the
 *     queue as rendering time; the runner would rather say "not measured"
 *     (a duration of 0) than state a time that was not the render's.
 *   - A run's record also names its handoff `frame`, the still a continued
 *     reel shot opens on, because the runner now completes the next shot's
 *     graph itself once this one has landed.
 */

/**
 * The value a reel shot's graph carries where the frame of the shot before it
 * belongs, while that frame does not exist yet. The page builds the whole
 * graph with this in the frame's place and the runner puts the real frame in
 * with {@link splice} once the shot before has landed. src/lib/runner.ts
 * spells the same literal; keep the two in step.
 */
export const CHAIN_TOKEN = 'switchgen:chain:previous-frame'

/**
 * The node id of the handoff tap a chainable video graph carries: a SaveImage
 * of the decode's last frame (NODE_IDS.frameSave in src/lib/continuation.ts).
 */
export const FRAME_NODE = '__cont_frame'

// ------------------------------------------------------------- the files --

/**
 * What counts as moving pictures by name alone. Wide on purpose, as in the
 * page: it is one of three signals, and GIF and WebP writers make clips too.
 */
const MOVING_EXT = /\.(webm|mp4|mkv|gif|webp|avi|mov)$/i

/**
 * Turn one node's `ui` output into typed files.
 *
 * SaveWEBM reports its clip under `images` beside `animated: [true]`, so the
 * container key alone would call every clip a still. Any one of three signals
 * makes a file a clip: a key other than `images`, the `animated` flag, or the
 * extension. A port of collectFiles in src/lib/comfy.ts; keep them in step.
 */
export function collectFiles(output, cached = false) {
  const animated = Array.isArray(output?.animated) ? output.animated : []
  const anyAnimated = animated.some(Boolean)
  const out = []
  for (const [key, val] of Object.entries(output ?? {})) {
    if (key === 'animated' || !Array.isArray(val)) continue
    val.forEach((f, i) => {
      if (!f?.filename) return
      const isVideo = key !== 'images' || anyAnimated || animated[i] === true || MOVING_EXT.test(String(f.filename))
      out.push({
        filename: String(f.filename),
        subfolder: String(f.subfolder ?? ''),
        type: String(f.type ?? 'output'),
        kind: isVideo ? 'video' : 'image',
        ...(cached ? { cached: true } : {}),
      })
    })
  }
  return out
}

/**
 * Every file across every node of a history entry's `outputs`. Files of a
 * node ComfyUI answered from its cache are marked `cached`: they are the files
 * an earlier run wrote, returned again, not new ones.
 */
export function filesOfOutputs(outputs, cachedNodes = new Set()) {
  const out = []
  for (const [nodeId, node] of Object.entries(outputs ?? {})) {
    out.push(...collectFiles(node, cachedNodes.has(nodeId)))
  }
  return out
}

/** The node ids an `execution_cached` message names. */
export function cachedNodesOf(d) {
  return Array.isArray(d?.nodes) ? d.nodes.map((n) => String(n)) : []
}

/**
 * The node an `execution_error` or `execution_interrupted` names, from the
 * socket message or the copy of it kept in /history. Both carry the class as
 * `node_type`, which is what lets a failure say which kind of node it was.
 */
export function nodeOf(d) {
  return {
    node: d?.node_id == null ? null : String(d.node_id),
    nodeType: typeof d?.node_type === 'string' && d.node_type ? d.node_type : null,
  }
}

/** The timestamp ComfyUI stamped on the first `event` message of a history status. */
export function timestampOf(status, event) {
  const messages = Array.isArray(status?.messages) ? status.messages : []
  for (const m of messages) {
    if (Array.isArray(m) && m[0] === event && typeof m[1]?.timestamp === 'number') return m[1].timestamp
  }
  return null
}

/**
 * Normalise one raw /history entry, or null when it is not a run record.
 *
 * `prompt` is the tuple `[number, promptId, graph, extraData, outputNodeIds]`.
 * An entry without the graph is refused, as the page refuses it, so the two
 * readers agree on what counts as a record.
 *
 * `startedAt` is ComfyUI's `execution_start` stamp and nothing else: the
 * queue time the page falls back to is not when the render began (see the
 * head of this file). `finishedAt` is the stamp of whichever ending ComfyUI
 * recorded.
 */
export function readPastRun(promptId, raw) {
  const tuple = raw?.prompt
  if (!Array.isArray(tuple)) return null
  const graph = tuple[2]
  if (!graph || typeof graph !== 'object') return null

  const statusStr = raw?.status?.status_str
  const messages = Array.isArray(raw?.status?.messages) ? raw.status.messages : []
  const interrupted = messages.some((m) => Array.isArray(m) && m[0] === 'execution_interrupted')
  const failed = messages.find((m) => Array.isArray(m) && m[0] === 'execution_error')?.[1]
  const cached = new Set(
    messages.filter((m) => Array.isArray(m) && m[0] === 'execution_cached').flatMap((m) => cachedNodesOf(m[1])),
  )
  const status = interrupted ? 'cancelled' : statusStr === 'success' ? 'success' : statusStr === 'error' ? 'error' : 'unknown'
  const files = filesOfOutputs(raw?.outputs, cached)
  // The tap names itself; only a graph without one falls back to the
  // filename rule the page has always used.
  const tapped = collectFiles(raw?.outputs?.[FRAME_NODE], cached.has(FRAME_NODE)).filter((f) => f.kind === 'image')

  return {
    promptId,
    status,
    files,
    frame: tapped[tapped.length - 1] ?? chainFrameOf(files),
    startedAt: timestampOf(raw?.status, 'execution_start'),
    finishedAt:
      timestampOf(raw?.status, 'execution_success') ??
      timestampOf(raw?.status, 'execution_error') ??
      timestampOf(raw?.status, 'execution_interrupted'),
    error:
      status === 'error' && failed
        ? { message: String(failed.exception_message ?? failed.exception_type ?? ''), ...nodeOf(failed) }
        : null,
  }
}

// --------------------------------------------------------- paths and refs --

/**
 * The handoff frame among a finished run's outputs.
 *
 * A chainable video graph writes exactly one still: the tap. Preference goes
 * to a filename that looks like the tap, then to the last image, so a family
 * that saves a poster frame of its own cannot quietly become the chain
 * source. A port of chainFrameOf in src/lib/continuation.ts.
 */
export function chainFrameOf(files) {
  const images = files.filter((f) => f.kind === 'image')
  if (!images.length) return null
  const tagged = images.filter((f) => /(^chain|\.frame)/.test(f.filename))
  const pick = tagged.length ? tagged : images
  return pick[pick.length - 1] ?? null
}

/**
 * ComfyUI's annotated path, which LoadImage accepts for a file outside its
 * input folder. The format is exact: one space, then the bracket, because
 * ComfyUI cuts the suffix off by length. A port of annotatedRef in
 * src/lib/continuation.ts.
 */
export function annotatedRef(f) {
  const path = f.subfolder ? `${f.subfolder}/${f.filename}` : f.filename
  const type = f.type || 'output'
  return type === 'input' ? path : `${path} [${type}]`
}

/** A file's path under its root, `subfolder/filename`: what the archive names it by. */
export function relOf(f) {
  return f.subfolder ? `${f.subfolder}/${f.filename}` : f.filename
}

// ------------------------------------------------------------- the graph --

const ADVANCED_SAMPLER = 'KSamplerAdvanced'

function samplersIn(graph) {
  if (!graph || typeof graph !== 'object') return 0
  return Object.values(graph).filter((n) => n?.class_type === ADVANCED_SAMPLER).length
}

/**
 * Which of a shot's two sampling passes `node` is, or null.
 *
 * The Wan 2.2 14B pairs sample in two KSamplerAdvanced passes, one per model,
 * and ComfyUI counts each pass's steps from one again, so a progress bar that
 * read the steps alone would go back to empty half way. The first pass hands
 * its leftover noise on. A port of samplerPass in src/lib/continuation.ts.
 */
export function samplerPass(graph, node) {
  const n = node && graph && typeof graph === 'object' ? graph[node] : undefined
  if (!n || n.class_type !== ADVANCED_SAMPLER) return null
  if (samplersIn(graph) < 2) return null
  return { index: n.inputs?.return_with_leftover_noise === 'enable' ? 1 : 2, count: 2 }
}

/**
 * The least the server takes a job's weight to be: a graph that samples in two
 * KSamplerAdvanced passes loads a pair of models, which is exactly the case
 * the page's memory rule releases memory for (clipMemory's release follows the
 * family's dualModel). So the server never runs such a graph as a light job,
 * whatever a page said about it.
 */
export function heavyFloor(graph) {
  return samplersIn(graph) >= 2
}

/**
 * Every place a graph holds {@link CHAIN_TOKEN} as an input's whole value, as
 * `[nodeId, input]`, in the graph's own order.
 */
export function tokenSites(graph) {
  const sites = []
  if (!graph || typeof graph !== 'object') return sites
  for (const [id, node] of Object.entries(graph)) {
    const inputs = node?.inputs
    if (!inputs || typeof inputs !== 'object') continue
    for (const [name, value] of Object.entries(inputs)) {
      if (value === CHAIN_TOKEN) sites.push([id, name])
    }
  }
  return sites
}

/**
 * True when the token appears anywhere in the graph: as a whole value, inside
 * a longer string, in a key or deeper in a nested value. It is the check for
 * intake and for {@link splice}, since a token left anywhere would reach
 * ComfyUI as a file name.
 */
export function hasToken(graph) {
  return JSON.stringify(graph ?? null).includes(CHAIN_TOKEN)
}

/**
 * A copy of `graph` with the real handoff frame `ref` put in at `sites`.
 *
 * The payload saved at intake is never rewritten; the graph sent is made from
 * it here each time, so a send that has to be made again starts from the same
 * bytes. Every site is checked against the original before anything is
 * written, which also makes a site listed twice harmless. It throws when a
 * site does not hold the token, or when the token is left anywhere
 * afterwards: either means the graph is not the one the page built, and a
 * shot sent with a placeholder for its opening frame would spend minutes of
 * the card on the wrong thing.
 */
export function splice(graph, sites, ref) {
  if (typeof ref !== 'string' || !ref) throw new Error('No frame was given to open the shot on.')
  if (!Array.isArray(sites)) throw new Error('The places for the frame are not a list.')
  for (const site of sites) {
    const [id, input] = Array.isArray(site) ? site : []
    if (graph?.[id]?.inputs?.[input] !== CHAIN_TOKEN) {
      throw new Error(`Node ${String(id)} input ${String(input)} does not hold the place for the frame.`)
    }
  }
  const copy = JSON.parse(JSON.stringify(graph))
  for (const [id, input] of sites) copy[id].inputs[input] = ref
  if (hasToken(copy)) throw new Error('The graph still holds a place for the frame after the frame was put in.')
  return copy
}
