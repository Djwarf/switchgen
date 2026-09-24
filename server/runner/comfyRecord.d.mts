/** ComfyUI's node map, as it is sent to /prompt. */
export type ApiNode = { class_type: string; inputs: Record<string, unknown> }
export type ApiWorkflow = Record<string, ApiNode>

/** Anything ComfyUI can serve from /view. */
export type FileRef = { filename: string; subfolder: string; type: string }

/** A file a run produced. `cached` is set when ComfyUI answered its node from the cache. */
export type OutputFile = FileRef & { kind: 'image' | 'video'; cached?: true }

export type PastRunError = { message: string; node: string | null; nodeType: string | null }

/** One /history entry, read. */
export type PastRun = {
  promptId: string
  status: 'success' | 'error' | 'cancelled' | 'unknown'
  files: OutputFile[]
  /** The handoff still: the tap's last image, else chainFrameOf(files). */
  frame: OutputFile | null
  /** ComfyUI's execution_start stamp, epoch ms. Never the queue time. */
  startedAt: number | null
  /** The stamp of the ending ComfyUI recorded, epoch ms. */
  finishedAt: number | null
  error: PastRunError | null
}

export type SamplerPass = { index: number; count: number }

/** The frame's placeholder in a chained reel shot's graph. */
export const CHAIN_TOKEN: 'switchgen:chain:previous-frame'
/** The handoff tap's node id. */
export const FRAME_NODE: '__cont_frame'

export function collectFiles(output: Record<string, unknown> | null | undefined, cached?: boolean): OutputFile[]
export function filesOfOutputs(outputs: Record<string, unknown> | null | undefined, cachedNodes?: ReadonlySet<string>): OutputFile[]
export function cachedNodesOf(d: unknown): string[]
export function nodeOf(d: unknown): { node: string | null; nodeType: string | null }
export function timestampOf(status: unknown, event: string): number | null
export function readPastRun(promptId: string, raw: unknown): PastRun | null

export function chainFrameOf(files: readonly OutputFile[]): OutputFile | null
/** `subfolder/name.png [output]`, the form LoadImage takes for a file outside its input folder. */
export function annotatedRef(f: FileRef): string
/** `subfolder/filename`, or `filename`. */
export function relOf(f: FileRef): string

export function samplerPass(graph: ApiWorkflow, node: string | null): SamplerPass | null
/** True for two or more KSamplerAdvanced nodes: a job the server always runs as heavy. */
export function heavyFloor(graph: ApiWorkflow): boolean
/** Every `[nodeId, input]` whose whole value is CHAIN_TOKEN. */
export function tokenSites(graph: ApiWorkflow): [string, string][]
/** True when CHAIN_TOKEN appears anywhere in the graph, a substring included. */
export function hasToken(graph: unknown): boolean
/**
 * A deep copy with `ref` at every site. Throws when a site does not hold
 * CHAIN_TOKEN, or when the token is left anywhere afterwards.
 */
export function splice(graph: ApiWorkflow, sites: readonly (readonly [string, string])[], ref: string): ApiWorkflow
