/**
 * What the local server can do, from its own probes.
 *
 * The app runs against two servers: ComfyUI, which is always there or the
 * app is offline, and the SwitchGen middleware in server/*.mjs, which is only
 * there when the page is served by Vite rather than by a static host. Every
 * feature that needs the second one (deleting files, joining a reel, fetching
 * a model) asks here before it is offered, and prints the reason when it is
 * not. The answer is the server's own: each binary it would spawn is run once
 * and reported as a path or as null.
 */
import { useEffect, useState } from 'react'

export type ServerTools = {
  aria2c: string | null
  ffmpeg: string | null
  ffprobe: string | null
}

export type ServerCapabilities = {
  server: string | null
  deleteFiles: boolean
  /** ffmpeg and ffprobe both run. */
  stitch: boolean
  /** aria2c runs. */
  downloads: boolean
  models: boolean
  hardware: boolean
  hardwareStream: boolean
  /** nvidia-smi answers. */
  gpu: boolean
  /** The shared archive is served here. */
  archive: boolean
  tools: ServerTools
  roots: { models?: string; outputs?: string }
  /** Why everything is false, when it is. Null when the server answered. */
  reason: string | null
}

const NONE: ServerCapabilities = {
  server: null,
  deleteFiles: false,
  stitch: false,
  downloads: false,
  models: false,
  hardware: false,
  hardwareStream: false,
  gpu: false,
  archive: false,
  tools: { aria2c: null, ffmpeg: null, ffprobe: null },
  roots: {},
  reason: 'the local server did not answer, so this page can only talk to ComfyUI',
}

let cache: Promise<ServerCapabilities> | null = null

/**
 * How long a failed probe stands before the next caller asks again. Long
 * enough that the components mounting together on one page share a single
 * answer, short enough that a server which was restarting is found once it
 * is back.
 */
export const RETRY_FAILED_MS = 10_000

/**
 * An answer is cached for the page's lifetime: it changes when a binary is
 * installed, which is not something that happens while a tab is open. A
 * failure is not an answer, so it is held only briefly: the server may have
 * been restarting, and a page that asked once at the wrong moment would
 * otherwise hide deleting, joining and fetching until it was reloaded. A
 * non-JSON body means Vite's SPA fallback answered for an unmounted
 * middleware, which is reported as "did not answer" rather than as a parse
 * error, because that is what it means.
 */
export function serverCapabilities(): Promise<ServerCapabilities> {
  if (cache) return cache
  const probe: Promise<ServerCapabilities> = fetch('/api/capabilities', { headers: { Accept: 'application/json' } })
    .then(async r => {
      const type = r.headers.get('content-type') ?? ''
      if (!r.ok || !type.includes('json')) throw new Error(`capabilities: ${r.status} ${type}`)
      const data = (await r.json()) as Partial<ServerCapabilities>
      return {
        ...NONE,
        ...data,
        tools: { ...NONE.tools, ...(data.tools ?? {}) },
        roots: data.roots ?? {},
        reason: null,
      }
    })
    .catch(() => {
      setTimeout(() => {
        if (cache === probe) cache = null
      }, RETRY_FAILED_MS)
      return NONE
    })
  cache = probe
  return probe
}

/**
 * Ask, hand the answer on, and while the answer is a failure ask again once
 * it has stopped standing. The short hold on a failure only helps whoever
 * asks next; a component that is already on screen with one would keep it,
 * and go on saying the server cannot do what it never got to check, until it
 * was mounted again. The retry waits the same span as the hold, so it lands
 * after the failure has been let go and makes a real request, and every
 * component retrying in the same window shares that one request.
 *
 * `ask` must not reject; both probes turn a failure into an answer. Returns
 * the function that stops asking.
 */
export function askUntilAnswered<T>(
  ask: () => Promise<T>,
  failed: (answer: T) => boolean,
  onAnswer: (answer: T) => void,
): () => void {
  let live = true
  let timer: ReturnType<typeof setTimeout> | undefined
  const once = () => {
    void ask().then(answer => {
      if (!live) return
      onAnswer(answer)
      if (failed(answer)) timer = setTimeout(once, RETRY_FAILED_MS)
    })
  }
  once()
  return () => {
    live = false
    clearTimeout(timer)
  }
}

/**
 * The same answer for a component. Null until it arrives. A failure is
 * replaced by the real answer once the server is back, so a panel opened
 * during a restart offers what the server can do without being reopened.
 */
export function useServerCapabilities(): ServerCapabilities | null {
  const [caps, setCaps] = useState<ServerCapabilities | null>(null)
  useEffect(() => askUntilAnswered(serverCapabilities, c => c.reason !== null, setCaps), [])
  return caps
}
