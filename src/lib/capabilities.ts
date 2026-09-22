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
  tools: { aria2c: null, ffmpeg: null, ffprobe: null },
  roots: {},
  reason: 'the local server did not answer, so this page can only talk to ComfyUI',
}

let cache: Promise<ServerCapabilities> | null = null

/**
 * Cached for the page's lifetime: the answer changes when a binary is
 * installed, which is not something that happens while a tab is open. A
 * non-JSON body means Vite's SPA fallback answered for an unmounted
 * middleware, which is reported as "did not answer" rather than as a parse
 * error, because that is what it means.
 */
export function serverCapabilities(): Promise<ServerCapabilities> {
  if (!cache) {
    cache = fetch('/api/capabilities', { headers: { Accept: 'application/json' } })
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
      .catch(() => NONE)
  }
  return cache
}

export function refreshServerCapabilities(): Promise<ServerCapabilities> {
  cache = null
  return serverCapabilities()
}

/** The same answer for a component. Null until it arrives. */
export function useServerCapabilities(): ServerCapabilities | null {
  const [caps, setCaps] = useState<ServerCapabilities | null>(null)
  useEffect(() => {
    let live = true
    void serverCapabilities().then(c => {
      if (live) setCaps(c)
    })
    return () => {
      live = false
    }
  }, [])
  return caps
}
