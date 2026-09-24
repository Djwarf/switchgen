/**
 * Where the lab keeps things, and the one refusal that protects them.
 *
 * Private things (the user's photos, scores, the sealed key, run ledgers) live
 * in SWITCHGEN_LAB_DIR, by default /mnt/storage/ai/lab. That folder must be
 * outside the repo, so nothing private can be committed, and outside the
 * outputs folder, because ComfyUI's /view serves every file under outputs to
 * anyone on the tailnet. Only the pictures ComfyUI itself must read or write
 * go under <outputs>/.lab, and the dot keeps them out of the app's History.
 */
import { existsSync, realpathSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

export type LabEnv = {
  repoRoot: string
  labDir: string
  outputs: string
  appUrl: string
  comfyUrl: string
  port: number
  host: string
}

export const DEFAULTS = {
  labDir: '/mnt/storage/ai/lab',
  outputs: '/mnt/storage/ai/outputs',
  appUrl: 'http://127.0.0.1:5273',
  comfyUrl: 'http://127.0.0.1:8188',
  port: 5274,
  host: '127.0.0.1',
} as const

/** The repo this file sits in: lab/core/env.ts is two folders down. */
export const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..')

/**
 * The real path of `p`, following symlinks as far as the path exists. A
 * folder that does not exist yet is resolved through its nearest existing
 * parent, so a symlink higher up still counts.
 */
function realish(p: string): string {
  const abs = path.resolve(p)
  const rest: string[] = []
  let cur = abs
  for (;;) {
    if (existsSync(cur)) {
      try {
        return path.join(realpathSync(cur), ...rest.reverse())
      } catch {
        return abs
      }
    }
    const parent = path.dirname(cur)
    if (parent === cur) return abs
    rest.push(path.basename(cur))
    cur = parent
  }
}

/** True when `child` is `parent` or inside it. */
export function isInside(child: string, parent: string): boolean {
  const rel = path.relative(realish(parent), realish(child))
  return rel === '' || (!rel.startsWith('..') && !path.isAbsolute(rel))
}

function portOf(v: string | undefined, fallback: number): number {
  if (v === undefined || v === '') return fallback
  const n = Number(v)
  if (!Number.isInteger(n) || n < 1 || n > 65535) throw new Error(`SWITCHGEN_LAB_PORT must be a port number, not "${v}".`)
  return n
}

/**
 * The lab's settings, read from the environment. Throws when the lab folder
 * or the outputs folder would sit inside the repo, or the lab folder inside
 * the outputs folder.
 */
export function labEnv(env: NodeJS.ProcessEnv = process.env): LabEnv {
  const repoRoot = REPO_ROOT
  const labDir = path.resolve(env.SWITCHGEN_LAB_DIR || DEFAULTS.labDir)
  const outputs = path.resolve(env.SWITCHGEN_OUTPUTS || DEFAULTS.outputs)
  if (isInside(labDir, repoRoot)) {
    throw new Error(
      `The lab folder ${labDir} is inside the repo (${repoRoot}). It holds your photos and scores, so it must live outside it. Set SWITCHGEN_LAB_DIR to a folder elsewhere.`,
    )
  }
  if (isInside(outputs, repoRoot)) {
    throw new Error(`The outputs folder ${outputs} is inside the repo (${repoRoot}). Set SWITCHGEN_OUTPUTS to the folder ComfyUI writes to.`)
  }
  if (isInside(labDir, outputs)) {
    throw new Error(
      `The lab folder ${labDir} is inside the outputs folder ${outputs}, which ComfyUI serves to the whole tailnet. Set SWITCHGEN_LAB_DIR to a folder outside it.`,
    )
  }
  return {
    repoRoot,
    labDir,
    outputs,
    appUrl: (env.SWITCHGEN_URL || DEFAULTS.appUrl).replace(/\/+$/, ''),
    comfyUrl: (env.COMFY_URL || DEFAULTS.comfyUrl).replace(/\/+$/, ''),
    port: portOf(env.SWITCHGEN_LAB_PORT, DEFAULTS.port),
    host: env.SWITCHGEN_LAB_HOST || DEFAULTS.host,
  }
}

/** SaveImage's filename_prefix for a cell. The name carries no model or family. */
export function cellPrefix(cellId: string): string {
  return '.lab/cells/' + cellId
}

/** Where a cell's picture lands under outputs, for the first (and normally only) save. */
export function cellRel(cellId: string): string {
  return cellPrefix(cellId) + '_00001_.png'
}

/** A reference's copy under outputs, which ComfyUI loads. For a mask, `ext` is 'mask.png'. */
export function refRel(sha12: string, ext: string): string {
  return '.lab/refs/' + sha12 + '.' + ext
}

/** The mask's copy under outputs, named by the mask file's own hash (cells.ts maskKey). */
export function maskRel(sha12: string): string {
  return refRel(sha12, 'mask.png')
}

/** How LoadImage names a file under outputs: one space, then the bracket. */
export function annotated(rel: string): string {
  return rel.endsWith(' [output]') ? rel : rel + ' [output]'
}

/** A run id is a folder name: letters, digits and dashes. */
export const RUN_ID = /^[A-Za-z0-9][A-Za-z0-9-]{0,63}$/

export function runDir(e: LabEnv, run: string): string {
  if (!RUN_ID.test(run)) throw new Error(`"${run}" is not a run name. Use letters, digits and dashes, like core-1.`)
  return path.join(e.labDir, 'runs', run)
}
