/**
 * The reference photos: the user's own pictures a night's tests start from
 * (the cat, the room or table scene). They live in the lab's private folder,
 * <lab>/refs/<id>.<ext>, listed in <lab>/refs/index.json with their upright
 * size, a hash, the marked area and the user's description.
 *
 * A photo arrives either from the lab page (addRef) or by being put in the
 * refs folder by hand; listRefs finds a file put there and takes it in by its
 * name (scene.jpg is the 'scene' photo).
 *
 * Every size here is the upright one: a phone's portrait photo is stored
 * sideways with an EXIF orientation flag, and the browser, ComfyUI's
 * LoadImage and LoadImageMask all turn it upright, so the marked area and its
 * mask are drawn in that frame.
 *
 * ComfyUI can only read files under its outputs folder, so a run copies the
 * photos it needs to <outputs>/.lab/refs/<sha12>.<ext> (ensureRefCopies). That
 * copy has the photo's metadata (camera, place, time) taken out, since
 * ComfyUI serves the outputs folder to the tailnet, and keeps the orientation.
 */
import { createHash } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import type { LabEnv } from '../core/env.ts'
import { maskRel, refRel } from '../core/env.ts'
import { maskKey } from '../core/cells.ts'
import type { RefInfo, RefRect } from '../core/types.ts'
import { EXT_OF, MIME_OF, cleanCopy, sizeOf, sniff } from './imagesize.ts'
import { normalizeRect, rectMaskPng } from './masks.ts'

/** A ref id: lowercase letters, digits and dashes. */
export const REF_ID = /^[a-z0-9][a-z0-9-]{0,31}$/
/** The largest photo the lab takes (the lab page's upload is capped at 20 MB). */
export const REF_MAX_BYTES = 40 * 1024 * 1024
const MIN_SIDE = 64
const MAX_PIXELS = 80_000_000
const IMAGE_FILE = /^(.+)\.(jpe?g|png|webp)$/i
/**
 * Names a photo put in the refs folder by hand may have, for the id the
 * suites ask for: the user calls the scene photo "the room or table photo".
 */
const DROP_ALIASES: Readonly<Record<string, string>> = { room: 'scene', table: 'scene' }

type Entry = RefInfo & {
  /** The file's name in the refs folder. */
  file: string
  bytes: number
  mtimeMs: number
  orientation: number
  /**
   * When the photo or its marked area last changed, epoch ms. A run checks
   * its photos by their hashes, not by this; a new description leaves it alone.
   */
  updatedAt: number
  maskSha12?: string | null
}

type IndexFile = { v: 1; refs: Record<string, Entry> }

export const refsDir = (env: LabEnv) => path.join(env.labDir, 'refs')
const indexPath = (env: LabEnv) => path.join(refsDir(env), 'index.json')
const maskPath = (env: LabEnv, id: string) => path.join(refsDir(env), `${id}.mask.png`)
const sha12Of = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex').slice(0, 12)

/** A photo's id from a name the user gave or a file name: 'Scene.JPG' → 'scene'. Throws when nothing is left. */
export function refIdFrom(name: string): string {
  const stem = String(name ?? '')
    .trim()
    .replace(/^.*[/\\]/, '')
    .replace(/\.(jpe?g|png|webp)$/i, '')
  const id = stem
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 32)
    .replace(/-+$/, '')
  if (!REF_ID.test(id)) throw new Error(`"${name}" is not a usable photo name. Use a short word such as cat or scene.`)
  return id
}

function writeAtomic(file: string, bytes: Uint8Array | string): void {
  fs.mkdirSync(path.dirname(file), { recursive: true })
  const tmp = path.join(path.dirname(file), `.tmp-${process.pid}-${Date.now()}-${path.basename(file)}`)
  const fd = fs.openSync(tmp, 'w', 0o600)
  try {
    fs.writeSync(fd, typeof bytes === 'string' ? Buffer.from(bytes) : bytes)
    fs.fsyncSync(fd)
  } finally {
    fs.closeSync(fd)
  }
  fs.renameSync(tmp, file)
}

function readIndex(env: LabEnv): IndexFile {
  try {
    const v = JSON.parse(fs.readFileSync(indexPath(env), 'utf8'))
    if (v && v.v === 1 && v.refs && typeof v.refs === 'object') return v as IndexFile
  } catch {
    /* none yet, or unreadable: start again from the files */
  }
  return { v: 1, refs: {} }
}

function writeIndex(env: LabEnv, idx: IndexFile): void {
  writeAtomic(indexPath(env), JSON.stringify(idx, null, 2) + '\n')
}

function infoOf(e: Entry): RefInfo {
  return {
    id: e.id,
    sha12: e.sha12,
    ext: e.ext,
    width: e.width,
    height: e.height,
    mask: e.mask,
    rect: e.rect ?? null,
    maskSha12: e.maskSha12 ?? null,
    describe: e.describe ?? null,
  }
}

/** Measure a photo, or say in words why the lab cannot use it. */
function measure(bytes: Uint8Array) {
  const kind = sniff(bytes)
  if (!kind) throw new Error('The photo must be a JPEG, PNG or WebP picture.')
  const size = sizeOf(bytes)
  if (size.width < MIN_SIDE || size.height < MIN_SIDE) throw new Error(`The photo is too small: it must be at least ${MIN_SIDE} pixels on each side.`)
  if (size.width * size.height > MAX_PIXELS) throw new Error('The photo is too large: at most 80 megapixels.')
  return { kind, size, ext: EXT_OF[kind] }
}

function removeQuietly(file: string): void {
  try {
    fs.unlinkSync(file)
  } catch {
    /* already gone */
  }
}

/**
 * Take in the photos put in the refs folder by hand, and forget any whose
 * file has gone. A photo whose bytes changed loses its marked area, which was
 * drawn on the old one.
 */
function discover(env: LabEnv, now: () => number = Date.now): IndexFile {
  const dir = refsDir(env)
  const idx = readIndex(env)
  let names: string[]
  try {
    names = fs.readdirSync(dir)
  } catch {
    names = []
  }
  // The newest file for each id, when two share one (cat.jpg and cat.png).
  const found = new Map<string, { file: string; st: fs.Stats }>()
  for (const file of names) {
    if (file.startsWith('.') || /\.mask\.png$/i.test(file) || !IMAGE_FILE.test(file)) continue
    let id: string
    try {
      id = refIdFrom(file)
    } catch {
      continue
    }
    id = DROP_ALIASES[id] ?? id
    let st: fs.Stats
    try {
      st = fs.statSync(path.join(dir, file))
    } catch {
      continue
    }
    if (!st.isFile()) continue
    const had = found.get(id)
    if (!had || st.mtimeMs > had.st.mtimeMs) found.set(id, { file, st })
  }

  let changed = false
  for (const id of Object.keys(idx.refs)) {
    if (!found.has(id)) {
      delete idx.refs[id]
      changed = true
    }
  }
  for (const [id, { file, st }] of found) {
    const e = idx.refs[id]
    if (e && e.file === file && e.bytes === st.size && e.mtimeMs === st.mtimeMs) {
      // A mask file deleted by hand: the area is no longer marked.
      if (e.mask && !fs.existsSync(maskPath(env, id))) {
        idx.refs[id] = { ...e, mask: false, rect: null, maskSha12: null, updatedAt: now() }
        changed = true
      }
      continue
    }
    let bytes: Buffer
    let m: ReturnType<typeof measure>
    try {
      bytes = fs.readFileSync(path.join(dir, file))
      m = measure(bytes)
    } catch {
      // Not a picture the lab can use: left out, and forgotten if it was one before.
      if (e) {
        delete idx.refs[id]
        changed = true
      }
      continue
    }
    const sha12 = sha12Of(bytes)
    const same = e && e.sha12 === sha12
    idx.refs[id] = {
      id,
      sha12,
      ext: m.ext,
      width: m.size.width,
      height: m.size.height,
      mask: same ? e.mask : false,
      rect: same ? e.rect ?? null : null,
      describe: e?.describe ?? null,
      file,
      bytes: st.size,
      mtimeMs: st.mtimeMs,
      orientation: m.size.orientation,
      updatedAt: same ? e.updatedAt : now(),
      maskSha12: same ? e.maskSha12 ?? null : null,
    }
    if (!same) removeQuietly(maskPath(env, id))
    changed = true
  }
  if (changed) writeIndex(env, idx)
  return idx
}

/** Every photo the lab has, the ones put in its folder by hand included. */
export function listRefs(env: LabEnv): RefInfo[] {
  return Object.values(discover(env).refs)
    .map(infoOf)
    .sort((a, b) => a.id.localeCompare(b.id))
}

/** The photos by id, as expand() takes them. */
export function refIndex(env: LabEnv): Record<string, RefInfo> {
  return Object.fromEntries(listRefs(env).map((r) => [r.id, r]))
}

function entry(env: LabEnv, id: string): { idx: IndexFile; e: Entry } {
  const idx = discover(env)
  const e = Object.hasOwn(idx.refs, id) ? idx.refs[id] : undefined
  if (!e) throw new Error(missingSentence(env, id))
  return { idx, e }
}

/** Store a photo under `name` (its id), replacing any photo of that id. */
export function addRef(env: LabEnv, bytes: Uint8Array, name: string, now: () => number = Date.now): RefInfo {
  const id = refIdFrom(name)
  if (bytes.length > REF_MAX_BYTES) throw new Error('The photo is over 40 MB.')
  const m = measure(bytes)
  const dir = refsDir(env)
  const file = `${id}.${m.ext}`
  const idx = discover(env, now)
  const had = Object.hasOwn(idx.refs, id) ? idx.refs[id] : undefined
  const sha12 = sha12Of(bytes)
  writeAtomic(path.join(dir, file), bytes)
  // Any other file of this id (cat.png beside a new cat.jpg) would be found again by discovery.
  for (const other of fs.readdirSync(dir)) {
    if (other === file || other.startsWith('.') || /\.mask\.png$/i.test(other) || !IMAGE_FILE.test(other)) continue
    let otherId: string | null = null
    try {
      otherId = refIdFrom(other)
    } catch {
      otherId = null
    }
    if (otherId && (DROP_ALIASES[otherId] ?? otherId) === id) removeQuietly(path.join(dir, other))
  }
  const same = !!had && had.sha12 === sha12
  if (!same) removeQuietly(maskPath(env, id))
  const st = fs.statSync(path.join(dir, file))
  const e: Entry = {
    id,
    sha12,
    ext: m.ext,
    width: m.size.width,
    height: m.size.height,
    mask: same ? had.mask : false,
    rect: same ? had.rect ?? null : null,
    describe: had?.describe ?? null,
    file,
    bytes: st.size,
    mtimeMs: st.mtimeMs,
    orientation: m.size.orientation,
    updatedAt: same ? had.updatedAt : now(),
    maskSha12: same ? had.maskSha12 ?? null : null,
  }
  idx.refs[id] = e
  writeIndex(env, idx)
  return infoOf(e)
}

/**
 * Mark the area a region test redraws: a rectangle in the photo's upright
 * pixels, written as refs/<id>.mask.png (white inside, black outside).
 */
export function setMask(env: LabEnv, id: string, rect: RefRect, now: () => number = Date.now): RefInfo {
  const { idx, e } = entry(env, id)
  const r = normalizeRect(e.width, e.height, rect)
  const png = rectMaskPng(e.width, e.height, r)
  writeAtomic(maskPath(env, id), png)
  const sha = sha12Of(png)
  const changed = !e.mask || e.maskSha12 !== sha
  idx.refs[id] = { ...e, mask: true, rect: r, maskSha12: sha, updatedAt: changed ? now() : e.updatedAt }
  writeIndex(env, idx)
  return infoOf(idx.refs[id])
}

/**
 * The user's own description of the photo, or null to go back to the suite's.
 * A night not started yet is planned again when Start is pressed, so it takes
 * these words; a night already under way keeps the words it started with (its
 * prompts are in its plan), so a new description is no change to the photo
 * and leaves its change time alone. `_now` is taken and not used, for callers
 * written when it did move that time.
 */
export function setDescribe(env: LabEnv, id: string, text: string | null, _now?: () => number): RefInfo {
  const { idx, e } = entry(env, id)
  const t = text == null ? null : String(text).replace(/\s+/g, ' ').trim()
  if (t !== null && (t.length < 3 || t.length > 300)) throw new Error('The description must be 3 to 300 characters.')
  idx.refs[id] = { ...e, describe: t || null }
  writeIndex(env, idx)
  return infoOf(idx.refs[id])
}

/**
 * Give a photo another id: a file put in the refs folder under its camera
 * name (img-2034) becomes the 'scene' photo. Its marked area and
 * description go with it; any photo already under the new id is replaced.
 */
export function renameRef(env: LabEnv, from: string, to: string, now: () => number = Date.now): RefInfo {
  const target = refIdFrom(to)
  const { e } = entry(env, from)
  if (target === e.id) return infoOf(e)
  const dir = refsDir(env)
  const bytes = fs.readFileSync(path.join(dir, e.file))
  const info = addRef(env, bytes, target, now)
  const idx = discover(env, now)
  const moved = idx.refs[target]
  if (moved && e.mask && e.rect && fs.existsSync(maskPath(env, e.id))) {
    fs.copyFileSync(maskPath(env, e.id), maskPath(env, target))
    idx.refs[target] = { ...moved, mask: true, rect: e.rect, maskSha12: e.maskSha12 ?? null }
  }
  if (moved && e.describe) idx.refs[target] = { ...idx.refs[target], describe: e.describe }
  removeQuietly(path.join(dir, e.file))
  removeQuietly(maskPath(env, e.id))
  delete idx.refs[e.id]
  writeIndex(env, idx)
  return idx.refs[target] ? infoOf(idx.refs[target]) : info
}

/** The photo's own file, for the lab page to show (the browser turns it upright). */
export function refFile(env: LabEnv, id: string): { path: string; mime: string; info: RefInfo } | null {
  const idx = discover(env)
  const e = Object.hasOwn(idx.refs, id) ? idx.refs[id] : undefined
  if (!e) return null
  const kind = e.ext === 'jpg' ? 'jpeg' : (e.ext as 'png' | 'webp')
  return { path: path.join(refsDir(env), e.file), mime: MIME_OF[kind] ?? 'application/octet-stream', info: infoOf(e) }
}

/** When a photo or its marked area last changed, epoch ms, or null when there is none. */
export function refUpdatedAt(env: LabEnv, id: string): number | null {
  const idx = discover(env)
  return Object.hasOwn(idx.refs, id) ? idx.refs[id].updatedAt ?? null : null
}

/** The sentence for a photo that has not arrived. */
export function missingSentence(env: LabEnv, id: string): string {
  return `The "${id}" photo is not in the lab yet. Upload it on the lab page, or put it in ${refsDir(env)} as ${id}.jpg (or .png, .webp).`
}

/** The sentence for a region test's photo with no area marked. */
export function noAreaSentence(id: string): string {
  return `The "${id}" photo has no area marked for redrawing yet. Draw the box on the lab page.`
}

/**
 * What stops a night that needs these photos, in words: a photo that has not
 * arrived, or a region test's photo with no area marked. Empty when all is there.
 */
export function refProblems(env: LabEnv, needs: readonly { id: string; mask?: boolean }[]): string[] {
  const idx = discover(env)
  const out: string[] = []
  const seen = new Set<string>()
  for (const n of needs) {
    const key = `${n.id}:${n.mask ? 1 : 0}`
    if (seen.has(key)) continue
    seen.add(key)
    const e = Object.hasOwn(idx.refs, n.id) ? idx.refs[n.id] : undefined
    if (!e) out.push(missingSentence(env, n.id))
    else if (n.mask && !e.mask) out.push(noAreaSentence(n.id))
  }
  return out
}

export type RefCopy = { id: string; sha12: string; ext: string; ref: string; mask: string | null }

/** Write `bytes` to `full` unless it already holds exactly them. */
function writeIfDifferent(full: string, bytes: Uint8Array): void {
  try {
    const cur = fs.readFileSync(full)
    if (cur.length === bytes.length && cur.equals(Buffer.from(bytes.buffer, bytes.byteOffset, bytes.length))) return
  } catch {
    /* not there yet */
  }
  writeAtomic(full, bytes)
}

/**
 * Copy these photos (and their masks) to <outputs>/.lab/refs/, where ComfyUI
 * reads them as '.lab/refs/<sha12>.<ext> [output]'. A mask's copy is named by
 * the mask's own hash (maskKey), as the region cells' 'mask:' placeholders
 * name it, so a rectangle drawn again is a new file and never overwrites the
 * one an earlier plan used. Throws, in words, for a photo that has not
 * arrived or has changed on disk since it was listed.
 */
export function ensureRefCopies(env: LabEnv, ids: readonly string[]): Record<string, RefCopy> {
  const idx = discover(env)
  const out: Record<string, RefCopy> = {}
  for (const id of new Set(ids)) {
    const e = Object.hasOwn(idx.refs, id) ? idx.refs[id] : undefined
    if (!e) throw new Error(missingSentence(env, id))
    const bytes = fs.readFileSync(path.join(refsDir(env), e.file))
    if (sha12Of(bytes) !== e.sha12) throw new Error(`The "${id}" photo changed while the lab was reading it. Try again.`)
    const ref = refRel(e.sha12, e.ext)
    writeIfDifferent(path.join(env.outputs, ref), cleanCopy(bytes))
    let mask: string | null = null
    if (e.mask) {
      let png: Buffer
      try {
        png = fs.readFileSync(maskPath(env, id))
      } catch {
        throw new Error(`The marked area of the "${id}" photo is missing. Draw it again on the lab page.`)
      }
      if (e.maskSha12 && sha12Of(png) !== e.maskSha12) throw new Error(`The marked area of the "${id}" photo is not the one drawn on the lab page. Draw it again on the lab page.`)
      mask = maskRel(maskKey(e))
      writeIfDifferent(path.join(env.outputs, mask), png)
    }
    out[id] = { id, sha12: e.sha12, ext: e.ext, ref, mask }
  }
  return out
}
