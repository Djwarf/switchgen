/**
 * Thumbnail addresses for files in the outputs folder.
 *
 * A grid card or a picker tile a few hundred pixels wide used to load the
 * original render, a PNG of a megabyte or more, and decode all of it. GET
 * /api/thumb (server/thumbs.mjs) answers with a WebP a small fraction of that
 * size, made once on the server and kept there. It is a path of this app's
 * own server, so it works on every device the app does.
 *
 * When the server cannot make one:
 *   - a picture is redirected to the original, so an <img> shows what it
 *     showed before thumbnails existed, with nothing for the caller to do;
 *   - a clip is answered 503, so the <img> fires onError, and the caller
 *     decides what to show (the archive grabs a frame in the browser).
 * Files in ComfyUI's input and temp folders have no thumbnails; for those
 * thumbUrl gives the file's own address.
 *
 * A thumbnail is named by its file, as the file is, and a name can be handed
 * to a new render. The server marks each answer with the file's size and date
 * and has the browser ask again every time, so a replaced file never shows
 * the old picture.
 */
import { fileUrl, relPath, type FileRef } from './comfy'

/** The widths the server makes. Keep in step with WIDTHS in server/thumbs.mjs. */
export const THUMB_WIDTHS = [256, 512, 1024] as const
export type ThumbWidth = (typeof THUMB_WIDTHS)[number]

/** Does the server make thumbnails of this file? Only of files in the outputs folder. */
export function hasThumb(file: FileRef): boolean {
  return !file.type || file.type === 'output'
}

/** The smallest width made that covers `width`, or the largest when none does. */
function widthFor(width: number): ThumbWidth {
  for (const w of THUMB_WIDTHS) if (w >= width) return w
  return THUMB_WIDTHS[THUMB_WIDTHS.length - 1]
}

/**
 * Where to load `file` at least `width` pixels across, rounded up to a width
 * the server makes. Pass the width it is drawn at in device pixels: a tile's
 * CSS width times devicePixelRatio. Where the tile's width changes with the
 * layout, give the <img> thumbSrcSet and `sizes` as well, and the browser
 * picks.
 */
export function thumbUrl(file: FileRef, width: number): string {
  if (!hasThumb(file)) return fileUrl(file)
  const q = new URLSearchParams({ rel: relPath(file), w: String(widthFor(width)) })
  return `/api/thumb?${q}`
}

/** Every width made, as an <img srcSet>. Undefined for a file without thumbnails. */
export function thumbSrcSet(file: FileRef): string | undefined {
  if (!hasThumb(file)) return undefined
  return THUMB_WIDTHS.map((w) => `${thumbUrl(file, w)} ${w}w`).join(', ')
}
