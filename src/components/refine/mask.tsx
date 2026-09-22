/**
 * The mask model behind the region brush.
 *
 * WHAT THE GRAPH NEEDS, AND WHY THIS FILE IS SHAPED THE WAY IT IS.
 *
 * src/lib/refine.ts wires LoadImageMask on the RED channel. So the file this
 * module produces is a PNG the exact size of the source picture, WHITE where
 * the region is and BLACK everywhere else, fully opaque. The opacity is not a
 * detail: LoadImageMask on the alpha channel returns 1 minus alpha, so an
 * RGB PNG read as alpha comes back as an empty mask and the refine pass
 * silently does nothing at all. Two canvases keep that honest here. Strokes are
 * painted on a transparent LAYER canvas, which is what makes an eraser possible
 * at all, and the layer is then flattened onto an opaque black canvas at export
 * time. Nothing with an alpha channel ever leaves this module.
 *
 * WHY STROKES ARE KEPT AS DATA.
 * The layer is re rendered from the stroke list on every change. That is O(n)
 * per edit and therefore O(n squared) over a session, which for the few dozen
 * strokes a person actually draws is far below a frame. In exchange, undo and
 * redo are a slice of an array rather than a stack of bitmaps, and the bounding
 * box can be recomputed exactly rather than tracked incrementally.
 *
 * WHY THE BOUNDS ARE MEASURED TWO DIFFERENT WAYS.
 * With paint strokes only, the bounding box is the union of the stroke
 * geometry: exact, and free. Once anything has been erased, geometry no longer
 * describes what is on the canvas, so the alpha channel is scanned instead. The
 * scan is strided on large pictures and the result is then widened by the
 * stride, so the measured box is never SMALLER than the real mask. A box that
 * is a few pixels generous costs nothing. A box that clips the mask would
 * composite a cut off region back into the picture.
 *
 * Note this file is .tsx and contains no components. It sits inside the folder
 * this agent owns, which is scoped to .tsx, and the hook has to live beside the
 * geometry it serves.
 */
import { useCallback, useMemo, useState } from 'react'
import type { Rect } from '../../lib/refine'

export type MaskPoint = { x: number; y: number }

/**
 * One editing action, in SOURCE pixel coordinates. Screen coordinates are never
 * stored: the canvas is resized by the layout, by a window resize and by the
 * device pixel ratio, and a stroke recorded in screen space would drift away
 * from the picture the moment any of those changed.
 */
export type MaskStroke =
  | { kind: 'paint'; size: number; points: MaskPoint[] }
  | { kind: 'erase'; size: number; points: MaskPoint[] }
  | { kind: 'rect'; rect: Rect }

export type MaskTool = 'paint' | 'erase'

/** Pixels below which the scan runs unstrided. */
const SCAN_BUDGET = 1_400_000

export function clampRect(r: Rect, width: number, height: number): Rect {
  const x = Math.max(0, Math.min(Math.floor(r.x), width - 1))
  const y = Math.max(0, Math.min(Math.floor(r.y), height - 1))
  return {
    x,
    y,
    width: Math.max(1, Math.min(Math.ceil(r.width), width - x)),
    height: Math.max(1, Math.min(Math.ceil(r.height), height - y)),
  }
}

/** Paint one stroke into a context already sized to the source picture. */
export function drawStroke(ctx: CanvasRenderingContext2D, s: MaskStroke): void {
  ctx.save()
  if (s.kind === 'rect') {
    ctx.globalCompositeOperation = 'source-over'
    ctx.fillStyle = '#ffffff'
    ctx.fillRect(s.rect.x, s.rect.y, s.rect.width, s.rect.height)
    ctx.restore()
    return
  }
  // destination-out is what makes the eraser cut a hole in the layer rather
  // than painting black over it. Painting black would survive the flatten and
  // read as "do not refine here", which is the same result by accident, but it
  // would also defeat the alpha scan that measures the bounding box.
  ctx.globalCompositeOperation = s.kind === 'erase' ? 'destination-out' : 'source-over'
  ctx.strokeStyle = '#ffffff'
  ctx.fillStyle = '#ffffff'
  ctx.lineWidth = Math.max(1, s.size)
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  const pts = s.points
  if (pts.length === 1) {
    ctx.beginPath()
    ctx.arc(pts[0].x, pts[0].y, Math.max(0.5, s.size / 2), 0, Math.PI * 2)
    ctx.fill()
  } else if (pts.length > 1) {
    ctx.beginPath()
    ctx.moveTo(pts[0].x, pts[0].y)
    for (let i = 1; i < pts.length; i += 1) ctx.lineTo(pts[i].x, pts[i].y)
    ctx.stroke()
  }
  ctx.restore()
}

/** Re render the whole layer from the stroke list. */
export function renderMask(layer: HTMLCanvasElement, strokes: readonly MaskStroke[]): void {
  const ctx = layer.getContext('2d')
  if (!ctx) return
  ctx.setTransform(1, 0, 0, 1, 0, 0)
  ctx.clearRect(0, 0, layer.width, layer.height)
  for (const s of strokes) drawStroke(ctx, s)
}

/** Union of the painted geometry. Erase strokes are not subtracted here. */
export function geometricBounds(strokes: readonly MaskStroke[]): Rect | null {
  let x0 = Infinity
  let y0 = Infinity
  let x1 = -Infinity
  let y1 = -Infinity
  for (const s of strokes) {
    if (s.kind === 'erase') continue
    if (s.kind === 'rect') {
      x0 = Math.min(x0, s.rect.x)
      y0 = Math.min(y0, s.rect.y)
      x1 = Math.max(x1, s.rect.x + s.rect.width)
      y1 = Math.max(y1, s.rect.y + s.rect.height)
      continue
    }
    const r = s.size / 2
    for (const p of s.points) {
      x0 = Math.min(x0, p.x - r)
      y0 = Math.min(y0, p.y - r)
      x1 = Math.max(x1, p.x + r)
      y1 = Math.max(y1, p.y + r)
    }
  }
  if (!Number.isFinite(x0) || x1 <= x0 || y1 <= y0) return null
  return { x: x0, y: y0, width: x1 - x0, height: y1 - y0 }
}

/**
 * Bounding box of everything still opaque on the layer. Used once anything has
 * been erased, where geometry alone would over report.
 */
export function scanBounds(layer: HTMLCanvasElement): Rect | null {
  const ctx = layer.getContext('2d', { willReadFrequently: true })
  if (!ctx || layer.width < 1 || layer.height < 1) return null
  let data: Uint8ClampedArray
  try {
    data = ctx.getImageData(0, 0, layer.width, layer.height).data
  } catch {
    // A tainted canvas cannot be read back. Nothing cross origin is ever drawn
    // into the layer, so this should not happen, and if it does the geometry
    // path is the right answer rather than a crash.
    return null
  }
  const pixels = layer.width * layer.height
  const stride = pixels > SCAN_BUDGET ? Math.ceil(Math.sqrt(pixels / SCAN_BUDGET)) : 1
  let x0 = Infinity
  let y0 = Infinity
  let x1 = -Infinity
  let y1 = -Infinity
  for (let y = 0; y < layer.height; y += stride) {
    const row = y * layer.width
    for (let x = 0; x < layer.width; x += stride) {
      // 16 is a quarter of full alpha: below that the blurred edge of a stroke
      // contributes nothing the grow and feather in the graph will not restore.
      if (data[(row + x) * 4 + 3] > 16) {
        if (x < x0) x0 = x
        if (x > x1) x1 = x
        if (y < y0) y0 = y
        if (y > y1) y1 = y
      }
    }
  }
  if (!Number.isFinite(x0)) return null
  // Widen by the stride so a strided scan never reports a box inside the mask.
  return clampRect(
    { x: x0 - stride, y: y0 - stride, width: x1 - x0 + stride * 2, height: y1 - y0 + stride * 2 },
    layer.width,
    layer.height,
  )
}

/** The mask's bounding box in source pixels, or null when nothing is marked. */
export function maskBounds(
  layer: HTMLCanvasElement | null,
  strokes: readonly MaskStroke[],
): Rect | null {
  if (!strokes.some(s => s.kind !== 'erase')) return null
  if (layer && strokes.some(s => s.kind === 'erase')) return scanBounds(layer)
  const geo = geometricBounds(strokes)
  if (!geo) return null
  return layer ? clampRect(geo, layer.width, layer.height) : geo
}

/**
 * Flatten the layer onto opaque black and encode it as a PNG.
 *
 * The result is exactly the size of the source picture, which
 * ImageCompositeMasked and CropMask both assume: a mask of any other size is
 * interpolated to fit and the region moves.
 */
export function toMaskBlob(layer: HTMLCanvasElement): Promise<Blob> {
  const out = document.createElement('canvas')
  out.width = layer.width
  out.height = layer.height
  const ctx = out.getContext('2d')
  if (!ctx) {
    return Promise.reject(new Error('This browser gave us no 2D canvas, so the mask cannot be built.'))
  }
  ctx.fillStyle = '#000000'
  ctx.fillRect(0, 0, out.width, out.height)
  ctx.drawImage(layer, 0, 0)
  return new Promise((resolve, reject) => {
    out.toBlob(
      b => (b ? resolve(b) : reject(new Error('The mask image could not be encoded as a PNG.'))),
      'image/png',
    )
  })
}

// ---------------------------------------------------------------------------
// The editor
// ---------------------------------------------------------------------------

export type MaskEditor = {
  /** The transparent stroke layer, at source resolution. Null until sized. */
  layer: HTMLCanvasElement | null
  /** Every committed stroke. Its identity changes on every edit, so views can
   *  use it as their redraw signal. */
  strokes: readonly MaskStroke[]
  marked: boolean
  bounds: Rect | null
  canUndo: boolean
  canRedo: boolean
  commit: (s: MaskStroke) => void
  undo: () => void
  redo: () => void
  clear: () => void
  /** The finished PNG: source sized, white on black, opaque. */
  blob: () => Promise<Blob>
}

type Doc = { strokes: MaskStroke[]; undone: MaskStroke[] }

/** The layer and the edit history belong to one picture, so they reset together. */
type Session = { signature: string; layer: HTMLCanvasElement | null; doc: Doc }

const EMPTY: Doc = { strokes: [], undone: [] }

function makeLayer(width: number, height: number): HTMLCanvasElement | null {
  if (width < 1 || height < 1) return null
  const c = document.createElement('canvas')
  c.width = width
  c.height = height
  return c
}

/**
 * Hold the mask for one source picture.
 *
 * There are no effects in here on purpose. When the picture changes, the layer
 * and the history are replaced DURING render, which is React's own answer to
 * "adjust state when a prop changes": the component renders again and nothing
 * intermediate is ever committed. An effect would paint one frame of the
 * previous picture's mask over the new one, and on a refined result that came
 * back at exactly the same size nobody would notice until it queued.
 */
export function useMaskEditor(
  size: { width: number; height: number } | null,
  /**
   * Anything that identifies the picture, usually its URL. A refined result is
   * composited back into the source, so the next picture very often has exactly
   * the same dimensions as this one: without an identity the strokes drawn on
   * the old frame would silently survive onto the new one.
   */
  identity?: string,
): MaskEditor {
  const width = Math.max(0, Math.round(size?.width ?? 0))
  const height = Math.max(0, Math.round(size?.height ?? 0))
  const signature = `${width}x${height}:${identity ?? ''}`

  const [session, setSession] = useState<Session>(() => ({
    signature,
    layer: makeLayer(width, height),
    doc: EMPTY,
  }))
  if (session.signature !== signature) {
    setSession({ signature, layer: makeLayer(width, height), doc: EMPTY })
  }
  // This render pass is thrown away when the signature just changed, so it runs
  // against nothing rather than against the outgoing picture's strokes.
  const live: Session =
    session.signature === signature ? session : { signature, layer: null, doc: EMPTY }
  const { layer } = live
  const { strokes, undone } = live.doc

  // Render the layer and measure it in one step, both derived from the stroke
  // list rather than tracked alongside it. renderMask clears first, so running
  // it twice for the same strokes is the same as running it once.
  const bounds = useMemo(() => {
    if (!layer) return null
    renderMask(layer, strokes)
    return maskBounds(layer, strokes)
  }, [layer, strokes])

  const edit = useCallback((fn: (d: Doc) => Doc) => {
    setSession(s => {
      const next = fn(s.doc)
      return next === s.doc ? s : { ...s, doc: next }
    })
  }, [])

  const commit = useCallback(
    (s: MaskStroke) => edit(d => ({ strokes: [...d.strokes, s], undone: [] })),
    [edit],
  )

  const undo = useCallback(
    () =>
      edit(d =>
        d.strokes.length
          ? {
              strokes: d.strokes.slice(0, -1),
              undone: [...d.undone, d.strokes[d.strokes.length - 1]],
            }
          : d,
      ),
    [edit],
  )

  const redo = useCallback(
    () =>
      edit(d =>
        d.undone.length
          ? {
              strokes: [...d.strokes, d.undone[d.undone.length - 1]],
              undone: d.undone.slice(0, -1),
            }
          : d,
      ),
    [edit],
  )

  /** Clearing is undoable: the strokes go onto the redo stack in reverse. */
  const clear = useCallback(
    () => edit(d => (d.strokes.length ? { strokes: [], undone: d.strokes.slice().reverse() } : d)),
    [edit],
  )

  const blob = useCallback(() => {
    if (!layer) return Promise.reject(new Error('There is no picture to mask yet.'))
    return toMaskBlob(layer)
  }, [layer])

  return {
    layer,
    strokes,
    marked: bounds !== null,
    bounds,
    canUndo: strokes.length > 0,
    canRedo: undone.length > 0,
    commit,
    undo,
    redo,
    clear,
    blob,
  }
}

/** A sensible starting brush for a picture of this size, in source pixels. */
export function defaultBrush(size: { width: number; height: number } | null): number {
  if (!size) return 64
  return Math.max(8, Math.round(Math.min(size.width, size.height) * 0.06))
}
