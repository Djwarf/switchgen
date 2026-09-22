/**
 * The brush.
 *
 * WHY A BRUSH AND NOT A RECTANGLE. The regions this exists for are not
 * rectangular. A breast, a vulva, a penis, a foot: each is a shape sitting on
 * skin that is already correct. The mask does two jobs in the derived graph. It
 * drives SetLatentNoiseMask, so only the marked cells actually move while the
 * sampler still sees the body around them, and it drives ImageCompositeMasked,
 * so only the marked cells are pasted back. A rectangle would hand both of
 * those a box of surrounding skin and re render that too, which is how a
 * correct nipple ends up on a slightly different breast. One tool, done
 * properly, beats a brush and a box done half way.
 *
 * WHAT THE KEYBOARD GETS. A pointer is not available on a television, and a
 * pass with no keyboard path is a pass that does not exist there. The numeric
 * rectangle under the picture, in `refine/index.tsx`, is that path: four
 * numbers committed as a `rect` stroke through the same `editor.commit` the
 * brush uses. It is second best on purpose and the copy beside it says so, a
 * box takes the skin around the region with it, but second best beats nothing.
 * The canvas's own keys resize the brush, swap the tool and undo.
 *
 * WHAT IS ON SCREEN. The picture, at whatever size the column allows, with a
 * canvas laid exactly over it. The canvas is transparent except where the mask
 * is, which is tinted burgundy so the anatomy underneath stays readable while
 * you paint. The dashed rectangle is the CROP: the region plus its padding,
 * which is what actually gets cut out and re rendered. Seeing it matters,
 * because that rectangle is what the sampler will use as context.
 *
 * COORDINATES. Every stroke is recorded in source pixels. The canvas backing
 * store is sized in device pixels and the drawing transform scales source to
 * device, so a stroke drawn on a phone, on a 4K television and on the desk all
 * land in the same place in the file. Pointer positions are converted through
 * the canvas's own bounding rectangle, which is the only measurement that
 * survives a zoom, a resize and a device pixel ratio change.
 */
import { useCallback, useEffect, useRef } from 'react'
import type { KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent } from 'react'
import type { Rect } from '../../lib/refine'
import { drawStroke, type MaskEditor, type MaskPoint, type MaskStroke, type MaskTool } from './mask'
import { clamp } from './bits'

/** Burgundy 900 at just under half, which reads over both light and dark skin. */
const TINT = 'rgba(46, 0, 0, 0.45)'

/** Device pixel ratio is capped: a 4K television does not need a 4x backing store. */
const MAX_DPR = 2

export function MaskCanvas({
  image,
  editor,
  brush,
  tool,
  crop,
  disabled = false,
  onBrush,
  onTool,
}: {
  image: { url: string; width: number; height: number }
  editor: MaskEditor
  /** Brush diameter in SOURCE pixels. */
  brush: number
  tool: MaskTool
  /** The crop rectangle the current mask would produce, drawn as context. */
  crop: Rect | null
  disabled?: boolean
  onBrush: (n: number) => void
  onTool: (t: MaskTool) => void
}) {
  const wrapRef = useRef<HTMLDivElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const draftRef = useRef<MaskStroke | null>(null)
  const hoverRef = useRef<MaskPoint | null>(null)
  const activeRef = useRef<number | null>(null)
  const frameRef = useRef(0)

  const redraw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.setTransform(1, 0, 0, 1, 0, 0)
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    const sx = canvas.width / image.width
    const sy = canvas.height / image.height
    if (!Number.isFinite(sx) || !Number.isFinite(sy) || sx <= 0 || sy <= 0) return
    ctx.setTransform(sx, 0, 0, sy, 0, 0)

    // The committed mask, plus the stroke currently under the pointer. An
    // erasing draft uses destination-out, so it previews as a real hole rather
    // than as paint of another colour.
    if (editor.layer) ctx.drawImage(editor.layer, 0, 0)
    const draft = draftRef.current
    if (draft) drawStroke(ctx, draft)

    // Tint what is marked, and only what is marked. source-in multiplies the
    // fill by the alpha already on the canvas, so the picture underneath stays
    // visible through it and nothing outside the mask is touched.
    ctx.globalCompositeOperation = 'source-in'
    ctx.fillStyle = TINT
    ctx.fillRect(0, 0, image.width, image.height)
    ctx.globalCompositeOperation = 'source-over'

    // One screen pixel, whatever the scale. Lines specified in source units
    // would be hairlines on a big picture and slabs on a small one.
    const px = 1 / Math.max(sx, 0.0001)

    if (crop) {
      ctx.save()
      ctx.lineWidth = px
      ctx.strokeStyle = 'rgba(250, 250, 248, 0.9)'
      ctx.strokeRect(crop.x, crop.y, crop.width, crop.height)
      ctx.setLineDash([6 * px, 4 * px])
      ctx.strokeStyle = 'rgba(46, 0, 0, 0.95)'
      ctx.strokeRect(crop.x, crop.y, crop.width, crop.height)
      ctx.restore()
    }

    const hover = hoverRef.current
    if (hover && !disabled) {
      ctx.save()
      ctx.lineWidth = px
      ctx.beginPath()
      ctx.arc(hover.x, hover.y, Math.max(1, brush / 2), 0, Math.PI * 2)
      ctx.strokeStyle = 'rgba(250, 250, 248, 0.95)'
      ctx.stroke()
      ctx.setLineDash([4 * px, 3 * px])
      ctx.strokeStyle = tool === 'erase' ? 'rgba(153, 27, 27, 0.95)' : 'rgba(46, 0, 0, 0.95)'
      ctx.stroke()
      ctx.restore()
    }
  }, [brush, crop, disabled, editor.layer, image.height, image.width, tool])

  /** One redraw per frame, however many pointer samples arrive. */
  const schedule = useCallback(() => {
    if (frameRef.current) return
    frameRef.current = requestAnimationFrame(() => {
      frameRef.current = 0
      redraw()
    })
  }, [redraw])

  useEffect(() => () => { if (frameRef.current) cancelAnimationFrame(frameRef.current) }, [])

  // Redraw on every committed change, and on every change to what is drawn on
  // top of the mask. The stroke list gets a new identity on each edit, which is
  // what makes it a usable signal here.
  useEffect(() => { schedule() }, [schedule, editor.strokes])

  // The backing store follows the element's real size. Without this the canvas
  // keeps its 300x150 default and the mask lands nowhere near the picture.
  useEffect(() => {
    const el = wrapRef.current
    const canvas = canvasRef.current
    if (!el || !canvas) return
    const apply = () => {
      const r = el.getBoundingClientRect()
      if (r.width < 1 || r.height < 1) return
      const dpr = Math.min(MAX_DPR, window.devicePixelRatio || 1)
      const w = Math.max(1, Math.round(r.width * dpr))
      const h = Math.max(1, Math.round(r.height * dpr))
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w
        canvas.height = h
        schedule()
      }
    }
    apply()
    const ro = new ResizeObserver(apply)
    ro.observe(el)
    return () => ro.disconnect()
  }, [schedule])

  const toSource = useCallback(
    (clientX: number, clientY: number): MaskPoint | null => {
      const canvas = canvasRef.current
      if (!canvas) return null
      const r = canvas.getBoundingClientRect()
      if (r.width < 1 || r.height < 1) return null
      return {
        x: clamp(((clientX - r.left) / r.width) * image.width, 0, image.width),
        y: clamp(((clientY - r.top) / r.height) * image.height, 0, image.height),
      }
    },
    [image.height, image.width],
  )

  const extend = useCallback((p: MaskPoint) => {
    const d = draftRef.current
    if (!d || d.kind === 'rect') return
    const last = d.points[d.points.length - 1]
    // Points closer together than a tenth of the brush add nothing to the
    // rendered path and cost memory on a long stroke.
    const min = Math.max(0.75, d.size * 0.1)
    if (!last || Math.hypot(p.x - last.x, p.y - last.y) >= min) d.points.push(p)
  }, [])

  const onDown = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    if (disabled || e.button > 0) return
    const p = toSource(e.clientX, e.clientY)
    if (!p) return
    e.preventDefault()
    wrapRef.current?.focus()
    try {
      e.currentTarget.setPointerCapture(e.pointerId)
    } catch {
      // Capture is a convenience. Without it the stroke simply ends at the edge.
    }
    activeRef.current = e.pointerId
    draftRef.current = { kind: tool, size: brush, points: [p] }
    hoverRef.current = p
    schedule()
  }

  const onMove = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    if (disabled) return
    const drawing = activeRef.current === e.pointerId && draftRef.current !== null
    if (drawing) {
      // A fast stroke between two frames is reported as one move event with the
      // intermediate samples folded into it. Reading them back is the
      // difference between a smooth curve and a polygon.
      const native = e.nativeEvent
      const samples =
        typeof native.getCoalescedEvents === 'function' ? native.getCoalescedEvents() : []
      if (samples.length > 1) {
        for (const s of samples) {
          const q = toSource(s.clientX, s.clientY)
          if (q) extend(q)
        }
      }
    }
    const p = toSource(e.clientX, e.clientY)
    if (p) {
      hoverRef.current = p
      if (drawing) extend(p)
    }
    schedule()
  }

  const finish = (e: ReactPointerEvent<HTMLCanvasElement>, keep: boolean) => {
    if (activeRef.current !== e.pointerId) return
    activeRef.current = null
    const d = draftRef.current
    draftRef.current = null
    if (keep && d && (d.kind === 'rect' || d.points.length > 0)) editor.commit(d)
    schedule()
  }

  /**
   * Every key this canvas claims is claimed exclusively.
   *
   * The wrapper is a `role="application"` div, which the shell's typing guard
   * correctly does not count as a text field, so without `stopPropagation` the
   * key also runs whatever the window listeners have bound to it: `e` reached
   * the shell and toggled expert mode, collapsing the expert column and re
   * laying out the very canvas the stroke was being drawn on, and `[` and `]`
   * reached the player's speed control. Taking the key here means taking it,
   * so a key is stopped on exactly the branches that handle it.
   */
  const take = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const onKey = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    if (e.altKey) return
    // Ctrl+Z and Cmd+Z undo the last stroke, the same as bare Z. A mask is a
    // document being edited, and the undo key people reach for first is the
    // one their word processor uses.
    const mod = e.ctrlKey || e.metaKey
    const key = e.key.length === 1 ? e.key.toLowerCase() : e.key

    if (!mod && key === '[') {
      take(e)
      onBrush(Math.max(4, brush - Math.max(2, Math.round(brush * 0.25))))
    } else if (!mod && key === ']') {
      take(e)
      onBrush(brush + Math.max(2, Math.round(brush * 0.25)))
    } else if (!mod && key === 'e') {
      take(e)
      onTool(tool === 'erase' ? 'paint' : 'erase')
    } else if (key === 'z' && !e.shiftKey) {
      take(e)
      editor.undo()
    } else if (key === 'y' || (key === 'z' && e.shiftKey)) {
      take(e)
      editor.redo()
    } else if (key === 'Escape') {
      // Escape only belongs to the canvas while there is a stroke to abandon.
      // Otherwise it belongs to whatever is open around it, so it travels.
      if (draftRef.current) {
        take(e)
        draftRef.current = null
        activeRef.current = null
        schedule()
      }
    }
  }

  return (
    <div
      ref={wrapRef}
      role="application"
      tabIndex={0}
      aria-label={
        `Mask the region to refine. Drag on the picture to paint over it. ` +
        `Brush ${Math.round(brush)} pixels, ${tool === 'erase' ? 'erasing' : 'painting'}. ` +
        `Square bracket keys resize the brush, E switches between painting and erasing, Z undoes.`
      }
      onKeyDown={onKey}
      className={`relative inline-block max-w-full border border-grey-300 bg-newsprint-aged align-top ${
        disabled ? '' : 'cursor-crosshair'
      } focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900`}
    >
      <img
        src={image.url}
        alt="The picture you are marking a region on"
        draggable={false}
        className="block max-h-[58vh] w-auto max-w-full select-none"
      />
      <canvas
        ref={canvasRef}
        className="absolute inset-0 h-full w-full touch-none"
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={e => finish(e, true)}
        onPointerCancel={e => finish(e, false)}
        onPointerLeave={() => {
          hoverRef.current = null
          schedule()
        }}
      />
    </div>
  )
}
