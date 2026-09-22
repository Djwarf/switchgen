/**
 * REGION REFINE: the surface, and the direct answer to bad anatomy.
 *
 * A diffusion model spends its capacity evenly over the latent grid. A region
 * that occupies four percent of a 1024 frame is about 650 latent cells, and the
 * feature inside it that has to be right, a nipple, a set of labia, a glans, a
 * knuckle, is a few dozen. Nothing reconstructs correct anatomy from a few
 * dozen numbers, which is why genitalia, nipples, hands and faces are always
 * the first things to fall apart, and why a different checkpoint does not fix
 * it: the same cells move to a different model. More pixels at the region is
 * the fix. Mark it, cut it out, enlarge it to a full working resolution, draw
 * it again, composite it back.
 *
 * WHAT THIS COMPONENT DOES AND DOES NOT DO.
 * It owns the mask, the plan and the settings. It does not touch the network,
 * the family registry or the job engine: it hands the caller a finished PNG and
 * a finished plan, and the desk that owns the press queues it. That boundary is
 * what keeps this folder testable and keeps the GPU in one place.
 *
 * WIRING IT UP, IN FULL:
 *
 *   import { deriveRefine, instantiateRefine } from '../lib/refine'
 *   import { RegionRefine, type RefineRequest } from '../components/refine'
 *
 *   const refinable = deriveRefine(family)      // null means: do not offer it
 *
 *   async function onRun(req: RefineRequest) {
 *     if (!refinable) return
 *     const mask = await uploadImage(req.mask, req.maskName)
 *     const wf = instantiateRefine(refinable, baseParams, {
 *       image: sourceNameInComfyInput,   // the SOURCE, already uploaded
 *       mask,                            // what this component just produced
 *       crop: req.plan.crop,
 *       target: req.plan.target,
 *       denoise: req.denoise,
 *       grow: req.grow,
 *       feather: req.feather,
 *       prompt: req.prompt,
 *       seed: req.seed,                  // undefined means: keep the picture's
 *     })
 *     await run(wf, onProgress)
 *   }
 *
 * The mask is a PNG the exact size of the source, WHITE on BLACK and fully
 * opaque, because the derived graph reads it through LoadImageMask on the RED
 * channel. Upload it exactly as handed over. Re encoding it with an alpha
 * channel would make LoadImageMask return one minus alpha, the mask would read
 * as empty, and the pass would complete having changed nothing at all.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { planRefine, type RefinePlan, type Rect } from '../../lib/refine'
import { randomSeed } from '../../lib/session'
import { Caution, Chips, Head, Kicker, Quiet, clamp, times } from './bits'
import { clampRect, defaultBrush, useMaskEditor, type MaskTool } from './mask'
import { MaskCanvas } from './MaskCanvas'
import { Compare } from './Compare'
import { RefinePanel, REFINE_SETTINGS, type RefineSettings, type RegionAddOns } from './RefinePanel'

export { Compare } from './Compare'
export { RefinePanel, REFINE_SETTINGS, type RefineSettings, type RegionAddOns } from './RefinePanel'
export { useMaskEditor, toMaskBlob, type MaskStroke, type MaskTool } from './mask'

/** Everything the desk needs to queue one refine pass. */
export type RefineRequest = {
  /** PNG, source sized, white on black, opaque. Upload it unchanged. */
  mask: Blob
  /** A unique name to upload it under. */
  maskName: string
  plan: RefinePlan
  prompt: string
  denoise: number
  grow: number
  feather: number
  /** Undefined means: keep the seed the picture was made with. */
  seed?: number
}

export type RegionRefineProps = {
  /**
   * The picture to work on. Null prints the empty state.
   *
   * Keep this pointing at the picture that was refined. Swapping it for the
   * result the moment one lands is what turns a before and after into two
   * copies of the same frame. When the person asks to carry on from the
   * result, THEN make the result the source: the mask resets, because the
   * identity changed, and the comparison starts again from there.
   */
  source: { url: string; width: number; height: number } | null
  /** The prompt the picture was made with. Seeds the region prompt. */
  parentPrompt?: string
  /** The refined picture, once it lands. Drives the comparison. */
  result?: { url: string } | null
  expert?: boolean
  busy?: boolean
  /** 0 to 1 while the pass runs, null while it is merely queued. */
  progress?: number | null
  /** A reason the pass cannot run, from the desk. Printed, and blocks the button. */
  blocked?: string | null
  /**
   * Which model draws the region, and the alternatives.
   *
   * Plain `{id, label, group}` rows rather than registry types, so this folder
   * still knows nothing about families or graphs. Omit it and nothing is
   * printed, which is what a bench with only one capable model should do.
   *
   * Offered because the picture's own model often cannot redraw a region at
   * all, and the bench used to resolve that by silently taking whichever
   * capable model happened to come first.
   */
  model?: {
    options: readonly { id: string; label: string; group?: string }[]
    value: string
    onChange: (id: string) => void
    /** Why the picture's own model is not the one drawing. */
    note?: string | null
  }
  /**
   * Add-ons made for close framing that fit the model drawing the region,
   * offered unticked. Plain rows, like `model`, so this folder still knows
   * nothing about the add-on library. The desk chains the ticked ones and
   * names them on the record; nothing is applied without a tick.
   */
  addOns?: RegionAddOns
  onRun: (req: RefineRequest) => void
  onStop?: () => void
}

/** Four brush diameters, proportional to the picture rather than fixed. */
function brushSizes(size: { width: number; height: number } | null) {
  const base = size ? Math.min(size.width, size.height) : 1024
  return [0.02, 0.05, 0.1, 0.18].map(f => Math.max(6, Math.round(base * f)))
}

const BRUSH_LABELS = ['Fine', 'Small', 'Medium', 'Broad']

export function RegionRefine({
  source,
  parentPrompt = '',
  result = null,
  expert = false,
  busy = false,
  progress = null,
  blocked = null,
  model,
  addOns,
  onRun,
  onStop,
}: RegionRefineProps) {
  const editor = useMaskEditor(source, source?.url)
  const [settings, setSettings] = useState<RefineSettings>({ ...REFINE_SETTINGS, prompt: parentPrompt })
  const [tool, setTool] = useState<MaskTool>('paint')
  const [brush, setBrush] = useState(() => defaultBrush(source))
  const [fault, setFault] = useState<string | null>(null)
  const lastSource = useRef<string | null>(null)

  const patch = useCallback((p: Partial<RefineSettings>) => setSettings(s => ({ ...s, ...p })), [])

  // A new picture means a new region prompt and a brush sized for it. The
  // strength, the context and the expert numbers are the person's working
  // preferences and deliberately survive.
  useEffect(() => {
    if (!source || lastSource.current === source.url) return
    lastSource.current = source.url
    setBrush(defaultBrush(source))
    setSettings(s => ({ ...s, prompt: parentPrompt }))
    setFault(null)
  }, [source, parentPrompt])

  const plan = useMemo<RefinePlan | null>(() => {
    if (!source || !editor.bounds) return null
    return planRefine(editor.bounds, source, {
      padding: settings.padding,
      targetLongEdge: settings.targetLongEdge,
    })
  }, [source, editor.bounds, settings.padding, settings.targetLongEdge])

  const sizes = brushSizes(source)

  const start = useCallback(async () => {
    if (!source || !plan) return
    try {
      const mask = await editor.blob()
      setFault(null)
      onRun({
        mask,
        maskName: `refine-mask-${Date.now().toString(36)}.png`,
        plan,
        prompt: settings.prompt.trim() || parentPrompt,
        denoise: settings.denoise,
        grow: settings.grow,
        feather: settings.feather,
        seed: settings.newSeed ? randomSeed() : undefined,
      })
    } catch (err) {
      setFault(err instanceof Error ? err.message : 'The mask could not be built.')
    }
  }, [editor, onRun, parentPrompt, plan, settings, source])

  if (!source) {
    return (
      <section>
        <Head
          title="Refine a region"
          note="Mark one region of a picture and render just that region at full resolution."
        />
        <p className="text-small leading-snug text-grey-700">
          Make a picture first, or open one from the archive. A refine pass works on a finished
          picture: it cuts the region you mark out of it, enlarges the crop, draws it again and
          puts it back.
        </p>
      </section>
    )
  }

  return (
    <section>
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_23rem]">
        {/* ---- the picture and the brush -------------------------------- */}
        <div className="min-w-0">
          <div className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-2">
            <Chips<MaskTool>
              ariaLabel="Brush or eraser"
              value={tool}
              onChange={setTool}
              options={[
                { value: 'paint', label: 'Paint', title: 'Mark the region. Key: E toggles' },
                { value: 'erase', label: 'Erase', title: 'Take a mark back. Key: E toggles' },
              ]}
            />
            <Chips<number>
              ariaLabel="Brush size"
              value={sizes.reduce((best, s) => (Math.abs(s - brush) < Math.abs(best - brush) ? s : best), sizes[0])}
              onChange={setBrush}
              options={sizes.map((s, i) => ({
                value: s,
                label: BRUSH_LABELS[i],
                title: `${s} pixels across`,
              }))}
            />
            <div className="flex gap-1">
              <Quiet onClick={editor.undo} disabled={!editor.canUndo} title="Key: Z">
                Undo
              </Quiet>
              <Quiet onClick={editor.redo} disabled={!editor.canRedo} title="Key: Y">
                Redo
              </Quiet>
              <Quiet onClick={editor.clear} disabled={!editor.canUndo} danger>
                Clear
              </Quiet>
            </div>
            {expert ? (
              <label className="flex items-center gap-1.5 text-caption text-grey-700">
                <span className="uppercase tracking-[0.14em]">brush px</span>
                <input
                  type="number"
                  className="field w-20 tabular-nums"
                  value={brush}
                  min={2}
                  max={Math.max(8, Math.round(Math.min(source.width, source.height)))}
                  step={1}
                  onChange={e => {
                    const v = Number(e.target.value)
                    if (Number.isFinite(v)) setBrush(Math.round(clamp(v, 2, Math.min(source.width, source.height))))
                  }}
                />
              </label>
            ) : null}
          </div>

          {model && model.options.length > 1 && (
            <div className="mb-3">
              <label className="block">
                <span className="text-overline font-semibold uppercase tracking-[0.18em] text-grey-700">
                  Drawn by
                </span>
                <select
                  className="field mt-1 block max-w-[42ch]"
                  value={model.value}
                  disabled={busy}
                  onChange={e => model.onChange(e.target.value)}
                >
                  {model.options.map(o => (
                    <option key={o.id} value={o.id}>
                      {o.group && o.group !== o.label ? `${o.group} - ${o.label}` : o.label}
                    </option>
                  ))}
                </select>
              </label>
              {model.note ? (
                <p className="mt-1.5 max-w-[62ch] text-caption text-grey-500">{model.note}</p>
              ) : null}
            </div>
          )}

          <div className="flex justify-center">
            <MaskCanvas
              image={source}
              editor={editor}
              brush={brush}
              tool={tool}
              crop={plan?.crop ?? null}
              disabled={busy}
              onBrush={n => setBrush(Math.round(clamp(n, 2, Math.min(source.width, source.height))))}
              onTool={setTool}
            />
          </div>

          <p className="mt-2 text-caption text-grey-700">
            {plan ? (
              <>
                <Kicker tone="ink">Marked</Kicker>{' '}
                <span className="tabular-nums">
                  {plan.gain.sharePct.toFixed(1)}% of the frame. The dashed rectangle is what gets
                  cut out: {times(plan.crop.width, plan.crop.height)} at {Math.round(plan.crop.x)},{' '}
                  {Math.round(plan.crop.y)}.
                </span>
              </>
            ) : (
              <span className="italic">
                Drag on the picture to paint over the region you want redrawn. The brush follows
                the shape, so only what you paint is changed and the skin around it is kept. With
                no pointer to hand, type a rectangle below instead.
              </span>
            )}
          </p>
          <p aria-live="polite" className="sr-only">
            {plan
              ? `Region marked, ${plan.gain.sharePct.toFixed(1)} percent of the frame, ${Math.round(plan.gain.gain)} times the detail after refining.`
              : 'No region marked. Paint on the picture, or type a rectangle in the four fields below it.'}
          </p>

          <p className="mt-1 text-caption italic text-grey-500">
            Keys: square brackets resize the brush, E swaps paint and erase, Z undoes, Y redoes.
          </p>

          <RectangleMark
            key={source.url}
            source={source}
            disabled={busy}
            onCommit={r => editor.commit({ kind: 'rect', rect: r })}
          />
        </div>

        {/* ---- the controls --------------------------------------------- */}
        <div className="min-w-0">
          <RefinePanel
            settings={settings}
            onPatch={patch}
            plan={plan}
            parentPrompt={parentPrompt}
            marked={editor.marked}
            expert={expert}
            busy={busy}
            progress={progress}
            blocked={blocked}
            addOns={addOns}
            onRun={() => void start()}
            onStop={onStop}
          />
          {fault ? (
            <div className="mt-2">
              <Caution>{fault}</Caution>
            </div>
          ) : null}
        </div>
      </div>

      {result ? (
        <div className="mt-6 border-t-2 border-burgundy-900 pt-4">
          <Compare before={source} after={result} crop={plan?.crop ?? null} busy={busy} />
        </div>
      ) : null}
    </section>
  )
}

/**
 * The keyboard path to a region.
 *
 * This machine drives a television and boots into Big Picture, so a controller
 * or keyboard only session is the ordinary session rather than the exotic one.
 * Everything else on this surface is pointer input, and without this block the
 * whole quality layer was unreachable there: no mask, no bounds, no plan, and
 * a run button that never unblocked with nothing on screen saying why.
 *
 * A rectangle is worse than a brush and the note beside it says so rather than
 * pretending otherwise. It goes through exactly the same `editor.commit` the
 * brush uses, as the `rect` stroke the mask model has always carried, so undo,
 * redo, the bounds scan and the exported PNG all treat it as one more stroke.
 */
function RectangleMark({
  source,
  onCommit,
  disabled,
}: {
  source: { width: number; height: number }
  onCommit: (r: Rect) => void
  disabled?: boolean
}) {
  const short = Math.min(source.width, source.height)
  const [rect, setRect] = useState<Rect>(() => {
    // A centred box at two fifths of the short edge: large enough to be worth
    // refining, small enough that the gain figure is not already 1.0x. One
    // press of the button from here produces a usable mask.
    const side = Math.max(16, Math.round(short * 0.4))
    return {
      x: Math.round((source.width - side) / 2),
      y: Math.round((source.height - side) / 2),
      width: side,
      height: side,
    }
  })

  const set = (patch: Partial<Rect>) => setRect(r => ({ ...r, ...patch }))
  const committed = clampRect(rect, source.width, source.height)

  return (
    <div className="mt-3 border-t border-grey-300 pt-2">
      <Kicker tone="ink">Mark a rectangle by number</Kicker>
      <div className="mt-1.5 flex flex-wrap items-end gap-2">
        <Num
          label="Left"
          value={rect.x}
          min={0}
          max={Math.max(0, source.width - 1)}
          disabled={disabled}
          onChange={n => set({ x: n })}
        />
        <Num
          label="Top"
          value={rect.y}
          min={0}
          max={Math.max(0, source.height - 1)}
          disabled={disabled}
          onChange={n => set({ y: n })}
        />
        <Num
          label="Width"
          value={rect.width}
          min={1}
          max={source.width}
          disabled={disabled}
          onChange={n => set({ width: n })}
        />
        <Num
          label="Height"
          value={rect.height}
          min={1}
          max={source.height}
          disabled={disabled}
          onChange={n => set({ height: n })}
        />
        <Quiet onClick={() => onCommit(committed)} disabled={disabled}>
          Mark it
        </Quiet>
      </div>
      <p className="mt-1.5 text-caption italic leading-snug text-grey-500">
        Source pixels, with the origin at the top left. The arrow keys step each field by eight.
        A rectangle takes the skin around the region with it, so where there is a pointer the
        brush is the better tool. Marking adds to what is already there: use Clear to start again.
      </p>
      <p className="mt-1 text-caption tabular-nums text-grey-700">
        {times(committed.width, committed.height)} at {committed.x}, {committed.y}
        {committed.width !== Math.round(rect.width) ||
        committed.height !== Math.round(rect.height) ||
        committed.x !== Math.round(rect.x) ||
        committed.y !== Math.round(rect.y)
          ? ', trimmed to fit the picture'
          : ''}
      </p>
    </div>
  )
}

/** One field of the rectangle. Labelled, stepped for a remote control, clamped. */
function Num({
  label,
  value,
  min,
  max,
  disabled,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  disabled?: boolean
  onChange: (n: number) => void
}) {
  const id = `sg-refine-rect-${label.toLowerCase()}`
  return (
    <label htmlFor={id} className="block">
      <span className="mb-1 block text-overline font-semibold uppercase tracking-[0.18em] text-grey-700">
        {label}
      </span>
      <input
        id={id}
        type="number"
        className="field w-20 tabular-nums"
        value={value}
        min={min}
        max={max}
        step={8}
        disabled={disabled}
        onChange={e => {
          const v = Number(e.target.value)
          if (Number.isFinite(v)) onChange(Math.round(clamp(v, min, max)))
        }}
      />
    </label>
  )
}

export default RegionRefine
