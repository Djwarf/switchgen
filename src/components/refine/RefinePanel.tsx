/**
 * The controls for one region refine, and the honest arithmetic beside them.
 *
 * Every number shown here comes from planRefine and detailGain in
 * src/lib/refine.ts rather than from a promise about quality. A region that
 * occupies four percent of the frame has about a thousand latent cells to hold
 * a nipple, a set of labia or a glans, and no checkpoint can reconstruct those
 * from a thousand cells. Cropped and re rendered at 1024 the same region gets
 * roughly sixteen thousand. That multiplier is the product, so it is printed
 * next to the button rather than buried in a tooltip, and when it falls to
 * nothing the panel says so instead of selling a pass that will not help.
 *
 * The controls are named for what they do. "How much to change" is the
 * denoise, and the word denoise appears only in the expert row where the raw
 * number is also shown. Padding is called context, because that is its job: the
 * sampler needs to see the skin, the shadow and the body line around a region
 * to draw it as part of the same body.
 */
import type { ReactNode } from 'react'
import {
  MASK_ONLY_REGIONS,
  REFINE_DEFAULTS,
  REFINE_DENOISE,
  type RefinePlan,
} from '../../lib/refine'
import { Caution, Chips, Field, Head, Kicker, Leader, Link, Note, Quiet, Rail, clamp, grouped, multiple, times } from './bits'

export type RefineSettings = {
  /** 0.35 to 0.55 is the useful band. The expert row can leave it. */
  denoise: number
  /** Context in source pixels added around the mask before cropping. */
  padding: number
  /** Long edge the crop is re rendered at. */
  targetLongEdge: number
  /** Pixels the mask is widened by before rendering. */
  grow: number
  /** Blur on the mask edge, which is what hides the composite seam. */
  feather: number
  /** The prompt for this region. Names the anatomy, not the scene. */
  prompt: string
  /** A fresh seed each pass, so a second try is a different try. */
  newSeed: boolean
}

/**
 * Add-ons offered for one region: plain rows, so this folder knows nothing
 * about the add-on library. `picked` holds the ids that are ticked.
 */
export type RegionAddOns = {
  options: readonly { id: string; label: string; strength: number; why: string }[]
  picked: readonly string[]
  onToggle: (id: string) => void
}

export const REFINE_SETTINGS: RefineSettings = {
  denoise: REFINE_DENOISE.default,
  padding: REFINE_DEFAULTS.padding,
  targetLongEdge: REFINE_DEFAULTS.targetLongEdge,
  grow: REFINE_DEFAULTS.grow,
  feather: REFINE_DEFAULTS.feather,
  prompt: '',
  newSeed: true,
}

type Stop = { key: string; label: string; denoise: number; help: string }

const STOPS: Stop[] = [
  {
    key: 'soften',
    label: 'Sharpen',
    denoise: REFINE_DENOISE.min,
    help: 'Keeps the shapes that are there and resolves them. Use when the region is soft rather than wrong.',
  },
  {
    key: 'rebuild',
    label: 'Rebuild',
    denoise: REFINE_DENOISE.default,
    help: 'Draws the region again from your prompt, keeping its place, its lighting and its skin tone. The usual choice.',
  },
  {
    key: 'redraw',
    label: 'Redraw',
    denoise: REFINE_DENOISE.max,
    help: 'Starts the region close to scratch. Use when the anatomy is wrong, not merely blurred.',
  },
]

/**
 * Clinical descriptions for the regions people actually come here to fix.
 * A region prompt names the anatomy: the scene, the lighting and the style are
 * already in the picture and in the parent prompt carried below.
 */
const PRESETS: { key: string; label: string; text: string }[] = [
  {
    key: 'breasts',
    label: 'Breasts',
    text: 'breasts, natural shape and weight, areola and nipple correctly placed and proportioned, skin tone continuous with the chest',
  },
  {
    key: 'vulva',
    label: 'Vulva',
    text: 'vulva, labia majora and labia minora anatomically correct and symmetrical, natural skin tone, correct proportion to the hips',
  },
  {
    key: 'penis',
    label: 'Penis',
    text: 'penis, anatomically correct shaft and glans, correct proportion, natural skin tone',
  },
  {
    key: 'hands',
    label: 'Hands',
    text: 'hand, five fingers, correct knuckles and fingernails, natural pose',
  },
  {
    key: 'feet',
    label: 'Feet',
    text: 'foot, five toes, correct toenails and arch, natural pose',
  },
  {
    key: 'face',
    label: 'Face',
    text: 'face, symmetrical eyes with clean irises and lashes, natural skin texture',
  },
]

const PADDINGS = [32, 64, 96, 128]
const LONG_EDGES = [768, 1024, 1280]

/** `a, b, c and d`. */
function list(items: readonly string[]): string {
  if (items.length < 2) return items[0] ?? ''
  return `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`
}

export function RefinePanel({
  settings,
  onPatch,
  plan,
  parentPrompt,
  marked,
  expert,
  busy,
  progress,
  blocked,
  addOns,
  onRun,
  onStop,
}: {
  settings: RefineSettings
  onPatch: (patch: Partial<RefineSettings>) => void
  /** Null until a region has been marked. */
  plan: RefinePlan | null
  parentPrompt: string
  marked: boolean
  expert: boolean
  busy: boolean
  /** 0 to 1 while a pass runs, null otherwise. */
  progress: number | null
  /** Set when the pass cannot run at all, with the reason to print. */
  blocked: string | null
  /** Region add-ons to offer, unticked. Omit it, or pass none, and nothing is printed. */
  addOns?: RegionAddOns
  onRun: () => void
  onStop?: () => void
}) {
  const stop = STOPS.reduce((best, s) =>
    Math.abs(s.denoise - settings.denoise) < Math.abs(best.denoise - settings.denoise) ? s : best,
  )
  const onStopExactly = Math.abs(stop.denoise - settings.denoise) < 0.005
  const gain = plan?.gain ?? null
  const thin = gain !== null && gain.gain < 1.2

  const addPreset = (text: string) => {
    const current = settings.prompt.trim()
    if (current.toLowerCase().includes(text.slice(0, 18).toLowerCase())) return
    onPatch({ prompt: current ? `${text}, ${current}` : text })
  }

  return (
    <div>
      <Head
        title="Refine a region"
        figure={plan ? times(plan.target.width, plan.target.height) : null}
        note="Mark the region on the picture. It is cut out, enlarged to a full working resolution, drawn again, and composited back."
      />

      {/* ---- the region prompt ------------------------------------------- */}
      <Field
        label="What is in this region"
        id="sg-refine-prompt"
        hint={settings.prompt.trim() ? `${settings.prompt.trim().length} characters` : 'empty'}
      >
        <div className="mb-1.5 flex flex-wrap gap-1">
          {PRESETS.map(p => (
            <Quiet key={p.key} onClick={() => addPreset(p.text)} title={p.text}>
              {p.label}
            </Quiet>
          ))}
        </div>
        <textarea
          id="sg-refine-prompt"
          className="field min-h-[5.5rem]"
          value={settings.prompt}
          placeholder="breasts, natural shape and weight, areola and nipple correctly placed"
          onChange={e => onPatch({ prompt: e.target.value })}
        />
        <Note>
          Name the anatomy, not the scene. The picture already carries the scene, and a region
          prompt describing a whole composition pulls the crop away from the body it belongs to.
        </Note>
        {parentPrompt.trim() && settings.prompt.trim() !== parentPrompt.trim() ? (
          <p className="mt-1">
            <Link onClick={() => onPatch({ prompt: parentPrompt })}>
              Put the picture's own prompt back
            </Link>
          </p>
        ) : null}
      </Field>

      {/* ---- add-ons for this region ------------------------------------- */}
      {addOns && addOns.options.length ? <AddOnPicks addOns={addOns} busy={busy} /> : null}

      {/* ---- strength ----------------------------------------------------- */}
      <Field
        label="How much to change"
        hint={expert ? `denoise ${settings.denoise.toFixed(2)}` : undefined}
      >
        <Chips
          ariaLabel="How much to change"
          grow
          value={onStopExactly ? stop.key : ''}
          options={STOPS.map(s => ({ value: s.key, label: s.label, title: s.help }))}
          onChange={key => {
            const picked = STOPS.find(s => s.key === key)
            if (picked) onPatch({ denoise: picked.denoise })
          }}
        />
        <Note>{stop.help}</Note>
        {expert ? (
          <div className="mt-2 flex items-center gap-2">
            <label htmlFor="sg-refine-denoise" className="text-caption text-grey-700">
              denoise
            </label>
            <input
              id="sg-refine-denoise"
              type="number"
              className="field w-24 tabular-nums"
              value={settings.denoise}
              min={0.05}
              max={0.95}
              step={0.01}
              onChange={e => {
                const v = Number(e.target.value)
                if (Number.isFinite(v)) onPatch({ denoise: Math.round(clamp(v, 0.05, 0.95) * 100) / 100 })
              }}
            />
          </div>
        ) : null}
        {settings.denoise < REFINE_DENOISE.min - 0.001 ? (
          <div className="mt-2">
            <Caution>
              Under {REFINE_DENOISE.min.toFixed(2)} the sampler moves the region too little to
              correct anatomy. It will look sharper and be shaped the same.
            </Caution>
          </div>
        ) : settings.denoise > REFINE_DENOISE.max + 0.001 ? (
          <div className="mt-2">
            <Caution>
              Over {REFINE_DENOISE.max.toFixed(2)} the region stops agreeing with the body around
              it: expect a different skin tone or a different pose inside the mask.
            </Caution>
          </div>
        ) : null}
      </Field>

      {/* ---- context ------------------------------------------------------ */}
      <Field label="Context around the region" hint={`${settings.padding} px`}>
        <Chips
          ariaLabel="Context around the region"
          grow
          value={settings.padding}
          options={PADDINGS.map(p => ({ value: p, label: `${p} px` }))}
          onChange={p => onPatch({ padding: p })}
        />
        <Note>
          The crop is widened by this much on every side. The sampler needs to see the skin, the
          shadow and the body line around a region to draw it as part of the same body. Too little
          context gives a correct region that does not belong to the picture.
        </Note>
      </Field>

      {/* ---- expert ------------------------------------------------------- */}
      {expert ? (
        <div className="mb-4 border-t border-grey-300 pt-3">
          <Kicker className="mb-2 block">The raw numbers</Kicker>

          <Field label="Render the crop at" hint={`${settings.targetLongEdge} px on the long edge`}>
            <Chips
              ariaLabel="Render the crop at"
              grow
              value={settings.targetLongEdge}
              options={LONG_EDGES.map(e => ({ value: e, label: `${e}` }))}
              onChange={e => onPatch({ targetLongEdge: e })}
            />
            <Note>
              Higher puts more latent cells on the region and costs more time and more VRAM. 1024 is
              the working resolution these checkpoints were trained around.
            </Note>
          </Field>

          <div className="mb-3 flex gap-3">
            <label className="block w-28">
              <span className="mb-1 block text-overline font-semibold uppercase tracking-[0.18em] text-grey-700">
                Grow
              </span>
              <input
                type="number"
                className="field tabular-nums"
                value={settings.grow}
                min={0}
                max={128}
                step={1}
                onChange={e => {
                  const v = Number(e.target.value)
                  if (Number.isFinite(v)) onPatch({ grow: Math.round(clamp(v, 0, 128)) })
                }}
              />
            </label>
            <label className="block w-28">
              <span className="mb-1 block text-overline font-semibold uppercase tracking-[0.18em] text-grey-700">
                Feather
              </span>
              <input
                type="number"
                className="field tabular-nums"
                value={settings.feather}
                min={0}
                max={100}
                step={1}
                onChange={e => {
                  const v = Number(e.target.value)
                  if (Number.isFinite(v)) onPatch({ feather: Math.round(clamp(v, 0, 100)) })
                }}
              />
            </label>
          </div>
          <Note>
            Grow widens the mask so the sampler owns the boundary instead of meeting it. Feather
            blurs that boundary, and it is the setting that decides whether the composite shows a
            seam.
          </Note>

          <label className="mt-3 flex items-center gap-2 text-caption text-grey-700">
            <input
              type="checkbox"
              checked={settings.newSeed}
              onChange={e => onPatch({ newSeed: e.target.checked })}
            />
            A new seed on every pass
          </label>
          <Note>
            Off, a second pass over the same mask with the same settings returns the same region.
            On, each pass is a fresh attempt at the same instruction.
          </Note>

          {plan ? (
            <div className="mt-3 border-t border-grey-300 pt-2">
              <Leader label="Crop, in the source" value={`${times(plan.crop.width, plan.crop.height)} at ${Math.round(plan.crop.x)}, ${Math.round(plan.crop.y)}`} />
              <Leader label="Re rendered at" value={times(plan.target.width, plan.target.height)} />
              <Leader label="Share of the frame" value={`${plan.gain.sharePct.toFixed(1)}%`} />
              <Leader label="Latent cells now" value={grouped(plan.gain.before)} />
              <Leader label="Latent cells after" value={grouped(plan.gain.after)} />
            </div>
          ) : null}
        </div>
      ) : null}

      {/* ---- the honest arithmetic ---------------------------------------- */}
      <div className="mb-3 border-t-2 border-burgundy-900 pt-2">
        {gain ? (
          <Measure>
            This region is <b className="font-semibold">{gain.sharePct.toFixed(1)}%</b> of the
            frame, which is {grouped(gain.before)} latent cells today. Cropped and rendered at{' '}
            {plan ? times(plan.target.width, plan.target.height) : ''} it gets{' '}
            {grouped(gain.after)}: about{' '}
            <b className="font-semibold">{multiple(gain.gain)}</b> the detail.
          </Measure>
        ) : (
          <Measure>
            Paint over the region on the picture. Nothing is marked yet, so there is nothing to
            refine.
          </Measure>
        )}
        {thin ? (
          <div className="mt-2">
            <Caution>
              This region is already about the size it would be re rendered at. The pass will change
              the detail it has: it will not add any. Mark a tighter region, or raise the render
              size in the expert row.
            </Caution>
          </div>
        ) : null}
      </div>

      <p className="mb-3 text-caption leading-snug text-grey-700">
        <Kicker tone="ink">Cost</Kicker>{' '}
        <span className="italic">
          a refine pass renders a whole frame at {plan ? times(plan.target.width, plan.target.height) : `${settings.targetLongEdge} px`}, so
          it takes about as long as making a fresh picture and holds the card for the whole of it.
        </span>
      </p>

      {blocked ? (
        <div className="mb-3">
          <Caution>{blocked}</Caution>
        </div>
      ) : null}

      <button type="button" className="press ring" disabled={busy || !marked || !!blocked} onClick={onRun}>
        {busy ? 'Refining the region' : 'Refine this region'}
      </button>

      {busy ? (
        <div className="mt-2">
          <Rail value={progress ?? 0} />
          <div className="mt-1 flex items-baseline justify-between">
            <span className="text-caption tabular-nums text-grey-700">
              {progress === null ? 'queued' : `${Math.round(progress * 100)}%`}
            </span>
            {onStop ? <Link onClick={onStop}>Stop</Link> : null}
          </div>
        </div>
      ) : null}

      <p className="mt-3 text-caption italic leading-snug text-grey-500">
        Faces and hands can also be found automatically, because a detector exists for each of them.
        {' '}
        {list(MASK_ONLY_REGIONS.slice(0, -1))} have no detector at all, which is why they are marked
        by hand here.
      </p>
    </div>
  )
}

/**
 * The region add-ons, folded away and unticked.
 *
 * These are made for close framing, which is the crop this pass renders, so
 * this is where they belong. But they are not free: every one ticked is
 * another file loaded on top of the model, and a slider at full strength
 * reshapes whatever it is aimed at. So none is on until it is ticked, and each
 * row says what it does, the word it adds to the prompt, and whose strength it
 * is.
 */
function AddOnPicks({ addOns, busy }: { addOns: RegionAddOns; busy: boolean }) {
  const on = addOns.options.filter(o => addOns.picked.includes(o.id)).length
  return (
    <Field label="Add-ons for this region" hint={on ? `${on} on` : 'none on'}>
      <details>
        <summary className="cursor-pointer text-caption text-grey-700">
          {addOns.options.length} fit the model drawing this. None is used until you tick it.
        </summary>
        <ul className="mt-1 border-t border-grey-300">
          {addOns.options.map(o => {
            const id = `sg-refine-addon-${o.id}`
            return (
              <li key={o.id} className="border-b border-grey-300 py-1.5">
                <label htmlFor={id} className="flex cursor-pointer items-baseline gap-2">
                  <input
                    id={id}
                    type="checkbox"
                    checked={addOns.picked.includes(o.id)}
                    disabled={busy}
                    onChange={() => addOns.onToggle(o.id)}
                  />
                  <span className="min-w-0 flex-1 text-caption text-ink">{o.label}</span>
                  <span className="shrink-0 text-caption tabular-nums text-grey-700">
                    {o.strength.toFixed(2)}
                  </span>
                </label>
                <span className="mt-0.5 block pl-5 text-caption italic leading-snug text-grey-500">
                  {o.why}
                </span>
              </li>
            )
          })}
        </ul>
      </details>
    </Field>
  )
}

function Measure({ children }: { children: ReactNode }) {
  return <p className="text-small leading-snug text-ink">{children}</p>
}
