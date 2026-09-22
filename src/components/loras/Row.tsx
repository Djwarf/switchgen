/**
 * One LoRA in the stack.
 *
 * A row has to answer four questions without being asked: what this LoRA does,
 * whether it belongs on the checkpoint that is loaded, how hard it is being
 * applied, and where it sits in the chain. The last one is not decoration.
 * Every LoraLoader patches the model the previous one produced, so a style
 * LoRA placed after an anatomy LoRA paints over it. Order is a control, so it
 * gets buttons rather than a drag handle that a keyboard cannot reach.
 *
 * Disabling keeps the row and its strength and leaves it out of the graph.
 * That is the difference between "not this picture" and "not this LoRA", and
 * conflating them is how a carefully tuned strength gets lost.
 *
 * THE CAP ON THE RAIL. An anatomy LoRA's rail stops at the measured cap rather
 * than at the band's 1.5, because the measurement is unambiguous about what
 * lies past it: anatomy-helper holds 0.81 of base sharpness at 0.3 and 0.44 at
 * 0.8, and no restorer in the set buys that back. The cap can be lifted on any
 * row in one press, because this is one person's machine and they may know
 * something the reading does not, starting with the fact that the reading was
 * taken on Pony V6 and the checkpoint loaded here may not be. A stack saved at
 * a higher strength is never silently rewritten: the row shows it as it is,
 * prints what was measured there, and offers to bring it down.
 */
import { useState } from 'react'
import {
  ARCH_LABEL,
  USAGE_LABEL,
  boundsFor,
  size,
  type Fit,
  type LoraInfo,
  type StackEntry,
} from '../../lib/loras'
import { Badge, Rail, RING, Tap, clamp } from './bits'
import { MEASURED_SINGLES, capFor, ratioText } from './measured'

export function Row({
  entry,
  info,
  fit,
  index,
  count,
  expert,
  clipPatched,
  onPatch,
  onMove,
  onRemove,
}: {
  entry: StackEntry
  info?: LoraInfo
  fit: Fit
  index: number
  count: number
  expert?: boolean
  /** True on checkpoint families, where the text encoder is patched as well. */
  clipPatched?: boolean
  onPatch: (patch: Partial<StackEntry>) => void
  onMove: (to: number) => void
  onRemove: () => void
}) {
  const bounds = boundsFor(info)
  const label = info?.label ?? entry.file
  const off = !entry.enabled
  const blocked = fit.level === 'mismatch' || info?.installed === false
  const id = `lora-${index}`

  const cap = capFor(info)
  const [lifted, setLifted] = useState(false)
  const overCap = cap !== null && entry.strength > cap + 1e-9
  const capped = cap !== null && !lifted && !overCap
  const max = capped ? cap : bounds.max
  const points = info ? MEASURED_SINGLES[info.file] : undefined

  return (
    <li className={`border-b border-grey-300 py-2 ${off ? 'opacity-55' : ''}`}>
      <div className="flex items-start gap-2">
        <input
          id={id}
          type="checkbox"
          checked={entry.enabled}
          onChange={(e) => onPatch({ enabled: e.target.checked })}
          className={`mt-1 h-3 w-3 shrink-0 accent-burgundy-900 ${RING}`}
          aria-label={`Use ${label}`}
        />

        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
            <label htmlFor={id} className="cursor-pointer text-small leading-tight text-ink">
              {label}
            </label>
            <Badge tone={fit.level} title={fit.why}>
              {fit.level === 'match' ? 'fits' : fit.level === 'untested' ? 'untested' : 'wrong model'}
            </Badge>
            {info && !info.installed ? <Badge tone="mismatch">not downloaded</Badge> : null}
            {info?.slider ? <Badge tone="plain">slider</Badge> : null}
          </div>

          <p className="mt-0.5 text-[0.6875rem] leading-snug text-grey-500">
            {info ? ARCH_LABEL[info.arch] : 'model not known'}
            <span className="px-1 text-grey-300">|</span>
            {info ? USAGE_LABEL[info.usage] : 'either pass'}
            {info?.installed ? (
              <>
                <span className="px-1 text-grey-300">|</span>
                <span className="tabular-nums">{size(info.bytes)}</span>
              </>
            ) : null}
          </p>
        </div>

        <div className="flex shrink-0 items-center gap-1">
          <Tap label="Move earlier" onClick={() => onMove(index - 1)} disabled={index === 0}>
            &uarr;
          </Tap>
          <Tap
            label="Move later"
            onClick={() => onMove(index + 1)}
            disabled={index === count - 1}
          >
            &darr;
          </Tap>
          <Tap label={`Remove ${label}`} onClick={onRemove}>
            &times;
          </Tap>
        </div>
      </div>

      <div className="mt-1.5 flex items-center gap-3">
        <div className="flex-1">
          <Rail
            value={entry.strength}
            min={bounds.min}
            max={max}
            step={bounds.step}
            // With the cap on, the rail's own end IS the recommendation, and a
            // second mark on top of it says nothing. With the cap lifted the
            // author's figure earns its place again: it is the number the card
            // asked for, next to the number that was measured.
            recommended={capped ? undefined : info?.recommended}
            disabled={off}
            label={`How strongly ${label} applies`}
            onChange={(n) => onPatch({ strength: n })}
          />
        </div>
        <input
          type="number"
          className="field w-16 shrink-0 px-1 py-0.5 text-center tabular-nums"
          value={entry.strength}
          min={bounds.min}
          max={max}
          step={bounds.step}
          disabled={off}
          aria-label={`${label} strength, number`}
          onChange={(e) => {
            const v = Number(e.target.value)
            if (Number.isFinite(v)) onPatch({ strength: clamp(v, bounds.min, max) })
          }}
        />
      </div>

      {cap !== null ? (
        <div className="mt-1 flex flex-wrap items-baseline justify-end gap-2">
          {overCap ? (
            <>
              <span className="text-caption leading-snug text-warning">
                Above the measured cap of {cap.toFixed(2)}. Saved stacks are left alone, so this is
                yours to keep or to lower.
              </span>
              <Tap label={`Lower ${label} to ${cap.toFixed(2)}`} onClick={() => onPatch({ strength: cap })}>
                to {cap.toFixed(2)}
              </Tap>
            </>
          ) : (
            <Tap
              active={lifted}
              label={
                lifted
                  ? `Hold ${label} at the measured cap of ${cap.toFixed(2)}`
                  : `Allow ${label} past the measured cap of ${cap.toFixed(2)}`
              }
              onClick={() => setLifted((v) => !v)}
            >
              {lifted ? `cap at ${cap.toFixed(2)}` : `past ${cap.toFixed(2)}`}
            </Tap>
          )}
        </div>
      ) : null}

      {points && points.length ? (
        <p className="mt-1 text-caption leading-snug text-grey-700">
          <span className="uppercase tracking-[0.14em] text-[0.5625rem] text-grey-500">
            measured here
          </span>{' '}
          <span className="tabular-nums">
            {points
              .map((p) => `${ratioText(p.ratio)} of base at ${p.strength.toFixed(1)}`)
              .join(', ')}
          </span>
          . Sharpness, on Pony Diffusion V6 XL at one held seed. It says nothing about whether the
          anatomy comes out right.
        </p>
      ) : null}

      {expert && clipPatched ? (
        <label className="mt-1 flex items-center justify-end gap-2 text-caption text-grey-700">
          <span className="uppercase tracking-[0.14em]">effect on wording</span>
          <input
            type="number"
            className="field w-16 px-1 py-0.5 text-center tabular-nums"
            value={entry.clipStrength ?? entry.strength}
            min={bounds.min}
            max={max}
            step={bounds.step}
            disabled={off}
            onChange={(e) => {
              const v = Number(e.target.value)
              if (Number.isFinite(v)) onPatch({ clipStrength: clamp(v, bounds.min, max) })
            }}
          />
        </label>
      ) : null}

      {info?.trigger ? (
        <p className="mt-1 text-caption text-grey-700">
          <span className="uppercase tracking-[0.14em] text-[0.5625rem] text-grey-500">needs this in the prompt</span>{' '}
          <span className="italic">{info.trigger}</span>
        </p>
      ) : null}

      {info?.slider ? (
        <p className="mt-1 text-caption italic text-grey-500">
          A slider. The sign sets the direction: above zero enlarges, below zero shrinks. No
          trigger word.
        </p>
      ) : null}

      {expert && info ? <p className="mt-1 text-caption italic text-grey-700">{info.does}</p> : null}

      {blocked || fit.level === 'untested' ? (
        <p
          className={`mt-1 text-caption leading-snug ${
            fit.level === 'mismatch' ? 'text-error' : 'text-warning'
          }`}
        >
          {info?.installed === false
            ? 'Not downloaded yet. It is left out of the graph until the file is on disk.'
            : fit.why}
        </p>
      ) : null}

      {info?.caution ? <p className="mt-1 text-caption italic text-warning">{info.caution}</p> : null}
    </li>
  )
}
