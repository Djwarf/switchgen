/**
 * The strip.
 *
 * A storyboard, set as a numbered table: one row a shot, a hairline rule
 * between them, four columns that always mean the same thing.
 *
 *   gutter   the shot number, its state, how many generations deep it sits
 *   plate    the frame it ends on, or the frame it opens on, or nothing yet
 *   body     the line the reader wrote, and where this shot starts from
 *   margin   length, and the handful of things you can do to one shot alone
 *
 * Enter in the prompt field starts the next shot. That is the whole writing
 * gesture: type a line, press Enter, type the next. Shift with Enter puts a
 * line break inside one shot's prompt, for the rare shot that needs a
 * paragraph.
 */
import { useRef, useState } from 'react'

import { fileUrl } from '../../lib/comfy'
import { clipFrames, type ShotJob } from '../../lib/continuation'
import { Chips, Mark, Quiet, RING, duration, seconds } from './bits'
import type { Currency, RunState, ShotState } from './engine'
import type { ReelShot } from './store'

export type StripProps = {
  shots: readonly ReelShot[]
  jobs: readonly ShotJob[]
  run: RunState
  fps: number
  /** The reel's own frame count, which a shot follows unless it overrides it. */
  reelLength: number
  /** Frame lengths offered as chips, already snapped to what the node accepts. */
  lengthOptions: readonly number[]
  expert: boolean
  busy: boolean
  /** Per shot, in strip order: whether its clip still matches its line (engine currencyOf). */
  currency: readonly (Currency | null)[]
  /** Per shot, in strip order: true when the desk will not queue it as it stands. */
  refused: readonly boolean[]
  /** Null when this family can pin a closing frame, otherwise the reason it cannot. */
  bookendBlocked: string | null
  onEdit: (id: string, patch: Partial<ReelShot>) => void
  onMove: (id: string, delta: number) => void
  onRemove: (id: string) => void
  onDuplicate: (id: string) => void
  /** Returns the new shot's id, so the strip can put the caret in it. */
  onAdd: (after?: string) => string | null
  onRender: (index: number) => void
  onPin: (id: string, which: 'start' | 'end') => void
  onWatch: (id: string) => void
}

export function Strip(props: StripProps) {
  const { shots, jobs, run, onAdd } = props
  const fields = useRef(new Map<string, HTMLTextAreaElement>())

  const focus = (id: string | null) => {
    if (!id) return
    requestAnimationFrame(() => {
      const el = fields.current.get(id)
      el?.focus()
      el?.setSelectionRange(el.value.length, el.value.length)
    })
  }

  return (
    <div>
      <ol className="border-t border-grey-300">
        {shots.map((shot, i) => (
          <ShotRow
            key={shot.id}
            index={i}
            shot={shot}
            job={jobs[i] ?? null}
            state={run.states[shot.id] ?? null}
            currencyNow={props.currency[i] ?? null}
            refusedNow={props.refused[i] ?? false}
            locked={
              props.busy &&
              run.queue.includes(shot.id) &&
              !['done', 'error', 'stopped'].includes(run.states[shot.id]?.status ?? '')
            }
            live={run.currentShotId === shot.id}
            registerField={(el) => {
              if (el) fields.current.set(shot.id, el)
              else fields.current.delete(shot.id)
            }}
            onEnter={() => focus(onAdd(shot.id))}
            first={i === 0}
            last={i === shots.length - 1}
            {...props}
          />
        ))}
      </ol>

      <div className="flex flex-wrap items-center gap-3 py-3">
        <Quiet onClick={() => focus(onAdd())}>Add a shot</Quiet>
        <span className="text-caption italic text-grey-500">
          Or press Enter at the end of any line.
        </span>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// One row
// ---------------------------------------------------------------------------

type RowProps = StripProps & {
  index: number
  shot: ReelShot
  job: ShotJob | null
  state: ShotState | null
  currencyNow: Currency | null
  refusedNow: boolean
  /**
   * True while this shot is in the running pass and has not finished. The walk
   * renders the job it was handed when the button was pressed, so an edit here
   * would be ignored and the clip would land beside words it never saw.
   */
  locked: boolean
  live: boolean
  first: boolean
  last: boolean
  registerField: (el: HTMLTextAreaElement | null) => void
  onEnter: () => void
}

function ShotRow(p: RowProps) {
  const { shot, job, state, index, fps, reelLength, expert, busy, locked, currencyNow, refusedNow, bookendBlocked } = p
  const [confirming, setConfirming] = useState(false)
  const frames = job ? clipFrames(job) : (shot.length ?? reelLength)
  const status = state?.status ?? 'waiting'
  const lockedTitle = locked ? 'This shot is in the queue. It can be changed once it has rendered.' : undefined

  return (
    <li className="grid grid-cols-1 gap-x-5 gap-y-3 border-b border-grey-300 py-4 sm:grid-cols-[2.75rem_minmax(0,1fr)] lg:grid-cols-[2.75rem_11rem_minmax(0,1fr)_11rem]">
      {/* gutter ------------------------------------------------------------ */}
      <div className="flex items-baseline gap-2 lg:block">
        <span className="font-serif text-h3 leading-none tabular-nums text-burgundy-900">
          {String(index + 1).padStart(2, '0')}
        </span>
        <span className="mt-2 flex items-center gap-1.5 lg:mt-3">
          <Mark state={markFor(status, p.live)} />
          <span className="text-[0.625rem] uppercase tracking-[0.14em] text-grey-500">
            {stateWord(state, p.live, currencyNow)}
          </span>
        </span>
        {job && job.hops > 0 ? (
          <span
            className="mt-1 block text-[0.625rem] uppercase tracking-[0.14em] text-grey-400"
            title="Generations between this shot and the last frame a person actually chose."
          >
            hop {job.hops}
          </span>
        ) : null}
      </div>

      {/* plate -------------------------------------------------------------- */}
      <ShotPlate shot={shot} state={state} onWatch={() => p.onWatch(shot.id)} />

      {/* body --------------------------------------------------------------- */}
      <div className="min-w-0">
        <textarea
          ref={p.registerField}
          className="field min-h-[3.75rem] resize-y text-small disabled:cursor-not-allowed disabled:text-grey-500"
          rows={2}
          disabled={locked}
          title={lockedTitle}
          placeholder={index === 0 ? 'What happens in the opening shot' : 'What happens next'}
          value={shot.prompt}
          onChange={(e) => p.onEdit(shot.id, { prompt: e.target.value })}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey && !e.metaKey) {
              e.preventDefault()
              p.onEnter()
            }
          }}
        />

        <p className="mt-1.5 text-caption text-grey-700">
          <span className="font-semibold uppercase tracking-[0.12em] text-grey-500">Opens on</span>{' '}
          {openingLine(job, shot, index)}
        </p>

        {shot.start ? (
          <p className="mt-1 text-caption italic text-grey-500">
            Pinned: {shot.start.label}.{' '}
            <button
              type="button"
              className={`sg-link ${RING}`}
              disabled={locked}
              onClick={() => p.onEdit(shot.id, { start: null })}
            >
              Unpin it
            </button>
          </p>
        ) : null}
        {shot.end ? (
          <p className="mt-1 text-caption italic text-grey-500">
            Has to land on: {shot.end.label}.{' '}
            <button
              type="button"
              className={`sg-link ${RING}`}
              disabled={locked}
              onClick={() => p.onEdit(shot.id, { end: null })}
            >
              Unpin it
            </button>
          </p>
        ) : null}

        {job?.notes.map((note) => (
          <p key={note} className="mt-1 border-l-2 border-grey-300 pl-2 text-caption italic text-grey-700">
            {note}
          </p>
        ))}

        {job?.blocked ? (
          <p className="mt-1 border-l-2 border-error pl-2 text-caption text-ink-error">{job.blocked}</p>
        ) : null}

        {currencyNow === 'stale' ? (
          <p className="mt-1 border-l-2 border-warning pl-2 text-caption text-ink-warning">
            Rendered, then the shot before it changed. This clip still opens on the old frame.
          </p>
        ) : currencyNow === 'changed' ? (
          <p className="mt-1 border-l-2 border-warning pl-2 text-caption text-ink-warning">
            Changed since it was rendered. The clip on disk still shows it as it was.
          </p>
        ) : null}

        {state?.error ? (
          <p className="mt-1 border-l-2 border-error pl-2 text-caption text-ink-error">
            {state.error}
            {state.detail ? <span className="block text-grey-700">{state.detail}</span> : null}
          </p>
        ) : null}

        {expert ? (
          <div className="mt-2 grid gap-2 sm:grid-cols-2">
            <label className="block">
              <span className="mb-1 block text-[0.625rem] font-semibold uppercase tracking-[0.14em] text-grey-500">
                This shot's seed
              </span>
              <input
                type="number"
                className="field tabular-nums text-caption disabled:cursor-not-allowed disabled:text-grey-500"
                placeholder="follows the reel"
                value={shot.seed ?? ''}
                min={0}
                disabled={locked}
                onChange={(e) => {
                  const v = e.target.value
                  p.onEdit(shot.id, { seed: v === '' ? null : Math.max(0, Math.floor(Number(v))) })
                }}
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-[0.625rem] font-semibold uppercase tracking-[0.14em] text-grey-500">
                This shot's negative
              </span>
              <input
                type="text"
                className="field text-caption disabled:cursor-not-allowed disabled:text-grey-500"
                placeholder="follows the reel"
                value={shot.negative ?? ''}
                disabled={locked}
                onChange={(e) => p.onEdit(shot.id, { negative: e.target.value || null })}
              />
            </label>
          </div>
        ) : null}
      </div>

      {/* margin ------------------------------------------------------------- */}
      <div className="lg:text-right">
        <p className="mb-1.5 text-caption tabular-nums text-ink">
          {seconds(frames, fps)}
          <span className="text-grey-500"> · {frames} frames</span>
        </p>

        <div className="mb-2 lg:flex lg:justify-end">
          <Chips
            ariaLabel={`Length of shot ${index + 1}`}
            value={shot.length ?? 0}
            disabled={locked}
            onChange={(v) => p.onEdit(shot.id, { length: v === 0 ? null : v })}
            options={[
              { value: 0, label: 'Reel', title: `Follow the reel: ${reelLength} frames` },
              ...p.lengthOptions.map((n) => ({ value: n, label: seconds(n, fps), title: `${n} frames` })),
            ]}
          />
        </div>

        {state?.durationMs ? (
          <p className="mb-2 text-caption italic tabular-nums text-grey-500">Took {duration(state.durationMs)}</p>
        ) : null}

        <div className="flex flex-wrap gap-1 lg:justify-end">
          <Quiet
            onClick={() => p.onRender(index)}
            disabled={busy || !shot.prompt.trim() || refusedNow}
            title={refusedNow ? 'This shot cannot be rendered as it stands. The note under the strip says why.' : undefined}
          >
            {state?.status === 'done' ? 'Render again' : 'Render'}
          </Quiet>
          <Quiet
            onClick={() => p.onPin(shot.id, 'start')}
            disabled={locked}
            title={lockedTitle ?? 'Pin the frame this shot opens on'}
          >
            Pin start
          </Quiet>
          <Quiet
            onClick={() => p.onPin(shot.id, 'end')}
            disabled={bookendBlocked !== null || locked}
            title={bookendBlocked ?? lockedTitle ?? 'Pin the frame this shot has to reach'}
          >
            Pin end
          </Quiet>
          <Quiet onClick={() => p.onMove(shot.id, -1)} disabled={p.first || busy} title="Move this shot earlier">
            Up
          </Quiet>
          <Quiet onClick={() => p.onMove(shot.id, 1)} disabled={p.last || busy} title="Move this shot later">
            Down
          </Quiet>
          <Quiet onClick={() => p.onDuplicate(shot.id)} title="Copy this shot's line and settings">
            Copy
          </Quiet>
          <Quiet
            danger
            disabled={busy}
            onClick={() => {
              if (confirming) {
                p.onRemove(shot.id)
                setConfirming(false)
                return
              }
              setConfirming(true)
              setTimeout(() => setConfirming(false), 3000)
            }}
          >
            {confirming ? 'Really?' : 'Cut'}
          </Quiet>
        </div>
      </div>
    </li>
  )
}

// ---------------------------------------------------------------------------
// The plate
// ---------------------------------------------------------------------------

function ShotPlate({ shot, state, onWatch }: { shot: ReelShot; state: ShotState | null; onWatch: () => void }) {
  const live = state?.previewUrl ?? null
  const frame = state?.frame ? fileUrl(state.frame) : null
  const pinned = shot.start?.previewUrl ?? null
  const src = live ?? frame ?? pinned
  const watchable = Boolean(state?.clip)

  const body = src ? (
    <img
      src={src}
      alt=""
      loading="lazy"
      className={`h-full w-full object-cover ${live ? 'opacity-90' : ''}`}
    />
  ) : (
    <span className="flex h-full w-full items-center justify-center text-[0.625rem] uppercase tracking-[0.14em] text-grey-400">
      No frame yet
    </span>
  )

  return (
    <div className="max-w-[16rem] lg:max-w-none">
      {watchable ? (
        <button
          type="button"
          onClick={onWatch}
          className={`${RING} block w-full border border-grey-300 bg-grey-200 transition-colors hover:border-ink`}
          title="Watch this shot"
        >
          <span className="block aspect-[16/9] overflow-hidden">{body}</span>
        </button>
      ) : (
        <div className="aspect-[16/9] overflow-hidden border border-grey-300 bg-grey-200">{body}</div>
      )}
      <p className="mt-1 text-[0.625rem] uppercase tracking-[0.14em] text-grey-400">
        {live ? 'Drawing' : frame ? 'Last frame' : pinned ? 'Pinned opening' : 'Empty'}
      </p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Words
// ---------------------------------------------------------------------------

function markFor(status: ShotState['status'], live: boolean): 'idle' | 'live' | 'ok' | 'off' {
  if (live || status === 'running' || status === 'queued') return 'live'
  if (status === 'done') return 'ok'
  if (status === 'error') return 'off'
  return 'idle'
}

function stateWord(state: ShotState | null, live: boolean, currency: Currency | null): string {
  if (!state) return 'unwritten'
  if (live && state.status === 'queued') return 'queued'
  switch (state.status) {
    case 'running':
      return 'drawing'
    case 'queued':
      return 'queued'
    case 'done':
      return currency === 'stale' ? 'out of date' : currency === 'changed' ? 'changed' : 'rendered'
    case 'error':
      return 'failed'
    case 'stopped':
      return 'stopped'
    default:
      return 'waiting'
  }
}

function openingLine(job: ShotJob | null, shot: ReelShot, index: number): string {
  if (!job) {
    if (shot.start) return `a pinned frame: ${shot.start.label}.`
    return index === 0 ? 'words alone.' : 'the shot before it.'
  }
  switch (job.start.from) {
    case 'previous':
      return `shot ${index}'s last frame. The take carries on.`
    case 'anchor':
      return 'the anchor frame, which resets the look.'
    case 'given':
      return `a pinned frame: ${shot.start?.label ?? 'a picture you chose'}.`
    default:
      return 'words alone. Nothing carries over.'
  }
}
