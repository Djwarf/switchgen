/**
 * Before and after.
 *
 * The whole point of a refine pass is a region a few hundred pixels across, so
 * a pair of thumbnails side by side proves nothing: at that size the change is
 * two pixels wide and everyone nods. Three views, each answering a different
 * question.
 *
 *   REGION   both pictures cropped to the rectangle that was re rendered and
 *            blown up to the column width. This is where the anatomy is
 *            actually judged, so it is the default whenever a crop is known.
 *   WIPE     one picture with the other revealed under a draggable rule. This
 *            is what catches a seam at the composite edge or a shift in skin
 *            tone, which the region view crops away.
 *   BOTH     the two full frames, for the composition.
 *
 * The crop maths is done in percentages against the container rather than in
 * measured pixels, so the enlargement is correct at any column width with no
 * resize observer and no layout pass to wait for.
 */
import { useState } from 'react'
import type { KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent } from 'react'
import type { Rect } from '../../lib/refine'
import { Chips, Kicker, clamp, times } from './bits'

type View = 'region' | 'wipe' | 'both'

export type ComparePicture = { url: string; width: number; height: number }

export function Compare({
  before,
  after,
  crop,
  busy = false,
}: {
  before: ComparePicture
  after: { url: string }
  /** The rectangle that was re rendered, when it is known. */
  crop: Rect | null
  busy?: boolean
}) {
  const [view, setView] = useState<View>(crop ? 'region' : 'wipe')
  const [split, setSplit] = useState(50)
  const mode: View = view === 'region' && !crop ? 'wipe' : view

  return (
    <div>
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
        <Kicker tone="burgundy">Before and after</Kicker>
        <Chips<View>
          ariaLabel="How to compare"
          value={mode}
          onChange={setView}
          options={[
            { value: 'region', label: 'The region', disabled: !crop, title: crop ? `${times(crop.width, crop.height)} of the frame` : 'Draw a region first' },
            { value: 'wipe', label: 'Wipe' },
            { value: 'both', label: 'Both frames' },
          ]}
        />
      </div>

      {mode === 'region' && crop ? (
        <div className="grid gap-3 sm:grid-cols-2">
          <Cropped title="Before" picture={before} url={before.url} crop={crop} />
          <Cropped title="After" picture={before} url={after.url} crop={crop} />
        </div>
      ) : mode === 'wipe' ? (
        <Wipe before={before} after={after} split={split} onSplit={setSplit} />
      ) : (
        <div className="grid gap-3 sm:grid-cols-2">
          <Whole title="Before" url={before.url} />
          <Whole title="After" url={after.url} />
        </div>
      )}

      <p className="mt-2 text-caption italic text-grey-700">
        {busy
          ? 'The pass is running. What you see is the picture you started from.'
          : mode === 'region'
            ? 'Both panels show the same rectangle at the same scale. Judge the anatomy here.'
            : mode === 'wipe'
              ? 'Drag the rule across, or use the arrow keys. A seam at the edge of the region shows up here first.'
              : 'The whole frame, so you can see the region still belongs to the body it sits on.'}
      </p>
    </div>
  )
}

/** One picture cropped to `crop` and enlarged to the column width. */
function Cropped({
  title,
  picture,
  url,
  crop,
}: {
  title: string
  picture: ComparePicture
  url: string
  crop: Rect
}) {
  return (
    <figure className="m-0">
      <figcaption className="mb-1">
        <Kicker>{title}</Kicker>
      </figcaption>
      <div
        className="relative overflow-hidden border border-grey-300 bg-newsprint-aged"
        style={{ aspectRatio: `${crop.width} / ${crop.height}` }}
      >
        <img
          src={url}
          alt={`${title}, the marked region enlarged`}
          draggable={false}
          className="absolute max-w-none select-none"
          style={{
            width: `${(picture.width / crop.width) * 100}%`,
            left: `${(-crop.x / crop.width) * 100}%`,
            top: `${(-crop.y / crop.height) * 100}%`,
          }}
        />
      </div>
    </figure>
  )
}

function Whole({ title, url }: { title: string; url: string }) {
  return (
    <figure className="m-0">
      <figcaption className="mb-1">
        <Kicker>{title}</Kicker>
      </figcaption>
      <div className="flex items-center justify-center border border-grey-300 bg-newsprint-aged">
        <img src={url} alt={title} draggable={false} className="max-h-[40vh] max-w-full select-none" />
      </div>
    </figure>
  )
}

function Wipe({
  before,
  after,
  split,
  onSplit,
}: {
  before: ComparePicture
  after: { url: string }
  split: number
  onSplit: (n: number) => void
}) {
  const fromPointer = (e: ReactPointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    if (r.width < 1) return
    onSplit(clamp(((e.clientX - r.left) / r.width) * 100, 0, 100))
  }

  const onKey = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    const step = e.shiftKey ? 10 : 2
    if (e.key === 'ArrowLeft') {
      e.preventDefault()
      onSplit(clamp(split - step, 0, 100))
    } else if (e.key === 'ArrowRight') {
      e.preventDefault()
      onSplit(clamp(split + step, 0, 100))
    } else if (e.key === 'Home') {
      e.preventDefault()
      onSplit(0)
    } else if (e.key === 'End') {
      e.preventDefault()
      onSplit(100)
    }
  }

  return (
    <div
      role="slider"
      tabIndex={0}
      aria-label="Reveal the refined picture"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(split)}
      aria-valuetext={`${Math.round(split)} percent refined`}
      onKeyDown={onKey}
      onPointerDown={e => {
        e.currentTarget.setPointerCapture(e.pointerId)
        fromPointer(e)
      }}
      onPointerMove={e => {
        if (e.buttons) fromPointer(e)
      }}
      className="relative mx-auto w-full touch-none select-none border border-grey-300 bg-newsprint-aged focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900"
      style={{ aspectRatio: `${before.width} / ${before.height}`, maxHeight: '58vh' }}
    >
      <img
        src={before.url}
        alt="Before"
        draggable={false}
        className="absolute inset-0 h-full w-full object-contain"
      />
      <img
        src={after.url}
        alt="After"
        draggable={false}
        className="absolute inset-0 h-full w-full object-contain"
        style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }}
      />
      <span
        aria-hidden="true"
        className="absolute inset-y-0 w-px bg-newsprint"
        style={{ left: `${split}%` }}
      />
      <span
        aria-hidden="true"
        className="absolute inset-y-0 w-px bg-burgundy-900"
        style={{ left: `calc(${split}% + 1px)` }}
      />
      <span
        aria-hidden="true"
        className="absolute top-1/2 h-6 w-[3px] -translate-x-1/2 -translate-y-1/2 bg-burgundy-900"
        style={{ left: `${split}%` }}
      />
      <span className="absolute bottom-1 left-1 bg-newsprint/85 px-1">
        <Kicker>Before</Kicker>
      </span>
      <span className="absolute right-1 bottom-1 bg-newsprint/85 px-1">
        <Kicker>After</Kicker>
      </span>
    </div>
  )
}
