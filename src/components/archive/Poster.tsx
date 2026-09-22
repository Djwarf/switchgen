/**
 * Poster frames for the archive.
 *
 * A card shows a thumbnail the server made (see src/lib/thumbs.ts), not the
 * original: a render is often a megabyte or more, and a card draws it a few
 * hundred pixels wide. A clip's thumbnail is its first frame.
 *
 * A grid of clips must never be a grid of `<video>` elements: fifty of them
 * decoding at once will stall a machine that is already giving its card to a
 * generation. So when the server cannot make a clip's frame (no ffmpeg there,
 * or a file it cannot read), the frame is grabbed here instead, once, into a
 * small JPEG, by a detached video element that is never added to the
 * document. A picture the server cannot shrink is shown as it is.
 *
 * The grabbed frames live for the tab. Captures are limited to two at a time
 * and are only started when a card is actually near the viewport, so
 * scrolling past a thousand records costs nothing.
 */
import { useEffect, useRef, useState } from 'react'
import { fileUrl, relPath, type FileRef } from '../../lib/comfy'
import type { HistoryEntry } from '../../lib/history'
import { hasThumb, thumbSrcSet, thumbUrl } from '../../lib/thumbs'
import { clipSlug } from './query'

type State = 'idle' | 'working' | 'ready' | 'failed'

const cache = new Map<string, string>()
const failures = new Set<string>()
const waiting: (() => void)[] = []
const MAX_CACHED = 240
const MAX_PARALLEL = 2
let running = 0

/**
 * The poster for a clip: the frame grabbed in this tab if there is one, which
 * means the server could not make it, and otherwise the server's.
 */
export function posterUrl(file: FileRef): string | undefined {
  return cache.get(relPath(file)) ?? (hasThumb(file) ? thumbUrl(file, 1024) : undefined)
}

function remember(key: string, url: string): void {
  cache.set(key, url)
  while (cache.size > MAX_CACHED) {
    const oldest = cache.keys().next()
    if (oldest.done) break
    const stale = cache.get(oldest.value)
    cache.delete(oldest.value)
    if (stale) URL.revokeObjectURL(stale)
  }
}

function pump(): void {
  while (running < MAX_PARALLEL && waiting.length) {
    const next = waiting.shift()
    if (next) next()
  }
}

/**
 * Grab frame one of a clip. Detached element, hard timeout, always cleaned up
 * — a file the browser cannot decode must not leave a video decoding forever.
 */
function grab(file: FileRef): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    const video = document.createElement('video')
    let done = false
    const finish = (url: string | null, err?: unknown) => {
      if (done) return
      done = true
      clearTimeout(timer)
      video.removeAttribute('src')
      video.load()
      running--
      pump()
      if (url) resolve(url)
      else reject(err ?? new Error('no frame'))
    }
    const timer = setTimeout(() => finish(null, new Error('timed out')), 12_000)

    video.muted = true
    video.preload = 'auto'
    video.playsInline = true
    video.crossOrigin = 'anonymous'

    const draw = () => {
      try {
        const w = video.videoWidth
        const h = video.videoHeight
        if (!w || !h) return finish(null, new Error('no dimensions'))
        const scale = Math.min(1, 320 / Math.max(w, h))
        const canvas = document.createElement('canvas')
        canvas.width = Math.max(1, Math.round(w * scale))
        canvas.height = Math.max(1, Math.round(h * scale))
        const ctx = canvas.getContext('2d')
        if (!ctx) return finish(null, new Error('no 2d context'))
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
        canvas.toBlob(
          (blob) => finish(blob ? URL.createObjectURL(blob) : null, new Error('no blob')),
          'image/jpeg',
          0.72,
        )
      } catch (err) {
        finish(null, err)
      }
    }

    video.addEventListener('seeked', draw, { once: true })
    video.addEventListener('error', () => finish(null, new Error('decode failed')), { once: true })
    video.addEventListener(
      'loadeddata',
      () => {
        const at = Number.isFinite(video.duration) ? Math.min(0.04, video.duration / 2) : 0.04
        try {
          video.currentTime = at
        } catch {
          draw()
        }
      },
      { once: true },
    )

    running++
    video.src = fileUrl(file)
    video.load()
  })
}

function request(file: FileRef): Promise<string> {
  const key = relPath(file)
  const held = cache.get(key)
  if (held) return Promise.resolve(held)
  return new Promise<string>((resolve, reject) => {
    // `grab` takes the slot synchronously and gives it back in `finish`, so
    // `pump` can count honestly without a second counter here.
    const start = () => {
      const again = cache.get(key)
      if (again) return resolve(again)
      grab(file).then(
        (url) => {
          remember(key, url)
          resolve(url)
        },
        (err) => {
          failures.add(key)
          reject(err)
        },
      )
    }
    waiting.push(start)
    pump()
  })
}

/**
 * How wide a grid card is drawn, for the browser to pick a thumbnail width by:
 * the grid's columns at each of its breakpoints (Grid.tsx).
 */
const GRID_SIZES = '(min-width: 1400px) 25vw, (min-width: 1100px) 33vw, (min-width: 780px) 50vw, 100vw'

type Props = {
  entry: HistoryEntry
  /** Extra classes for the frame. The frame is always a 1 px grey rule. */
  className?: string
  /** `cover` in a grid, `contain` in a dialog. */
  fit?: 'cover' | 'contain'
  /** Grab the poster immediately rather than waiting for the card to be seen. */
  eager?: boolean
  /** How wide the frame is drawn, as an <img sizes>. A grid card's by default. */
  sizes?: string
}

/**
 * One archive thumbnail: a picture, or a clip's first frame with its length in
 * the corner. Never a `<video>`.
 */
export function Poster({ entry, className = '', fit = 'cover', eager = false, sizes = GRID_SIZES }: Props) {
  const key = relPath(entry.file)
  const isVideo = entry.kind === 'video'

  // Each keyed, so what happened to the record that used to be here is never
  // held against a different one: the server's thumbnail failing, a grabbed
  // poster, the last fallback failing too.
  const [noThumb, setNoThumb] = useState<string | null>(null)
  const [got, setGot] = useState<{ key: string; url: string | null } | null>(null)
  const [broken, setBroken] = useState<string | null>(null)
  const frame = useRef<HTMLDivElement | null>(null)

  /** A frame grabbed earlier in this tab, because the server could not make one then. */
  const grabbedBefore = isVideo ? cache.get(key) : undefined
  /** The server's thumbnail, until it has failed for this record. */
  const fromServer = !grabbedBefore && hasThumb(entry.file) && noThumb !== key
  /** Without it, a picture is shown as it is; a clip's frame has to be grabbed. */
  const grabbing = isVideo && !fromServer
  const held = fromServer ? thumbUrl(entry.file, 512) : isVideo ? grabbedBefore : fileUrl(entry.file)

  const grabbed = got && got.key === key ? got.url : undefined
  const url = held ?? grabbed ?? undefined
  const failed =
    broken === key ||
    (isVideo && entry.missing === true) ||
    (grabbing && (failures.has(key) || grabbed === null))
  const state: State = failed ? 'failed' : url ? 'ready' : 'working'

  useEffect(() => {
    if (!grabbing || cache.has(key) || failures.has(key) || entry.missing) return

    let live = true
    const begin = () => {
      if (!live) return
      request(entry.file).then(
        (found) => {
          if (live) setGot({ key, url: found })
        },
        () => {
          if (live) setGot({ key, url: null })
        },
      )
    }

    if (eager || typeof IntersectionObserver === 'undefined') {
      begin()
      return () => {
        live = false
      }
    }

    const el = frame.current
    if (!el) return
    const io = new IntersectionObserver(
      (rows) => {
        if (rows.some((r) => r.isIntersecting)) {
          io.disconnect()
          begin()
        }
      },
      { rootMargin: '300px' },
    )
    io.observe(el)
    return () => {
      live = false
      io.disconnect()
    }
  }, [key, grabbing, entry.file, entry.missing, eager])

  const slug = isVideo ? clipSlug(entry.length, entry.fps) : null

  return (
    <div
      ref={frame}
      className={`relative flex items-center justify-center overflow-hidden bg-newsprint-aged ${className}`}
    >
      {url && !failed ? (
        <img
          // A new element for a new source, so a fallback never inherits the
          // failed thumbnail's srcSet or its pending error.
          key={url}
          src={url}
          srcSet={fromServer ? thumbSrcSet(entry.file) : undefined}
          sizes={fromServer ? sizes : undefined}
          alt={entry.prompt ? `Frame from: ${entry.prompt}` : 'Archive thumbnail'}
          loading={eager ? 'eager' : 'lazy'}
          decoding="async"
          onError={() => (fromServer ? setNoThumb(key) : setBroken(key))}
          className={`h-full w-full ${fit === 'cover' ? 'object-cover' : 'object-contain'}`}
        />
      ) : (
        <span className="px-3 text-center text-[0.625rem] font-semibold uppercase tracking-[0.18em] text-grey-500">
          {state === 'failed'
            ? entry.missing
              ? 'Not on disk'
              : isVideo
                ? 'Clip'
                : 'No preview'
            : isVideo
              ? 'Reading the clip'
              : 'Loading'}
        </span>
      )}

      {slug && (
        <span className="absolute right-0 bottom-0 bg-newsprint/90 px-1.5 py-0.5 text-[0.625rem] font-semibold tracking-[0.12em] text-grey-700 tabular-nums">
          {slug}
        </span>
      )}
    </div>
  )
}
