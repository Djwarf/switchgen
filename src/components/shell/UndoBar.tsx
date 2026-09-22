/**
 * The undo bar.
 *
 * Removing a record should feel cheap, because nine times in ten "delete" here
 * means "stop showing me this". The price of cheap is a window to change your
 * mind, and a window you can watch closing: a single burgundy hairline drains
 * left to right across eight seconds.
 *
 * Anything in the app can offer an undo:
 *
 *   const removed = history.remove(id)
 *   if (removed) offerUndo({
 *     body: <>One record removed. The file is still on disk at <code>{rel}</code>.</>,
 *     undo: () => history.restore(removed),
 *   })
 */
import { useSyncExternalStore, type ReactNode } from 'react'

export type UndoOffer = {
  id: string
  title: string
  body: ReactNode
  undo: () => void
  /** How long the window stays open. */
  ms: number
  at: number
}

export type UndoInput = {
  body: ReactNode
  undo: () => void
  title?: string
  ms?: number
  /** Run when the window closes without an undo. Rarely needed; the record is already gone. */
  onExpire?: () => void
}

export const UNDO_MS = 8000

let offer: UndoOffer | null = null
let expire: (() => void) | null = null
let timer = 0
const listeners = new Set<() => void>()

function emit(): void {
  for (const fn of [...listeners]) {
    try {
      fn()
    } catch {
      /* ignore */
    }
  }
}

function subscribe(fn: () => void): () => void {
  listeners.add(fn)
  return () => {
    listeners.delete(fn)
  }
}

function close(run: 'expire' | 'silent'): void {
  clearTimeout(timer)
  timer = 0
  const onExpire = expire
  expire = null
  offer = null
  emit()
  if (run === 'expire') onExpire?.()
}

/**
 * Offer an undo. A second offer commits the first — you get one window at a
 * time, which is what a person can actually hold in their head.
 */
export function offerUndo(input: UndoInput): string {
  if (offer) close('expire')
  const id = `u-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`
  const ms = input.ms ?? UNDO_MS
  offer = {
    id,
    title: input.title ?? 'Correction',
    body: input.body,
    undo: input.undo,
    ms,
    at: Date.now(),
  }
  expire = input.onExpire ?? null
  emit()
  timer = setTimeout(() => close('expire'), ms) as unknown as number
  return id
}

/** Take the offer. Returns false when there was nothing to take — for `Ctrl+Z`. */
export function undoLast(): boolean {
  if (!offer) return false
  const fn = offer.undo
  close('silent')
  try {
    fn()
  } catch {
    /* an undo that throws is still an undo that was asked for */
  }
  return true
}

/** Let the window close now, without undoing. */
export function commitUndo(): void {
  if (offer) close('expire')
}

const getOffer = () => offer

export function useUndo(): UndoOffer | null {
  return useSyncExternalStore(subscribe, getOffer, getOffer)
}

/**
 * The bar itself. Mounted once by the shell; it renders nothing until there is
 * something to take back.
 */
export function UndoBar() {
  const current = useUndo()
  if (!current) return null

  return (
    <div className="fixed bottom-[calc(1rem+var(--sg-safe-b))] left-1/2 z-[55] w-[min(34rem,calc(100vw-2rem))] -translate-x-1/2">
      <div className="bg-newsprint border border-grey-300">
        <div className="notice notice-correction text-small">
          <div className="flex items-baseline gap-3">
            <div className="m-0 flex-1">
              <strong className="block not-italic">{current.title}</strong>
              <p className="mt-1 mb-0">{current.body}</p>
            </div>
            <button
              type="button"
              className="sg-link ring shrink-0 not-italic"
              onClick={() => undoLast()}
            >
              Undo
            </button>
          </div>
        </div>
        {/* Keyed on the offer so each new window restarts the drain. */}
        <div className="sg-drain" aria-hidden>
          <div
            key={current.id}
            className="sg-drain-fill"
            style={{ animationDuration: `${current.ms}ms` }}
          />
        </div>
      </div>
    </div>
  )
}
