/**
 * Notices.
 *
 * A newspaper does not pop up a toast; it prints a correction. Every notice in
 * this application is a ruled block with a left rule, a short uppercase label
 * and a sentence that says what happened and what to do next.
 *
 * The component can be used inline anywhere. The store behind it exists so any
 * corner of the app — a desk, the archive, a background audit — can post a line
 * that appears in the shell's notice column without plumbing a callback through
 * four components.
 */
import { useSyncExternalStore, type ReactNode } from 'react'

export type NoticeTone = 'info' | 'correction' | 'error' | 'warning' | 'success'

export type NoticeAction = {
  label: string
  run: () => void
  /** Close the notice after running. Default true. */
  closes?: boolean
}

export type NoticeProps = {
  tone?: NoticeTone
  /** Short, uppercase by the stylesheet. "CORRECTION", "THAT JOB WAS REJECTED". */
  title?: string
  children?: ReactNode
  actions?: readonly NoticeAction[]
  onDismiss?: () => void
  className?: string
}

const TONE_CLASS: Record<NoticeTone, string> = {
  info: 'notice-info',
  correction: 'notice-correction',
  error: 'notice-error',
  warning: 'notice-warning',
  success: 'notice-success',
}

/** One notice, ruled and quiet. */
export function Notice({
  tone = 'info',
  title,
  children,
  actions,
  onDismiss,
  className = '',
}: NoticeProps) {
  return (
    <div
      className={`notice ${TONE_CLASS[tone]} text-small ${className}`}
      role={tone === 'error' ? 'alert' : 'status'}
    >
      <div className="flex items-start gap-3">
        <div className="m-0 flex-1">
          {title && <strong className="block not-italic">{title}</strong>}
          {children && <p className={title ? 'mt-1 mb-0' : 'm-0'}>{children}</p>}
        </div>
        {onDismiss && (
          <button
            type="button"
            onClick={onDismiss}
            className="sg-link ring shrink-0 text-caption not-italic"
            aria-label="Dismiss this notice"
          >
            Close
          </button>
        )}
      </div>
      {actions && actions.length > 0 && (
        <p className="mt-2 mb-0 flex flex-wrap items-baseline gap-x-4 gap-y-1 not-italic">
          {actions.map((a) => (
            <button
              key={a.label}
              type="button"
              className="sg-link ring text-small"
              onClick={() => {
                a.run()
                if (a.closes !== false) onDismiss?.()
              }}
            >
              {a.label}
            </button>
          ))}
        </p>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// The store
// ---------------------------------------------------------------------------

export type PostedNotice = {
  id: string
  tone: NoticeTone
  title?: string
  body: ReactNode
  actions?: readonly NoticeAction[]
  /** Milliseconds before it takes itself away. 0 means it waits to be dismissed. */
  ttl: number
  at: number
}

export type NoticeInput = {
  tone?: NoticeTone
  title?: string
  body: ReactNode
  actions?: readonly NoticeAction[]
  ttl?: number
  /**
   * A stable key. Posting again with the same key replaces the standing notice
   * rather than stacking a second copy — the missing-file audit and the quota
   * warning both need this.
   */
  key?: string
}

let posted: readonly PostedNotice[] = []
const listeners = new Set<() => void>()
const timers = new Map<string, number>()

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

/** Default lives: corrections and errors wait to be read; the rest fade. */
const DEFAULT_TTL: Record<NoticeTone, number> = {
  info: 9000,
  success: 6000,
  correction: 0,
  warning: 0,
  error: 0,
}

/** Post a notice into the shell's column. Returns its id. */
export function postNotice(input: NoticeInput): string {
  const tone = input.tone ?? 'info'
  const id = input.key ?? `n-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`
  const ttl = input.ttl ?? DEFAULT_TTL[tone]
  const next: PostedNotice = {
    id,
    tone,
    title: input.title,
    body: input.body,
    actions: input.actions,
    ttl,
    at: Date.now(),
  }
  const existing = timers.get(id)
  if (existing) {
    clearTimeout(existing)
    timers.delete(id)
  }
  posted = [...posted.filter((n) => n.id !== id), next]
  emit()
  if (ttl > 0) {
    timers.set(id, setTimeout(() => dismissNotice(id), ttl) as unknown as number)
  }
  return id
}

export function dismissNotice(id: string): void {
  const t = timers.get(id)
  if (t) {
    clearTimeout(t)
    timers.delete(id)
  }
  const before = posted.length
  posted = posted.filter((n) => n.id !== id)
  if (posted.length !== before) emit()
}

export function clearNotices(): void {
  for (const t of timers.values()) clearTimeout(t)
  timers.clear()
  if (posted.length) {
    posted = []
    emit()
  }
}

const getPosted = () => posted

export function useNotices(): readonly PostedNotice[] {
  return useSyncExternalStore(subscribe, getPosted, getPosted)
}

/**
 * The standing column of notices, bottom left, above the undo bar.
 *
 * Bottom left rather than top right: the top of this application is the
 * masthead and the section bar, and nothing may cover those.
 */
export function NoticeRail() {
  const all = useNotices()
  if (all.length === 0) return null
  return (
    <div
      className="pointer-events-none fixed bottom-[calc(1rem+var(--sg-safe-b))] left-[calc(1rem+var(--sg-safe-l))] z-50 flex w-[min(28rem,calc(100vw-2rem))] flex-col gap-2 max-xl:bottom-[calc(6.5rem+var(--sg-safe-b))]"
      aria-live="polite"
    >
      {all.map((n) => (
        <div key={n.id} className="pointer-events-auto bg-newsprint sg-unfold">
          <Notice
            tone={n.tone}
            title={n.title}
            actions={n.actions}
            onDismiss={() => dismissNotice(n.id)}
          >
            {n.body}
          </Notice>
        </div>
      ))}
    </div>
  )
}
