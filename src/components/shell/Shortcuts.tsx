/**
 * The shortcuts card.
 *
 * Set as a newspaper index: two columns, leader dots, grouped by where the key
 * works. `?` opens it from anywhere, `Esc` closes it. The whole map lives here
 * as data so there is exactly one place to change when a key changes.
 *
 * EVERY ROW HERE IS A CLAIM, and a card documenting a key nothing handles is
 * worse than no card: it teaches a key, the key does nothing, and the reader
 * concludes the application is broken rather than the page. So each row was
 * read back against the handler that owns it. Where the two disagreed the row
 * was corrected or deleted, never the other way round.
 *
 * IT ALSO OWNS THE KEYBOARD WHILE IT IS UP. Five modules install their own
 * bare `window` keydown listeners: the three desks, the archive and the player.
 * None of them can see this card, and one of the archive's single keys is
 * Backspace, bound to removing a record. On a television, where Backspace is
 * the natural back key, that meant reading a shortcut and losing a record
 * behind the scrim. The capture phase listener below stops every key except
 * the ones needed to read and close the card, so nothing behind it fires.
 */
import { useEffect, useRef, useSyncExternalStore } from 'react'

// ---------------------------------------------------------------------------
// The map
// ---------------------------------------------------------------------------

export type Shortcut = { keys: string; action: string; note?: string }
export type ShortcutGroup = { title: string; items: Shortcut[]; note?: string }

export const SHORTCUTS: readonly ShortcutGroup[] = [
  {
    title: 'Getting about',
    items: [
      { keys: '1 2 3 4', action: 'Pictures, Video, Reel, Archive' },
      { keys: 'g then p v r a', action: 'The same four, for the Vim-minded' },
      { keys: 'e', action: 'Simple settings, or all controls' },
      { keys: '/', action: 'Search the archive' },
      { keys: '?', action: 'This card' },
      { keys: 'Esc', action: 'Close what is open, then clear the search, then let go of the field' },
    ],
  },
  {
    title: 'At a desk',
    items: [
      {
        keys: 'Ctrl Enter',
        action: 'Make it',
        note: 'The one key that still works while you are typing',
      },
      { keys: 'u', action: 'Choose a source picture' },
      { keys: 'Ctrl V', action: 'Paste a picture from the clipboard' },
    ],
  },
  {
    title: 'In the archive',
    note: 'Deleting a file has no key. It is a menu and a confirmation, on purpose.',
    items: [
      { keys: 'j k', action: 'Move between records' },
      { keys: 'Enter', action: 'Open the record' },
      { keys: 'r', action: 'Use these settings' },
      { keys: 'Shift R', action: 'Make another, with a fresh seed' },
      { keys: 'u', action: 'Use as a source' },
      { keys: 's', action: 'Star it' },
      { keys: 'x', action: 'Select it' },
      { keys: 'v', action: 'Grid or list' },
      {
        keys: 'Backspace',
        action: 'Remove the record',
        note: 'The file stays. Eight seconds to undo.',
      },
      {
        keys: 'Ctrl Z',
        action: 'Put the last removal back',
        note: 'The archive only, and only while the offer is still up.',
      },
    ],
  },
  {
    title: 'In the player',
    items: [
      { keys: 'Space k', action: 'Play or pause' },
      { keys: ', .', action: 'Step one frame back or forward' },
      {
        keys: '← →',
        action: 'One frame',
        note: 'On the scrub rail these move the rail itself.',
      },
      { keys: 'Shift ← →', action: 'One second' },
      { keys: 'Home End', action: 'First frame, last frame' },
      { keys: '[ ]', action: 'Playback speed', note: 'A quarter, a half, 1×, 2×.' },
      { keys: 'i o', action: 'Set the loop in and out' },
      { keys: 'Shift I', action: 'Clear the loop' },
      { keys: 'l r', action: 'Loop on or off' },
      { keys: 'f', action: 'Fullscreen' },
      { keys: 's', action: 'Save the clip' },
      { keys: 'g', action: 'Save this frame as a PNG' },
      { keys: 'u', action: 'Use this frame' },
    ],
  },
]

// ---------------------------------------------------------------------------
// Open state, so `?` can reach it from anywhere
// ---------------------------------------------------------------------------

let open = false
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

const getOpen = () => open

function set(next: boolean): void {
  if (open === next) return
  open = next
  emit()
}

export function openShortcuts(): void {
  set(true)
}
export function closeShortcuts(): void {
  set(false)
}
export function toggleShortcuts(): void {
  set(!open)
}
/** True while the card is up — the shell uses it to scope `Esc`. */
export function shortcutsOpen(): boolean {
  return open
}
export function useShortcutsOpen(): boolean {
  return useSyncExternalStore(subscribe, getOpen, getOpen)
}

// ---------------------------------------------------------------------------
// The card
// ---------------------------------------------------------------------------

/**
 * Keys that still reach the page while the card is up.
 *
 * `Escape` and `?` close it, and both are handled by the shell's own window
 * listener, so they have to travel. The rest scroll the sheet, which is taller
 * than a television screen at this type size, and scrolling is a default
 * action rather than a listener, so they are let through for the reader rather
 * than for any handler. `ArrowLeft` and `ArrowRight` are deliberately not in
 * this set: they scroll nothing vertical and the player seeks on them.
 */
const TRAVELS = new Set([
  'Escape',
  '?',
  'ArrowUp',
  'ArrowDown',
  'PageUp',
  'PageDown',
  'Home',
  'End',
])

/** What a focus trap counts as somewhere focus can land. */
const FOCUSABLE =
  'button:not(:disabled), [href], input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex]:not([tabindex="-1"])'

export function Shortcuts() {
  const isOpen = useShortcutsOpen()
  const sheet = useRef<HTMLDivElement>(null)
  const restore = useRef<HTMLElement | null>(null)

  useEffect(() => {
    if (!isOpen) return
    const previous = document.activeElement as HTMLElement | null
    restore.current = previous
    sheet.current?.focus()

    const onKey = (e: KeyboardEvent) => {
      const panel = sheet.current
      if (!panel) return

      if (e.key === 'Tab') {
        // The trap. Without it one Tab leaves the card and lands on the
        // section bar behind the scrim, where the next Enter changes desk
        // under a dialog that is still on screen.
        const focusable = [...panel.querySelectorAll<HTMLElement>(FOCUSABLE)]
        e.preventDefault()
        e.stopPropagation()
        if (!focusable.length) {
          panel.focus()
          return
        }
        const here = focusable.indexOf(document.activeElement as HTMLElement)
        const step = e.shiftKey ? -1 : 1
        const next = here < 0 ? (e.shiftKey ? focusable.length - 1 : 0) : here + step
        focusable[(next + focusable.length) % focusable.length].focus()
        return
      }

      if (TRAVELS.has(e.key)) return

      // A control inside the card needs its own activation keys. Stopping the
      // event here would stop it reaching the button, and the Close button
      // would then be focusable and inert, which is worse than no trap.
      const target = e.target as HTMLElement | null
      const onControl = !!target && target !== panel && panel.contains(target)
      if (onControl && (e.key === 'Enter' || e.key === ' ')) return

      // Everything else stops here, in the capture phase, before it reaches
      // the five window listeners that cannot see this card.
      e.stopPropagation()
    }

    document.addEventListener('keydown', onKey, true)
    return () => {
      document.removeEventListener('keydown', onKey, true)
      // A restore onto an element that has since been unmounted silently drops
      // focus onto the body, which strands a keyboard session at the top of
      // the document. The page element is the fallback: it is never unmounted
      // and it carries tabIndex -1 for exactly this.
      const back = restore.current
      if (back && back.isConnected) back.focus()
      else document.querySelector<HTMLElement>('#page')?.focus?.()
    }
  }, [isOpen])

  if (!isOpen) return null

  return (
    <div
      className="sg-scrim"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) closeShortcuts()
      }}
    >
      <div
        ref={sheet}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-label="Keyboard shortcuts"
        className="sg-sheet ring my-auto"
      >
        <div className="px-8 py-6">
          <div className="flex items-baseline justify-between section-rule">
            <h2 className="m-0 text-h3 font-semibold text-burgundy-900">Keyboard shortcuts</h2>
            <button type="button" className="sg-link ring" onClick={closeShortcuts}>
              Close
            </button>
          </div>

          <p className="dropcap mt-0 mb-6 max-w-[62ch] text-body">
            Everything here can be done with the mouse as well. These are for the things you do
            twenty times an hour. Every single-key shortcut is off while you are typing, so a prompt
            about a rain-slicked tram stop cannot change desks behind your back.
          </p>

          <div className="columns-1 gap-10 md:columns-2">
            {SHORTCUTS.map((group) => (
              <section key={group.title} className="mb-7 break-inside-avoid">
                <h3 className="kicker m-0 mb-2 border-b border-grey-300 pb-1">{group.title}</h3>
                <dl className="m-0">
                  {group.items.map((item) => (
                    // The key is the term and the action is its description;
                    // the leader dots run between them, as in a printed index.
                    <div key={item.keys} className="sg-leader py-1">
                      <dt className="order-2 m-0 shrink-0">
                        <span className="sg-key">{item.keys}</span>
                      </dt>
                      <dd className="order-first m-0 max-w-[26ch] text-small text-ink">
                        {item.action}
                        {item.note && (
                          <span className="block text-caption text-grey-700 italic">
                            {item.note}
                          </span>
                        )}
                      </dd>
                      <span className="sg-leader-dots order-1" aria-hidden />
                    </div>
                  ))}
                </dl>
                {group.note && (
                  <p className="mt-2 mb-0 max-w-[36ch] text-small text-grey-700 italic">{group.note}</p>
                )}
              </section>
            ))}
          </div>

          <p className="mt-2 mb-0 border-t border-grey-300 pt-3 text-small text-grey-700 italic">
            Press <span className="sg-key not-italic">?</span> at any time to bring this back.
          </p>
        </div>
      </div>
    </div>
  )
}
