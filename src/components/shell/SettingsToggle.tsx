/**
 * Simple ⇄ All controls.
 *
 * One switch, global to both desks, persisted. The state lives in
 * `lib/session.ts`; this file owns only the control and the hook that reads it.
 *
 * Expert mode never replaces simple mode: it fills the left margin that simple
 * mode leaves as page gutter, and every revealed field already carries the
 * value simple mode was quietly using. Nothing on the page moves when it is
 * turned on, which is why this is a toggle and not a different screen.
 */
import { useSyncExternalStore } from 'react'
import { settings, type Settings } from '../../lib/session'

/** The global display settings. */
export function useSettings(): Settings {
  return useSyncExternalStore(settings.subscribe, settings.get, settings.get)
}

/** Just the expert flag, for the many components that need only that. */
export function useExpert(): boolean {
  return useSettings().expert
}

export function toggleExpert(): void {
  settings.toggleExpert()
}

export type SettingsToggleProps = {
  className?: string
}

export function SettingsToggle({ className = '' }: SettingsToggleProps) {
  const expert = useExpert()
  return (
    <button
      type="button"
      className={`sg-quiet ring ${className}`}
      aria-pressed={expert}
      title={expert ? 'Back to the five controls that matter (E)' : 'Show every control (E)'}
      onClick={toggleExpert}
    >
      {expert ? (
        <>
          <span aria-hidden>◂ </span>
          <span className="sm:hidden">All</span>
          <span className="hidden sm:inline">All controls</span>
        </>
      ) : (
        <>
          <span className="sm:hidden">Simple</span>
          <span className="hidden sm:inline">Setting: Simple</span>
          <span aria-hidden> ▸</span>
        </>
      )}
    </button>
  )
}

/**
 * The link form, for the italic line under the run button:
 * "Using the maker's settings: 30 steps · CFG 5.0. *Show all controls →*"
 */
export function ShowAllControlsLink({ className = '' }: { className?: string }) {
  const expert = useExpert()
  if (expert) return null
  return (
    <button type="button" className={`sg-link ring ${className}`} onClick={toggleExpert}>
      Show all controls →
    </button>
  )
}
