/**
 * The hash router.
 *
 * No router library is installed and none may be added. Navigation is
 * `location.hash` and nothing else, which buys three things for free: the
 * browser's back and forward buttons, reload, and a link you can send someone.
 *
 *   #/pictures                       the image desk
 *   #/video                          the video desk
 *   #/reel                           the reel desk, where shots chain
 *   #/archive?q=rain%20model:krea    the archive, already searched
 *
 * The store is a module singleton read through `useSyncExternalStore`, so a
 * hundred components can ask where they are without a hundred listeners
 * disagreeing about it.
 */
import { useSyncExternalStore } from 'react'

export type Section = 'pictures' | 'video' | 'reel' | 'archive'

export type Route =
  | { name: 'pictures' }
  | { name: 'video' }
  | { name: 'reel' }
  | { name: 'archive'; q: string }

export const SECTIONS: readonly Section[] = ['pictures', 'video', 'reel', 'archive']

export const SECTION_LABEL: Record<Section, string> = {
  pictures: 'Pictures',
  video: 'Video',
  reel: 'Reel',
  archive: 'Archive',
}

/** The standfirst under each section name in the bar. */
export const SECTION_STANDFIRST: Record<Section, string> = {
  pictures: 'Seconds, not minutes',
  video: 'Five seconds at a time',
  reel: 'Each shot opens where the last one closed',
  archive: 'Everything you have made',
}

const HOME: Route = { name: 'pictures' }

/** Parse a location hash. Anything unrecognised lands on the pictures desk. */
export function parseRoute(hash: string): Route {
  const raw = hash.replace(/^#/, '') || '/pictures'
  const [path, qs] = raw.split('?')
  const q = new URLSearchParams(qs ?? '').get('q') ?? ''
  if (path === '/video') return { name: 'video' }
  if (path === '/reel') return { name: 'reel' }
  if (path === '/archive') return { name: 'archive', q }
  return HOME
}

/** The href for a route, `#` included, ready for an anchor. */
export function routeHref(route: Route): string {
  if (route.name === 'video') return '#/video'
  if (route.name === 'reel') return '#/reel'
  if (route.name === 'archive') return route.q ? `#/archive?q=${encodeURIComponent(route.q)}` : '#/archive'
  return '#/pictures'
}

export function sectionHref(section: Section): string {
  return section === 'archive' ? '#/archive' : `#/${section}`
}

// ---------------------------------------------------------------------------
// The store
// ---------------------------------------------------------------------------

const listeners = new Set<() => void>()
let lastHash = typeof window === 'undefined' ? '' : window.location.hash
let current: Route = parseRoute(lastHash)

function emit(): void {
  for (const fn of [...listeners]) {
    try {
      fn()
    } catch {
      /* one bad subscriber must not strand the rest */
    }
  }
}

/** Refresh from the address bar. Pure enough to call during render. */
function getSnapshot(): Route {
  const hash = window.location.hash
  if (hash !== lastHash) {
    lastHash = hash
    current = parseRoute(hash)
  }
  return current
}

function onLocationChange(): void {
  const before = current
  const next = getSnapshot()
  if (next !== before) emit()
}

function subscribe(fn: () => void): () => void {
  if (listeners.size === 0) {
    window.addEventListener('hashchange', onLocationChange)
    window.addEventListener('popstate', onLocationChange)
  }
  listeners.add(fn)
  return () => {
    listeners.delete(fn)
    if (listeners.size === 0) {
      window.removeEventListener('hashchange', onLocationChange)
      window.removeEventListener('popstate', onLocationChange)
    }
  }
}

/**
 * Go somewhere.
 *
 * `replace` rewrites the current entry instead of adding one — used for
 * search-as-you-type, so the back button does not have to walk through every
 * keystroke the reader made.
 */
export function go(route: Route, opts: { replace?: boolean } = {}): void {
  const href = routeHref(route)
  if (href === (window.location.hash || '#/pictures') && window.location.hash) return
  if (opts.replace) {
    window.history.replaceState(window.history.state, '', href)
    onLocationChange()
  } else {
    // Assigning the hash pushes a history entry and fires `hashchange` for us.
    window.location.hash = href.slice(1)
  }
}

/** Jump to a section, keeping the archive's current query if we are in it. */
export function goToSection(section: Section): void {
  if (section === 'archive') {
    const here = getSnapshot()
    go({ name: 'archive', q: here.name === 'archive' ? here.q : '' })
    return
  }
  go({ name: section })
}

/** Write the archive query into the URL. Replaces, so back still works. */
export function setArchiveQuery(q: string, opts: { replace?: boolean } = { replace: true }): void {
  go({ name: 'archive', q }, opts)
}

/** Current route without a subscription, for code outside React. */
export function currentRoute(): Route {
  return getSnapshot()
}

/** Put a bare `/` or an empty hash onto the pictures desk without a history entry. */
export function normaliseHash(): void {
  const hash = window.location.hash
  if (hash === '' || hash === '#' || hash === '#/') {
    window.history.replaceState(window.history.state, '', '#/pictures')
    onLocationChange()
  }
}

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------

/** Where we are. Re-renders on back, forward, reload and every navigation. */
export function useRoute(): Route {
  return useSyncExternalStore(subscribe, getSnapshot, () => HOME)
}

/** Just the section name, for the bar and for keyboard scoping. */
export function useSection(): Section {
  return useRoute().name
}

// ---------------------------------------------------------------------------
// The search line
// ---------------------------------------------------------------------------

/**
 * `/` focuses the archive search from anywhere. The archive owns the input, so
 * the shell navigates and then asks for focus by event rather than by reaching
 * into someone else's ref.
 */
export const FOCUS_SEARCH_EVENT = 'switchgen:focus-search'

export const SEARCH_INPUT_ID = 'archive-search'

/**
 * `/` from anywhere: go to the archive, then put the caret in the search line.
 *
 * Two mechanisms, deliberately. The event is the contract — the archive can
 * listen for it and do something cleverer, such as selecting the existing
 * query. The direct focus by id is the fallback that works today, and it stops
 * trying the moment something has taken focus.
 */
export function requestSearchFocus(): void {
  goToSection('archive')
  let tries = 0
  const ask = () => {
    window.dispatchEvent(new CustomEvent(FOCUS_SEARCH_EVENT))
    const field = document.getElementById(SEARCH_INPUT_ID) as HTMLInputElement | null
    if (field && document.activeElement !== field) {
      field.focus()
      field.select?.()
    }
    // The archive may still be mounting on the first pass.
    if ((!field || document.activeElement !== field) && ++tries < 4) {
      setTimeout(ask, 60 * tries)
    }
  }
  requestAnimationFrame(ask)
}

// ---------------------------------------------------------------------------
// Desks and sections
// ---------------------------------------------------------------------------

/**
 * A desk id (`lib/session.ts`) to the section that holds it.
 *
 * The two vocabularies are deliberately different: `images` is the store's
 * name for a desk, `Pictures` is the room's name on the door. The reel has no
 * store of its own but is a room, so a job from it goes back to it.
 */
export function sectionForDesk(desk: 'images' | 'video' | 'reel'): Section {
  if (desk === 'reel') return 'reel'
  return desk === 'video' ? 'video' : 'pictures'
}

/** The desk a section drives, or null for the archive, which drives neither. */
export function deskForSection(section: Section): 'images' | 'video' | null {
  if (section === 'pictures') return 'images'
  if (section === 'video' || section === 'reel') return 'video'
  return null
}
