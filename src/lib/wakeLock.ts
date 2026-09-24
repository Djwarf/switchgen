/**
 * Keeping the screen on while work waits in the page.
 *
 * Some work lives only in the page until the page sends it: clips waiting in
 * the Video desk's lane, the rest of a Pictures batch, the shots a reel has
 * not reached. A phone that locks its screen suspends the page, and nothing
 * more is sent until it wakes. Work already sent to ComfyUI carries on either
 * way; it is only the next send that waits.
 *
 * The Screen Wake Lock API asks the browser to keep the screen on. It exists
 * only in a secure context (https, or localhost), so over plain http on a
 * Tailscale address there is nothing to ask, and holdAwake does nothing. That
 * is why every desk also says in words, while work waits, that nothing more
 * is sent while the page is closed, hidden or the phone is locked.
 *
 * Where the server runs its own queue, a desk hands its waiting work to it
 * and takes no hold at all: the server sends each job in turn whatever the
 * page is doing, and the desk says so with WAITS_ON_SERVER instead.
 *
 * Holds are counted. Each desk takes one with holdAwake and lets it go with
 * the function it returns; the lock goes when the last hold does. The browser
 * drops the lock itself whenever the page is hidden, so it is asked for again
 * each time the page becomes visible while any hold is kept.
 */

type Sentinel = {
  readonly released: boolean
  release(): Promise<void>
  addEventListener?(type: 'release', listener: () => void): void
}
type WakeLockApi = { request(type: 'screen'): Promise<Sentinel> }

/**
 * The line a desk shows while work waits in the page, so every desk says it
 * the same way.
 */
export const WAITS_IN_PAGE =
  'Nothing more is sent while this page is closed or hidden, or the phone is locked. Work already sent to ComfyUI carries on.'

/**
 * The line a desk shows instead while its waiting work is held by the queue
 * on the SwitchGen server (src/lib/runner.ts). That work needs no page and no
 * screen kept on, so no wake lock is taken for it, and this line says why the
 * reader can put the phone down. It lives beside WAITS_IN_PAGE so the two are
 * read, and changed, together.
 */
export const WAITS_ON_SERVER =
  'Waiting work is kept on the SwitchGen server and sent in turn, so it goes on while this page is closed or the phone is locked. If the SwitchGen server stops, it waits and carries on when the server is back.'

/** Hold id to the reason it was taken, so a hold can only be let go once. */
const holds = new Map<number, string>()
let nextHold = 1
let sentinel: Sentinel | null = null
let asking: Promise<void> | null = null
let listening = false

function api(): WakeLockApi | null {
  const nav = (globalThis as { navigator?: { wakeLock?: WakeLockApi } }).navigator
  const wl = nav?.wakeLock
  return wl && typeof wl.request === 'function' ? wl : null
}

/** True when this page can keep the screen on at all (a secure context that offers it). */
export function wakeLockAvailable(): boolean {
  return api() !== null
}

/**
 * Keep the screen on until the returned function is called. `reason` names
 * the hold ("video lane", "reel") for anyone reading the holds while
 * debugging. A no-op where the browser offers no wake lock.
 */
export function holdAwake(reason: string): () => void {
  if (!api()) return () => {}
  const id = nextHold++
  holds.set(id, reason)
  listen()
  void acquire()
  return () => {
    if (!holds.delete(id)) return
    if (holds.size === 0) void letGo()
  }
}

/** What is holding the screen on now, by reason. */
export function awakeReasons(): string[] {
  return [...holds.values()]
}

function visible(): boolean {
  return typeof document === 'undefined' || document.visibilityState === 'visible'
}

async function acquire(): Promise<void> {
  const wl = api()
  if (!wl || holds.size === 0) return
  if (sentinel && !sentinel.released) return
  if (asking) return asking
  // The browser refuses a hidden page; the visibility listener asks again.
  if (!visible()) return
  asking = (async () => {
    try {
      const s = await wl.request('screen')
      // Every hold went while the request was on its way.
      if (holds.size === 0) {
        await s.release().catch(() => undefined)
        return
      }
      sentinel = s
      s.addEventListener?.('release', () => {
        if (sentinel === s) sentinel = null
      })
    } catch {
      // Refused: a battery saver, a policy, or the page hid in the meantime.
      // The desk's own words still say what happens while the phone locks.
    } finally {
      asking = null
    }
  })()
  return asking
}

async function letGo(): Promise<void> {
  const s = sentinel
  sentinel = null
  if (s && !s.released) await s.release().catch(() => undefined)
}

function listen(): void {
  if (listening || typeof document === 'undefined') return
  listening = true
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible' && holds.size > 0) void acquire()
  })
}
