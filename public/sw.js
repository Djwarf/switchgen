/*
 * SwitchGen service worker.
 *
 * This app drives a ComfyUI server on the same host, so it is NOT useful
 * offline — you cannot generate without the backend. The worker therefore aims
 * at a fast, reliable shell on flaky wifi and over Tailscale, not at offline
 * generation.
 *
 * Caching rules, in order of importance:
 *   /comfy/*, /api/*  NEVER cached. These are live state: the job queue, device
 *                     status, the model inventory. A stale answer here is worse
 *                     than no answer.
 *   /comfy/view?...   Cached. Generated files are immutable once written and are
 *                     addressed by filename, so the archive stays browsable and
 *                     scrolling it does not re-fetch megabytes.
 *   everything else   Shell assets: cache-first with a background refresh.
 */
const VERSION = 'switchgen-v1'
const SHELL = `${VERSION}-shell`
const MEDIA = `${VERSION}-media`
const MEDIA_MAX = 200 // generated files kept locally; oldest evicted past this

const PRECACHE = [
  '/',
  '/manifest.webmanifest',
  '/icon-192.png',
  '/fonts/CrimsonText-Regular.ttf',
  '/fonts/CrimsonText-SemiBold.ttf',
  '/fonts/CrimsonText-Bold.ttf',
  '/fonts/CrimsonText-Italic.ttf',
]

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(SHELL)
      .then((c) => Promise.allSettled(PRECACHE.map((u) => c.add(u))))
      .then(() => self.skipWaiting()),
  )
})

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => !k.startsWith(VERSION)).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  )
})

async function trimCache(name, max) {
  const c = await caches.open(name)
  const keys = await c.keys()
  if (keys.length <= max) return
  await Promise.all(keys.slice(0, keys.length - max).map((k) => c.delete(k)))
}

self.addEventListener('fetch', (event) => {
  const { request } = event
  if (request.method !== 'GET') return

  const url = new URL(request.url)
  if (url.origin !== self.location.origin) return

  // Generated output files are immutable — safe and worthwhile to cache.
  if (url.pathname === '/comfy/view') {
    event.respondWith((async () => {
      const cache = await caches.open(MEDIA)
      const hit = await cache.match(request)
      if (hit) return hit
      const res = await fetch(request)
      if (res.ok) {
        cache.put(request, res.clone()).then(() => trimCache(MEDIA, MEDIA_MAX))
      }
      return res
    })())
    return
  }

  // Live backend state must never be served stale.
  if (url.pathname.startsWith('/comfy') || url.pathname.startsWith('/api')) return

  // Vite's dev-server plumbing must pass straight through, or HMR breaks and a
  // stale module gets served on the next reload.
  if (url.pathname.startsWith('/@') || url.pathname.startsWith('/src/') ||
      url.pathname.startsWith('/node_modules/')) return

  // A navigation asks the network for the current index.html, because its
  // asset hashes change every build and the cached copy names the previous
  // ones. The cache is the fallback, not the answer: that is what keeps the
  // app booting over flaky wifi without pinning it to an old release.
  if (request.mode === 'navigate') {
    event.respondWith((async () => {
      const cache = await caches.open(SHELL)
      try {
        const res = await fetch(request)
        if (res.ok && res.type === 'basic') cache.put('/', res.clone())
        return res
      } catch {
        return (await cache.match('/')) ??
          new Response('Offline', { status: 503, statusText: 'Offline' })
      }
    })())
    return
  }

  // App shell: serve fast from cache, refresh in the background.
  event.respondWith((async () => {
    const cache = await caches.open(SHELL)
    const hit = await cache.match(request, { ignoreSearch: false })
    const network = fetch(request)
      .then((res) => {
        if (res.ok && res.type === 'basic') cache.put(request, res.clone())
        return res
      })
      .catch(() => null)
    if (hit) return hit
    const res = await network
    if (res) return res
    // Navigations fall back to the cached shell so the app still boots.
    if (request.mode === 'navigate') {
      const shell = await cache.match('/')
      if (shell) return shell
    }
    return new Response('Offline', { status: 503, statusText: 'Offline' })
  })())
})
