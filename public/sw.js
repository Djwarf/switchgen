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
 *   /comfy/view?...   Network first, the cache as the fallback. A file is
 *                     addressed by its name, and a name can come back: ComfyUI
 *                     numbers a new file one past the highest it can see, so
 *                     deleting the newest picture hands its name to the next
 *                     render. A copy kept by name is therefore not proof of
 *                     what is on disk now. The cache keeps the archive
 *                     browsable when the connection drops.
 *   everything else   Shell assets: cache-first with a background refresh.
 */
// v2 retired the v1 media cache, which served files cache-first and could
// still be holding a deleted picture under a name that has since been reused.
const VERSION = 'switchgen-v2'
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

  // Generated files: ask the server first (see the rules above). `no-cache`
  // makes the browser revalidate its own HTTP copy too, because ComfyUI sends
  // /view with a Last-Modified and an ETag but no Cache-Control, which leaves
  // the browser free to reuse a stale copy without asking. An unchanged file
  // comes back as a 304 with no body.
  if (url.pathname === '/comfy/view') {
    // A file opened in a tab of its own is a navigation, and not every browser
    // lets a navigation be re-issued with other cache settings, so it is left
    // to the browser.
    if (request.mode === 'navigate') return
    event.respondWith((async () => {
      const cache = await caches.open(MEDIA)
      try {
        const res = await fetch(request, { cache: 'no-cache' })
        // Whole files only: a 206 is part of a clip, and the cache refuses it.
        // An unchanged file is not written again.
        if (res.status === 200) {
          const hit = await cache.match(request)
          const etag = res.headers.get('etag')
          if (!hit || !etag || hit.headers.get('etag') !== etag) {
            cache.put(request, res.clone())
              .then(() => trimCache(MEDIA, MEDIA_MAX))
              .catch(() => { /* storage full or refused: the page has the file */ })
          }
        }
        return res
      } catch {
        return (await cache.match(request)) ??
          new Response('Offline', { status: 503, statusText: 'Offline' })
      }
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
