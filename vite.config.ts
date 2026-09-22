import os from 'node:os'
import path from 'node:path'
import { createHash } from 'node:crypto'
import { promises as fs } from 'node:fs'
import type { IncomingMessage } from 'node:http'
import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { switchgenApi } from './server/api.mjs'
import { switchgenArchive } from './server/archive.mjs'
import { switchgenDownloads } from './server/downloads.mjs'
import { switchgenReel } from './server/reel.mjs'
import { switchgenThumbs } from './server/thumbs.mjs'
import { switchgenVision } from './server/vision.mjs'
// @ts-expect-error the server is plain ESM without a declaration for its helpers
import { upgradeAllowed } from './server/guard.mjs'

// ComfyUI runs as a systemd user service on :8188.
// Proxy through Vite so the browser sees one origin (no CORS, no mixed content).
const COMFY = process.env.COMFY_URL ?? 'http://127.0.0.1:8188'

// ComfyUI refuses requests whose Origin doesn't match its Host ("non matching
// host and origin ... returning 403"). changeOrigin rewrites Host but leaves the
// browser's Origin intact, so we must rewrite Origin to match. Without that,
// every request is rejected, the WebSocket included.
const headers = { Origin: COMFY }

const proxy = {
  '/comfy-ws': {
    target: COMFY, ws: true, changeOrigin: true, headers,
    rewrite: (p: string) => p.replace(/^\/comfy-ws/, '/ws'),
    // Vite checks allowedHosts on HTTP requests only; a WebSocket upgrade is
    // proxied before that check runs, and the Origin rewrite above means
    // ComfyUI's own check passes too. So the handshake is checked here, for
    // Host and for origin (see upgradeAllowed in server/guard.mjs). False
    // makes Vite answer 404 and close the socket.
    bypass: (req: IncomingMessage) => (upgradeAllowed(req, allowedHosts) ? undefined : false),
  },
  '/comfy': {
    target: COMFY, changeOrigin: true, headers,
    rewrite: (p: string) => p.replace(/^\/comfy/, ''),
  },
}

// Reachable from the LAN and over Tailscale, not just this machine.
// Vite binds to loopback by default, and separately refuses requests whose Host
// header it does not recognise (DNS-rebinding protection) - so binding alone is
// not enough: the hostnames have to be allowed too. Nothing here is specific to
// one machine: the hostname and the addresses are read at start, any tailnet
// name is allowed, and SWITCHGEN_ALLOWED_HOSTS adds the rest.
const host = true // 0.0.0.0 + [::]
const ownAddresses = Object.values(os.networkInterfaces())
  .flat()
  .filter((i): i is os.NetworkInterfaceInfo => !!i && !i.internal && i.family === 'IPv4')
  .map((i) => i.address)
const allowedHosts = [
  'localhost',
  os.hostname(),
  '.ts.net', // any host on the tailnet (Tailscale MagicDNS)
  ...ownAddresses,
  ...(process.env.SWITCHGEN_ALLOWED_HOSTS ?? '').split(',').map((h) => h.trim()).filter(Boolean),
]
const port = Number(process.env.SWITCHGEN_PORT) || 5273

/**
 * Stamp the service worker with the build it belongs to.
 *
 * public/sw.js is copied into dist byte for byte, and its bytes did not change
 * from one build to the next, so no browser ever installed a newer worker and
 * the shell cache kept every build's scripts for good. The copy in dist gets a
 * stamp of this build's file names in place of `const BUILD = 'dev'`, so a
 * build with new files is a new worker, and that worker's activate step drops
 * the shells of the builds before it. A rebuild that changes nothing keeps its
 * stamp and costs no reinstall. The dev server serves public/sw.js untouched.
 */
function stampServiceWorker(): Plugin {
  const placeholder = /const BUILD = '[^']*'/
  return {
    name: 'switchgen-sw-stamp',
    apply: 'build',
    async writeBundle(options, bundle) {
      if (!options.dir) return
      const file = path.join(options.dir, 'sw.js')
      let source: string
      try {
        source = await fs.readFile(file, 'utf8')
      } catch {
        return // no worker in this build
      }
      if (!placeholder.test(source)) {
        this.warn(`${file} has no "const BUILD = '...'" line to stamp; old shells will not be dropped`)
        return
      }
      const stamp = createHash('sha1').update(Object.keys(bundle).sort().join('\n')).digest('hex').slice(0, 12)
      await fs.writeFile(file, source.replace(placeholder, `const BUILD = '${stamp}'`))
    },
  }
}

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    switchgenApi(),
    switchgenArchive(),
    switchgenThumbs(),
    switchgenDownloads(),
    switchgenReel(),
    switchgenVision(),
    stampServiceWorker(),
  ],
  build: {
    rolldownOptions: {
      output: {
        // Everything used to ship as one script of about 940 KB, so any change
        // to any room made every device fetch React and the model tables
        // again. These groups change far less often than the rooms do, and as
        // chunks of their own they keep their hashes, and their cached copies,
        // across a build that does not touch them. Nothing is loaded later
        // than before: all of it is still imported from the start. The two
        // tables import nothing and touch nothing outside themselves when
        // loaded, so where they sit cannot change what runs first.
        codeSplitting: {
          groups: [
            { name: 'react', test: /node_modules[\\/](react|react-dom|scheduler)[\\/]/, priority: 3 },
            { name: 'vendor', test: /node_modules[\\/]/, priority: 2 },
            { name: 'tables', test: /src[\\/]lib[\\/](registry|loraIndex)\.ts$/, priority: 1 },
          ],
        },
      },
    },
  },
  server: { host, port, proxy, allowedHosts },
  // strictPort: with the port taken, preview used to move to the next free
  // one without a word, and the launcher found the old server still answering
  // on this one. Failing is the honest answer; bin/switchgen reports it.
  preview: { host, port, strictPort: true, proxy, allowedHosts },
})
