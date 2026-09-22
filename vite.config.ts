import os from 'node:os'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { switchgenApi } from './server/api.mjs'
import { switchgenArchive } from './server/archive.mjs'
import { switchgenDownloads } from './server/downloads.mjs'
import { switchgenReel } from './server/reel.mjs'
import { switchgenVision } from './server/vision.mjs'

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

export default defineConfig({
  plugins: [react(), tailwindcss(), switchgenApi(), switchgenArchive(), switchgenDownloads(), switchgenReel(), switchgenVision()],
  server: { host, port, proxy, allowedHosts },
  preview: { host, port, proxy, allowedHosts },
})
