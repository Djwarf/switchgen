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
// not enough: the hostnames have to be allowed too.
const host = true // 0.0.0.0 + [::]
const allowedHosts = [
  'localhost',
  'freya',
  'freya.tail8bf383.ts.net', // Tailscale MagicDNS
  '.ts.net',                 // any host on the tailnet
  '192.168.1.12',
  '100.93.117.89',
]

export default defineConfig({
  plugins: [react(), tailwindcss(), switchgenApi(), switchgenArchive(), switchgenDownloads(), switchgenReel(), switchgenVision()],
  server: { host, port: 5273, proxy, allowedHosts },
  preview: { host, port: 5273, proxy, allowedHosts },
})
