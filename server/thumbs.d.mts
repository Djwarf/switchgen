import type { Plugin } from 'vite'
/** GET /api/thumb: small WebP thumbnails of the outputs, made once and kept. */
export function switchgenThumbs(): Plugin
/** Remove every thumbnail made from `rel`, a path under the outputs folder. Never rejects. */
export function forgetThumbs(rel: string): Promise<void>
