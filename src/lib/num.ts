/**
 * Three arithmetic helpers that eight files had each written for themselves.
 */

export const clamp = (n: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, n))

/** Two decimals, so 0.7000000000000001 never reaches the screen. */
export const round2 = (n: number) => Math.round(n * 100) / 100

/** The nearest multiple of 16 at or above 16: what every latent size has to be. */
export const snap16 = (n: number) => Math.max(16, Math.round(n / 16) * 16)
