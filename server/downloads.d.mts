import type { Plugin } from 'vite'
/** Hardware-aware model catalogue and aria2c-backed downloader. */
export function switchgenDownloads(): Plugin

/** A family fetch as GET /api/download/status lists it. */
export type PublicPlan = {
  family: string
  state: 'running' | 'done' | 'error' | 'cancelled'
  files: { filename: string; dest: string; sizeBytes: number | null; gated: boolean }[]
  current: {
    jobId: string
    filename: string
    index: number
    count: number
    state: 'starting' | 'downloading'
    done: number
    total: number
    pct: number
    speed: number
    etaSec: number | null
  } | null
  finished: string[]
  error: string | null
  startedAt: number
  endedAt: number | null
}

/**
 * What GET /api/download/status says about a set of plan records, one per
 * family, leaving out those that ended more than ten minutes before `now`.
 */
export function publicPlans(list: Iterable<Record<string, unknown>>, now?: number): PublicPlan[]
