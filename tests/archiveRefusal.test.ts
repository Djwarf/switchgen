import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { ArchiveSyncState } from '../src/lib/archiveSync'

/**
 * What the page is told when the server answers but will not share the
 * archive: another SwitchGen server holds it (server/archive.mjs, the lock).
 * The sentence is the server's own, shown as it is, where a server that did
 * not answer at all is only offline.
 */
const SAID = 'Another SwitchGen server (process 1) is using this archive, so this one cannot share it. Stop the other server, or give this one an archive of its own.'

beforeEach(() => {
  // The retry after a failure waits on a timer that never needs to run here.
  vi.useFakeTimers()
  vi.resetModules()
  vi.stubGlobal('EventSource', class { onmessage = null; close() {} })
})

afterEach(() => {
  vi.clearAllTimers()
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

/** The sync's state as a component on the page reads it. */
async function seen(): Promise<ArchiveSyncState> {
  const sync = await import('../src/lib/archiveSync')
  await sync.startArchiveSync()
  let state: ArchiveSyncState | null = null
  const Probe = () => {
    state = sync.useArchiveSync()
    return null
  }
  renderToStaticMarkup(createElement(Probe))
  return state!
}

describe('a server that will not share the archive', () => {
  it('is offline, and its own sentence is kept to show', async () => {
    vi.stubGlobal('fetch', async () =>
      new Response(JSON.stringify({ error: SAID, busy: 'archive' }), { status: 503, headers: { 'content-type': 'application/json' } }),
    )
    const state = await seen()
    expect(state.mode).toBe('offline')
    expect(state.refusal).toBe(SAID)
  })

  it('is told apart from a server that did not answer, which has nothing to show', async () => {
    vi.stubGlobal('fetch', async () => {
      throw new TypeError('Failed to fetch')
    })
    const state = await seen()
    expect(state.mode).toBe('offline')
    expect(state.refusal).toBeNull()
  })

  it('and from a server that failed for a reason of its own', async () => {
    vi.stubGlobal('fetch', async () =>
      new Response(JSON.stringify({ error: 'EACCES' }), { status: 503, headers: { 'content-type': 'application/json' } }),
    )
    const state = await seen()
    expect(state.mode).toBe('offline')
    expect(state.refusal).toBeNull()
  })
})
