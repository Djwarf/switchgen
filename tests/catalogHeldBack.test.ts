import { describe, expect, it } from 'vitest'
import { heldBack, noToken, type CatalogPlan } from '../src/lib/catalog'

// The question the catalogue asks before "Fetch anyway". A plan can be held
// back for a reason that has nothing to do with fitting, and the question
// quotes the server's own reason rather than calling every one "will not fit".
const plan = (over: Partial<CatalogPlan>): CatalogPlan => ({
  family: { id: 'flux1-dev', label: 'Flux.1 dev', mode: 't2i', models: ['flux1-dev.gguf'] },
  chosenModel: 'flux1-dev.gguf',
  download: [],
  totalBytes: 0,
  alreadyInstalled: [],
  asIs: [],
  incomplete: [],
  gated: { files: [], tokenPresent: false },
  hardware: { ramTotal: 0, ramFree: 0, diskFree: null, diskTotal: null },
  fits: true,
  verdict: 'Fits.',
  reasons: [],
  blockers: [],
  ...over,
})

describe('why a fetch was held back', () => {
  it('quotes a disk blocker as the reason, not a fit it never claimed', () => {
    const text = heldBack(plan({ fits: false, blockers: ['Only 3.1 GB free on the models disk; this needs 12 GB.'] }))
    expect(text).toBe('The server holds this back. Only 3.1 GB free on the models disk; this needs 12 GB.')
    expect(text).not.toMatch(/will not fit/)
  })

  it('falls back to the verdict when the server gives no blocker', () => {
    expect(heldBack(plan({ fits: false, verdict: 'Too little RAM free right now.' }))).toBe(
      'The server holds this back. Too little RAM free right now.',
    )
  })

  it('names a missing token alone when the plan fits', () => {
    const text = heldBack(plan({ gated: { files: ['ae.safetensors'], tokenPresent: false } }))
    expect(text).not.toMatch(/holds this back/)
    expect(text).toBe(
      'ae.safetensors is gated on HuggingFace and no token is on the server, so the fetch would stop at it.',
    )
  })

  it('adds the missing token after a blocker', () => {
    const text = heldBack(plan({ fits: false, blockers: ['No disk.'], gated: { files: ['a', 'b'], tokenPresent: false } }))
    expect(text).toBe(
      'The server holds this back. No disk. a, b are gated on HuggingFace and no token is on the server, so the fetch would stop at the first of them.',
    )
  })
})

describe('a gated file with no token', () => {
  it('says nothing when there is a token, or nothing gated', () => {
    expect(noToken(plan({ gated: { files: ['ae.safetensors'], tokenPresent: true } }))).toBeNull()
    expect(noToken(plan({}))).toBeNull()
  })
})

describe('a fetch held back with a file on disk that is not the catalogue\'s', () => {
  it('says the fetch leaves that file as it is', () => {
    const text = heldBack(plan({ fits: false, blockers: ['Too little RAM.'], asIs: ['c.bin'] }))
    expect(text).toContain('The fetch leaves c.bin as it is on disk')
  })
})
