import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { PassPanel } from '../src/components/advanced/PassPanel'
import { detailRedraw, detailSentence } from '../src/lib/refine'

// The detail passes offered before the picture, and the note under them about
// the region pass, which the finished picture offers along with whichever of
// these it can run. A pass whose detector is missing is not listed, and the
// finished picture does not offer it either.
type Props = Parameters<typeof PassPanel>[0]

function render(opts: { face: boolean; hand: boolean; hires: boolean }): string {
  const pass = (available: boolean) => ({ available, blocked: available ? null : 'needs a detector file' })
  const plan = {
    label: 'Illustrious',
    capabilities: { faceDetail: true, handDetail: true, hires: opts.hires, refine: true },
    passes: { face: pass(opts.face), hand: pass(opts.hand), refine: pass(true) },
  } as unknown as Props['plan']
  const settled = {
    passes: { face: false, hand: false, hires: false },
    params: { steps: 20, width: 1024, height: 1024 },
    cost: 1,
  } as unknown as Props['settled']
  return renderToStaticMarkup(createElement(PassPanel, { plan, settled, onPasses: () => {} }))
}

describe('the note on the region pass', () => {
  it('counts only the passes listed above it', () => {
    const html = render({ face: false, hand: true, hires: true })
    expect(html).toContain('along with these two')
    expect(html).not.toContain('these three')
  })

  it('names none when nothing is listed above it', () => {
    const html = render({ face: false, hand: false, hires: false })
    expect(html).toContain('The finished picture offers it.')
    expect(html).not.toContain('along with')
  })

  it('names all three when all three are listed', () => {
    expect(render({ face: true, hand: true, hires: true })).toContain('along with these three')
  })
})

describe('what the face pass says it does', () => {
  // The Impact Pack's arithmetic, not a measurement: the padded crop is
  // capped at 1024 pixels, so a small face comes back at about 1024 over the
  // crop factor, and a large one is never scaled down.
  it('draws a face 80 pixels across again at about 410', () => {
    const r = detailRedraw('face', 80)
    expect(r.redrawn).toBeCloseTo(409.6, 1)
    expect(r.cells).toBe(2621)
  })

  it('never shrinks a face larger than the cap allows', () => {
    expect(detailRedraw('face', 600).redrawn).toBe(600)
  })

  it('says so in the panel, in figures the arithmetic gives', () => {
    const said = detailSentence('face')
    expect(said).toContain('about 410')
    expect(said).not.toContain('nine thousand')
    expect(render({ face: true, hand: false, hires: false })).toContain(said)
  })
})
