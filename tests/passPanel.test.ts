import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { PassPanel } from '../src/components/advanced/PassPanel'

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
