import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { Detail } from '../src/components/archive/Detail'
import type { HistoryEntry } from '../src/lib/history'

const entry = (variant: HistoryEntry['variant']): HistoryEntry => ({
  id: 'r1',
  no: 40,
  at: 1_700_000_000_000,
  desk: 'images',
  kind: 'image',
  mode: 'i2i',
  file: { filename: 'r1.png', subfolder: '', type: 'output' },
  familyId: 'sdxl-illustrious',
  familyLabel: 'Illustrious',
  variant,
  model: 'm.safetensors',
  modelLabel: 'M',
  prompt: 'a lighthouse at dusk',
  negative: null,
  seed: 1,
  steps: 20,
  cfg: 5,
  sampler: 'euler',
  scheduler: 'normal',
  width: 1024,
  height: 1024,
  promptId: 'p1',
  durationMs: 1000,
})

const none = () => {}
const render = (variant: HistoryEntry['variant']) =>
  renderToStaticMarkup(
    createElement(Detail, {
      entry: entry(variant),
      canDeleteFile: false,
      hasPrev: false,
      hasNext: false,
      onPrev: none,
      onNext: none,
      onClose: none,
      onOpenEntry: none,
      onOpen: none,
      onReuse: none,
      onAnother: none,
      onSource: none,
      onStar: none,
      onDownload: none,
      onRemove: none,
      onDeleteFile: none,
      onSelect: none,
    }),
  )

describe('the record in full', () => {
  it('names the region pass in words, not by its machine name', () => {
    const html = render('refine')
    expect(html).toContain('One region of a finished picture, redrawn')
    expect(html).not.toMatch(/>refine</)
  })

  it('shows no variant row for a plain picture', () => {
    expect(render(null)).not.toContain('Variant')
  })
})
