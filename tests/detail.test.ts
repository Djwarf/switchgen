import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { CardActions } from '../src/components/archive/CardActions'
import { DeleteDialog } from '../src/components/archive/DeleteDialog'
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
const render = (variant: HistoryEntry['variant'], over: Partial<HistoryEntry> = {}) =>
  renderToStaticMarkup(
    createElement(Detail, {
      entry: { ...entry(variant), ...over },
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

  it('says how a record was made in words, not by its machine name', () => {
    const html = render(null, { mode: 't2i' })
    expect(html).toMatch(/Mode<\/dt>.*?From words/s)
    expect(html).not.toContain('>t2i<')
  })

  it('says which values are links, since not all of them are', () => {
    expect(render(null)).toContain('Underlined values are links')
  })

  // On a phone the overlay is the whole screen and Close the only way out.
  const COARSE = '[@media(pointer:coarse)]:min-h-11'
  const classOf = (html: string, pattern: RegExp) => pattern.exec(html)?.[1] ?? ''
  it('gives Close, Previous and Next a finger-sized target on a touch screen', () => {
    const html = render(null)
    for (const label of ['Close', '← Previous', 'Next →']) {
      expect(classOf(html, new RegExp(`<button[^>]*class="([^"]*)"[^>]*>${label}</button>`)), label).toContain(COARSE)
    }
  })

  it('writes the note at a size a phone will not zoom into', () => {
    const html = render(null)
    expect(classOf(html, /<textarea[^>]*class="([^"]*)"/)).toContain('[@media(pointer:coarse)]:text-body')
  })
})

describe('the dialog before files are deleted', () => {
  const clip = { filename: 'shot_00001_.webm', subfolder: 'reel', type: 'output' }
  const frame = { filename: 'shot_00001_.webm.frame.png', subfolder: 'reel', type: 'output' }
  const shot: HistoryEntry = { ...entry(null), id: 'shot', kind: 'video', desk: 'video', file: clip, files: [clip, frame] }
  const dialog = (keep?: (rel: string) => boolean) =>
    renderToStaticMarkup(
      createElement(DeleteDialog, { records: [shot], keep, busy: false, error: null, onCancel: none, onConfirm: none }),
    )

  it('counts and names every file the run wrote', () => {
    const html = dialog()
    expect(html).toContain('Delete 2 files')
    expect(html).toContain('reel/shot_00001_.webm.frame.png')
  })

  it('says a last frame the reel still opens a shot on stays', () => {
    const html = dialog((rel) => rel.endsWith('.frame.png'))
    expect(html).toContain('Delete the file')
    expect(html).not.toContain('Delete 2 files')
    expect(html).toContain('stays, because the reel still opens a shot on it')
  })
})

describe('the verbs on a record', () => {
  const actions = (variant: HistoryEntry['variant'], reuse: boolean) =>
    renderToStaticMarkup(
      createElement(CardActions, {
        entry: entry(variant),
        canDeleteFile: false,
        selected: false,
        onOpen: none,
        ...(reuse ? { onReuse: none, onAnother: none } : {}),
        onSource: none,
        onStar: none,
        onDownload: none,
        onRemove: none,
        onDeleteFile: none,
        onSelect: none,
      }),
    )

  it('offers a region pass once, as drawing the region again, not as a whole picture to make again', () => {
    const html = actions('refine', true)
    expect(html).toContain('Draw the region again')
    expect(html).not.toContain('Make another')
    expect(html).not.toContain('Use these settings')
  })

  it('offers neither when the picture the region was drawn on has gone', () => {
    const html = actions('refine', false)
    expect(html).not.toContain('Draw the region again')
    expect(html).not.toContain('Make another')
    expect(html).not.toContain('Use these settings')
  })

  it('offers a plain picture both verbs', () => {
    const html = actions(null, true)
    expect(html).toContain('Use these settings')
    expect(html).toContain('Make another')
    expect(html).not.toContain('Draw the region again')
  })
})
