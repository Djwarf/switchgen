import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { AdvancedPanel, NO_OVERRIDES } from '../src/components/advanced'
import { PromptField } from '../src/components/compose/PromptField'
import { RunButton, STOPPING, waitingLine } from '../src/components/compose/RunButton'
import { intentReport } from '../src/lib/intent'
import { decide } from '../src/lib/recipe'
import { WAITS_IN_PAGE } from '../src/lib/wakeLock'

/**
 * What the Pictures desk says to a reader on a phone: that the rest of a
 * batch waits in the page, why a weight file on disk is not offered, and no
 * keyboard hint on a screen with no keyboard.
 */
const none = () => {}

describe('the line under a batch', () => {
  it('says how many pictures wait in the page, and that a locked phone sends none of them', () => {
    const line = waitingLine({ index: 1, total: 4 }, false)!
    expect(line).toContain('3 more pictures')
    expect(line).toContain(WAITS_IN_PAGE)
    expect(line).not.toContain('keep the screen on')
  })

  it('says the screen is kept on only where the page can keep it on', () => {
    expect(waitingLine({ index: 1, total: 2 }, true)).toContain('keep the screen on')
  })

  it('says nothing once the last picture is on the press', () => {
    expect(waitingLine({ index: 4, total: 4 }, true)).toBeNull()
    expect(waitingLine(null, true)).toBeNull()
  })

  it('says nothing once Stop is pressed, though the job still counts the rest', () => {
    expect(waitingLine({ index: 1, total: 4, stage: STOPPING }, true)).toBeNull()
    expect(waitingLine({ index: 1, total: 4, stage: 'Drawing' }, false)).toContain('3 more pictures')
  })

  it('is left out under the button while the batch is stopping', () => {
    const button = (stage: string) =>
      renderToStaticMarkup(createElement(RunButton, { running: true, job: { index: 1, total: 3, stage }, onRun: none, onStop: none }))
    const stopping = button(STOPPING)
    expect(stopping).toContain('Picture 1 of 3 · Stopping')
    expect(stopping).not.toContain('more pictures wait')
    expect(button('Drawing')).toContain('2 more pictures wait')
  })
})

describe('a desk where nothing installed can run', () => {
  it('names the files that cannot load here and why, and points to the catalogue', () => {
    const recipe = decide({ prompt: 'a lighthouse at dusk', look: 'anime', anatomy: 'off', installed: [] })
    expect(recipe.ok).toBe(false)
    const html = renderToStaticMarkup(
      createElement(AdvancedPanel, {
        recipe,
        overrides: NO_OVERRIDES,
        onOverrides: none,
        onClose: none,
        unloadable: [{ model: 'Z-Image-Turbo-fp8mix.safetensors', label: 'Z-Image Turbo', why: 'needs qwen_3_4b.safetensors' }],
      }),
    )
    expect(html).toContain('Cannot load here (1)')
    expect(html).toContain('Needs qwen_3_4b.safetensors.')
    expect(intentReport({ intent: 'anime', explicit: false }, { installed: [] }).note).toContain('catalogue')
  })
})

describe('the prompt field', () => {
  it('hides the keyboard shortcut on a screen with no pointer to hover', () => {
    const html = renderToStaticMarkup(createElement(PromptField, { value: '', onChange: none, onSubmit: none }))
    expect(html).toContain('[@media(hover:none)]:hidden')
  })
})
