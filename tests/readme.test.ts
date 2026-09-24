import { readFileSync } from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'

/**
 * What the README says of the behaviour this suite pins elsewhere, read as
 * text, as the launcher's advice is in launcher.test.ts. Each sentence here
 * was missing or said something else once; it is the reader's only account
 * of what happens on a second device or after a reload. Lines are wrapped in
 * the file, so runs of spaces and newlines are read as one space.
 */
const readme = readFileSync(path.resolve(import.meta.dirname, '..', 'README.md'), 'utf8').replace(/\s+/g, ' ')

describe('the README on the archive', () => {
  it('says one server holds the archive at a time, and what a second one answers', () => {
    expect(readme).toContain('archive.json.lock')
    expect(readme).toContain('503 for the archive')
  })

  it('says a record filed after the fact is folded into the desk\'s record on every device', () => {
    expect(readme).toContain('folds a record the recovery pass filed')
  })

  it('says what deleting the file leaves on disk', () => {
    expect(readme).toContain('except a last frame the reel in this browser')
  })
})

describe('the README on a fetch that outlives its page', () => {
  it('says which fetches the page finds again by itself, and how the rest are taken up', () => {
    expect(readme).toContain("finds a family's fetch again")
    expect(readme).toContain('pressing its fetch once more takes up the fetch still')
  })
})
