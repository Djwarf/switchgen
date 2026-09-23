import { readFileSync } from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'

/**
 * The advice for installing the sample ComfyUI unit, as the launcher prints it
 * and as the README gives it. The unit ships with paths that suit the author's
 * machine; enabled as shipped, where those paths are wrong it fails to start
 * and Restart=always tries again every five seconds. So the lines a reader
 * would paste together must not copy the unit and enable it in one go: the
 * edit has to sit between them, in words, not as a comment the paste runs past.
 */
const root = path.resolve(import.meta.dirname, '..')
const COPY = /cp .*comfyui\.service/
const ENABLE = /systemctl --user .*enable/

/** Runs of consecutive lines that pass `inBlock`, each a block one paste would take. */
function blocks(lines: string[], inBlock: (line: string) => boolean): string[][] {
  const out: string[][] = []
  let run: string[] = []
  for (const line of lines) {
    if (inBlock(line)) run.push(line)
    else if (run.length) {
      out.push(run)
      run = []
    }
  }
  if (run.length) out.push(run)
  return out
}

describe('the advice for the sample ComfyUI unit', () => {
  it('in the launcher, puts the edit between the copy and the enable', () => {
    const script = readFileSync(path.join(root, 'bin', 'switchgen'), 'utf8')
    const msg = /<<MSG\n([\s\S]*?)\nMSG\n/.exec(script)?.[1]
    expect(msg).toBeDefined()
    const lines = msg!.split('\n')
    // The commands are the indented lines; the prose around them is not.
    const commands = blocks(lines, (l) => /^\s+\S/.test(l))
    expect(commands.some((b) => b.some((l) => COPY.test(l)))).toBe(true)
    expect(commands.some((b) => b.some((l) => ENABLE.test(l)))).toBe(true)
    for (const block of commands) expect(block.some((l) => COPY.test(l)) && block.some((l) => ENABLE.test(l))).toBe(false)

    const copy = lines.findIndex((l) => COPY.test(l))
    const enable = lines.findIndex((l) => ENABLE.test(l))
    expect(copy).toBeLessThan(enable)
    expect(lines.slice(copy + 1, enable).some((l) => /^\S/.test(l))).toBe(true)
  })

  it('in the README, gives the copy and the enable as separate blocks, with the edit between them', () => {
    const readme = readFileSync(path.join(root, 'README.md'), 'utf8')
    const fenced = [...readme.matchAll(/```[a-z]*\n([\s\S]*?)```/g)].map((m) => m[1].split('\n'))
    expect(fenced.some((b) => b.some((l) => COPY.test(l)))).toBe(true)
    expect(fenced.some((b) => b.some((l) => ENABLE.test(l)))).toBe(true)
    for (const block of fenced) expect(block.some((l) => COPY.test(l)) && block.some((l) => ENABLE.test(l))).toBe(false)

    const lines = readme.split('\n')
    const copy = lines.findIndex((l) => COPY.test(l))
    const enable = lines.findIndex((l) => ENABLE.test(l))
    expect(copy).toBeLessThan(enable)
    // Prose between them, outside any fence, that names what to change.
    const between = lines.slice(copy + 1, enable).filter((l) => l && !l.startsWith('```')).join(' ')
    expect(between).toContain('%h/ComfyUI')
    expect(between).toContain('%h/ai/outputs')
  })
})
