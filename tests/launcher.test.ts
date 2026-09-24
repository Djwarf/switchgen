import { execFileSync, spawnSync } from 'node:child_process'
import { chmodSync, copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'

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

const launcher = () => readFileSync(path.join(root, 'bin', 'switchgen'), 'utf8')
/** One arm of the launcher's `case`, from its label to its `;;`. */
function arm(name: string): string[] {
  const lines = launcher().split('\n')
  const start = lines.findIndex((l) => new RegExp(`^  ${name}\\)`).test(l))
  expect(start).toBeGreaterThan(-1)
  const end = lines.findIndex((l, i) => i > start && /^\s*;;\s*$/.test(l))
  return lines.slice(start + 1, end).map((l) => l.trim()).filter((l) => l && !l.startsWith('#'))
}

describe('the launcher', () => {
  it('will not start a dev server beside the running app, on the same archive', () => {
    const dev = arm('dev')
    const check = dev.findIndex((l) => l.includes('app_up'))
    const run = dev.findIndex((l) => l === 'npm run dev')
    expect(check).toBeGreaterThan(-1)
    expect(check).toBeLessThan(run)
  })

  it('serves in the foreground with vite itself, the command line `stop` looks for', () => {
    const serve = arm('serve')
    expect(serve.at(-1)).toBe('exec "$APP/node_modules/.bin/vite" preview')
    expect(serve.some((l) => /^exec npm|npm run preview/.test(l))).toBe(false)
    // What `switchgen stop` matches, for a checkout in an awkward folder.
    const fn = /^server_pattern\(\) \{[\s\S]*?^\}/m.exec(launcher())![0]
    const app = '/home/a.b/switch+gen'
    const pattern = execFileSync('bash', ['-c', `${fn}\nserver_pattern`], { env: { ...process.env, APP: app } }).toString()
    // A script run by its #!/usr/bin/env node line runs as node with the script's path.
    expect(new RegExp(pattern).test(`node ${app}/node_modules/.bin/vite preview`)).toBe(true)
    expect(new RegExp(pattern).test(`/usr/bin/node ${app}/node_modules/.bin/vite preview`)).toBe(true)
    expect(new RegExp(pattern).test(`npm exec vite preview`)).toBe(false)
  })

  it('lists serve and phone in its usage line', () => {
    const usage = /usage: switchgen \{([^}]*)\}/.exec(launcher())![1].split('|').map((w) => w.split(' ')[0])
    expect(usage).toContain('serve')
    expect(usage).toContain('phone')
  })
})

describe('the address the launcher gives for the phone', () => {
  // The launcher is copied into a folder of its own, so no .env beside the
  // checkout can change its port, and `tailscale` is a script that answers
  // from files the test writes.
  const dir = mkdtempSync(path.join(os.tmpdir(), 'switchgen-launcher-'))
  const app = path.join(dir, 'app')
  const fake = path.join(dir, 'fakebin')
  mkdirSync(path.join(app, 'bin'), { recursive: true })
  mkdirSync(fake)
  copyFileSync(path.join(root, 'bin', 'switchgen'), path.join(app, 'bin', 'switchgen'))
  chmodSync(path.join(app, 'bin', 'switchgen'), 0o755)
  writeFileSync(
    path.join(fake, 'tailscale'),
    `#!/bin/sh\nif [ "$1" = serve ]; then cat ${JSON.stringify(path.join(dir, 'serve.json'))}; else cat ${JSON.stringify(path.join(dir, 'status.json'))}; fi\n`,
  )
  chmodSync(path.join(fake, 'tailscale'), 0o755)
  writeFileSync(path.join(dir, 'status.json'), JSON.stringify({ Self: { DNSName: 'freya.tail1234.ts.net.' } }))

  afterAll(() => rmSync(dir, { recursive: true, force: true }))

  const phone = (serve: unknown) => {
    writeFileSync(path.join(dir, 'serve.json'), JSON.stringify(serve))
    const env: NodeJS.ProcessEnv = { ...process.env, PATH: `${fake}${path.delimiter}${process.env.PATH}` }
    delete env.SWITCHGEN_PORT
    const r = spawnSync(path.join(app, 'bin', 'switchgen'), ['phone'], { env, encoding: 'utf8', timeout: 15_000 })
    expect(r.status).toBe(0)
    return r.stdout
  }

  it('is the https one Tailscale Serve gives this port', () => {
    const out = phone({
      TCP: { '443': { HTTPS: true } },
      Web: { 'host:443': { Handlers: { '/': { Proxy: 'http://127.0.0.1:5273' } } } },
    })
    expect(out).toContain('On the phone: https://host')
  })

  it('is the command that sets Serve up, and the name it will answer at, when it is not', () => {
    const out = phone({})
    expect(out).toContain('tailscale serve --bg 5273')
    expect(out).toContain('https://freya.tail1234.ts.net ')
    expect(out).not.toContain('ts.net.')
  })
})

describe('the sample units', () => {
  const unit = (name: string) => readFileSync(path.join(root, 'contrib', name), 'utf8')
  /** A unit's ExecStart with its continued lines joined. */
  const execStart = (text: string) => /^ExecStart=((?:.*\\\n)*.*)$/m.exec(text)![1].replace(/\\\n\s*/g, ' ')

  it('runs the app as `switchgen serve`, and restarts it only when that helps', () => {
    const text = unit('switchgen.service')
    expect(execStart(text)).toMatch(/bin\/switchgen serve$/)
    expect(text).toMatch(/^SuccessExitStatus=143$/m)
    expect(text).toMatch(/^RestartPreventExitStatus=3$/m)
  })

  it('has ComfyUI compress its answers', () => {
    expect(execStart(unit('comfyui.service'))).toContain('--enable-compress-response-body')
  })

  it('in the README, gives the app unit\'s copy and enable as separate blocks, with the edit between them', () => {
    const copyRe = /cp .*switchgen\.service/
    const enableRe = /systemctl --user .*enable.*switchgen/
    const readme = readFileSync(path.join(root, 'README.md'), 'utf8')
    const fenced = [...readme.matchAll(/```[a-z]*\n([\s\S]*?)```/g)].map((m) => m[1].split('\n'))
    expect(fenced.some((b) => b.some((l) => copyRe.test(l)))).toBe(true)
    expect(fenced.some((b) => b.some((l) => enableRe.test(l)))).toBe(true)
    for (const block of fenced) expect(block.some((l) => copyRe.test(l)) && block.some((l) => enableRe.test(l))).toBe(false)

    const lines = readme.split('\n')
    const copy = lines.findIndex((l) => copyRe.test(l))
    const enable = lines.findIndex((l) => enableRe.test(l))
    expect(copy).toBeLessThan(enable)
    const between = lines.slice(copy + 1, enable).filter((l) => l && !l.startsWith('```')).join(' ')
    expect(between).toContain('%h/switchgen')
  })
})
