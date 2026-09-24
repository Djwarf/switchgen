/**
 * The boundary (TEST PLAN, BOUNDARY): the app never loads the lab, the app's
 * tests never run the lab's, the app's type check and build never see lab/,
 * and the lab's tests run with the guards that keep them off the real
 * ComfyUI, the running app and the author's folders.
 */
import fs from 'node:fs'
import path from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { labImports } from '../bin/lab.ts'
import { REPO, removeTemp, tempDir } from './helpers.ts'

afterAll(removeTemp)

/** Every import, export-from, dynamic import and require in a source file, by its specifier. */
function specifiers(text: string): string[] {
  const out: string[] = []
  const res = [
    /\bimport\s+(?:type\s+)?(?:[\w*{}\s,$]+\s+from\s+)?['"]([^'"]+)['"]/g,
    /\bexport\s+(?:type\s+)?[\w*{}\s,$]+\s+from\s+['"]([^'"]+)['"]/g,
    /\bimport\s*\(\s*['"]([^'"]+)['"]\s*\)/g,
    /\brequire\s*\(\s*['"]([^'"]+)['"]\s*\)/g,
  ]
  for (const re of res) for (const m of text.matchAll(re)) out.push(m[1])
  return out
}

function sources(dir: string): string[] {
  const out: string[] = []
  for (const d of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, d.name)
    if (d.isDirectory()) {
      if (d.name !== 'node_modules') out.push(...sources(full))
    } else if (/\.(m?[jt]sx?|c[jt]s)$/.test(d.name)) out.push(full)
  }
  return out
}

describe('the app never loads the lab', () => {
  it('no file under src/ or server/ imports from lab/', () => {
    const lab = path.join(REPO, 'lab') + path.sep
    const found: string[] = []
    for (const top of ['src', 'server']) {
      for (const file of sources(path.join(REPO, top))) {
        for (const spec of specifiers(fs.readFileSync(file, 'utf8'))) {
          const target = spec.startsWith('.') ? path.resolve(path.dirname(file), spec) : spec.startsWith('/') ? spec : null
          if (/^(\.\/)?lab\//.test(spec) || (target && (target + path.sep).startsWith(lab))) found.push(`${path.relative(REPO, file)} → ${spec}`)
        }
      }
    }
    expect(found).toEqual([])
    // `lab/lab check` makes the same scan.
    expect(labImports(REPO)).toEqual([])
  })

  it('labImports finds an import, a dynamic import, a require and an export-from of lab/, and nothing else', () => {
    const fake = tempDir('lab-fake-repo-')
    fs.mkdirSync(path.join(fake, 'src', 'deep'), { recursive: true })
    fs.mkdirSync(path.join(fake, 'server'), { recursive: true })
    fs.mkdirSync(path.join(fake, 'lab', 'core'), { recursive: true })
    fs.writeFileSync(path.join(fake, 'src', 'a.ts'), "import { x } from '../lab/core/env.ts'\n")
    fs.writeFileSync(path.join(fake, 'src', 'deep', 'b.tsx'), "const y = await import('../../lab/x.ts')\nimport z from './label.ts'\n")
    fs.writeFileSync(path.join(fake, 'server', 'c.cjs'), "const q = require('../lab/run/driver.ts')\n")
    fs.writeFileSync(path.join(fake, 'server', 'd.mjs'), "export { w } from '../lab/judge/seal.ts'\n")
    fs.writeFileSync(path.join(fake, 'server', 'e.mjs'), "import { v } from './laboratory.mjs'\nimport u from '../labels/u.mjs'\n")
    expect(labImports(fake).map((x) => x.file).sort()).toEqual(['server/c.cjs', 'server/d.mjs', 'src/a.ts', 'src/deep/b.tsx'])
  })

  it('the build starts at index.html, and the launcher does not rebuild for lab/ edits', () => {
    const html = fs.readFileSync(path.join(REPO, 'index.html'), 'utf8')
    expect(html).not.toMatch(/lab\//)
    const launcher = fs.readFileSync(path.join(REPO, 'bin', 'switchgen'), 'utf8')
    const needsBuild = /needs_build\(\)\s*\{([\s\S]*?)\n\}/.exec(launcher)?.[1] ?? ''
    expect(needsBuild).toMatch(/find src server/)
    expect(needsBuild).not.toMatch(/\blab\b/)
  })
})

describe('the app\'s tests never run the lab\'s, and the lab\'s command runs only its own', () => {
  it('the root vitest include does not match lab/tests, and the lab\'s include matches only lab/tests', async () => {
    const root = (await import('../../vitest.config.ts')).default as { test: { include: string[] } }
    const labCfg = (await import('../vitest.config.ts')).default as { root: string; test: { include: string[] } }
    const labFiles = fs.readdirSync(path.join(REPO, 'lab', 'tests')).filter((f) => f.endsWith('.test.ts')).map((f) => `lab/tests/${f}`)
    expect(labFiles).toContain("lab/tests/boundary.test.ts")
    const appFiles = fs.readdirSync(path.join(REPO, 'tests')).filter((f) => f.endsWith('.test.ts')).map((f) => `tests/${f}`)
    const matches = (file: string, globs: string[]) => globs.some((g) => path.matchesGlob(file, g))
    for (const f of labFiles) {
      expect(matches(f, root.test.include), f).toBe(false)
      expect(matches(f, labCfg.test.include), f).toBe(true)
    }
    for (const f of appFiles) expect(matches(f, labCfg.test.include), f).toBe(false)
    expect(path.resolve(labCfg.root)).toBe(REPO)
    const pkg = JSON.parse(fs.readFileSync(path.join(REPO, 'package.json'), 'utf8'))
    expect(pkg.scripts.test).toBe('vitest run')
    expect(JSON.stringify(pkg)).not.toMatch(/lab\//)
  })

  it('the lab\'s launcher runs the lab\'s config for test and typecheck', () => {
    const sh = fs.readFileSync(path.join(REPO, 'lab', 'lab'), 'utf8')
    expect(sh).toMatch(/npx vitest run --config lab\/vitest\.config\.ts/)
    expect(sh).toMatch(/npx tsc -p lab\/tsconfig\.json/)
  })

  it('every folder a lab test makes goes through tempDir, so removeTemp takes it away when the file ends', () => {
    // A bare mkdtemp is never removed: each run of the tests would leave one behind.
    // Spelled in two parts, so this file does not find itself.
    const bare = new RegExp('\\bmkd' + 'temp(Sync)?\\s*\\(')
    const found: string[] = []
    for (const f of fs.readdirSync(path.join(REPO, 'lab', 'tests'), { recursive: true }) as string[]) {
      if (!/\.(m?[jt]s)$/.test(f) || f === 'helpers.ts') continue
      if (bare.test(fs.readFileSync(path.join(REPO, 'lab', 'tests', f), 'utf8'))) found.push(f)
    }
    expect(found).toEqual([])
  })

  it('these tests run with the guards on: ComfyUI and the app on port 9, the runner off, the lab folder outside the repo', () => {
    expect(process.env.COMFY_URL).toBe('http://127.0.0.1:9')
    expect(process.env.SWITCHGEN_URL).toBe('http://127.0.0.1:9')
    expect(process.env.SWITCHGEN_RUNNER).toBe('off')
    for (const k of ['SWITCHGEN_LAB_DIR', 'SWITCHGEN_OUTPUTS', 'SWITCHGEN_MODELS']) {
      const v = process.env[k]!
      expect(v, k).toMatch(/switchgen-lab-vitest-unset/)
      expect(path.relative(REPO, v).startsWith('..'), k).toBe(true)
    }
  })
})

describe('the app\'s type check never sees lab/', () => {
  const read = (f: string) => fs.readFileSync(path.join(REPO, f), 'utf8')
  it('the root tsconfig does not reference lab, and no referenced config includes it', () => {
    const rootCfg = JSON.parse(read('tsconfig.json')) as { files?: string[]; include?: string[]; references: { path: string }[] }
    expect(JSON.stringify(rootCfg)).not.toMatch(/lab/)
    for (const r of rootCfg.references) {
      const text = read(r.path.replace(/^\.\//, ''))
      const include = /"include"\s*:\s*(\[[^\]]*\])/.exec(text)?.[1]
      expect(include, r.path).toBeDefined()
      expect(include, r.path).not.toMatch(/lab|"\.\/?\*\*|"\*\*|"\."/)
    }
  })
  it('the lab\'s own tsconfig type-checks without emitting and is not a project reference', () => {
    const text = read('lab/tsconfig.json')
    expect(text).toMatch(/"noEmit": true/)
    expect(text).toMatch(/"strict": true/)
    expect(text).not.toMatch(/"composite"/)
  })
})
