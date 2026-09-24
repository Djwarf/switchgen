import { readFileSync, rmSync } from 'node:fs'
import path from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { tempRoots } from './http'

/**
 * What the README says of the behaviour this suite pins elsewhere, read as
 * text, as the launcher's advice is in launcher.test.ts. Each sentence here
 * was missing or said something else once; it is the reader's only account
 * of what happens on a second device or after a reload. Lines are wrapped in
 * the file, so runs of spaces and newlines are read as one space.
 */
const raw = readFileSync(path.resolve(import.meta.dirname, '..', 'README.md'), 'utf8')
const readme = raw.replace(/\s+/g, ' ')

/** The temporary folders a test here made, removed once the file is done. */
const temps: string[] = []
afterAll(() => {
  for (const dir of temps) rmSync(dir, { recursive: true, force: true })
})

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

describe('the README on the queue on the server', () => {
  it('says where work that waits now waits, and that a closed page or a locked phone stops nothing', () => {
    expect(readme).toContain('### Work that waits')
    expect(readme).toContain('Work that waits its turn waits on the SwitchGen server, in a queue of its own')
    expect(readme).toContain('Closing the page, reloading it or locking the phone stops nothing')
    expect(readme).toContain('If the SwitchGen server itself stops, the work waits and carries on when the server is back.')
  })

  it('says how a heavy job is sent, what a lost one holds, and that a send is never made twice', () => {
    expect(readme).toContain('checks that the queue is still empty, and only then is sent')
    expect(readme).toContain('nothing heavy is sent until you send the waiting work or call it off')
    expect(readme).toContain('marked as not sent, never sent a second time')
    expect(readme).toContain('A restart of the SwitchGen server alone holds nothing')
  })

  it('lists what the queue changes', () => {
    for (const change of [
      'One heavy lane covers the Video desk and the Reel on every device',
      'takes turns between batches, reels and clips',
      "A job's time is ComfyUI's own",
      'One batch of pictures is made at a time',
      'After the machine restarts, waiting work is held until you say.',
    ]) expect(readme).toContain(change)
  })

  it('keeps the account of a server without the queue, where the desks work as before', () => {
    expect(readme).toContain('On a server without the queue, or with SWITCHGEN_RUNNER=off, the desks work as before')
    expect(readme).toContain('Three kinds of work then wait in the page, not in ComfyUI')
  })

  it('says the queue\'s work survives a change of address between http and https', () => {
    expect(readme).toContain('so is the work waiting in the queue on the server')
  })

  it('lists every route of the queue and every setting it reads', () => {
    for (const route of ['`GET /api/runner`', '`GET /api/runner/stream`', '`GET /api/runner/jobs/:id`', '`/jobs/:id/preview`', '`POST /api/runner/groups`', '`POST /api/runner/jobs/:id/stop`', '`/groups/:id/stop`', '`POST /api/runner/lane`', '`POST /api/runner/dismiss`']) {
      expect(readme, route).toContain(route)
    }
    for (const name of ['SWITCHGEN_RUNNER', 'SWITCHGEN_RUNNER_DIR', 'SWITCHGEN_RUNNER_DESKS', 'SWITCHGEN_RUNNER_SETTLE_MS']) {
      expect(readme, name).toContain(`| \`${name}\` |`)
    }
  })

  it('says the queue files its own work once, and the recovery pass leaves it to it', () => {
    expect(readme).toContain('Work the queue on the server makes is filed by the server itself, once, under the job\'s own id')
  })
})

describe('the README on the notice about held work, and the queue off', () => {
  it('says the notice shows on every room and device, why, what it counts, and its two answers', () => {
    expect(readme).toContain('every room shows the same notice, on every device')
    expect(readme).toContain('the machine restarted, or the queue was off while work waited')
    expect(readme).toContain('Send them and Call them off')
  })

  it('says turning the queue off loses nothing, shows the saved list, and holds it when the queue runs again', () => {
    expect(readme).toContain('Turning the queue off loses nothing it holds')
    expect(readme).toContain('still answers that list, read-only')
    expect(readme).toContain('So is work that waited while the queue was off, once it runs again.')
    const lane = raw.split('\n').find((l) => l.startsWith('| `POST /api/runner/lane` |'))
    expect(lane).toContain('a time with the queue off')
  })

  it('says the word answers the hold the page showed, and no later one', () => {
    expect(readme).toContain('Either answer is for the hold the page showed')
    expect(readme).toContain('A notice left from an earlier hold cannot answer a later one')
    const lane = raw.split('\n').find((l) => l.startsWith('| `POST /api/runner/lane` |'))
    expect(lane).toContain('refused when that hold has changed since')
  })

  it('says a desk leaves work the queue keeps while it is off to the queue, with no Stop, and its press free', () => {
    expect(readme).toContain('offers no Stop for it, and leaves its press free for work the page sends itself')
    expect(readme).toContain('the server refuses every change to that list until a queue runs the folder again')
  })
})

describe('the README as a file', () => {
  it('keeps its prose within 80 columns; only table rows and code run longer', () => {
    let code = false
    const long: string[] = []
    raw.split('\n').forEach((line, i) => {
      if (line.startsWith('```')) {
        code = !code
        return
      }
      if (code || line.startsWith('|')) return
      if (line.length > 80) long.push(`${i + 1}: ${line.length}`)
    })
    expect(long).toEqual([])
  })
})

describe('the README on the queue\'s folder', () => {
  it('says it belongs to one archive, names it as the server does, and says what its lock is', async () => {
    expect(readme).toContain("The queue's folder belongs to one archive")
    expect(readme).toContain('`runner` beside the default `archive.json`')
    expect(readme).toContain('(`archive-b.json.runner` for `archive-b.json`)')
    expect(readme).toContain('the `lock` file in that folder')
    // The server module reads its roots as it loads: temporary ones, never the author's.
    temps.push(tempRoots().root)
    const { runnerDirFor } = await import('../server/runner.mjs')
    expect(path.basename(runnerDirFor('/x/.switchgen/archive.json'))).toBe('runner')
    expect(path.basename(runnerDirFor('/x/.switchgen/archive-b.json'))).toBe('archive-b.json.runner')
  })

  it('gives a reel pass no page has taken in the time and count the queue keeps it for', () => {
    expect(readme).toContain('kept apart, for 30 days and up to 2000 such jobs')
    // The queue's own numbers, read from its source, since they are its own.
    const engine = readFileSync(path.resolve(import.meta.dirname, '..', 'server', 'runner', 'engine.mjs'), 'utf8')
    const value = (name: string) => {
      const expr = new RegExp(`const ${name} = ([0-9_ *]+)\\n`).exec(engine)?.[1]
      expect(expr, name).toBeTruthy()
      return expr!.split('*').reduce((n, f) => n * Number(f.trim().replaceAll('_', '')), 1)
    }
    expect(value('KEEP_PASS_MS') / 86_400_000).toBe(30)
    expect(value('KEEP_PASS_JOBS')).toBe(2000)
    expect(value('KEEP_ENDED_MS') / 3_600_000).toBe(48)
    expect(value('KEEP_ENDED')).toBe(500)
    expect(readme).toContain('forgets a job 48 hours after it ended, or sooner once 500 have ended')
  })
})

describe('.env.example', () => {
  const example = readFileSync(path.resolve(import.meta.dirname, '..', '.env.example'), 'utf8')

  it('lists every setting the README\'s table does, as the README says it does', () => {
    expect(readme).toContain('`.env.example` lists them all')
    const table = raw.slice(raw.indexOf('### Environment'), raw.indexOf('## Usage'))
    const names = [...new Set([...table.matchAll(/`([A-Z][A-Z0-9_]+)`/g)].map((m) => m[1]!))]
    expect(names.length).toBeGreaterThan(20)
    for (const name of names) expect(example, name).toMatch(new RegExp(`^#?${name}=`, 'm'))
  })

  it('says what the queue had saved is kept, read-only, while it is off, and held once it is on again', () => {
    // Comment lines read as one run of text.
    const said = example.replace(/^# ?/gm, '').replace(/\s+/g, ' ')
    expect(said).toContain('stays listed, read-only, as waiting')
    expect(said).toContain('when the queue is on again it is held until you send it or call it off')
  })

  it('offers the queue\'s settings commented out, at their defaults', () => {
    for (const line of ['#SWITCHGEN_RUNNER=off', '#SWITCHGEN_RUNNER_DIR=', '#SWITCHGEN_RUNNER_DESKS=video,images,reel,lab', '#SWITCHGEN_RUNNER_SETTLE_MS=1000']) {
      expect(example.split('\n'), line).toContain(line)
    }
  })

  it('gives the queue\'s desks at the default the queue itself has, the lab\'s among them', () => {
    // The queue's own list, read from its source: a copy of the old default in
    // .env silently turns off a desk added since, and the lab then refuses to start.
    const engine = readFileSync(path.resolve(import.meta.dirname, '..', 'server', 'runner', 'engine.mjs'), 'utf8')
    const list = /^export const DESKS = \[([^\]]*)\]/m.exec(engine)?.[1]
    expect(list).toBeTruthy()
    const desks = [...list!.matchAll(/'([a-z]+)'/g)].map((m) => m[1]).join(',')
    expect(desks).toBe('video,images,reel,lab')
    expect(example.split('\n')).toContain(`#SWITCHGEN_RUNNER_DESKS=${desks}`)
    expect(raw).toContain(`| \`SWITCHGEN_RUNNER_DESKS\` | \`${desks}\` |`)
    expect(readme).toContain("`lab` is the model lab's own desk")
  })
})
