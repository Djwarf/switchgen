/**
 * Two shot chain, end to end, headless.
 * Not part of the app. A verification harness for the reel feature.
 */
import { FAMILIES, defaultsFor } from '../src/lib/workflows.ts'
import {
  annotatedRef,
  chainFrameOf,
  checkReel,
  instantiateShot,
  shotPlan,
  snapLength,
} from '../src/lib/continuation.ts'
import type { OutputFile } from '../src/lib/comfy.ts'

const COMFY = 'http://127.0.0.1:8188'

const base = FAMILIES.find(f => f.id === 'wan22-5b')
if (!base) throw new Error('wan22-5b not in the registry')

const model = base.models[0]!
const d = defaultsFor(base, model)

const params = {
  ...d,
  model,
  positive: '',
  negative: d.negative ?? '',
  width: 320,
  height: 192,
  length: snapLength(17),
  fps: d.fps ?? 24,
  seed: 12345,
}

const plan = shotPlan({
  base,
  params,
  prefix: 'switchgen/e2e',
  shots: [
    { prompt: 'a red sports car driving along an empty coastal road, sunny', seed: 12345 },
    { prompt: 'the red sports car continues along the coastal road past a lighthouse', seed: 12346 },
  ],
})

console.log('PLAN', JSON.stringify({
  jobs: plan.jobs.map(j => ({
    index: j.index, label: j.label, start: j.start, hops: j.hops,
    prefix: j.outputPrefix, def: j.def.id, notes: j.notes,
  })),
  frames: plan.frames, seconds: plan.seconds, warnings: plan.warnings,
  issues: checkReel(plan.jobs),
}, null, 2))

async function submit(wf: unknown): Promise<string> {
  const r = await fetch(`${COMFY}/prompt`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt: wf, client_id: 'chain-e2e' }),
  })
  const j = await r.json() as any
  if (!r.ok) throw new Error(JSON.stringify(j))
  return j.prompt_id as string
}

async function wait(id: string): Promise<OutputFile[]> {
  const t0 = Date.now()
  for (;;) {
    await new Promise(r => setTimeout(r, 2000))
    const h = await (await fetch(`${COMFY}/history/${id}`)).json() as any
    const rec = h[id]
    if (!rec) {
      if (Date.now() - t0 > 900_000) throw new Error('timed out')
      continue
    }
    const st = rec.status
    if (st && st.completed === false && st.status_str === 'error') {
      throw new Error('ComfyUI reported an error: ' + JSON.stringify(st.messages).slice(0, 2000))
    }
    if (!st?.completed) continue
    // Same rule as lib/comfy.ts: SaveWEBM files arrive under "images".
    const VIDEO_EXT = /\.(webm|mp4|mkv|gif|webp|avi|mov)$/i
    const files: OutputFile[] = []
    for (const [key, out] of Object.entries(rec.outputs ?? {}) as any[]) {
      void key
      for (const bucket of ['images', 'gifs', 'videos'] as const) {
        for (const f of (out as any)[bucket] ?? []) {
          files.push({ ...f, kind: VIDEO_EXT.test(String(f.filename)) ? 'video' : 'image' } as OutputFile)
        }
      }
    }
    console.log(`  took ${((Date.now() - t0) / 1000).toFixed(1)}s`)
    return files
  }
}

const results: { clip: string; frame: string | null }[] = []
let handoff: OutputFile | null = null

// Resume: SHOT1_CLIP / SHOT1_FRAME skip a shot that already rendered.
const skipFirst = process.env.SHOT1_FRAME
if (skipFirst) {
  const [sub, name] = [skipFirst.split('/').slice(0, -1).join('/'), skipFirst.split('/').pop()!]
  handoff = { filename: name, subfolder: sub, type: 'output', kind: 'image' } as OutputFile
  results.push({ clip: process.env.SHOT1_CLIP!, frame: skipFirst })
  console.log('\nresuming: shot 1 already rendered, handoff', annotatedRef(handoff))
}

for (const job of plan.jobs) {
  if (skipFirst && job.index === 0) continue
  console.log(`\n=== ${job.label} (start ${job.start.from}, hops ${job.hops}) ===`)
  const wf = instantiateShot(job, handoff ? annotatedRef(handoff) : null)
  if (job.start.from === 'previous') console.log('  opens on', annotatedRef(handoff!))
  const id = await submit(wf)
  console.log('  prompt', id)
  const files = await wait(id)
  console.log('  files', JSON.stringify(files.map(f => ({ k: f.kind, s: f.subfolder, n: f.filename }))))
  const clip = files.find(f => f.kind === 'video')
  if (!clip) throw new Error('no clip came back')
  handoff = chainFrameOf(files)
  results.push({
    clip: clip.subfolder ? `${clip.subfolder}/${clip.filename}` : clip.filename,
    frame: handoff ? (handoff.subfolder ? `${handoff.subfolder}/${handoff.filename}` : handoff.filename) : null,
  })
}

console.log('\nRESULT ' + JSON.stringify(results))
