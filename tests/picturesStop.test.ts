import { beforeEach, describe, expect, it, vi } from 'vitest'
import { STOPPING, waitingLine } from '../src/components/compose/RunButton'

// The Pictures desk's Stop as the section bar and the running slug call it.
// ComfyUI is stood in for; nothing is sent anywhere.
const m = vi.hoisted(() => ({ run: vi.fn(), cancelJob: vi.fn() }))
vi.mock('../src/lib/comfy', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../src/lib/comfy')>()),
  run: m.run,
  cancelJob: m.cancelJob,
}))

beforeEach(() => {
  for (const f of Object.values(m)) f.mockReset()
  vi.resetModules()
})

describe('Stop from outside the Pictures desk', () => {
  it('does nothing when the desk has nothing on the press', async () => {
    const desk = await import('../src/routes/Pictures')
    const before = desk.pressSnapshot()
    desk.stopPress()
    await new Promise((resolve) => setTimeout(resolve, 10))
    expect(m.cancelJob).not.toHaveBeenCalled()
    expect(desk.pressSnapshot()).toBe(before)
    expect(desk.pressSnapshot().job).toBeNull()
  })

  it('stops a batch of three while the first picture runs: cancels it and sends no more', async () => {
    const desk = await import('../src/routes/Pictures')
    let rejectFirst: (e: unknown) => void = () => {}
    m.run.mockImplementation((_graph: unknown, onProgress: (ev: unknown) => void) => {
      onProgress({ phase: 'queued', promptId: 'p1' })
      return new Promise((_resolve, reject) => {
        rejectFirst = reject
      })
    })
    m.cancelJob.mockImplementation(async (id: string) => {
      rejectFirst(Object.assign(new Error('Job stopped. Nothing was saved.'), { cancelled: true }))
      return id
    })
    const one = (label: string) => ({ graph: {}, label }) as unknown as import('../src/routes/Pictures').RunPlan
    desk.startRuns([one('first'), one('second'), one('third')])
    await new Promise((resolve) => setTimeout(resolve, 10))
    expect(desk.pressSnapshot().job?.status).toBe('queued')

    desk.stopPress()
    await new Promise((resolve) => setTimeout(resolve, 10))
    expect(m.cancelJob).toHaveBeenCalledWith('p1')
    expect(m.run).toHaveBeenCalledTimes(1)
    expect(desk.pressSnapshot().job?.status).toBe('cancelled')
  })

  it('ends the batch when Stop lands just as the first picture saves, with nothing left to cancel', async () => {
    const desk = await import('../src/routes/Pictures')
    let finishFirst: () => void = () => {}
    const saved = { filename: 'one.png', subfolder: '', type: 'output', kind: 'image' }
    m.run.mockImplementation((_graph: unknown, onProgress: (ev: unknown) => void) => {
      onProgress({ phase: 'queued', promptId: `p${m.run.mock.calls.length}` })
      return new Promise((resolve) => {
        finishFirst = () => resolve([saved])
      })
    })
    // ComfyUI has already finished the prompt, so the cancel finds nothing.
    m.cancelJob.mockResolvedValue(undefined)
    const composition = {
      desk: 'images', mode: 't2i', familyId: 'sdxl-illustrious', model: 'm.safetensors',
      prompt: 'a lighthouse at dusk', negative: null, seed: 1, steps: 20, cfg: 5,
      sampler: 'euler', scheduler: 'normal', width: 1024, height: 1024, noLora: false,
    }
    const one = (label: string) =>
      ({ graph: {}, label, composition, seed: 1, familyLabel: 'Illustrious', modelLabel: 'M', variant: null, passes: {}, loras: [] }) as unknown as import('../src/routes/Pictures').RunPlan
    desk.startRuns([one('first'), one('second'), one('third')])
    await new Promise((resolve) => setTimeout(resolve, 10))

    desk.stopPress()
    finishFirst()
    await new Promise((resolve) => setTimeout(resolve, 20))
    expect(m.cancelJob).toHaveBeenCalledWith('p1')
    expect(m.run).toHaveBeenCalledTimes(1)
    expect(desk.pressSnapshot().job?.status).toBe('done')
  })
})

describe('the line under a batch once Stop is pressed', () => {
  // Stop drops the rest of the batch at once, but the picture on the press can
  // take seconds to let go. Until it does, the job still counts the pictures
  // behind it, and the line under the button said they waited to be sent.
  it('says no pictures wait while the one on the press is stopping', async () => {
    const desk = await import('../src/routes/Pictures')
    m.run.mockImplementation((_graph: unknown, onProgress: (ev: unknown) => void) => {
      onProgress({ phase: 'queued', promptId: 'p1' })
      return new Promise(() => {})
    })
    // A cancel that has not landed yet, as on a long node.
    m.cancelJob.mockImplementation(() => new Promise(() => {}))
    const one = (label: string) => ({ graph: {}, label }) as unknown as import('../src/routes/Pictures').RunPlan
    desk.startRuns([one('first'), one('second'), one('third')])
    await new Promise((resolve) => setTimeout(resolve, 10))
    expect(waitingLine(desk.pressSnapshot().job, true)).toContain('2 more pictures')

    desk.stopPress()
    await new Promise((resolve) => setTimeout(resolve, 10))
    const job = desk.pressSnapshot().job!
    expect(job.status).toBe('queued')
    expect(job.stage).toBe(STOPPING)
    expect([job.index, job.total]).toEqual([1, 3])
    expect(waitingLine(job, true)).toBeNull()
    expect(m.run).toHaveBeenCalledTimes(1)
  })
})
