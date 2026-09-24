import { describe, expect, it } from 'vitest'
import { ComfyError, LostJob } from '../src/lib/comfy'
import { faultBody, faultOf, faultTitle, faultWhere } from '../src/lib/faults'

describe('faults', () => {
  it('carries ComfyUI\'s node and per-input detail through', () => {
    const err = new ComfyError('Prompt rejected', {
      node: '11',
      nodeType: 'LoadImage',
      nodeErrors: { '11': { errors: [{ message: 'Invalid image file', details: "'missing.png'" }] } },
    })
    const f = faultOf(err)
    expect(f.node).toBe('11')
    expect(f.detail).toBe("Invalid image file: 'missing.png'")
    expect(faultTitle(f)).toBe('That job was rejected')
    expect(faultBody(f)).toContain('Invalid image file')
    expect(faultWhere(f)).toBe('The trouble is in LoadImage (node 11).')
  })

  it('recognises a stopped job and a memory failure', () => {
    expect(faultTitle(faultOf(new ComfyError('stopped', { cancelled: true })))).toBe('Correction')
    const oom = faultOf(new Error('CUDA out of memory. Tried to allocate 2 GiB'))
    expect(faultTitle(oom)).toBe('The card ran out of memory')
    expect(faultWhere(oom)).toBeNull()
  })

  it('reads a plain object the way the desks used to throw them', () => {
    const f = faultOf({ message: 'lost', cancelled: false })
    expect(f.message).toBe('lost')
    expect(faultTitle(f)).toBe('That job did not finish')
  })

  it('calls a job that broke with no detail and no node one that did not finish, and points at nothing', () => {
    const f = faultOf(new ComfyError('bad frame'))
    expect(f.detail).toBeNull()
    expect(f.nodeType).toBeNull()
    expect(faultTitle(f)).toBe('That job did not finish')
    expect(faultBody(f)).toBe('bad frame')
    expect(faultBody(f)).not.toContain('highlighted setting')
    expect(faultWhere(f)).toBeNull()
  })

  it('names the node of a job that broke while running without calling it rejected', () => {
    const f = faultOf(new ComfyError('bad frame', { node: '11', nodeType: 'LoadImage' }))
    expect(faultTitle(f)).toBe('That job did not finish')
    expect(faultWhere(f)).toBe('The trouble is in LoadImage (node 11).')
  })
})

describe('a lost job whose desk has held what waits behind it', () => {
  // The Video desk holds its lane after a heavy clip is lost, because what
  // took ComfyUI down will likely take the next one down too. Saying the desk
  // is free to try again then was false, and invited the same loss.
  const lostPlain = faultOf(new LostJob('We lost track of this job.', 'p1'))
  const lostMayExist = faultOf(new LostJob('Ended without its result.', 'p2', { mayExist: true }))

  it('does not invite another go, and says the rest is held for the reader', () => {
    const said = faultBody(lostPlain, { held: true })
    expect(said).toContain('We lost track of this job.')
    expect(said).toContain('held until you send it or call it off')
    expect(said).not.toContain('The desk is free again')
    expect(said).not.toContain('try again')
  })

  it('still asks the reader to look first when the job may have finished', () => {
    const said = faultBody(lostMayExist, { held: true })
    expect(said).toContain('Look there before you run it again')
    expect(said).toContain('held until you send it')
    expect(said).not.toContain('The desk is free again')
  })

  it('says what it always said when nothing is held', () => {
    expect(faultBody(lostPlain)).toContain('The desk is free again, so you can try again.')
    expect(faultBody(lostMayExist)).toContain('look there before you run it again')
    expect(faultBody(lostPlain, { held: false })).toBe(faultBody(lostPlain))
  })

  it('changes nothing for a fault that is not a lost job', () => {
    expect(faultBody(faultOf({ cancelled: true, lost: true, message: 'x' }), { held: true })).toBe('Job stopped. Nothing was saved.')
    expect(faultBody(faultOf(new Error('boom')), { held: true })).toBe('boom')
  })

  it('reads a plain lost object the way a desk may still throw one', () => {
    const said = faultBody(faultOf({ lost: true, message: 'm' }), { held: true })
    expect(said).not.toContain('The desk is free again')
    expect(said).toContain('held until you send it or call it off')
    expect(faultBody(faultOf({ lost: true, mayExist: true, message: 'm' }), { held: true })).toContain('Look there')
  })
})
