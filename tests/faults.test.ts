import { describe, expect, it } from 'vitest'
import { ComfyError } from '../src/lib/comfy'
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
