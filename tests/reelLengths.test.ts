import { describe, expect, it } from 'vitest'
import { lengthChips, lengthsFor } from '../src/components/reel/lengths'
import type { Hardware } from '../src/lib/hardware'
import { FAMILIES } from '../src/lib/workflows'

// The shot lengths the reel offers, each with the verdict a shot that long
// gets. The chips used to offer 5 s and 7 s on the image-to-video 14B pair at
// its own size, which the plan then refused under the strip.
const GiB = 1024 ** 3
const choice = (id: string) => ({ def: FAMILIES.find((f) => f.id === id)!, frames: { min: 1, max: 9999, step: 4 } })
const reel = (width: number, height: number) => ({ fps: 16, length: 49, width, height })
const refusedAt = (id: string, width: number, height: number, hardware: Hardware | null = null) =>
  lengthsFor(choice(id), reel(width, height), hardware)
    .filter((c) => c.refused !== null)
    .map((c) => c.frames)

describe('the shot lengths on offer', () => {
  it('marks the lengths the image-to-video pair would refuse at its own size, with the reason', () => {
    const list = lengthsFor(choice('wan22-14b-i2v'), reel(832, 480), null)
    expect(list.map((c) => c.frames)).toEqual([33, 49, 81, 113])
    expect(list.filter((c) => c.refused === null).map((c) => c.frames)).toEqual([33, 49])
    for (const c of list.filter((x) => x.refused !== null)) expect(c.refused!.startsWith('Too large for memory')).toBe(true)
  })

  it('refuses fewer lengths at a smaller frame, and on the text-to-video pair', () => {
    expect(refusedAt('wan22-14b-i2v', 480, 480)).toEqual([113])
    expect(refusedAt('wan22-14b-t2v', 832, 480)).toEqual([113])
  })

  it('refuses nothing on a machine with clearly more memory than the one measured', () => {
    const roomy: Hardware = {
      cpu: { cores: 8, model: 'test' },
      ram: { total: 64 * GiB, free: 64 * GiB },
      gpu: { name: 'test', vramTotal: 16 * GiB, vramUsed: 0, vramFree: 16 * GiB },
      disk: null,
      platform: 'test',
    }
    expect(refusedAt('wan22-14b-i2v', 832, 480, roomy)).toEqual([])
  })

  it('greys out exactly the refused chips, with the reason as the chip\'s title', () => {
    const list = lengthsFor(choice('wan22-14b-i2v'), reel(832, 480), null)
    const chips = lengthChips(list, 16)
    expect(chips.map((c) => c.disabled)).toEqual(list.map((c) => c.refused !== null))
    for (const [i, c] of list.entries()) {
      if (c.refused) expect(chips[i]!.title).toBe(c.refused)
      else expect(chips[i]!.title).toBe(`${c.frames} frames`)
    }
  })
})
