/**
 * One driver at a time: runs/<run>/driver.lock names the process sending a
 * run's work, by pid and the machine's boot id, so a lock left by a process
 * that died, or by a machine that has since restarted, is seen to be stale
 * and taken over, while a live one is never shared. No two runs are made at
 * once either: after taking its own run's lock a driver looks for a live
 * lock on any other run and, if it finds one, lets its own go again.
 */
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import type { LabEnv } from '../core/env.ts'
import { runDir } from '../core/env.ts'

export type LockInfo = { pid: number; boot: string | null; host: string; run: string; at: number }

export class LockHeld extends Error {
  holder: LockInfo
  constructor(holder: LockInfo, message: string) {
    super(message)
    this.name = 'LockHeld'
    this.holder = holder
  }
}

/** Locks this process holds, by file. */
const heldHere = new Map<string, LockInfo>()

/** The kernel's id for this boot, or null where there is none. */
export function machineBoot(): string | null {
  try {
    return fs.readFileSync('/proc/sys/kernel/random/boot_id', 'utf8').trim() || null
  } catch {
    return null
  }
}

function pidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0)
    return true
  } catch (err) {
    return (err as NodeJS.ErrnoException).code === 'EPERM'
  }
}

/** Whether the process a lock names is still this boot's, and still running. */
export function lockAlive(info: LockInfo, file?: string, bootNow: string | null = machineBoot()): boolean {
  if (info.host !== os.hostname()) return true // another machine's: never taken over from here
  if (info.boot && bootNow && info.boot !== bootNow) return false
  if (info.pid === process.pid) {
    const mine = file ? heldHere.get(file) : undefined
    return !!mine && mine.at === info.at
  }
  return pidAlive(info.pid)
}

export function readLock(file: string): LockInfo | null {
  try {
    const v = JSON.parse(fs.readFileSync(file, 'utf8'))
    if (v && typeof v.pid === 'number' && typeof v.run === 'string') {
      return { pid: v.pid, boot: typeof v.boot === 'string' ? v.boot : null, host: String(v.host ?? ''), run: v.run, at: Number(v.at) || 0 }
    }
  } catch {
    /* missing or torn: no holder */
  }
  return null
}

export type Held = { info: LockInfo; file: string; release(): void }

/** Take `file` for `run`, taking over a stale lock. Throws LockHeld while a live process has it. */
export function acquireLock(file: string, run: string, now: () => number = Date.now): Held {
  fs.mkdirSync(path.dirname(file), { recursive: true })
  const info: LockInfo = { pid: process.pid, boot: machineBoot(), host: os.hostname(), run, at: now() }
  for (let attempt = 0; attempt < 3; attempt++) {
    let fd: number
    try {
      fd = fs.openSync(file, 'wx', 0o600)
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== 'EEXIST') throw err
      const holder = readLock(file)
      if (holder && lockAlive(holder, file)) {
        throw new LockHeld(holder, `The ${holder.run} run is already being sent by another lab process (pid ${holder.pid}).`)
      }
      try {
        fs.unlinkSync(file)
      } catch (e) {
        if ((e as NodeJS.ErrnoException).code !== 'ENOENT') throw e
      }
      continue
    }
    try {
      fs.writeSync(fd, JSON.stringify(info) + '\n')
      fs.fsyncSync(fd)
    } finally {
      fs.closeSync(fd)
    }
    heldHere.set(file, info)
    let released = false
    return {
      info,
      file,
      release() {
        if (released) return
        released = true
        heldHere.delete(file)
        const cur = readLock(file)
        if (cur && cur.pid === info.pid && cur.at === info.at) {
          try {
            fs.unlinkSync(file)
          } catch {
            /* already gone */
          }
        }
      },
    }
  }
  throw new Error(`Could not take the lab's lock at ${file}.`)
}

export const driverLockPath = (env: LabEnv, run: string) => path.join(runDir(env, run), 'driver.lock')

/** Every run whose driver is live now, by its lock. */
export function liveDrivers(env: LabEnv): LockInfo[] {
  const runs = path.join(env.labDir, 'runs')
  let names: string[]
  try {
    names = fs.readdirSync(runs)
  } catch {
    return []
  }
  const out: LockInfo[] = []
  for (const name of names) {
    const file = path.join(runs, name, 'driver.lock')
    const info = readLock(file)
    if (info && lockAlive(info, file)) out.push(info)
  }
  return out
}

/**
 * The driver's lock for `run`: its own run's lock, and no other run live.
 * Throws LockHeld naming the run that is.
 */
export function acquireDriverLock(env: LabEnv, run: string, now: () => number = Date.now): Held {
  const held = acquireLock(driverLockPath(env, run), run, now)
  const other = liveDrivers(env).find((i) => i.run !== run)
  if (other) {
    held.release()
    throw new LockHeld(other, `The ${other.run} run is being made now. Pause it before starting another.`)
  }
  return held
}
