import type { Plugin } from 'vite'
/** The shared archive: records beside the files, one copy for every device. */
export function switchgenArchive(): Plugin

/** What the archive holds of the record naming a file, for the queue's check of an answer from ComfyUI's cache. */
export type RecordNaming = {
  id: string
  no: number
  /** Filed after the fact by the recovery pass, not by the desk that made it. */
  recovered: boolean
  promptId: string | null
  /** 0 when the run was not timed. */
  durationMs: number
}

/**
 * The archive as the queue on the server (server/runner.mjs) uses it, in the
 * same process and under the same lock as the routes.
 */
export type ArchiveApi = {
  /** Whether this process holds the archive lock, taking it when it is free. */
  holds(): boolean
  /** The record naming `rel` (`subfolder/filename` under the outputs root), a desk's before a recovered one. Rejects while another server holds the archive. */
  recordNaming(rel: string): Promise<RecordNaming | null>
  /**
   * File a record under its own id, once: `existed` when that id is already
   * here, `removed` when the reader removed it. Rejects while another server
   * holds the archive, and with 'not a record' when it lacks an id, a time or
   * a file. The answer comes before the write is on disk.
   */
  fileOnce(record: object): Promise<{ entryId: string; no: number; existed: boolean } | { removed: true }>
  /** Write now; true once every change made before the call is on disk. */
  durable(): Promise<boolean>
  /** Mark files the queue is filing as filed in GET /api/outputs, so the recovery pass leaves them alone. */
  claim(rels: string[]): void
  unclaim(rels: string[]): void
  readonly outputsRoot: string
  readonly archiveFile: string
}

export const archiveApi: ArchiveApi
