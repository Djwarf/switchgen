/**
 * Fetching a file into the models tree through the local server.
 *
 * POST /api/download answers with an event stream rather than JSON, so this
 * reads the body as a stream instead of using EventSource, which cannot POST.
 * The server keeps the partial file when the request is aborted, so a
 * cancelled fetch resumes rather than starting again. The LoRA picker used to
 * own this reader; the vision tagger and the catalogue need it too.
 */

export type FetchProgress = {
  state: 'starting' | 'downloading' | 'done' | 'error' | 'cancelled'
  /** 0 to 1. Zero until the server has the content length. */
  pct: number
  done: number
  total: number
  /** Bytes per second, as aria2c reports it. */
  speed: number
  etaSec: number | null
  error: string | null
}

export type DownloadSpec = {
  url: string
  filename: string
  /** Destination relative to the models root, folder or full path. */
  dest: string
}

export async function downloadFile(
  spec: DownloadSpec,
  onProgress: (p: FetchProgress) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch('/api/download', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url: spec.url, filename: spec.filename, dest: spec.dest }),
    signal,
  })

  // A rejected plan answers JSON, not a stream: bad URL, a clashing download,
  // a destination outside the models root, a disk too full to take it.
  if (!res.ok || !res.body) {
    let message = `HTTP ${res.status}`
    try {
      const body = (await res.json()) as { error?: string }
      if (body.error) message = body.error
    } catch {
      /* not JSON either */
    }
    throw new Error(message)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let failure: string | null = null

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const blocks = buffer.split('\n\n')
    buffer = blocks.pop() ?? ''
    for (const block of blocks) {
      let event = 'message'
      let data = ''
      for (const line of block.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim()
        else if (line.startsWith('data:')) data += line.slice(5).trim()
      }
      if (!data) continue
      let payload: Record<string, unknown>
      try {
        payload = JSON.parse(data) as Record<string, unknown>
      } catch {
        continue
      }
      if (event === 'start' || event === 'progress' || event === 'file' || event === 'skip') {
        onProgress({
          state: event === 'start' ? 'starting' : event === 'progress' ? 'downloading' : 'done',
          pct: typeof payload.pct === 'number' ? payload.pct : 0,
          done: typeof payload.done === 'number' ? payload.done : 0,
          total: typeof payload.total === 'number' ? payload.total : 0,
          speed: typeof payload.speed === 'number' ? payload.speed : 0,
          etaSec: typeof payload.etaSec === 'number' ? payload.etaSec : null,
          error: null,
        })
      } else if (event === 'error') {
        failure = typeof payload.error === 'string' ? payload.error : 'the download failed'
        onProgress({ state: 'error', pct: 0, done: 0, total: 0, speed: 0, etaSec: null, error: failure })
      }
    }
  }

  if (failure) throw new Error(failure)
}

/**
 * Fetch the tagger's files, one after the other, reporting the whole as one
 * progress figure. `install` is what /api/vision/capabilities hands back when
 * the tagger is missing: verified URLs and byte counts.
 */
export async function installFiles(
  files: readonly { filename: string; dest: string; url: string; sizeBytes: number }[],
  onProgress: (p: FetchProgress) => void,
  signal?: AbortSignal,
): Promise<void> {
  const total = files.reduce((n, f) => n + f.sizeBytes, 0)
  let before = 0
  for (const f of files) {
    await downloadFile({ url: f.url, filename: f.filename, dest: f.dest }, (p) => {
      const done = before + p.done
      onProgress({ ...p, done, total, pct: total ? done / total : p.pct })
    }, signal)
    before += f.sizeBytes
  }
  onProgress({ state: 'done', pct: 1, done: total, total, speed: 0, etaSec: null, error: null })
}
