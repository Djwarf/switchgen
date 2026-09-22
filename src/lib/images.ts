/**
 * Read a picture's real pixel size off the decoded image, never off a record.
 *
 * A record's width and height are what the composer asked for; a hires pass
 * upscales after the record is written, so the two can disagree by half. Every
 * surface that maps a stroke or a crop onto pixels measures the file.
 */
export function measureImage(url: string): Promise<{ width: number; height: number } | null> {
  return new Promise((resolve) => {
    const img = new Image()
    img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight })
    img.onerror = () => resolve(null)
    img.src = url
  })
}
