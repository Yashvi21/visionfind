/** mm:ss, or h:mm:ss for an hour or more (matches the ML service's timestamp_label). */
export function formatTimestamp(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const mm = String(m).padStart(2, "0");
  const ss = String(s).padStart(2, "0");
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

export function formatDate(iso: string): string {
  // The API returns UTC; older rows may lack the "Z" suffix.
  const d = new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : `${iso}Z`);
  return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

/** Frames sampled at 0, N, 2N, ... before the end: ceil(duration / N). */
export function estimateFrames(durationSec: number, intervalSec: number): number {
  return Math.max(1, Math.ceil(durationSec / intervalSec - 1e-9));
}
