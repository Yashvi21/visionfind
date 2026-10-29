"use client";

import Link from "next/link";
import { useState } from "react";
import IntervalPicker from "@/components/IntervalPicker";
import Modal from "@/components/Modal";
import { api, isActive } from "@/lib/api";
import { formatBytes, formatDate, formatTimestamp } from "@/lib/format";
import type { Interval, Video, VideoStage } from "@/lib/types";

const STAGES: { key: VideoStage; label: string }[] = [
  { key: "extracting_frames", label: "Extracting frames" },
  { key: "embedding", label: "Embedding frames" },
  { key: "transcribing", label: "Transcribing audio" },
  { key: "building_index", label: "Saving index" },
];

export function StatusBadge({ video }: { video: Video }) {
  const styles = {
    queued: "bg-slate-100 text-slate-700",
    processing: "bg-amber-50 text-amber-800 ring-amber-200",
    ready: "bg-emerald-50 text-emerald-700 ring-emerald-200",
    failed: "bg-red-50 text-red-700 ring-red-200",
  }[video.status];
  const label = { queued: "Queued", processing: "Processing", ready: "Ready", failed: "Failed" }[video.status];
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset ring-slate-200 ${styles}`}>
      {isActive(video) && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current" />}
      {label}
    </span>
  );
}

function StageProgress({ video }: { video: Video }) {
  const idx = STAGES.findIndex((s) => s.key === video.stage);
  const label = video.status === "queued" || idx < 0 ? "Waiting in queue" : STAGES[idx].label;
  return (
    <div className="space-y-1">
      <div className="flex gap-1" aria-hidden>
        {STAGES.map((s, i) => (
          <div key={s.key} className={`h-1 flex-1 rounded-full ${i < idx ? "bg-indigo-600" : i === idx ? "animate-pulse bg-indigo-400" : "bg-slate-200"}`} />
        ))}
      </div>
      <p className="text-xs text-slate-500">
        {label}
        {idx >= 0 && ` (step ${idx + 1} of ${STAGES.length})`}
      </p>
    </div>
  );
}

interface Props {
  video: Video;
  onChanged: () => void;
}

export default function VideoCard({ video, onChanged }: Props) {
  const [dialog, setDialog] = useState<"none" | "reindex" | "delete">("none");
  const [interval, setSampleInterval] = useState<Interval>(video.sample_interval_sec as Interval);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const active = isActive(video);

  function openDialog(d: "reindex" | "delete") {
    setError(null);
    setDialog(d);
  }

  async function run(action: () => Promise<unknown>) {
    setBusy(true);
    setError(null);
    try {
      await action();
      setDialog("none");
      onChanged();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const btn = "rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50";

  return (
    <article className="flex flex-col gap-3 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-start justify-between gap-2">
        <h3 className="min-w-0 truncate font-medium" title={video.filename}>
          {video.filename}
        </h3>
        <StatusBadge video={video} />
      </div>

      <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs text-slate-500">
        <dt>Duration</dt>
        <dd className="text-right text-slate-700">{formatTimestamp(video.duration_sec)}</dd>
        <dt>Uploaded</dt>
        <dd className="text-right text-slate-700">{formatDate(video.created_at)}</dd>
        <dt>Frames</dt>
        <dd className="text-right text-slate-700">
          {video.status === "ready" || video.frame_count ? video.frame_count : "–"} · every {video.sample_interval_sec}s
        </dd>
        <dt>Size</dt>
        <dd className="text-right text-slate-700">{formatBytes(video.size_bytes)}</dd>
      </dl>

      {active && <StageProgress video={video} />}
      {video.status === "failed" && (
        <p className="rounded-md bg-red-50 px-3 py-2 text-xs text-red-700">{video.error_message ?? "Indexing failed."}</p>
      )}
      {error && dialog === "none" && (
        <p className="rounded-md bg-red-50 px-3 py-2 text-xs text-red-700">{error}</p>
      )}

      <div className="mt-auto flex flex-wrap gap-2 pt-1">
        <Link href={`/videos/${video.id}`} className="rounded-md bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-indigo-700">
          Open
        </Link>
        {video.status === "failed" ? (
          <button type="button" className={btn} disabled={busy} onClick={() => run(() => api.reindexVideo(video.id))}>
            {busy ? "Retrying…" : "Retry"}
          </button>
        ) : (
          <button type="button" className={btn} disabled={active} onClick={() => openDialog("reindex")}
            title={active ? "Wait until indexing finishes" : undefined}>
            Re-index
          </button>
        )}
        <button type="button" className={`${btn} text-red-700 hover:bg-red-50`} onClick={() => openDialog("delete")}>
          Delete
        </button>
      </div>

      <Modal open={dialog === "reindex"} onClose={() => setDialog("none")} title="Re-index video">
        <p className="text-sm text-slate-600">
          Re-indexing removes this video&apos;s current frames and search data, then processes it again. It won&apos;t be searchable until it finishes.
        </p>
        <IntervalPicker value={interval} onChange={setSampleInterval} durationSec={video.duration_sec} />
        {error && <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}
        <div className="flex justify-end gap-2">
          <button type="button" className={btn} onClick={() => setDialog("none")}>Cancel</button>
          <button type="button" disabled={busy} onClick={() => run(() => api.reindexVideo(video.id, interval))}
            className="rounded-md bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50">
            {busy ? "Starting…" : "Re-index"}
          </button>
        </div>
      </Modal>

      <Modal open={dialog === "delete"} onClose={() => setDialog("none")} title="Delete video?">
        <p className="text-sm text-slate-600">
          <strong className="break-all">{video.filename}</strong> and all its frames, transcript and search data will be permanently deleted.
        </p>
        {error && <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}
        <div className="flex justify-end gap-2">
          <button type="button" className={btn} onClick={() => setDialog("none")}>Cancel</button>
          <button type="button" disabled={busy} onClick={() => run(() => api.deleteVideo(video.id))}
            className="rounded-md bg-red-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-red-700 disabled:opacity-50">
            {busy ? "Deleting…" : "Delete"}
          </button>
        </div>
      </Modal>
    </article>
  );
}
