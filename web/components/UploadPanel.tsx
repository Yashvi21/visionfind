"use client";

import { useEffect, useRef, useState } from "react";
import IntervalPicker from "@/components/IntervalPicker";
import { api, errorFromResponse, ML_DOWN_MESSAGE } from "@/lib/api";
import { formatBytes, formatTimestamp } from "@/lib/format";
import { DEFAULT_INTERVAL, DEFAULT_MAX_UPLOAD_MB, type Interval, type Video } from "@/lib/types";

type Phase = "idle" | "uploading" | "validating";

/** Read the duration in the browser so we can estimate the frame count before uploading.
 *  Resolves null if the browser can't decode the file (e.g. MPEG-4 Part 2 or HEVC in Chrome);
 *  the server still validates and accepts it. */
function readDuration(file: File): Promise<number | null> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const v = document.createElement("video");
    v.preload = "metadata";
    let settled = false;
    const done = (d: number | null) => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timer);
      URL.revokeObjectURL(url);
      v.removeAttribute("src");
      resolve(d);
    };
    const timer = window.setTimeout(() => done(null), 5000);
    v.onloadedmetadata = () => done(Number.isFinite(v.duration) ? v.duration : null);
    v.onerror = () => done(null);
    v.src = url;
  });
}

export default function UploadPanel({ onUploaded }: { onUploaded: (v: Video) => void }) {
  const [file, setFile] = useState<File | null>(null);
  // undefined = still reading, null = browser couldn't read it, number = seconds
  const [duration, setDuration] = useState<number | null | undefined>(null);
  const [interval, setSampleInterval] = useState<Interval>(DEFAULT_INTERVAL);
  const [phase, setPhase] = useState<Phase>("idle");
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const xhrRef = useRef<XMLHttpRequest | null>(null);
  const pickedRef = useRef<File | null>(null);

  // The server's limit (VS_MAX_UPLOAD_MB) is the single source of truth.
  const [maxMb, setMaxMb] = useState(DEFAULT_MAX_UPLOAD_MB);
  useEffect(() => {
    api.health().then((h) => setMaxMb(h.max_upload_mb)).catch(() => {});
  }, []);

  useEffect(() => () => xhrRef.current?.abort(), []);

  const busy = phase !== "idle";

  async function pick(f: File | undefined) {
    setError(null);
    setNotice(null);
    setDuration(null);
    setFile(null);
    if (!f) return;
    if (!f.name.toLowerCase().endsWith(".mp4")) {
      setError("Only .mp4 files are supported.");
      return;
    }
    if (f.size > maxMb * 1024 * 1024) {
      setError(`This file is ${formatBytes(f.size)}. The limit is ${maxMb} MB.`);
      return;
    }
    setFile(f);
    pickedRef.current = f;
    setDuration(undefined);
    const d = await readDuration(f);
    if (pickedRef.current === f) setDuration(d); // ignore results for a file that was replaced
  }

  function reset() {
    setFile(null);
    setDuration(null);
    setProgress(0);
    setPhase("idle");
    if (inputRef.current) inputRef.current.value = "";
  }

  function upload() {
    if (!file) return;
    setError(null);
    setNotice(null);
    setProgress(0);
    setPhase("uploading");

    const form = new FormData();
    form.append("file", file);
    form.append("sample_interval_sec", String(interval));

    // XMLHttpRequest (not fetch) because it reports upload progress.
    const xhr = new XMLHttpRequest();
    xhrRef.current = xhr;
    xhr.open("POST", "/ml/videos");
    xhr.upload.onprogress = (e) => {
      if (!e.lengthComputable) return;
      const pct = Math.round((e.loaded / e.total) * 100);
      setProgress(pct);
      if (pct >= 100) setPhase("validating");
    };
    xhr.upload.onload = () => setPhase("validating");
    xhr.onload = async () => {
      xhrRef.current = null;
      const res = new Response(xhr.responseText, { status: xhr.status });
      if (xhr.status === 201) {
        const video = (await res.json()) as Video;
        setNotice(`"${video.filename}" uploaded. Indexing has started.`);
        reset();
        onUploaded(video);
      } else {
        setError((await errorFromResponse(res)).message);
        setPhase("idle");
      }
    };
    xhr.onerror = () => {
      xhrRef.current = null;
      setError(ML_DOWN_MESSAGE);
      setPhase("idle");
    };
    xhr.onabort = () => {
      xhrRef.current = null;
      setNotice("Upload cancelled.");
      setPhase("idle");
      setProgress(0);
    };
    xhr.send(form);
  }

  return (
    <section className="space-y-4 rounded-xl border border-slate-200 bg-white p-4 shadow-sm sm:p-5">
      <div>
        <h2 className="font-semibold">Upload a video</h2>
        <p className="text-sm text-slate-500">.mp4 only, up to {maxMb} MB.</p>
      </div>

      <label
        className={`flex cursor-pointer flex-col items-center justify-center gap-1 rounded-lg border-2 border-dashed px-4 py-6 text-center transition-colors ${
          busy ? "pointer-events-none opacity-60" : "border-slate-300 hover:border-indigo-400 hover:bg-indigo-50/40"
        }`}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          if (!busy) pick(e.dataTransfer.files[0]);
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="video/mp4,.mp4"
          className="sr-only"
          disabled={busy}
          onChange={(e) => pick(e.target.files?.[0])}
        />
        {file ? (
          <>
            <span className="max-w-full truncate font-medium">{file.name}</span>
            <span className="text-xs text-slate-500">
              {formatBytes(file.size)}
              {duration === undefined
                ? " · reading duration…"
                : duration === null
                  ? " · duration will be read on upload"
                  : ` · ${formatTimestamp(duration)}`}
            </span>
          </>
        ) : (
          <>
            <span className="font-medium text-indigo-700">Choose an .mp4 file</span>
            <span className="text-xs text-slate-500">or drag it here</span>
          </>
        )}
      </label>

      <IntervalPicker value={interval} onChange={setSampleInterval} durationSec={duration} disabled={busy} />

      {busy && (
        <div className="space-y-1">
          <div className="h-2 overflow-hidden rounded-full bg-slate-100">
            <div className="h-full bg-indigo-600 transition-[width]" style={{ width: `${progress}%` }} />
          </div>
          <p className="text-xs text-slate-500">
            {phase === "uploading" ? `Uploading… ${progress}%` : "Upload complete. Checking the video…"}
          </p>
        </div>
      )}

      {error && <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}
      {notice && <p className="rounded-md bg-emerald-50 px-3 py-2 text-sm text-emerald-700">{notice}</p>}

      <div className="flex gap-2">
        <button
          type="button"
          onClick={upload}
          disabled={!file || busy}
          className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Upload and index
        </button>
        {phase === "uploading" && (
          <button
            type="button"
            onClick={() => xhrRef.current?.abort()}
            className="rounded-lg border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
          >
            Cancel
          </button>
        )}
      </div>
    </section>
  );
}
