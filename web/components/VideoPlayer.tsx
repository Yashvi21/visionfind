"use client";

import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react";
import { formatTimestamp } from "@/lib/format";

export interface PlayerHandle {
  /** Seek to `seconds` (clamped), show "Jumped to mm:ss" and bring the player into view. */
  seek: (seconds: number) => void;
}

interface Props {
  src: string;
  /** Seek here once metadata has loaded (from ?t=). */
  initialTime?: number | null;
}

const VideoPlayer = forwardRef<PlayerHandle, Props>(function VideoPlayer({ src, initialTime }, ref) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [ready, setReady] = useState(false);
  const [jumpedTo, setJumpedTo] = useState<number | null>(null);
  const [playError, setPlayError] = useState<string | null>(null);
  const pendingSeek = useRef<number | null>(initialTime ?? null);
  const hideTimer = useRef<number | undefined>(undefined);

  const applySeek = useCallback((seconds: number) => {
    const v = videoRef.current;
    if (!v) return;
    const max = Number.isFinite(v.duration) ? Math.max(0, v.duration - 0.05) : seconds;
    const t = Math.min(Math.max(0, seconds), max);
    v.currentTime = t;
    setJumpedTo(t);
    window.clearTimeout(hideTimer.current);
    hideTimer.current = window.setTimeout(() => setJumpedTo(null), 4000);
  }, []);

  const seek = useCallback(
    (seconds: number) => {
      if (!ready) {
        pendingSeek.current = seconds; // applied in onLoadedMetadata
      } else {
        applySeek(seconds);
      }
      const rect = wrapperRef.current?.getBoundingClientRect();
      if (rect && (rect.top < 0 || rect.bottom > window.innerHeight)) {
        wrapperRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    },
    [ready, applySeek],
  );

  useImperativeHandle(ref, () => ({ seek }), [seek]);

  // A new ?t= while already on the page (e.g. back/forward) seeks again.
  useEffect(() => {
    if (initialTime != null && ready) applySeek(initialTime);
  }, [initialTime, ready, applySeek]);

  useEffect(() => () => window.clearTimeout(hideTimer.current), []);

  if (playError) {
    return (
      <div className="flex aspect-video w-full flex-col items-center justify-center gap-2 rounded-xl bg-slate-900 p-6 text-center text-slate-200">
        <p className="font-medium">This video can&apos;t be played here.</p>
        <p className="max-w-md text-sm text-slate-400">{playError}</p>
      </div>
    );
  }

  return (
    <div ref={wrapperRef} className="relative scroll-mt-20">
      <video
        ref={videoRef}
        src={src}
        controls
        playsInline
        preload="metadata"
        className="aspect-video w-full rounded-xl bg-black"
        onLoadedMetadata={() => {
          setReady(true);
          if (pendingSeek.current != null) {
            applySeek(pendingSeek.current);
            pendingSeek.current = null;
          }
        }}
        onError={async () => {
          // Browsers report a 404 the same way as an unsupported codec, so ask the server first.
          let status = 0;
          try {
            status = (await fetch(src, { headers: { Range: "bytes=0-0" }, cache: "no-store" })).status;
          } catch {
            /* network error: ML service unreachable */
          }
          if (status === 404) setPlayError("The original video file is missing from storage. Delete this video and upload it again.");
          else if (status === 0 || status >= 500) setPlayError("The video couldn't be loaded because the ML service isn't reachable. Make sure it is running on port 8000.");
          else
            setPlayError(
              "Its video codec isn't supported by this browser (common with older MPEG-4 Part 2 or HEVC files). Search, Ask and timestamps still work; re-encoding the file to H.264 makes it playable everywhere.",
            );
        }}
      />
      {jumpedTo != null && (
        <div role="status" className="pointer-events-none absolute left-3 top-3 rounded-md bg-slate-900/85 px-2.5 py-1 text-sm text-white shadow">
          Jumped to <span className="font-mono">{formatTimestamp(jumpedTo)}</span>
        </div>
      )}
    </div>
  );
});

export default VideoPlayer;
