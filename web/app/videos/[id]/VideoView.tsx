"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import AskPanel from "@/components/AskPanel";
import InVideoSearch from "@/components/InVideoSearch";
import { StatusBadge } from "@/components/VideoCard";
import VideoPlayer, { type PlayerHandle } from "@/components/VideoPlayer";
import { api, ApiError, isActive } from "@/lib/api";
import { formatDate, formatTimestamp } from "@/lib/format";
import type { Video } from "@/lib/types";

type Load = { kind: "loading" } | { kind: "missing" } | { kind: "error"; message: string } | { kind: "ok"; video: Video };

function parseT(raw: string | null): number | null {
  if (raw == null) return null;
  const t = Number(raw);
  return Number.isFinite(t) && t >= 0 ? t : null;
}

export default function VideoView({ id }: { id: string }) {
  const params = useSearchParams();
  const t = parseT(params.get("t"));
  const [load, setLoad] = useState<Load>({ kind: "loading" });
  const playerRef = useRef<PlayerHandle>(null);

  const fetchVideo = useCallback(async () => {
    try {
      setLoad({ kind: "ok", video: await api.getVideo(id) });
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) setLoad({ kind: "missing" });
      else setLoad({ kind: "error", message: (e as Error).message });
    }
  }, [id]);

  useEffect(() => {
    fetchVideo();
  }, [fetchVideo]);

  // Keep the status fresh while indexing so search/ask unlock when it's ready.
  const active = load.kind === "ok" && isActive(load.video);
  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(fetchVideo, 2000);
    return () => window.clearInterval(timer);
  }, [active, fetchVideo]);

  const back = (
    <Link href="/" className="text-sm text-indigo-700 hover:underline">
      ← Library
    </Link>
  );

  if (load.kind === "loading") {
    return <div className="aspect-video max-w-4xl animate-pulse rounded-xl bg-slate-200/60" />;
  }
  if (load.kind === "missing") {
    return (
      <div className="space-y-4">
        {back}
        <div className="rounded-xl border border-slate-200 bg-white px-6 py-10 text-center">
          <p className="font-medium">Video not found</p>
          <p className="mt-1 text-sm text-slate-500">It may have been deleted. Search results that pointed to it are no longer valid.</p>
        </div>
      </div>
    );
  }
  if (load.kind === "error") {
    return (
      <div className="space-y-4">
        {back}
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{load.message}</div>
      </div>
    );
  }

  const { video } = load;
  const ready = video.status === "ready";

  return (
    <div className="space-y-5">
      {back}
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <h1 className="truncate text-xl font-semibold sm:text-2xl" title={video.filename}>
            {video.filename}
          </h1>
          <p className="text-sm text-slate-500">
            {formatTimestamp(video.duration_sec)} · {ready ? `${video.frame_count} frames` : video.status === "failed" ? "not indexed" : "indexing"} · every {video.sample_interval_sec}s · uploaded{" "}
            {formatDate(video.created_at)}
          </p>
        </div>
        <StatusBadge video={video} />
      </div>

      {video.status === "failed" ? (
        <div className="rounded-xl border border-red-200 bg-red-50 px-5 py-6">
          <p className="font-medium text-red-800">Indexing failed for this video</p>
          <p className="mt-1 text-sm text-red-700">{video.error_message ?? "Unknown error."}</p>
          <p className="mt-3 text-sm text-red-700">
            Use <strong>Retry</strong> in the <Link href="/" className="underline">Library</Link> to index it again.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-5 lg:grid-cols-[minmax(0,1fr)_22rem]">
          <div className="min-w-0 space-y-5">
            <VideoPlayer ref={playerRef} src={`/ml/videos/${video.id}/file`} initialTime={t} />
            {ready ? (
              <InVideoSearch videoId={video.id} onSeek={(s) => playerRef.current?.seek(s)} />
            ) : (
              <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
                This video is still being indexed. Search and questions will be available when it&apos;s ready; this page updates automatically.
              </div>
            )}
          </div>

          <aside className="min-w-0 lg:sticky lg:top-20 lg:self-start">
            <AskPanel videoId={video.id} ready={ready} durationSec={video.duration_sec} onSeek={(s) => playerRef.current?.seek(s)} />
          </aside>
        </div>
      )}
    </div>
  );
}
