"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import UploadPanel from "@/components/UploadPanel";
import VideoCard from "@/components/VideoCard";
import { api, isActive } from "@/lib/api";
import type { Video } from "@/lib/types";

const POLL_MS = 2000;

export default function LibraryPage() {
  const [videos, setVideos] = useState<Video[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setVideos(await api.listVideos());
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  // Poll only while something is queued or processing (or while the service is unreachable).
  const shouldPoll = !!error || !!videos?.some(isActive);
  useEffect(() => {
    if (!shouldPoll) return;
    const id = window.setInterval(load, POLL_MS);
    return () => window.clearInterval(id);
  }, [shouldPoll, load]);

  const readyCount = videos?.filter((v) => v.status === "ready").length ?? 0;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h1 className="text-2xl font-semibold">Library</h1>
          <p className="text-sm text-slate-500">Upload videos, then search them by describing a scene or ask questions about them.</p>
        </div>
        {readyCount > 0 && (
          <Link href="/search" className="text-sm font-medium text-indigo-700 hover:underline">
            Search {readyCount} ready video{readyCount === 1 ? "" : "s"} →
          </Link>
        )}
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[22rem_minmax(0,1fr)]">
        <div className="lg:sticky lg:top-20 lg:self-start">
          <UploadPanel onUploaded={load} />
        </div>

        <section className="space-y-3">
          {error && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              {error} <span className="text-red-500">Retrying every few seconds…</span>
            </div>
          )}

          {videos === null && !error && (
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              {[0, 1].map((i) => (
                <div key={i} className="h-48 animate-pulse rounded-xl bg-slate-200/60" />
              ))}
            </div>
          )}

          {videos?.length === 0 && (
            <div className="rounded-xl border border-dashed border-slate-300 bg-white px-6 py-12 text-center">
              <p className="font-medium">No videos yet</p>
              <p className="mx-auto mt-1 max-w-md text-sm text-slate-500">
                Upload an .mp4 to get started. Each video is indexed once (frames, visual embeddings and transcript) and
                becomes searchable when its status shows <strong>Ready</strong>.
              </p>
            </div>
          )}

          {videos && videos.length > 0 && (
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              {videos.map((v) => (
                <VideoCard key={v.id} video={v} onChanged={load} />
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
