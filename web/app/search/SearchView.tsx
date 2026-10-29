"use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import ResultCard from "@/components/ResultCard";
import { api } from "@/lib/api";
import type { SearchResult, Video } from "@/lib/types";

const TOP_K_OPTIONS = [5, 10, 20] as const;
const DEFAULT_TOP_K = 10;
const EXAMPLES = [
  "a person writing on a whiteboard",
  "a red car beside a person",
  "people sitting around a table",
  "a laptop screen showing code",
  "trees and a blue sky",
];

type State =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "done"; results: SearchResult[]; query: string }
  | { kind: "error"; message: string };

function parseTopK(raw: string | null): number {
  const n = Number(raw);
  return (TOP_K_OPTIONS as readonly number[]).includes(n) ? n : DEFAULT_TOP_K;
}

export default function SearchView() {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();

  // The URL is the source of truth for the submitted search.
  const q = params.get("q")?.trim() ?? "";
  const topK = parseTopK(params.get("k"));
  const videoFilter = params.get("video") ?? "";

  const [draft, setDraft] = useState(q);
  const [videos, setVideos] = useState<Video[] | null>(null);
  const [videosError, setVideosError] = useState<string | null>(null);
  const [state, setState] = useState<State>({ kind: "idle" });

  useEffect(() => setDraft(q), [q]);

  useEffect(() => {
    api
      .listVideos()
      .then((v) => setVideos(v.filter((x) => x.status === "ready")))
      .catch((e: Error) => setVideosError(e.message));
  }, []);

  const filterIsValid = !videoFilter || !!videos?.some((v) => v.id === videoFilter);

  useEffect(() => {
    if (!q || videos === null || videos.length === 0) {
      setState({ kind: "idle" });
      return;
    }
    let cancelled = false;
    setState({ kind: "loading" });
    api
      .search({ query: q, top_k: topK, video_id: filterIsValid && videoFilter ? videoFilter : undefined })
      .then((results) => !cancelled && setState({ kind: "done", results, query: q }))
      .catch((e: Error) => !cancelled && setState({ kind: "error", message: e.message }));
    return () => {
      cancelled = true;
    };
  }, [q, topK, videoFilter, filterIsValid, videos]);

  function update(next: { q?: string; k?: number; video?: string }) {
    const sp = new URLSearchParams(params.toString());
    const set = (key: string, value: string | undefined) => (value ? sp.set(key, value) : sp.delete(key));
    if (next.q !== undefined) set("q", next.q.trim());
    if (next.k !== undefined) set("k", next.k === DEFAULT_TOP_K ? undefined : String(next.k));
    if (next.video !== undefined) set("video", next.video);
    const qs = sp.toString();
    router.push(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
  }

  const noneReady = videos !== null && videos.length === 0;
  const disabled = videos === null || noneReady;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Search</h1>
        <p className="text-sm text-slate-500">Describe what you see in the scene. VisionFind finds the matching moments in your videos.</p>
      </div>

      {videosError && <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{videosError}</div>}

      {noneReady && (
        <div className="rounded-xl border border-dashed border-slate-300 bg-white px-6 py-10 text-center">
          <p className="font-medium">No videos are ready to search yet</p>
          <p className="mx-auto mt-1 max-w-md text-sm text-slate-500">
            Upload a video in the Library and wait until it shows <strong>Ready</strong>. Search only covers fully indexed videos.
          </p>
          <Link href="/" className="mt-4 inline-block rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700">
            Go to Library
          </Link>
        </div>
      )}

      <form
        className={`space-y-3 rounded-xl border border-slate-200 bg-white p-4 shadow-sm ${disabled ? "opacity-60" : ""}`}
        onSubmit={(e) => {
          e.preventDefault();
          if (draft.trim()) update({ q: draft });
        }}
      >
        <fieldset disabled={disabled} className="space-y-3">
          <div className="flex flex-col gap-2 sm:flex-row">
            <label htmlFor="q" className="sr-only">Describe the scene</label>
            <input
              id="q"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              maxLength={500}
              placeholder="e.g. a person writing on a whiteboard"
              className="min-w-0 flex-1 rounded-lg border border-slate-300 px-3 py-2 outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200"
            />
            <button
              type="submit"
              disabled={!draft.trim()}
              className="rounded-lg bg-indigo-600 px-5 py-2 font-medium text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Search
            </button>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs text-slate-500">Try:</span>
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                type="button"
                onClick={() => {
                  setDraft(ex);
                  update({ q: ex });
                }}
                className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs text-slate-700 hover:border-indigo-300 hover:bg-indigo-50"
              >
                {ex}
              </button>
            ))}
          </div>

          <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-sm">
            <label className="flex items-center gap-2">
              <span className="text-slate-600">Results</span>
              <select
                value={topK}
                onChange={(e) => update({ k: Number(e.target.value) })}
                className="rounded-md border border-slate-300 bg-white px-2 py-1"
              >
                {TOP_K_OPTIONS.map((k) => (
                  <option key={k} value={k}>Top {k}</option>
                ))}
              </select>
            </label>
            <label className="flex min-w-0 items-center gap-2">
              <span className="text-slate-600">Video</span>
              <select
                value={filterIsValid ? videoFilter : ""}
                onChange={(e) => update({ video: e.target.value })}
                className="min-w-0 max-w-[16rem] truncate rounded-md border border-slate-300 bg-white px-2 py-1"
              >
                <option value="">All ready videos</option>
                {videos?.map((v) => (
                  <option key={v.id} value={v.id}>{v.filename}</option>
                ))}
              </select>
            </label>
          </div>
        </fieldset>
      </form>

      {videoFilter && videos !== null && !filterIsValid && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-2 text-sm text-amber-800">
          The video in this link is no longer available or isn&apos;t ready, so all ready videos are being searched.
        </div>
      )}

      <p className="text-xs text-slate-500">
        Search matches what is visible in individual frames: objects, scenes, text and how things are arranged. It does not reliably
        understand actions over time (like &ldquo;a person gets into a car&rdquo;) and does not identify people by face.
      </p>

      {state.kind === "loading" && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
          {Array.from({ length: Math.min(topK, 8) }, (_, i) => (
            <div key={i} className="aspect-video animate-pulse rounded-lg bg-slate-200/70" />
          ))}
        </div>
      )}

      {state.kind === "error" && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{state.message}</div>
      )}

      {state.kind === "done" && state.results.length > 0 && (
        <section className="space-y-3">
          <h2 className="text-sm text-slate-600">
            {state.results.length} result{state.results.length === 1 ? "" : "s"} for <strong className="text-slate-900">&lsquo;{state.query}&rsquo;</strong>
          </h2>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
            {state.results.map((r) => (
              <ResultCard key={r.frame_id} result={r} href={`/videos/${r.video_id}?t=${r.timestamp_sec}`} />
            ))}
          </div>
        </section>
      )}

      {state.kind === "done" && state.results.length === 0 && (
        <div className="rounded-xl border border-slate-200 bg-white px-6 py-8 text-center">
          <p className="font-medium">No matching frames for &lsquo;{state.query}&rsquo;</p>
          <p className="mx-auto mt-1 max-w-md text-sm text-slate-500">
            Try a shorter description, name the main object or setting (&ldquo;whiteboard&rdquo;, &ldquo;red car&rdquo;), or use different
            words.{videoFilter && " You can also search all videos instead of just one."}
          </p>
        </div>
      )}
    </div>
  );
}
