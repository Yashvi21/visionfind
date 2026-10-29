"use client";

import { useState } from "react";
import ResultCard from "@/components/ResultCard";
import { api } from "@/lib/api";
import type { SearchResult } from "@/lib/types";

interface Props {
  videoId: string;
  onSeek: (seconds: number) => void;
}

export default function InVideoSearch({ videoId, onSeek }: Props) {
  const [query, setQuery] = useState("");
  const [submitted, setSubmitted] = useState("");
  const [results, setResults] = useState<SearchResult[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run(e: React.FormEvent) {
    e.preventDefault();
    const q = query.trim();
    if (!q) return;
    setLoading(true);
    setError(null);
    try {
      setResults(await api.search({ query: q, top_k: 10, video_id: videoId }));
      setSubmitted(q);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="space-y-3 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <h2 className="font-semibold">Search within this video</h2>
      <form onSubmit={run} className="flex flex-col gap-2 sm:flex-row">
        <label htmlFor="in-video-q" className="sr-only">Describe the scene</label>
        <input
          id="in-video-q"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          maxLength={500}
          placeholder="Describe a scene in this video"
          className="min-w-0 flex-1 rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200"
        />
        <button
          type="submit"
          disabled={!query.trim() || loading}
          className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50"
        >
          {loading ? "Searching…" : "Find"}
        </button>
      </form>

      {error && <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {results && results.length > 0 && (
        <>
          <p className="text-xs text-slate-500">
            {results.length} moment{results.length === 1 ? "" : "s"} for &lsquo;{submitted}&rsquo;. Click one to jump there.
          </p>
          <div className="-mx-1 flex snap-x gap-2 overflow-x-auto pb-2">
            {results.map((r) => (
              <div key={r.frame_id} className="w-40 shrink-0 snap-start sm:w-48">
                <ResultCard result={r} showVideoName={false} compact onSelect={() => onSeek(r.timestamp_sec)} />
              </div>
            ))}
          </div>
        </>
      )}
      {results && results.length === 0 && (
        <p className="text-sm text-slate-500">No matching frames for &lsquo;{submitted}&rsquo;. Try shorter or different wording.</p>
      )}
    </section>
  );
}
