import Link from "next/link";
import type { SearchResult } from "@/lib/types";

interface Props {
  result: SearchResult;
  /** Link to the video page (search grid) ... */
  href?: string;
  /** ... or seek the current player (in-video strip). */
  onSelect?: () => void;
  showVideoName?: boolean;
  compact?: boolean;
}

export default function ResultCard({ result, href, onSelect, showVideoName = true, compact = false }: Props) {
  const body = (
    <>
      <div className="relative aspect-video overflow-hidden rounded-lg bg-slate-200">
        {/* Plain <img>: thumbnails are already 512px JPEGs served by the ML service. */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={result.thumbnail_url}
          alt={`Frame at ${result.timestamp_label}${showVideoName ? ` in ${result.video_filename}` : ""}`}
          loading="lazy"
          className="h-full w-full object-cover transition-transform group-hover:scale-[1.03]"
        />
        <span className="absolute bottom-1.5 left-1.5 rounded bg-slate-900/80 px-1.5 py-0.5 font-mono text-xs text-white">
          {result.timestamp_label}
        </span>
      </div>
      <div className={`flex items-baseline gap-2 ${showVideoName ? "justify-between" : "justify-end"} ${compact ? "mt-1" : "mt-2"}`}>
        {showVideoName && (
          <span className="min-w-0 truncate text-sm font-medium" title={result.video_filename}>
            {result.video_filename}
          </span>
        )}
        <span className="shrink-0 text-xs text-slate-500" title="Cosine similarity between your text and this frame (CLIP)">
          similarity <span className="font-mono text-slate-700">{result.similarity.toFixed(2)}</span>
        </span>
      </div>
    </>
  );

  const cls = "group block rounded-xl p-1.5 text-left outline-none transition-colors hover:bg-white focus-visible:ring-2 focus-visible:ring-indigo-500";
  if (href) {
    return (
      <Link href={href} className={cls}>
        {body}
      </Link>
    );
  }
  return (
    <button type="button" onClick={onSelect} className={`${cls} w-full`}>
      {body}
    </button>
  );
}
