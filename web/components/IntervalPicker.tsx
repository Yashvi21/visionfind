"use client";

import { estimateFrames } from "@/lib/format";
import { INTERVALS, type Interval } from "@/lib/types";

interface Props {
  value: Interval;
  onChange: (v: Interval) => void;
  durationSec?: number | null;
  disabled?: boolean;
}

export default function IntervalPicker({ value, onChange, durationSec, disabled }: Props) {
  return (
    <fieldset className="space-y-2" disabled={disabled}>
      <legend className="text-sm font-medium text-slate-700">Sample one frame every</legend>
      <div className="inline-flex rounded-lg border border-slate-300 bg-white p-0.5">
        {INTERVALS.map((n) => (
          <button
            key={n}
            type="button"
            onClick={() => onChange(n)}
            aria-pressed={value === n}
            className={`rounded-md px-4 py-1.5 text-sm font-medium transition-colors disabled:opacity-50 ${
              value === n ? "bg-indigo-600 text-white" : "text-slate-700 hover:bg-slate-100"
            }`}
          >
            {n}s
          </button>
        ))}
      </div>
      <p className="text-xs text-slate-500">
        A smaller interval catches brief moments but creates more frames, so indexing is slower and uses more storage.
        {durationSec ? (
          <>
            {" "}
            This video: about <strong className="text-slate-700">{estimateFrames(durationSec, value)}</strong> frames.
          </>
        ) : null}
      </p>
    </fieldset>
  );
}
