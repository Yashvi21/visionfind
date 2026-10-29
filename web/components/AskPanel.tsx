"use client";

import { Fragment, useEffect, useRef, useState } from "react";
import AnswerText from "@/components/AnswerText";
import { MAX_HISTORY_TURNS, MAX_QUESTION_CHARS, type AskEvent, type SourceChunk, type SourceFrame } from "@/lib/ask";

interface Turn {
  id: number;
  question: string;
  answer: string;
  frames: SourceFrame[];
  chunks: SourceChunk[];
  status: "thinking" | "streaming" | "done" | "error" | "stopped";
  error?: string;
}

interface Props {
  videoId: string;
  ready: boolean;
  durationSec: number;
  onSeek: (seconds: number) => void;
}

const EXAMPLES = ["What is shown at the start?", "Summarize what happens in this video.", "What text is visible on screen?"];

export default function AskPanel({ videoId, ready, durationSec, onSeek }: Props) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const busy = turns.some((t) => t.status === "thinking" || t.status === "streaming");

  useEffect(() => () => abortRef.current?.abort(), []);
  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "nearest" });
  }, [turns]);

  const patch = (id: number, p: Partial<Turn> | ((t: Turn) => Partial<Turn>)) =>
    setTurns((ts) => ts.map((t) => (t.id === id ? { ...t, ...(typeof p === "function" ? p(t) : p) } : t)));

  async function ask(questionRaw: string) {
    const question = questionRaw.trim();
    if (!question || busy || !ready) return;
    const id = Date.now();
    // Only completed turns are sent as history, and only the last few.
    const history = turns
      .filter((t) => t.status === "done" && t.answer)
      .slice(-MAX_HISTORY_TURNS)
      .map((t) => ({ question: t.question, answer: t.answer }));
    setTurns((ts) => [...ts, { id, question, answer: "", frames: [], chunks: [], status: "thinking" }]);
    setDraft("");

    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ videoId, question, history }),
        signal: ctrl.signal,
      });
      if (!res.ok || !res.body) {
        const body = await res.json().catch(() => null);
        patch(id, { status: "error", error: body?.error ?? `Request failed (${res.status}).` });
        return;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.trim()) continue;
          const ev = JSON.parse(line) as AskEvent;
          if (ev.type === "sources") patch(id, { frames: ev.frames, chunks: ev.chunks });
          else if (ev.type === "delta") patch(id, (t) => ({ answer: t.answer + ev.text, status: "streaming" }));
          else if (ev.type === "error") patch(id, { status: "error", error: ev.message });
          else if (ev.type === "done") patch(id, (t) => (t.status === "error" ? {} : { status: "done" }));
        }
      }
      patch(id, (t) => (t.status === "streaming" || t.status === "thinking" ? { status: "done" } : {}));
    } catch (e) {
      if (ctrl.signal.aborted) patch(id, { status: "stopped" });
      else patch(id, { status: "error", error: "Couldn't reach the server. Is the web app running?" });
      void e;
    } finally {
      if (abortRef.current === ctrl) abortRef.current = null;
    }
  }

  return (
    <section className="flex flex-col rounded-xl border border-slate-200 bg-white shadow-sm lg:max-h-[calc(100vh-7rem)]">
      <div className="border-b border-slate-100 px-4 py-3">
        <h2 className="font-semibold">Ask about this video</h2>
        <p className="text-xs text-slate-500">
          Answers come from the most relevant frames and transcript, via Gemini. Click a [mm:ss] citation to jump there.
        </p>
      </div>

      <div className="min-h-0 flex-1 space-y-5 overflow-y-auto px-4 py-4">
        {!ready && <p className="text-sm text-slate-500">Available once this video has finished indexing.</p>}
        {ready && turns.length === 0 && (
          <div className="space-y-2">
            <p className="text-sm text-slate-500">Try asking:</p>
            <div className="flex flex-wrap gap-2">
              {EXAMPLES.map((ex) => (
                <button key={ex} type="button" onClick={() => ask(ex)}
                  className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-left text-xs text-slate-700 hover:border-indigo-300 hover:bg-indigo-50">
                  {ex}
                </button>
              ))}
            </div>
          </div>
        )}

        {turns.map((t) => (
          <article key={t.id} className="space-y-2">
            <p className="ml-auto w-fit max-w-[90%] rounded-2xl rounded-br-sm bg-indigo-600 px-3 py-2 text-sm text-white">{t.question}</p>
            <div className="space-y-2 text-sm text-slate-800">
              {t.status === "thinking" && <p className="animate-pulse text-slate-500">Looking at the video…</p>}
              {t.answer && <AnswerText text={t.answer} durationSec={durationSec} onSeek={onSeek} />}
              {t.status === "streaming" && <span className="inline-block h-4 w-1.5 animate-pulse bg-slate-400 align-middle" />}
              {t.status === "stopped" && <p className="text-xs text-slate-500">Stopped.</p>}
              {t.status === "error" && <p className="rounded-md bg-red-50 px-3 py-2 text-red-700">{t.error}</p>}

              {t.frames.length > 0 && t.status !== "thinking" && (
                <div className="space-y-1 pt-1">
                  <p className="text-xs text-slate-500">Frames used</p>
                  <div className="flex gap-1.5 overflow-x-auto pb-1">
                    {t.frames.map((f) => (
                      <button key={f.frame_id} type="button" onClick={() => onSeek(f.timestamp_sec)} title={`Jump to ${f.timestamp_label}`}
                        className="relative w-20 shrink-0 overflow-hidden rounded-md ring-1 ring-slate-200 hover:ring-2 hover:ring-indigo-400">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img src={f.thumbnail_url} alt={`Frame at ${f.timestamp_label}`} className="aspect-video w-full object-cover" />
                        <span className="absolute bottom-0.5 left-0.5 rounded bg-slate-900/80 px-1 font-mono text-[10px] text-white">{f.timestamp_label}</span>
                      </button>
                    ))}
                  </div>
                  {t.chunks.length > 0 && (
                    <p className="text-xs text-slate-500">
                      Transcript used:{" "}
                      {t.chunks.map((c, i) => (
                        <Fragment key={i}>
                          {i > 0 && ", "}
                          <button type="button" onClick={() => onSeek(c.start_sec)} className="font-mono text-indigo-700 hover:underline">
                            {c.start_label}–{c.end_label}
                          </button>
                        </Fragment>
                      ))}
                    </p>
                  )}
                </div>
              )}
            </div>
          </article>
        ))}
        <div ref={endRef} />
      </div>

      <form
        className="flex items-end gap-2 border-t border-slate-100 p-3"
        onSubmit={(e) => {
          e.preventDefault();
          ask(draft);
        }}
      >
        <label htmlFor="ask-q" className="sr-only">Your question</label>
        <textarea
          id="ask-q"
          rows={2}
          value={draft}
          maxLength={MAX_QUESTION_CHARS}
          disabled={!ready}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              ask(draft);
            }
          }}
          placeholder={ready ? "Ask a question (Enter to send)" : "Not ready yet"}
          className="min-w-0 flex-1 resize-none rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 disabled:bg-slate-50"
        />
        {busy ? (
          <button type="button" onClick={() => abortRef.current?.abort()}
            className="rounded-lg border border-slate-300 px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50">
            Stop
          </button>
        ) : (
          <button type="submit" disabled={!ready || !draft.trim()}
            className="rounded-lg bg-indigo-600 px-3 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50">
            Ask
          </button>
        )}
      </form>
    </section>
  );
}
