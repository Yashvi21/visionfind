import { GoogleGenAI } from "@google/genai";
import { MAX_HISTORY_TURNS, MAX_QUESTION_CHARS, type AskEvent, type AskRequest, type ChatTurn } from "@/lib/ask";
import { formatTimestamp } from "@/lib/format";
import { buildContents, RateLimitError, startStreamWithRetry, SYSTEM_INSTRUCTION } from "@/lib/gemini";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const ML_URL = process.env.ML_URL ?? "http://localhost:8000";
const MAX_FRAMES = 6; // free-tier friendly: at most 6 images (512 px) per question
const K_CHUNKS = 4;

const ML_DOWN = "Can't reach the ML service. Make sure it is running on port 8000.";
const RATE_LIMITED = "Gemini's rate limit was reached (free tier). Please try again in a minute.";
const OVERLOADED = "Gemini is overloaded right now (high demand). Please try again in a minute.";

interface RetrieveResponse {
  video_id: string;
  video_filename: string;
  duration_sec: number;
  has_transcript: boolean;
  frames: { frame_id: number; timestamp_sec: number; timestamp_label: string; thumbnail_url: string }[];
  chunks: { start_sec: number; end_sec: number; start_label: string; end_label: string; text: string }[];
}

function jsonError(message: string, status: number) {
  return Response.json({ error: message }, { status });
}

async function ml<T>(path: string, init?: RequestInit): Promise<{ ok: true; data: T } | { ok: false; status: number; message: string }> {
  let res: Response;
  try {
    res = await fetch(`${ML_URL}${path}`, { cache: "no-store", ...init });
  } catch {
    return { ok: false, status: 503, message: ML_DOWN };
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    return { ok: false, status: res.status, message: typeof body?.detail === "string" ? body.detail : `ML service error (${res.status})` };
  }
  return { ok: true, data: (await res.json()) as T };
}

function sanitizeHistory(raw: unknown): ChatTurn[] {
  if (!Array.isArray(raw)) return [];
  return raw
    .filter((t): t is ChatTurn => typeof t?.question === "string" && typeof t?.answer === "string" && t.answer.trim() !== "")
    .slice(-MAX_HISTORY_TURNS)
    .map((t) => ({ question: t.question.slice(0, MAX_QUESTION_CHARS), answer: t.answer.slice(0, 4000) }));
}

export async function POST(req: Request) {
  const apiKey = process.env.GEMINI_API_KEY;
  const model = process.env.GEMINI_MODEL;
  if (!apiKey) return jsonError("GEMINI_API_KEY is not set. Add it to web/.env.local and restart the web server.", 500);
  if (!model) return jsonError("GEMINI_MODEL is not set. Add a Flash model name to web/.env.local and restart the web server.", 500);

  let body: AskRequest;
  try {
    body = await req.json();
  } catch {
    return jsonError("Invalid JSON body.", 400);
  }
  const question = typeof body?.question === "string" ? body.question.trim() : "";
  if (!body?.videoId || typeof body.videoId !== "string") return jsonError("videoId is required.", 400);
  if (!question) return jsonError("Please enter a question.", 400);
  if (question.length > MAX_QUESTION_CHARS) return jsonError(`Questions are limited to ${MAX_QUESTION_CHARS} characters.`, 400);
  const history = sanitizeHistory(body.history);

  const t0 = Date.now();
  // 1. Retrieve the most relevant frames + transcript chunks (no re-processing of the video).
  const retrieved = await ml<RetrieveResponse>("/retrieve", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ video_id: body.videoId, question, k_frames: MAX_FRAMES, k_chunks: K_CHUNKS }),
  });
  if (!retrieved.ok) return jsonError(retrieved.message, retrieved.status);
  const ctx = retrieved.data;
  const frames = ctx.frames.slice(0, MAX_FRAMES);

  // 2. Fetch the stored 512 px JPEGs as base64.
  const images = await Promise.all(frames.map((f) => ml<{ data: string }>(`/frames/${f.frame_id}/base64`)));
  const failed = images.find((r) => !r.ok);
  if (failed && !failed.ok) return jsonError(failed.message, failed.status);

  const contents = buildContents({
    filename: ctx.video_filename,
    durationLabel: formatTimestamp(ctx.duration_sec),
    frames: frames.map((f, i) => ({ label: f.timestamp_label, base64: (images[i] as { ok: true; data: { data: string } }).data.data })),
    chunks: ctx.chunks.map((c) => ({ startLabel: c.start_label, endLabel: c.end_label, text: c.text })),
    hasTranscript: ctx.has_transcript,
    question,
    history,
  });

  const tRetrieved = Date.now();
  // 3. Call Gemini with retries on 429/503, before anything is streamed to the client.
  const ai = new GoogleGenAI({ apiKey });
  let started;
  try {
    started = await startStreamWithRetry(
      ai,
      { model, contents, config: { systemInstruction: SYSTEM_INSTRUCTION, temperature: 0.2, abortSignal: req.signal } },
      { onRetry: (n, err) => console.warn(`[ask] Gemini busy/rate-limited, retry ${n}/3:`, (err as Error)?.message?.slice(0, 200)) },
    );
  } catch (err) {
    if (req.signal.aborted) return new Response(null, { status: 499 }); // client disconnected; nothing to report
    if (err instanceof RateLimitError) {
      console.warn(`[ask] giving up after retries (${err.overloaded ? "overloaded/timeout" : "rate limit"}): ${err.message.slice(0, 160)}`);
      return jsonError(err.overloaded ? OVERLOADED : RATE_LIMITED, err.overloaded ? 503 : 429);
    }
    console.error("[ask] Gemini error:", err);
    const status = (err as { status?: number })?.status;
    if (status === 400 || status === 403 || status === 401) {
      return jsonError(`Gemini rejected the request (${status}). Check GEMINI_API_KEY and GEMINI_MODEL in web/.env.local.`, 502);
    }
    if (status === 404) return jsonError(`Gemini model "${model}" was not found. Check GEMINI_MODEL in web/.env.local.`, 502);
    return jsonError("Gemini couldn't answer right now. Please try again.", 502);
  }

  const tFirst = Date.now();
  // 4. Stream: sources first (so thumbnails show immediately), then text deltas.
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const send = (e: AskEvent) => controller.enqueue(encoder.encode(JSON.stringify(e) + "\n"));
      send({
        type: "sources",
        frames: frames.map(({ frame_id, timestamp_sec, timestamp_label, thumbnail_url }) => ({ frame_id, timestamp_sec, timestamp_label, thumbnail_url })),
        chunks: ctx.chunks.map(({ start_sec, end_sec, start_label, end_label }) => ({ start_sec, end_sec, start_label, end_label })),
        hasTranscript: ctx.has_transcript,
      });
      let gotText = false;
      try {
        if (started.first?.text) {
          gotText = true;
          send({ type: "delta", text: started.first.text });
        }
        for await (const chunk of started.rest) {
          if (chunk.text) {
            gotText = true;
            send({ type: "delta", text: chunk.text });
          }
        }
        if (!gotText) send({ type: "error", message: "Gemini returned an empty answer (it may have been blocked by safety filters). Try rephrasing." });
        send({ type: "done" });
        console.info(
          `[ask] video=${body.videoId} frames=${frames.length} chunks=${ctx.chunks.length} history=${history.length} ` +
            `retrieve=${tRetrieved - t0}ms first_token=${tFirst - tRetrieved}ms total=${Date.now() - t0}ms`,
        );
      } catch (err) {
        if (!req.signal.aborted) {
          console.error("[ask] stream error:", err);
          send({ type: "error", message: "The answer was interrupted. Please try again." });
        }
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: { "content-type": "application/x-ndjson; charset=utf-8", "cache-control": "no-store", "x-accel-buffering": "no" },
  });
}
