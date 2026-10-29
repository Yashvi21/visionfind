// Server-only: imported solely by app/api/ask/route.ts. Never import this from a client component.
import { ApiError, GoogleGenAI, type Content, type GenerateContentResponse, type Part } from "@google/genai";
import type { ChatTurn } from "./ask";

export const SYSTEM_INSTRUCTION = `You answer questions about ONE video using ONLY the material in the user's latest message: still frames sampled from the video (each labelled "Frame at mm:ss") and transcript excerpts (each labelled with its [start–end] time range).

Rules:
- Use only what is visible in the frames or said in the transcript. Do not use outside knowledge and do not guess.
- Cite every moment you rely on as [mm:ss], using the timestamps given. For transcript excerpts use the start time, or a time inside the range.
- If the answer isn't visible in the frames or mentioned in the transcript, say so clearly (for example: "I can't see or hear that in the parts of the video I was given"). You may say what IS shown instead.
- The frames are samples, not the whole video: never claim something doesn't happen in the video, only that it isn't in the material provided.
- Be concise: a few sentences or a short list.`;

export interface PromptFrame {
  label: string;
  base64: string;
}

export interface PromptChunk {
  startLabel: string;
  endLabel: string;
  text: string;
}

export function buildContents(opts: {
  filename: string;
  durationLabel: string;
  frames: PromptFrame[];
  chunks: PromptChunk[];
  hasTranscript: boolean;
  question: string;
  history: ChatTurn[];
}): Content[] {
  const contents: Content[] = [];
  // Earlier turns are sent as text only (no images) to stay within free-tier limits.
  for (const turn of opts.history) {
    contents.push({ role: "user", parts: [{ text: turn.question }] });
    contents.push({ role: "model", parts: [{ text: turn.answer }] });
  }

  const parts: Part[] = [
    {
      text: `Video: "${opts.filename}" (length ${opts.durationLabel}). ${
        opts.frames.length
          ? `Here are ${opts.frames.length} frames from the moments most relevant to the question, in time order.`
          : "No frames were retrieved."
      }`,
    },
  ];
  for (const f of opts.frames) {
    parts.push({ text: `Frame at ${f.label}` });
    parts.push({ inlineData: { mimeType: "image/jpeg", data: f.base64 } });
  }
  if (opts.chunks.length) {
    parts.push({
      text: "Transcript excerpts:\n" + opts.chunks.map((c) => `[${c.startLabel}–${c.endLabel}] ${c.text}`).join("\n"),
    });
  } else {
    parts.push({
      text: opts.hasTranscript
        ? "No transcript excerpts matched this question."
        : "No transcript is available for this video (no audio or no speech).",
    });
  }
  parts.push({ text: `Question: ${opts.question}` });
  contents.push({ role: "user", parts });
  return contents;
}

/** Thrown when retries are exhausted. `overloaded` = 503 (model busy) rather than 429 (quota). */
export class RateLimitError extends Error {
  overloaded: boolean;
  constructor(message: string, overloaded: boolean) {
    super(message);
    this.overloaded = overloaded;
  }
}

/** 429 RESOURCE_EXHAUSTED (rate limit) and 503 UNAVAILABLE (overloaded) are worth retrying. */
export function isRetryable(err: unknown): boolean {
  if (err instanceof ApiError) return err.status === 429 || err.status === 503;
  const msg = String((err as Error)?.message ?? err);
  return /RESOURCE_EXHAUSTED|UNAVAILABLE|\b429\b|\b503\b/.test(msg);
}

export function isRateLimit(err: unknown): boolean {
  if (err instanceof ApiError) return err.status === 429;
  return /RESOURCE_EXHAUSTED|\b429\b/.test(String((err as Error)?.message ?? err));
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Gemini accepted the request but sent nothing within the per-attempt time limit. */
export class FirstChunkTimeoutError extends Error {}

/**
 * Start a streaming generation, retrying retryable errors with exponential backoff
 * (1 s, 2 s, 4 s + jitter; at most 3 retries). The first chunk is read inside the retry loop,
 * because some failures only surface once the stream is read. Nothing has been sent to the
 * client before this resolves, so retries are invisible to the user.
 *
 * Each attempt must produce its first chunk within `firstChunkTimeoutMs`: under heavy load Gemini
 * sometimes accepts a request and then never answers. A timed-out attempt counts as "overloaded"
 * and is retried. No new attempt starts after `deadlineMs`, so the user gets an answer or a clear
 * "try again" message in bounded time. The caller's `abortSignal` (client disconnect) is respected.
 */
export async function startStreamWithRetry(
  ai: GoogleGenAI,
  params: Parameters<GoogleGenAI["models"]["generateContentStream"]>[0],
  {
    retries = 3,
    baseDelayMs = 1000,
    firstChunkTimeoutMs = 45_000,
    deadlineMs = 90_000,
    onRetry,
  }: {
    retries?: number;
    baseDelayMs?: number;
    firstChunkTimeoutMs?: number;
    deadlineMs?: number;
    onRetry?: (attempt: number, err: unknown) => void;
  } = {},
): Promise<{ first: GenerateContentResponse | undefined; rest: AsyncGenerator<GenerateContentResponse> }> {
  const callerSignal = params.config?.abortSignal;
  const started = Date.now();
  for (let attempt = 0; ; attempt++) {
    const attemptCtrl = new AbortController();
    const timer = setTimeout(() => attemptCtrl.abort(new FirstChunkTimeoutError("no response from Gemini")), firstChunkTimeoutMs);
    const signal = callerSignal ? AbortSignal.any([callerSignal, attemptCtrl.signal]) : attemptCtrl.signal;
    try {
      const stream = await ai.models.generateContentStream({ ...params, config: { ...params.config, abortSignal: signal } });
      const first = await stream.next();
      return { first: first.done ? undefined : first.value, rest: stream };
    } catch (rawErr) {
      if (callerSignal?.aborted) throw rawErr; // client went away: don't retry
      const err = attemptCtrl.signal.aborted ? new FirstChunkTimeoutError("Gemini did not respond in time") : rawErr;
      const retryable = isRetryable(err) || err instanceof FirstChunkTimeoutError;
      const delay = baseDelayMs * 2 ** attempt + Math.random() * 250;
      const outOfTime = Date.now() - started + delay >= deadlineMs;
      if (!retryable || attempt >= retries || outOfTime) {
        if (retryable) throw new RateLimitError(String((err as Error)?.message ?? err), !isRateLimit(err));
        throw err;
      }
      onRetry?.(attempt + 1, err);
      await sleep(delay);
    } finally {
      clearTimeout(timer); // once streaming, the per-attempt limit no longer applies
    }
  }
}
