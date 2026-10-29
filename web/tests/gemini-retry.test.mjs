// Run with: npm test   (Node's built-in test runner + native TypeScript type stripping; no extra deps)
import assert from "node:assert/strict";
import { test } from "node:test";
import { ApiError } from "@google/genai";
import { isRetryable, RateLimitError, startStreamWithRetry } from "../lib/gemini.ts";

async function* gen(texts) {
  for (const t of texts) yield { text: t };
}

/** Fake GoogleGenAI whose nth call fails as scripted ("429" | "503" | "400" | "first-chunk-429" | "ok"). */
function fakeAi(script) {
  let calls = 0;
  const err = (status) => new ApiError({ message: `{"error":{"code":${status}}}`, status });
  return {
    calls: () => calls,
    models: {
      generateContentStream: async () => {
        const step = script[calls++];
        if (step === "429" || step === "503" || step === "400") throw err(Number(step));
        if (step === "first-chunk-429") return (async function* () { throw err(429); })();
        return gen(["Hello", " world"]);
      },
    },
  };
}

const fast = { baseDelayMs: 1 };

test("recovers after a 429 and a 503", async () => {
  const ai = fakeAi(["429", "503", "ok"]);
  const { first, rest } = await startStreamWithRetry(ai, {}, fast);
  const tail = [];
  for await (const c of rest) tail.push(c.text);
  assert.equal(ai.calls(), 3);
  assert.equal(first.text + tail.join(""), "Hello world");
});

test("an error on the first streamed chunk is retried", async () => {
  const ai = fakeAi(["first-chunk-429", "ok"]);
  const { first } = await startStreamWithRetry(ai, {}, fast);
  assert.equal(ai.calls(), 2);
  assert.equal(first.text, "Hello");
});

test("gives up after 3 retries with a rate-limit error", async () => {
  const ai = fakeAi(["429", "429", "429", "429", "ok"]);
  await assert.rejects(startStreamWithRetry(ai, {}, fast), (e) => e instanceof RateLimitError && !e.overloaded);
  assert.equal(ai.calls(), 4);
});

test("persistent 503 is reported as overloaded", async () => {
  const ai = fakeAi(["503", "503", "503", "503"]);
  await assert.rejects(startStreamWithRetry(ai, {}, fast), (e) => e instanceof RateLimitError && e.overloaded);
});

test("non-retryable errors are not retried", async () => {
  const ai = fakeAi(["400"]);
  await assert.rejects(startStreamWithRetry(ai, {}, fast), (e) => !(e instanceof RateLimitError) && e.status === 400);
  assert.equal(ai.calls(), 1);
});

test("backoff is exponential", async () => {
  const t0 = Date.now();
  await startStreamWithRetry(fakeAi(["429", "429", "ok"]), {}, { baseDelayMs: 100 });
  const ms = Date.now() - t0;
  assert.ok(ms >= 300 && ms < 1000, `took ${ms} ms, expected ~300 ms (100 + 200 + jitter)`);
});

test("isRetryable recognises status codes and messages", () => {
  assert.ok(isRetryable(new Error("got RESOURCE_EXHAUSTED from server")));
  assert.ok(isRetryable(new Error("503 UNAVAILABLE")));
  assert.ok(!isRetryable(new Error("INVALID_ARGUMENT")));
});

/** Fake whose nth call hangs forever ("hang") until aborted, otherwise behaves like fakeAi. */
function hangingAi(script) {
  let calls = 0;
  return {
    calls: () => calls,
    models: {
      generateContentStream: async (params) => {
        const step = script[calls++];
        if (step === "hang") {
          await new Promise((_, reject) => params.config.abortSignal.addEventListener("abort", () => reject(new Error("aborted"))));
        }
        return gen(["Hello"]);
      },
    },
  };
}

test("an attempt that never responds times out and is retried", async () => {
  const ai = hangingAi(["hang", "ok"]);
  const t0 = Date.now();
  const { first } = await startStreamWithRetry(ai, { config: {} }, { baseDelayMs: 1, firstChunkTimeoutMs: 100 });
  assert.equal(first.text, "Hello");
  assert.equal(ai.calls(), 2);
  assert.ok(Date.now() - t0 < 1000);
});

test("stops retrying at the overall deadline and reports overloaded", async () => {
  const ai = hangingAi(["hang", "hang", "hang", "hang"]);
  const t0 = Date.now();
  await assert.rejects(
    startStreamWithRetry(ai, { config: {} }, { baseDelayMs: 50, firstChunkTimeoutMs: 100, deadlineMs: 250 }),
    (e) => e instanceof RateLimitError && e.overloaded,
  );
  assert.ok(ai.calls() < 4, `made ${ai.calls()} attempts despite the deadline`);
  assert.ok(Date.now() - t0 < 600);
});

test("a client disconnect is not retried", async () => {
  const caller = new AbortController();
  const ai = hangingAi(["hang", "ok"]);
  setTimeout(() => caller.abort(), 50);
  await assert.rejects(startStreamWithRetry(ai, { config: { abortSignal: caller.signal } }, { baseDelayMs: 1, firstChunkTimeoutMs: 10_000 }));
  assert.equal(ai.calls(), 1);
});
