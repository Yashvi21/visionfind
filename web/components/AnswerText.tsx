"use client";

import { Fragment, type ReactNode } from "react";

const TIME = String.raw`(?:\d{1,2}:)?\d{1,2}:\d{2}`;
/** A bracket holding one or more times: [01:05], [01:05–01:20], [00:12, 00:31 and 01:02] */
const CITATION = new RegExp(String.raw`\[(${TIME}(?:\s*(?:[,;–—-]|and|to)\s*${TIME})*)\]`, "g");
const TIME_G = new RegExp(TIME, "g");

export function parseTime(label: string): number {
  return label.split(":").map(Number).reduce((acc, n) => acc * 60 + n, 0);
}

interface Props {
  text: string;
  durationSec: number;
  onSeek: (seconds: number) => void;
}

function Chip({ label, onSeek }: { label: string; onSeek: () => void }) {
  return (
    <button
      type="button"
      onClick={onSeek}
      title={`Jump to ${label}`}
      className="mx-0.5 inline-flex items-center rounded-md bg-indigo-50 px-1.5 py-0 align-baseline font-mono text-[0.8em] font-medium text-indigo-700 ring-1 ring-inset ring-indigo-200 hover:bg-indigo-100"
    >
      {label}
    </button>
  );
}

/** Citations -> chips, **bold** -> <strong>. Times past the end of the video stay plain text. */
function inline(text: string, durationSec: number, onSeek: (s: number) => void, keyBase: string): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0;
  let i = 0;
  for (const m of text.matchAll(CITATION)) {
    out.push(...bold(text.slice(last, m.index), `${keyBase}-t${i}`));
    const inner = m[1];
    let pos = 0;
    const pieces: ReactNode[] = [];
    for (const t of inner.matchAll(TIME_G)) {
      if (t.index! > pos) pieces.push(inner.slice(pos, t.index));
      const sec = parseTime(t[0]);
      pieces.push(
        sec <= durationSec + 1 ? (
          <Chip key={`${keyBase}-c${i}-${t.index}`} label={t[0]} onSeek={() => onSeek(sec)} />
        ) : (
          `[${t[0]}]`
        ),
      );
      pos = t.index! + t[0].length;
    }
    out.push(<Fragment key={`${keyBase}-g${i}`}>{pieces}</Fragment>);
    last = m.index! + m[0].length;
    i++;
  }
  out.push(...bold(text.slice(last), `${keyBase}-end`));
  return out;
}

function bold(text: string, keyBase: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith("**") && part.endsWith("**") && part.length > 4 ? (
      <strong key={`${keyBase}-${i}`}>{part.slice(2, -2)}</strong>
    ) : (
      part
    ),
  );
}

export default function AnswerText({ text, durationSec, onSeek }: Props) {
  const blocks: ReactNode[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;
  const flush = () => {
    if (!list) return;
    const Tag = list.ordered ? "ol" : "ul";
    const k = `l${blocks.length}`;
    blocks.push(
      <Tag key={k} className={`${list.ordered ? "list-decimal" : "list-disc"} space-y-1 pl-5`}>
        {list.items.map((it, j) => (
          <li key={j}>{inline(it, durationSec, onSeek, `${k}-${j}`)}</li>
        ))}
      </Tag>,
    );
    list = null;
  };

  text.split("\n").forEach((raw, idx) => {
    const line = raw.trimEnd();
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (bullet || numbered) {
      const ordered = !!numbered;
      if (list && list.ordered !== ordered) flush();
      list ??= { ordered, items: [] };
      list.items.push((bullet ?? numbered)![1]);
      return;
    }
    flush();
    if (!line.trim()) return;
    const heading = line.match(/^#{1,6}\s+(.*)$/);
    blocks.push(
      <p key={`p${idx}`} className={heading ? "font-semibold" : undefined}>
        {inline(heading ? heading[1] : line, durationSec, onSeek, `p${idx}`)}
      </p>,
    );
  });
  flush();
  return <div className="space-y-2 leading-relaxed">{blocks}</div>;
}
