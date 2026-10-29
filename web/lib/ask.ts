/** Types shared by the /api/ask route and the chat panel. The response body is NDJSON:
 *  one AskEvent per line, in order: "sources", then "delta"*, then "done" (or "error"). */

export interface ChatTurn {
  question: string;
  answer: string;
}

export interface AskRequest {
  videoId: string;
  question: string;
  history?: ChatTurn[];
}

export interface SourceFrame {
  frame_id: number;
  timestamp_sec: number;
  timestamp_label: string;
  thumbnail_url: string;
}

export interface SourceChunk {
  start_sec: number;
  end_sec: number;
  start_label: string;
  end_label: string;
}

export type AskEvent =
  | { type: "sources"; frames: SourceFrame[]; chunks: SourceChunk[]; hasTranscript: boolean }
  | { type: "delta"; text: string }
  | { type: "error"; message: string }
  | { type: "done" };

export const MAX_HISTORY_TURNS = 4;
export const MAX_QUESTION_CHARS = 1000;
