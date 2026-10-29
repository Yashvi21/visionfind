import { Suspense } from "react";
import VideoView from "./VideoView";

export default async function VideoPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  // useSearchParams() (for ?t=) in VideoView needs a Suspense boundary.
  return (
    <Suspense fallback={<div className="aspect-video max-w-4xl animate-pulse rounded-xl bg-slate-200/60" />}>
      <VideoView id={id} />
    </Suspense>
  );
}
