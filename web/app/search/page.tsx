import { Suspense } from "react";
import SearchView from "./SearchView";

export const metadata = { title: "Search · VisionFind" };

export default function SearchPage() {
  // useSearchParams() in SearchView needs a Suspense boundary.
  return (
    <Suspense fallback={<div className="h-40 animate-pulse rounded-xl bg-slate-200/60" />}>
      <SearchView />
    </Suspense>
  );
}
