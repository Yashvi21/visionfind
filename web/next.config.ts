import type { NextConfig } from "next";

const ML_URL = process.env.ML_URL ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  // The browser only talks to Next.js; /ml/* is proxied to the FastAPI service.
  async rewrites() {
    return [{ source: "/ml/:path*", destination: `${ML_URL}/:path*` }];
  },
  experimental: {
    // Large .mp4 uploads go through the proxy; the default timeout is too short for them.
    proxyTimeout: 10 * 60 * 1000,
    // Next.js truncates proxied request bodies at 10 MB by default; match the ML upload cap
    // (500 MB, plus headroom for multipart overhead).
    middlewareClientMaxBodySize: "520mb",
  },
};

export default nextConfig;
