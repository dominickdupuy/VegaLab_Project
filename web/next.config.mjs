import { PHASE_DEVELOPMENT_SERVER } from "next/constants.js";

/**
 * Production: a static export in web/out/ that optspread.server serves itself, beside
 * its /api. `trailingSlash` makes every route a folder with an index.html
 * (/runs/ -> out/runs/index.html), which a plain static file server resolves without
 * rewrite rules.
 *
 * Development (`make web-dev`): /api is proxied to a running `make serve`
 * (OPTSPREAD_API, default http://127.0.0.1:8600), so the browser sees one origin and
 * the server needs no CORS.
 */
export default function config(phase) {
  const base = {
    reactStrictMode: true,
    trailingSlash: true,
    images: { unoptimized: true },
  };
  if (phase !== PHASE_DEVELOPMENT_SERVER) return { ...base, output: "export" };

  const api = (process.env.OPTSPREAD_API || "http://127.0.0.1:8600").replace(/\/+$/, "");
  return {
    ...base,
    skipTrailingSlashRedirect: true, // keep /api/runs as /api/runs on its way to the server
    async rewrites() {
      return [{ source: "/api/:path*", destination: `${api}/api/:path*` }];
    },
  };
}
