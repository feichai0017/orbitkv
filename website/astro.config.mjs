import { defineConfig } from "astro/config";
const sourceRevision =
  process.env.PUBLIC_SOURCE_REF || process.env.GITHUB_SHA || "main";

const base = "/orbitkv";

export default defineConfig({
  site: "https://feichai0017.github.io",
  base,
  output: "static",
  trailingSlash: "always",
  redirects: { "/evidence/": `${base}/models/` },
  vite: {
    define: {
      "import.meta.env.PUBLIC_SOURCE_REF": JSON.stringify(sourceRevision),
    },
  },
});
