import { defineConfig } from "astro/config";
const sourceRevision =
  process.env.PUBLIC_SOURCE_REF || process.env.GITHUB_SHA || "main";

export default defineConfig({
  site: "https://feichai0017.github.io",
  base: "/orbitkv",
  output: "static",
  trailingSlash: "always",
  vite: {
    define: {
      "import.meta.env.PUBLIC_SOURCE_REF": JSON.stringify(sourceRevision),
    },
  },
});
