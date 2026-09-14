# OrbitKV website

A static Astro site maintained by [feichai](https://github.com/feichai0017).
The overview, architecture guide, and evidence page share one layout and content
catalog. The site ships no client-side JavaScript, remote fonts, or UI framework.

## Develop

Use the Node and npm versions declared in `package.json` (also pinned in CI).
From this directory:

```sh
npm ci
npm run dev
```

Open the local URL printed by Astro, under `/orbitkv/`.

```sh
npm run check
npm run build
npm run preview
```

## Layout

```text
src/data/site.ts             shared navigation, capabilities, records, and links
src/layouts/SiteLayout.astro document metadata, header, footer, and author credit
src/components/             state illustration and documentation navigation
src/pages/                  overview, architecture, evidence, and 404
src/styles/global.css       design tokens, layout, components, responsive rules
public/                     favicon and README banner as editable SVGs
```

Keep typography, colors, spacing, and breakpoints in the shared stylesheet.
Update capability statements against the repository documentation and recorded
evidence. Experimental plans belong in the roadmap, with their scope explicit.
Historical measurements retain their source and workload boundaries.

Deployed documentation links resolve to the source revision used for the build
through `GITHUB_SHA`. Local previews default to `main`; set `PUBLIC_SOURCE_REF`
to the branch or commit being presented when previewing unpublished work. This
also works for source archives without Git metadata. Local navigation uses
Astro's configured base path.

## Publish

The existing GitHub Pages workflow builds and deploys website changes pushed to
`main`, and supports manual dispatch. A push to a feature branch does not publish
the live website. The configured address is
[feichai0017.github.io/orbitkv](https://feichai0017.github.io/orbitkv/).

Before publishing, run the type check and production build, then inspect all
routes at desktop and mobile widths. Check keyboard navigation, contrast,
scrollable code/tables, and the `/orbitkv/` base path. No screenshot or browser
tool dependency is required in the production package.
