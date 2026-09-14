# OrbitKV website

Static Astro pages with shared content, an editable SVG identity, and no
client-side JavaScript or remote fonts.

## Develop

Use the Node and npm versions in `package.json`. From this directory:

```sh
npm ci
npm run dev
```

Open the printed URL under `/orbitkv/`. Before publishing:

```sh
npm run check
npm run build
npm run preview
```

Inspect all routes on desktop and mobile, including keyboard navigation,
contrast, overflow, and links. Browser tooling stays outside production dependencies.

## Layout

```text
src/data/site.ts             navigation, crate descriptions, and evidence links
src/layouts/SiteLayout.astro metadata, header, and footer
src/components/             compilation illustration
src/pages/                  overview, architecture, evidence, and 404
src/styles/global.css       shared tokens, layout, and responsive styles
public/                     SVG mark and README wordmark
```

Keep public copy brief; link to repository documents for detailed contracts.
Capability claims must match recorded evidence. Preserve historical results.

## Publish

The Pages workflow checks, builds, and deploys website changes pushed to `main`
at [feichai0017.github.io/orbitkv](https://feichai0017.github.io/orbitkv/).

Documentation links bind to the build's `GITHUB_SHA`. Local previews use `main`
unless `PUBLIC_SOURCE_REF` is set. Navigation respects Astro's `/orbitkv/` base.
