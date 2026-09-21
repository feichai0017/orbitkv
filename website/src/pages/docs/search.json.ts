import { getCollection } from "astro:content";
import { docPages } from "../../data/docs";
import { localUrl } from "../../data/site";

export async function GET() {
  const pages = await getCollection("docs");
  return new Response(
    JSON.stringify(
      pages.map((entry) => ({
        title:
          docPages.find((page) => page.slug === entry.id)?.title ?? entry.id,
        url: localUrl(`/docs/${entry.id}/`),
        body: entry.body ?? "",
      })),
    ),
    { headers: { "Content-Type": "application/json" } },
  );
}
