import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { EntryCards } from "@/components/coldopen/EntryCards";
import { MediaFrame } from "@/components/coldopen/MediaFrame";
import { previewLanes } from "@/components/coldopen/model";
import { storyData, storyLines } from "@/components/coldopen/story";
import { BoxesVisual, ClipVisual, MapVisual, SwapVisual, TruthVisual } from "@/components/coldopen/Visuals";
import { copy } from "@/lib/copy";
import { getServerSource } from "@/lib/data/server";
import { stillUrl, stillWebpUrl } from "@/lib/media";
import { VARIANT } from "@/lib/site";
import { loadSlide, slideIds, talkCopy, type Slide } from "@/lib/talk";
import { renderLine, type TokenData } from "@/lib/tokens";
import { ResultsSlide } from "../_components/ResultsSlide";
import { SlideView } from "../_components/SlideView";

interface Params {
  params: Promise<{ slide: string }>;
}

export const dynamicParams = false;

export function generateStaticParams(): { slide: string }[] {
  return slideIds().map((slide) => ({ slide }));
}

export async function generateMetadata({ params }: Params): Promise<Metadata> {
  const { slide } = await params;
  return { title: `${talkCopy.title}: ${slide}` };
}

async function siteData(): Promise<TokenData> {
  const source = getServerSource();
  const [manifest, scorecard, players, swap, frames] = await Promise.all([
    source.manifest(),
    source.scorecard(),
    source.players(),
    source.demoSwap(),
    source.demoFrames(),
  ]);
  return { manifest, scorecard, players, swap, frames };
}

/** What sits beside the copy of a slide: the same pictures as the cold open, live where they play. */
async function visualOf(slide: Slide, data: TokenData) {
  const story = storyData(data, VARIANT, stillUrl, stillWebpUrl);
  switch (slide.visual) {
    case "none":
      return null;
    case "still":
      return story.media ? <MediaFrame media={story.media} boxes={null} /> : null;
    case "clip":
      return <ClipVisual data={story} />;
    case "boxes":
      return <BoxesVisual data={story} stage={Number.MAX_SAFE_INTEGER} />;
    case "map":
      return <MapVisual data={story} />;
    case "truth":
      return <TruthVisual data={story} />;
    case "swap":
      return <SwapVisual data={story} />;
    case "next": {
      const source = getServerSource();
      const results = await source.results();
      const game = data.manifest.games.find((candidate) => candidate.n_featured > 0) ?? data.manifest.games[0];
      let preview: { gameSlug: string; possession: number } | null = null;
      if (game) {
        const index = await source.possessionIndex(game.slug);
        const row = index.possessions.find((candidate) => candidate.featured) ?? index.possessions[0];
        if (row) preview = { gameSlug: game.slug, possession: row.possession };
      }
      const next = storyLines("next", data);
      return (
        <EntryCards
          steps={[next[0] ?? "", next[1] ?? "", next[2] ?? ""]}
          preview={preview}
          player={data.players.players[0] ?? null}
          players={renderLine(copy.home.ui.players, null, data) ?? ""}
          lanes={previewLanes(results.lanes)}
          compact
        />
      );
    }
    case "results":
      return <ResultsSlide results={await getServerSource().results()} />;
  }
}

/** One slide of the talk (PLAN.md M10): the private build only, one static page per slide. */
export default async function TalkSlidePage({ params }: Params) {
  const { slide: id } = await params;
  const ids = slideIds();
  const position = ids.indexOf(id);
  if (position < 0) notFound();
  const data = await siteData();
  const slide = loadSlide(id, data);
  return (
    <SlideView
      slide={slide}
      total={ids.length}
      previous={ids[position - 1] ?? null}
      next={ids[position + 1] ?? null}
      first={ids[0] ?? id}
      last={ids[ids.length - 1] ?? id}
    >
      {await visualOf(slide, data)}
    </SlideView>
  );
}
