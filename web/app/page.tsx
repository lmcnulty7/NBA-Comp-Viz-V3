import { ColdOpen } from "@/components/coldopen/ColdOpen";
import { EntryCards } from "@/components/coldopen/EntryCards";
import { Hero } from "@/components/coldopen/Hero";
import { previewLanes } from "@/components/coldopen/model";
import { storyData, storyLines, storyScenes } from "@/components/coldopen/story";
import { copy } from "@/lib/copy";
import { getServerSource } from "@/lib/data/server";
import { stillUrl, stillWebpUrl } from "@/lib/media";
import { VARIANT } from "@/lib/site";
import { renderLine, type TokenData } from "@/lib/tokens";

const open = copy.home.coldOpen;

/**
 * The home page: the headline (D17), the scroll-driven cold open of docs/SCRIPT.md (D25, M4), and the three
 * ways in. Every number on it is a token of the script filled from the site data (lib/tokens.ts); an unknown
 * token fails the build.
 */
export default async function HomePage() {
  const source = getServerSource();
  const [manifest, scorecard, players, results, swap, frames] = await Promise.all([
    source.manifest(),
    source.scorecard(),
    source.players(),
    source.results(),
    source.demoSwap(),
    source.demoFrames(),
  ]);
  const data: TokenData = { manifest, scorecard, players, swap, frames };
  const story = storyData(data, VARIANT, stillUrl, stillWebpUrl);

  const firstGame = manifest.games.find((game) => game.n_featured > 0) ?? manifest.games[0];
  let preview: { gameSlug: string; possession: number } | null = null;
  if (firstGame) {
    const index = await source.possessionIndex(firstGame.slug);
    const row = index.possessions.find((candidate) => candidate.featured) ?? index.possessions[0];
    if (row) preview = { gameSlug: firstGame.slug, possession: row.possession };
  }
  const next = storyLines("next", data);

  return (
    <div className="stack stack-loose">
      <Hero
        title={open.headline.title}
        intro={storyLines("headline", data)}
        media={story.media}
        variant={VARIANT}
      />
      <ColdOpen scenes={storyScenes(data, VARIANT)} data={story} />
      <EntryCards
        steps={[next[0] ?? "", next[1] ?? "", next[2] ?? ""]}
        preview={preview}
        player={players.players[0] ?? null}
        players={renderLine(copy.home.ui.players, null, data) ?? ""}
        lanes={previewLanes(results.lanes)}
      />
    </div>
  );
}
