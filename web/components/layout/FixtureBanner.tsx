import { copy } from "@/lib/copy";

/**
 * The banner every page shows when the data build is synthetic (manifest.synthetic). Fixture output must
 * never be read as a result, so the banner stays at the top of the window while the page scrolls.
 */
export function FixtureBanner() {
  return (
    <aside className="banner" aria-label={copy.site.fixture.label} data-testid="fixture-banner">
      <p>
        <strong>{copy.site.fixture.title}</strong> <span className="banner-more">{copy.site.fixture.more}</span>
      </p>
    </aside>
  );
}
