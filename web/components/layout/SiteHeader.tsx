import Link from "next/link";
import { CourtGlyph } from "@/components/court/CourtMotif";
import { copy } from "@/lib/copy";
import { routes } from "@/lib/site";
import { SiteNav } from "./SiteNav";

export function SiteHeader() {
  return (
    <header className="site-header">
      <div className="site-header-inner">
        <Link href={routes.home} className="brand">
          <CourtGlyph />
          <span>{copy.site.name}</span>
        </Link>
        <SiteNav />
      </div>
    </header>
  );
}
