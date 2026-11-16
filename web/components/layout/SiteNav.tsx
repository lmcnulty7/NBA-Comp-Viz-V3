"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { copy } from "@/lib/copy";
import { isCurrentSection } from "@/lib/nav";
import { routes } from "@/lib/site";

const ITEMS = [
  { href: routes.home, text: copy.site.nav.home },
  { href: routes.review, text: copy.site.nav.review },
  { href: routes.prep, text: copy.site.nav.prep },
  { href: routes.results, text: copy.site.nav.results },
  { href: routes.methods, text: copy.site.nav.methods },
] as const;

export function SiteNav() {
  const pathname = usePathname() ?? "/";
  return (
    <nav className="site-nav" aria-label={copy.site.nav.label}>
      <ul>
        {ITEMS.map((item) => (
          <li key={item.href}>
            <Link href={item.href} aria-current={isCurrentSection(pathname, item.href) ? "page" : undefined}>
              {item.text}
            </Link>
          </li>
        ))}
      </ul>
    </nav>
  );
}
