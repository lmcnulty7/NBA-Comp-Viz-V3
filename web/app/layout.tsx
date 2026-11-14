import type { Metadata, Viewport } from "next";
import localFont from "next/font/local";
import { FixtureBanner } from "@/components/layout/FixtureBanner";
import { SiteFooter } from "@/components/layout/SiteFooter";
import { SiteHeader } from "@/components/layout/SiteHeader";
import { copy, fill } from "@/lib/copy";
import { getServerSource } from "@/lib/data/server";
import { VARIANT } from "@/lib/site";
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/site.css";

// The three faces of the design, served from this site (app/fonts, SIL Open Font License 1.1) so the pages
// make no request to another host and work offline. The faces every page uses are preloaded. The serif italic
// and the medium mono are declared under the same family names and fetched only by a page that uses them.
// Each face is declared under its real family name, and app/styles/tokens.css names those families (and the
// "newsreader Fallback" and "franklin Fallback" faces that next/font derives from the constant names below).
// A browser test checks that the three families really load.
const newsreader = localFont({
  src: "./fonts/newsreader-latin-opsz-normal.woff2",
  weight: "200 800",
  style: "normal",
  variable: "--font-newsreader",
  display: "swap",
  adjustFontFallback: "Times New Roman",
  declarations: [{ prop: "font-family", value: "Newsreader" }],
});

const newsreaderItalic = localFont({
  src: "./fonts/newsreader-latin-wght-italic.woff2",
  weight: "200 800",
  style: "italic",
  variable: "--font-newsreader-italic",
  display: "swap",
  preload: false,
  adjustFontFallback: false,
  declarations: [{ prop: "font-family", value: "Newsreader" }],
});

const franklin = localFont({
  src: "./fonts/libre-franklin-latin-wght-normal.woff2",
  weight: "100 900",
  variable: "--font-franklin",
  display: "swap",
  adjustFontFallback: "Arial",
  declarations: [{ prop: "font-family", value: "Libre Franklin" }],
});

const plexMono = localFont({
  src: "./fonts/ibm-plex-mono-latin-400-normal.woff2",
  weight: "400",
  style: "normal",
  variable: "--font-plex-mono",
  display: "swap",
  adjustFontFallback: false,
  declarations: [{ prop: "font-family", value: "IBM Plex Mono" }],
});

const plexMonoMedium = localFont({
  src: "./fonts/ibm-plex-mono-latin-500-normal.woff2",
  weight: "500",
  style: "normal",
  variable: "--font-plex-mono-medium",
  display: "swap",
  preload: false,
  adjustFontFallback: false,
  declarations: [{ prop: "font-family", value: "IBM Plex Mono" }],
});

const FONT_CLASSES = [newsreader, newsreaderItalic, franklin, plexMono, plexMonoMedium]
  .map((font) => font.variable)
  .join(" ");

export const metadata: Metadata = {
  title: {
    default: copy.site.name,
    template: fill(copy.site.titleTemplate, { page: "%s", site: copy.site.name }),
  },
  description: copy.site.description,
  // The private copy sits behind sign-in and is not for search engines.
  robots: VARIANT === "private" ? { index: false, follow: false } : undefined,
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

export default async function RootLayout({ children }: LayoutProps<"/">) {
  const manifest = await getServerSource().manifest();
  return (
    <html lang="en" className={FONT_CLASSES}>
      <body className={manifest.synthetic ? "has-banner" : undefined}>
        <a className="skip-link" href="#main">
          {copy.site.skipLink}
        </a>
        {manifest.synthetic ? <FixtureBanner /> : null}
        <SiteHeader />
        <main id="main" className="page" tabIndex={-1}>
          {children}
        </main>
        <SiteFooter manifest={manifest} />
      </body>
    </html>
  );
}
