import type { Metadata } from "next";
import Link from "next/link";
import { copy } from "@/lib/copy";
import { routes } from "@/lib/site";

export const metadata: Metadata = { title: copy.site.notFound.title };

export default function NotFound() {
  return (
    <div className="column stack">
      <h1>{copy.site.notFound.title}</h1>
      <p>{copy.site.notFound.body}</p>
      <p>
        <Link href={routes.home}>{copy.site.notFound.home}</Link>
      </p>
    </div>
  );
}
