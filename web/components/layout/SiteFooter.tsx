import { copy, fill } from "@/lib/copy";
import type { Manifest } from "@/lib/types/contract";

export function SiteFooter({ manifest }: { manifest: Manifest }) {
  return (
    <footer className="site-footer" aria-label={copy.site.footer.label}>
      <div className="site-footer-inner">
        <p>
          {fill(copy.site.footer.data, {
            build: manifest.build_id,
            freeze: manifest.freeze_tag,
            schema: manifest.schema_version,
          })}
        </p>
        <p>{manifest.synthetic ? copy.site.footer.fixture : copy.site.footer.trace}</p>
      </div>
    </footer>
  );
}
