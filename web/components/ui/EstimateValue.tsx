import { copy, fill } from "@/lib/copy";
import { formatCount, formatNumber } from "@/lib/format";
import type { Estimate } from "@/lib/types/contract";

/**
 * A value with its uncertainty: the interval and its method, or the words "point estimate". Every estimate
 * on the site goes through this component, so none can appear without one or the other.
 */
export function EstimateValue({ estimate, digits = 1 }: { estimate: Estimate; digits?: number }) {
  const hasInterval = !estimate.point_estimate && estimate.lo != null && estimate.hi != null;
  return (
    <span className="estimate">
      <span className="num">{formatNumber(estimate.value, digits)}</span>
      <span className="note">
        {hasInterval ? (
          <>
            <span className="num">
              {fill(copy.site.estimate.range, {
                lo: formatNumber(estimate.lo, digits),
                hi: formatNumber(estimate.hi, digits),
              })}
            </span>
            {estimate.interval ? <span className="method">{estimate.interval}</span> : null}
          </>
        ) : (
          copy.site.estimate.point
        )}
        {estimate.n != null ? (
          <span className="sample">
            {fill(copy.site.estimate.sample, { n: formatCount(estimate.n), unit: estimate.n_unit ?? "" }).trim()}
          </span>
        ) : null}
      </span>
    </span>
  );
}
