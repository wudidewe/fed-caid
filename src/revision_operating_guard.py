"""Assess a changed binary operating point on paired seed-level FNR effects."""

from .revision_report import summarize


def summarize_fnr_guard(records, planned_seeds, max_increase):
    """Report paired FNR differences and the predeclared upper-bound check.

    ``records`` must contain at most one row per planned seed. Each ``fnr_delta``
    is the candidate FNR minus the source operating point's FNR on the same
    evaluation rows. A missing pair or an unavailable interval cannot pass.
    """
    if type(max_increase) not in (float, int) or not 0 <= max_increase <= 1:
        raise ValueError("The maximum FNR increase must be a rate in [0, 1]")
    planned = tuple(planned_seeds)
    roster = set(planned)
    if len(roster) != len(planned):
        raise ValueError("The FNR guard requires distinct planned seeds")
    paired = {}
    for record in records:
        seed = record["seed"]
        if seed in paired or seed not in roster:
            raise ValueError("Every paired FNR observation requires a unique planned seed")
        paired[seed] = record["fnr_delta"]
    stats = summarize((paired.get(seed) for seed in planned), bounds=(-1, 1))
    upper = stats["ci95_high"]
    return {
        "operating_fnr_planned_seeds": len(planned),
        "operating_fnr_observed_seeds": len(paired),
        "operating_fnr_seeds_with_metric": stats["seeds_with_metric"],
        "operating_fnr_delta_mean": stats["mean"],
        "operating_fnr_delta_seed_sd": stats["seed_sd"],
        "operating_fnr_delta_ci95_low": stats["ci95_low"],
        "operating_fnr_delta_ci95_high": upper,
        "operating_fnr_max_increase": max_increase,
        "operating_fnr_guard_met": (
            stats["seeds_with_metric"] == len(planned) and upper is not None and upper <= max_increase
        ),
    }
