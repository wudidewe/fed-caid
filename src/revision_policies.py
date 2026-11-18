"""Declared repair policies and unambiguous report identities."""

from .revision_supplement import REPAIR_MODES, REPAIR_PAIRS

SENSITIVITY_POLICIES = ((0.5, 1.0), (0.25, 1.0), (1.0, 1.0), (0.5, 0.25), (0.5, 0.5))


def policy_specs(args):
    selected = (args.repair_gate_target_fraction, args.repair_residual_budget_fraction)
    policies = [selected]
    if getattr(args, "repair_sensitivity", False):
        policies.extend(pair for pair in SENSITIVITY_POLICIES if pair != selected)
    return [
        (g, r, "" if i == 0 else f"__g{g:g}_r{r:g}".replace(".", "p")) for i, (g, r) in enumerate(policies)
    ]


def repair_base(mode):
    return next((base for base in REPAIR_MODES if mode == base or mode.startswith(base + "__g")), None)


def repair_conditions(mode, settings):
    base = repair_base(mode)
    if base is None:
        return {}
    suffix = mode[len(base) :]
    if suffix:
        gate, residual = suffix[3:].split("_r", 1)
        return {
            "repair_gate_target_fraction": float(gate.replace("p", ".")),
            "repair_residual_budget_fraction": float(residual.replace("p", ".")),
        }
    return {
        name: settings.get(name, default)
        for name, default in (("repair_gate_target_fraction", 0.5), ("repair_residual_budget_fraction", 1.0))
    }


def repair_pairs_for_modes(modes):
    pairs = dict(REPAIR_PAIRS)
    for mode in modes:
        base = repair_base(mode)
        if base is None or mode == base:
            continue
        suffix = mode[len(base) :]
        for contrast, (left, right) in REPAIR_PAIRS.items():
            if left == base:
                label, reference = contrast.split("_minus_", 1)
                pairs[f"{label}{suffix}_minus_{reference}"] = (mode, right)
    return pairs
