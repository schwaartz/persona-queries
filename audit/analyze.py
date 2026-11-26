"""Recompute the source-persona audit results from the saved annotations.

    python audit/analyze.py

Two LLM auditors (Claude and GPT) labelled 500 Nemotron personas (250 Dutch, 250
English) for four error categories as present / absent / uncertain. A human then made
the final decision per persona and category (review.json): flagged contradictions and
real-world references were checked by hand, the other categories were accepted
automatically from the agents' flags.

Writes CSV tables and figures to audit/results/.
"""

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).parent / "data"
RESULTS = Path(__file__).parent / "results"
AGENTS = ("claude", "gpt")
TITLES = {"contradictory_information": "Contradictions", "language_defect": "Language defects",
          "broken_or_missing_content": "Broken / missing content",
          "incorrect_real_world_reference": "Real-world references", "any_error": "Any error"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def wilson(errors, n, z=1.959963984540054):
    """95% Wilson score interval for a proportion."""
    p = errors / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0, center - half), min(1, center + half)


def cohen_kappa(pairs):
    """Observed agreement, chance agreement and Cohen's kappa for a list of label pairs."""
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    labels = {label for pair in pairs for label in pair}
    expected = sum(sum(a == l for a, _ in pairs) * sum(b == l for _, b in pairs) for l in labels) / n ** 2
    kappa = (observed - expected) / (1 - expected) if expected < 1 else float("nan")
    return observed, expected, kappa


def write_csv(name, rows):
    with (RESULTS / name).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, name):
    fig.savefig(RESULTS / f"{name}.png", dpi=180)
    fig.savefig(RESULTS / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main():
    rubric = read(DATA / "rubric.json")
    categories = [c["id"] for c in rubric["categories"] if not c["ignored"]]
    language = {p["item_id"]: p["language"] for p in read(DATA / "personas.json")}
    agents = {agent: {item["item_id"]: item["labels"]
                      for path in sorted((DATA / "annotations" / agent).glob("batch-*.json"))
                      for item in read(path)["items"]}
              for agent in AGENTS}
    review = read(DATA / "review.json")["items"]
    for labels in [*agents.values(), {r["item_id"]: r for r in review}]:
        assert labels.keys() == language.keys(), "Every audited persona needs labels from every source."
    RESULTS.mkdir(exist_ok=True)

    # Prevalence of each error category, per language, from the final decisions.
    prevalence = []
    for group in ("all", "nl", "en"):
        rows = [r for r in review if group == "all" or language[r["item_id"]] == group]
        for category in [*categories, "any_error"]:
            errors = sum(any(r["labels"].values()) if category == "any_error" else r["labels"][category]
                         for r in rows)
            low, high = wilson(errors, len(rows))
            prevalence.append({"language": group, "category": category, "n": len(rows), "errors": errors,
                               "rate": errors / len(rows), "wilson_95_low": low, "wilson_95_high": high})
    write_csv("prevalence.csv", prevalence)

    # Agreement between the two auditors.
    agreement = []
    for category in categories:
        pairs = [(agents["claude"][i][category], agents["gpt"][i][category]) for i in language]
        certain = [pair for pair in pairs if "uncertain" not in pair]
        observed, expected, kappa = cohen_kappa(certain)
        flagged = [tuple(label != "absent" for label in pair) for pair in pairs]
        agreement.append({"category": category, "n": len(certain), "uncertain_excluded": len(pairs) - len(certain),
                          "observed_agreement": observed, "expected_agreement": expected, "kappa": kappa,
                          "kappa_uncertain_as_flagged": cohen_kappa(flagged)[2],
                          "claude_present": sum(a == "present" for a, _ in pairs),
                          "gpt_present": sum(b == "present" for _, b in pairs)})
    write_csv("agreement.csv", agreement)

    # Final review: how many agent flags were confirmed, and who decided.
    decisions = []
    for category in categories:
        flagged = [r for r in review if any(agents[a][r["item_id"]][category] != "absent" for a in AGENTS)]
        origins = [r["decision_origin"][category] for r in review]
        decisions.append({"category": category, "flagged_by_an_agent": len(flagged),
                          "confirmed": sum(r["labels"][category] for r in flagged),
                          "rejected": sum(not r["labels"][category] for r in flagged),
                          "human_decisions": origins.count("human"),
                          "automatic_acceptances": origins.count("automatic_agent_acceptance"),
                          "automatic_absences": origins.count("automatic_agent_absence")})
    write_csv("review.csv", decisions)

    plt.rcParams.update({"font.size": 10})
    shown = [*categories, "any_error"]
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    for offset, (group, label, color) in enumerate([("all", "Overall", "#1b6b64"), ("nl", "Dutch", "#dc8b2d"),
                                                     ("en", "English", "#5b70b5")]):
        rows = [next(r for r in prevalence if r["language"] == group and r["category"] == c) for c in shown]
        x = [i + (offset - 1) * 0.24 for i in range(len(shown))]
        y = [100 * r["rate"] for r in rows]
        error = [[100 * (r["rate"] - r["wilson_95_low"]) for r in rows],
                 [100 * (r["wilson_95_high"] - r["rate"]) for r in rows]]
        ax.bar(x, y, width=0.22, label=label, color=color)
        ax.errorbar(x, y, yerr=error, fmt="none", ecolor="#24333a", capsize=3)
    ax.set(xticks=range(len(shown)), xticklabels=[TITLES[c] for c in shown], ylim=(0, 106),
           ylabel="Personas with error (%)", title="Error prevalence in source personas (Wilson 95% intervals)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()
    save_figure(fig, "prevalence")

    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for offset, (key, label, color) in enumerate([("kappa", "Uncertain pairs excluded", "#1b6b64"),
                                                   ("kappa_uncertain_as_flagged", "Uncertain counted as flagged", "#dc8b2d")]):
        values = [row[key] for row in agreement]
        ax.barh([i + (offset - 0.5) * 0.3 for i in range(len(values))], values, height=0.28, label=label, color=color)
        for i, value in enumerate(values):
            ax.text(value + 0.02, i + (offset - 0.5) * 0.3, f"{value:.3f}", va="center", fontsize=9)
    ax.set(yticks=range(len(categories)), yticklabels=[TITLES[c] for c in categories], xlim=(0, 1.12),
           xlabel="Cohen's κ", title="Agreement between the Claude and GPT auditors")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    save_figure(fig, "agreement")

    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    confirmed = [row["confirmed"] for row in decisions]
    rejected = [row["rejected"] for row in decisions]
    ax.barh(range(len(categories)), confirmed, color="#1b6b64", label="Confirmed")
    ax.barh(range(len(categories)), rejected, left=confirmed, color="#dc8b2d", label="Rejected")
    for i, row in enumerate(decisions):
        how = "by hand" if row["human_decisions"] else "automatically"
        ax.text(row["flagged_by_an_agent"] + 3, i, f"{row['confirmed']}/{row['flagged_by_an_agent']} {how}", va="center")
    ax.set(yticks=range(len(categories)), yticklabels=[TITLES[c] for c in categories],
           xlim=(0, max(r["flagged_by_an_agent"] for r in decisions) * 1.3),
           xlabel="Personas flagged by at least one auditor", title="Final decision on flagged personas")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right")
    save_figure(fig, "review")
    print(f"Results written to {RESULTS}")


if __name__ == "__main__":
    main()
