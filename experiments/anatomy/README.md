# Error anatomy of the E13 candidate (2026-11-11)

One-off analysis scripts behind the "Where the E13 candidate lost its score" entry of
`EXPERIMENTS.md`. External open data only.

    python experiments/anatomy/anatomy_fit.py <repo> <folder>      # fit, dump predictions (25 min)
    python experiments/anatomy/anatomy_report.py <repo> <folder>   # where spellF1 is lost
    python experiments/anatomy/anatomy_offsets.py <repo> <folder>  # timing of matched boundaries
    python experiments/anatomy/anatomy_shift.py <repo> <folder>    # does a constant shift help?
    python experiments/anatomy/definition_cost.py <repo>           # cost of the possession threshold
