# Reproducibility Checklist

- [x] Canonical processed datasets are stored under `data/processed/`.
- [x] Dataset manifests are stored under `data/manifests/`.
- [x] Raw BM25 baselines are saved under `results/bm25/raw_*`.
- [x] Fixed-prompt rewrite artifacts are saved under `rewritten_queries/`.
- [x] Prompt evolution logs are saved under `evolution_logs/`.
- [x] Final evaluation JSON files are saved under `results/evaluations/`.
- [x] Phase 4 validation writes `results/phase4/validation/validation_report.json`.
- [x] Phase 4 traceability writes `results/phase4/validation/traceability.csv`.
- [x] Error analysis writes `results/phase4/error_analysis/`.
- [x] Qualitative review writes `results/phase4/qualitative_review/`.
- [x] Final results table writes `reports/final_results.csv`.
- [x] Paper tables write `paper/final_tables.tex`.
- [x] Figures write `figures/plots/` and `figures/venn/`.
- [x] The end-to-end command sequence is documented in `docs/full_experiment_runbook.md`.
