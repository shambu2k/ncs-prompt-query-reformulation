"""Prompt evolution engine for Phase 3.

Purpose:
    Automatically searches for an improved rewrite prompt by scoring candidates
    on a frozen dev subset and accepting only improvements above an epsilon gate.

Required inputs:
    - Seed prompt file under prompts/
    - Processed dataset splits under data/processed/
    - Optional: OPENAI_API_KEY for LLM-based candidate generation and rewriting.
      Falls back to heuristic mutations and normalization when not set.

Example command:
    python -m src.evolution.evolve_prompt \\
        --seed seed_prompt_v1 \\
        --run-name evolve_v1

Expected outputs:
    - prompts/best_prompt/best_prompt_evolve_v1.txt  (frozen best prompt)
    - evolution_logs/summaries/evolve_v1_summary.json
    - evolution_logs/summaries/evolve_v1_history.json
    - evolution_logs/iterations/evolve_v1_iter*.json  (per-iteration logs)
    - evolution_logs/evolve_v1.log
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.data.loader import DatasetLoader, DEFAULT_MANIFEST_ROOT, DEFAULT_PROCESSED_ROOT
from src.rewriting.prompt_templates import load_prompt_template

from .candidate_generator import CandidateGenerator
from .history import PromptHistory
from .scorer import compute_dev_mrr, DEFAULT_SCORE_CACHE_ROOT
from .selector import evaluate_candidate
from .stopping import StoppingState


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "src" / "configs" / "evolution.yaml"
DEFAULT_DEV_SUBSET_CONFIG = REPO_ROOT / "src" / "configs" / "dev_subset.yaml"
DEFAULT_EVOLUTION_LOG_ROOT = REPO_ROOT / "evolution_logs"
DEFAULT_PROMPTS_ROOT = REPO_ROOT / "prompts"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    return json.loads(text)


def _sample_dev_indices(n_total: int, dev_size: int, seed: int) -> list[int]:
    import random
    rng = random.Random(seed)
    indices = list(range(n_total))
    rng.shuffle(indices)
    return sorted(indices[:dev_size])


def _freeze_best_prompt(
    prompt_text: str,
    *,
    best_prompt_name: str,
    prompts_root: Path,
) -> Path:
    best_dir = prompts_root / "best_prompt"
    best_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = best_dir / f"{best_prompt_name}.txt"
    if frozen_path.exists():
        existing = frozen_path.read_text(encoding="utf-8").strip()
        if existing != prompt_text.strip():
            raise RuntimeError(
                f"Refusing to overwrite existing frozen prompt at {frozen_path}. "
                "Delete it manually if you intend to replace it."
            )
        return frozen_path
    frozen_path.write_text(prompt_text, encoding="utf-8")
    return frozen_path


def _log(log_path: Path, message: str) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{_utc_now()} {message}\n")


def _save_iteration_log(
    log_root: Path,
    run_name: str,
    iteration: int,
    payload: dict[str, Any],
) -> None:
    iters_dir = log_root / "iterations"
    iters_dir.mkdir(parents=True, exist_ok=True)
    path = iters_dir / f"{run_name}_iter{iteration:03d}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _save_run_state(log_root: Path, run_name: str, state: dict[str, Any]) -> None:
    summaries_dir = log_root / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    path = summaries_dir / f"{run_name}_state.json"
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_run_state(log_root: Path, run_name: str) -> dict[str, Any] | None:
    path = log_root / "summaries" / f"{run_name}_state.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run prompt evolution to find an improved rewrite prompt.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--seed",
        dest="seed_prompt",
        default="seed_prompt_v1",
        help="Seed prompt version to start evolution from (default: seed_prompt_v1).",
    )
    parser.add_argument(
        "--seed-path",
        default=None,
        help="Optional explicit path to the seed prompt file.",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Unique name for this evolution run (used in all output paths).",
    )
    parser.add_argument(
        "--best-prompt-name",
        default=None,
        help="Name for the frozen best prompt file (default: best_prompt_<run-name>).",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Evolution config JSON/YAML.",
    )
    parser.add_argument(
        "--dev-subset-config",
        default=str(DEFAULT_DEV_SUBSET_CONFIG),
        help="Dev subset config JSON/YAML.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last saved checkpoint if available.",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "heuristic", "openai"],
        default=None,
        help="Override provider from config.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override generator and scorer model from config.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override rewrite request concurrency for candidate scoring.",
    )
    parser.add_argument(
        "--score-batch-size",
        type=int,
        default=None,
        help="Override number of dev queries per OpenAI request during scoring.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override max iteration budget from config.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Override patience from config.",
    )
    parser.add_argument(
        "--dev-size",
        type=int,
        default=None,
        help="Override dev subset size from config.",
    )
    parser.add_argument(
        "--dev-seed",
        type=int,
        default=None,
        help="Override dev subset seed from config.",
    )
    parser.add_argument(
        "--processed-root",
        default=str(DEFAULT_PROCESSED_ROOT),
        help="Directory containing canonical processed JSONL files.",
    )
    parser.add_argument(
        "--manifest-root",
        default=str(DEFAULT_MANIFEST_ROOT),
        help="Directory containing dataset manifests.",
    )
    parser.add_argument(
        "--log-root",
        default=str(DEFAULT_EVOLUTION_LOG_ROOT),
        help="Root directory for evolution logs.",
    )
    parser.add_argument(
        "--prompts-root",
        default=str(DEFAULT_PROMPTS_ROOT),
        help="Root directory for prompt files.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    log_root = Path(args.log_root)
    prompts_root = Path(args.prompts_root)
    run_name = args.run_name
    log_path = log_root / f"{run_name}.log"
    log_root.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        _log(log_path, msg)
        print(f"[{_utc_now()}] {msg}")

    try:
        config = _load_json(Path(args.config))
        dev_config = _load_json(Path(args.dev_subset_config))

        max_iterations = int(args.max_iterations if args.max_iterations is not None else config.get("max_iterations", 25))
        patience = int(args.patience if args.patience is not None else config.get("patience", 5))
        epsilon = float(config.get("epsilon", 0.003))
        dev_split = dev_config.get("split", config.get("dev_split", "valid"))
        dev_condition = dev_config.get("condition", config.get("dev_condition", "adv"))
        dev_size = int(args.dev_size if args.dev_size is not None else dev_config.get("size", config.get("dev_subset_size", 300)))
        dev_seed = int(args.dev_seed if args.dev_seed is not None else dev_config.get("seed", config.get("dev_subset_seed", 42)))
        provider = args.provider or config.get("provider", "auto")
        model = args.model or config.get("proposer_model", "gpt-4o-mini")
        gen_temperature = float(config.get("generation_temperature", 0.7))
        gen_max_tokens = int(config.get("generation_max_tokens", 512))
        score_temperature = float(config.get("score_temperature", 0.0))
        score_max_tokens = int(config.get("score_max_tokens", 64))
        top_k = int(config.get("top_k", 100))
        num_workers = int(args.num_workers if args.num_workers is not None else config.get("num_workers", 10))
        score_batch_size = int(args.score_batch_size if args.score_batch_size is not None else config.get("score_batch_size", 1))

        log(
            f"start run_name={run_name} seed={args.seed_prompt} "
            f"max_iterations={max_iterations} patience={patience} epsilon={epsilon}"
        )

        seed_template = load_prompt_template(args.seed_prompt, args.seed_path)
        log(f"loaded seed prompt version={seed_template.version} path={seed_template.path}")

        loader = DatasetLoader(
            processed_root=Path(args.processed_root),
            manifest_root=Path(args.manifest_root),
        )
        all_records = loader.load_split(dev_split, dev_condition)
        log(f"loaded {len(all_records)} records split={dev_split} condition={dev_condition}")

        dev_indices = _sample_dev_indices(len(all_records), dev_size, dev_seed)
        actual_dev_size = len(dev_indices)
        indices_hash = hashlib.sha256(json.dumps(dev_indices).encode()).hexdigest()[:16]
        log(f"dev subset frozen: size={actual_dev_size} seed={dev_seed} hash={indices_hash}")

        dev_info: dict[str, Any] = {
            "split": dev_split,
            "condition": dev_condition,
            "total_records": len(all_records),
            "dev_size": actual_dev_size,
            "seed": dev_seed,
            "indices_hash": indices_hash,
        }
        dev_info_path = log_root / "summaries" / f"{run_name}_dev_subset.json"
        dev_info_path.parent.mkdir(parents=True, exist_ok=True)
        dev_info_path.write_text(
            json.dumps(dev_info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        history = PromptHistory(run_name=run_name, history_root=log_root)

        generator = CandidateGenerator(
            provider=provider,
            model=model,
            temperature=gen_temperature,
            max_tokens=gen_max_tokens,
            candidates_root=prompts_root / "candidates",
        )

        scorer_kwargs: dict[str, Any] = {
            "provider": provider,
            "model": model,
            "temperature": score_temperature,
            "max_tokens": score_max_tokens,
            "top_k": top_k,
            "score_cache_root": log_root / "score_cache",
            "num_workers": num_workers,
            "batch_size": score_batch_size,
        }

        resumed_state = _load_run_state(log_root, run_name) if args.resume else None

        if resumed_state is not None:
            log(f"resuming from checkpoint iteration={resumed_state['stopping_state']['iteration']}")
            stopping = StoppingState.from_dict(resumed_state["stopping_state"])
            best_prompt_id: str = resumed_state["best_prompt_id"]
            best_prompt_text: str = resumed_state["best_prompt_text"]
            best_mrr: float = float(resumed_state["best_mrr"])
        else:
            log("scoring seed prompt on dev subset...")
            seed_mrr = compute_dev_mrr(
                all_records,
                dev_indices,
                prompt_text=seed_template.text,
                **scorer_kwargs,
            )
            log(f"seed MRR={seed_mrr:.4f}")

            seed_record: dict[str, Any] = {
                "prompt_id": f"{run_name}_seed",
                "parent_prompt_id": None,
                "prompt_text": seed_template.text,
                "iteration": 0,
                "generator_model": "seed",
                "generation_method": "seed",
                "timestamp": _utc_now(),
                "status": "accepted",
                "dev_mrr": seed_mrr,
            }
            if not history.is_duplicate(seed_template.text):
                history.add(seed_record)

            stopping = StoppingState(max_iterations=max_iterations, patience=patience)
            best_prompt_id = seed_record["prompt_id"]
            best_prompt_text = seed_template.text
            best_mrr = seed_mrr

        log(f"initial best: prompt_id={best_prompt_id} mrr={best_mrr:.4f}")

        while True:
            stop, reason = stopping.should_stop()
            if stop:
                log(f"stopping: {reason}")
                break

            iteration = stopping.iteration + 1
            log(f"=== iteration {iteration}/{max_iterations} ===")

            candidate_record = generator.generate(
                parent_prompt_id=best_prompt_id,
                parent_text=best_prompt_text,
                parent_mrr=best_mrr,
                iteration=iteration,
                run_name=run_name,
            )
            candidate_text = candidate_record["prompt_text"]
            candidate_id = candidate_record["prompt_id"]
            log(f"generated candidate prompt_id={candidate_id} method={candidate_record['generation_method']}")

            if history.is_duplicate(candidate_text):
                log(f"duplicate candidate at iteration {iteration}, skipping")
                stopping.record_iteration(improved=False)
                _save_iteration_log(log_root, run_name, iteration, {
                    "iteration": iteration,
                    "candidate_id": candidate_id,
                    "status": "duplicate_skipped",
                    "best_mrr": best_mrr,
                })
                _save_run_state(log_root, run_name, {
                    "stopping_state": stopping.to_dict(),
                    "best_prompt_id": best_prompt_id,
                    "best_prompt_text": best_prompt_text,
                    "best_mrr": best_mrr,
                })
                continue

            history.add(candidate_record)

            log(f"scoring candidate prompt_id={candidate_id}...")
            candidate_mrr = compute_dev_mrr(
                all_records,
                dev_indices,
                prompt_text=candidate_text,
                **scorer_kwargs,
            )
            log(f"candidate MRR={candidate_mrr:.4f} (current best={best_mrr:.4f})")

            history.set_score(candidate_id, candidate_mrr)

            score_record = evaluate_candidate(
                prompt_id=candidate_id,
                candidate_mrr=candidate_mrr,
                current_best_mrr=best_mrr,
                epsilon=epsilon,
                iteration=iteration,
            )
            accepted = score_record["accepted"]

            if accepted:
                log(f"ACCEPTED delta={score_record['delta']:+.4f} > epsilon={epsilon}")
                history.update_status(candidate_id, "accepted")
                best_prompt_id = candidate_id
                best_prompt_text = candidate_text
                best_mrr = candidate_mrr
            else:
                log(f"rejected delta={score_record['delta']:+.4f} <= epsilon={epsilon}")
                history.update_status(candidate_id, "rejected")

            stopping.record_iteration(improved=accepted)

            _save_iteration_log(log_root, run_name, iteration, {
                "iteration": iteration,
                "candidate_id": candidate_id,
                "candidate_mrr": candidate_mrr,
                "score_record": score_record,
                "accepted": accepted,
                "best_prompt_id": best_prompt_id,
                "best_mrr": best_mrr,
            })
            _save_run_state(log_root, run_name, {
                "stopping_state": stopping.to_dict(),
                "best_prompt_id": best_prompt_id,
                "best_prompt_text": best_prompt_text,
                "best_mrr": best_mrr,
            })

        _, stop_reason = stopping.should_stop()
        log(f"evolution complete. best_prompt_id={best_prompt_id} best_mrr={best_mrr:.4f}")

        best_prompt_name = args.best_prompt_name or f"best_prompt_{run_name}"
        frozen_path = _freeze_best_prompt(
            best_prompt_text,
            best_prompt_name=best_prompt_name,
            prompts_root=prompts_root,
        )
        log(f"frozen best prompt to {frozen_path}")

        summary: dict[str, Any] = {
            "run_name": run_name,
            "seed_prompt_version": seed_template.version,
            "best_prompt_id": best_prompt_id,
            "best_prompt_name": best_prompt_name,
            "best_prompt_path": str(frozen_path),
            "best_mrr": best_mrr,
            "total_iterations": stopping.iteration,
            "stop_reason": stop_reason,
            "dev_subset": dev_info,
            "config": {
                "max_iterations": max_iterations,
                "patience": patience,
                "epsilon": epsilon,
                "provider": provider,
                "model": model,
                "num_workers": num_workers,
                "score_batch_size": score_batch_size,
            },
            "timestamp": _utc_now(),
        }
        summary_path = log_root / "summaries" / f"{run_name}_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        log(f"summary written to {summary_path}")
        print(json.dumps(summary, indent=2, sort_keys=True))

    except Exception as exc:
        print(f"evolve_prompt failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
