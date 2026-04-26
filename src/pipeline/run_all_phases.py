"""Run phases 1-4 end to end with disk-backed checkpoints and logs."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_ORCHESTRATION_ROOT = DEFAULT_RESULTS_ROOT / "orchestration"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve_python(explicit: str | None) -> str:
    if explicit:
        return explicit
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable or "python3"


@dataclass(frozen=True)
class Step:
    name: str
    command: list[str]
    expected_outputs: list[Path]
    description: str


def _expected_exists(paths: Sequence[Path]) -> bool:
    return all(path.exists() for path in paths)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _shell_join(parts: Sequence[str]) -> str:
    return shlex.join(list(parts))


def build_steps(args: argparse.Namespace, python_bin: str) -> list[Step]:
    provider = args.provider
    model = args.model
    rewrite_num_workers = str(args.rewrite_num_workers)
    rewrite_batch_size = str(args.rewrite_batch_size)
    evolution_num_workers = str(args.evolution_num_workers)
    evolution_score_batch_size = str(args.evolution_score_batch_size)
    phase3_best_prompt_name = f"best_prompt_{args.phase3_run_name}"
    phase3_best_prompt_path = REPO_ROOT / "prompts" / "best_prompt" / f"{phase3_best_prompt_name}.txt"
    rewrite_overwrite_args = ["--overwrite"] if args.restart else []

    steps: list[Step] = [
        Step(
            name="phase1_prepare_dataset",
            description="Prepare clean and adversarial processed datasets.",
            command=[python_bin, "-m", "src.data.prepare_dataset", "--condition", "all"],
            expected_outputs=[
                REPO_ROOT / "data" / "processed" / "clean" / "valid.jsonl",
                REPO_ROOT / "data" / "processed" / "clean" / "test.jsonl",
                REPO_ROOT / "data" / "processed" / "adv" / "valid.jsonl",
                REPO_ROOT / "data" / "processed" / "adv" / "test.jsonl",
            ],
        ),
        Step(
            name="phase1_validate_dataset",
            description="Validate all prepared dataset splits.",
            command=[python_bin, "-m", "src.data.validate_dataset", "--all"],
            expected_outputs=[
                REPO_ROOT / "data" / "manifests" / "clean_valid.json",
                REPO_ROOT / "data" / "manifests" / "clean_test.json",
                REPO_ROOT / "data" / "manifests" / "adv_valid.json",
                REPO_ROOT / "data" / "manifests" / "adv_test.json",
            ],
        ),
    ]

    for split in ("valid", "test"):
        for condition in ("clean", "adv"):
            raw_run_name = f"raw_{split}_{condition}"
            steps.extend(
                [
                    Step(
                        name=f"phase1_bm25_{split}_{condition}",
                        description=f"Run raw BM25 for {split}/{condition}.",
                        command=[
                            python_bin,
                            "-m",
                            "src.retrieval.run_bm25",
                            "--split",
                            split,
                            "--condition",
                            condition,
                            "--run-name",
                            raw_run_name,
                        ],
                        expected_outputs=[
                            REPO_ROOT / "results" / "bm25" / raw_run_name / "metadata.json",
                            REPO_ROOT / "results" / "bm25" / raw_run_name / "predictions.jsonl",
                        ],
                    ),
                    Step(
                        name=f"phase1_eval_{split}_{condition}",
                        description=f"Evaluate raw BM25 for {split}/{condition}.",
                        command=[python_bin, "-m", "src.evaluation.evaluate", "--run-name", raw_run_name],
                        expected_outputs=[REPO_ROOT / "results" / "evaluations" / f"{raw_run_name}.json"],
                    ),
                ]
            )

    for split in ("valid", "test"):
        for condition in ("clean", "adv"):
            rewrite_run_name = f"rewrite_{split}_{condition}"
            rewritten_run_name = f"rewritten_{split}_{condition}"
            steps.extend(
                [
                    Step(
                        name=f"phase2_rewrite_{split}_{condition}",
                        description=f"Generate fixed-prompt rewrites for {split}/{condition}.",
                        command=[
                            python_bin,
                            "-m",
                            "src.rewriting.rewrite_queries",
                            "--split",
                            split,
                            "--condition",
                            condition,
                            "--run-name",
                            rewrite_run_name,
                            "--prompt",
                            args.fixed_prompt_version,
                            "--provider",
                            provider,
                            "--model",
                            model,
                            "--num-workers",
                            rewrite_num_workers,
                            "--batch-size",
                            rewrite_batch_size,
                        ]
                        + rewrite_overwrite_args,
                        expected_outputs=[
                            REPO_ROOT / "rewritten_queries" / condition / rewrite_run_name / "metadata.json",
                            REPO_ROOT / "rewritten_queries" / condition / rewrite_run_name / "rewrites.jsonl",
                        ],
                    ),
                    Step(
                        name=f"phase2_bm25_{split}_{condition}",
                        description=f"Run BM25 over fixed-prompt rewrites for {split}/{condition}.",
                        command=[
                            python_bin,
                            "-m",
                            "src.retrieval.run_bm25",
                            "--split",
                            split,
                            "--condition",
                            condition,
                            "--query-source",
                            "rewritten",
                            "--rewrite-run-name",
                            rewrite_run_name,
                            "--run-name",
                            rewritten_run_name,
                        ],
                        expected_outputs=[
                            REPO_ROOT / "results" / "bm25" / rewritten_run_name / "metadata.json",
                            REPO_ROOT / "results" / "bm25" / rewritten_run_name / "predictions.jsonl",
                        ],
                    ),
                    Step(
                        name=f"phase2_eval_{split}_{condition}",
                        description=f"Evaluate fixed-prompt BM25 for {split}/{condition}.",
                        command=[python_bin, "-m", "src.evaluation.evaluate", "--run-name", rewritten_run_name],
                        expected_outputs=[REPO_ROOT / "results" / "evaluations" / f"{rewritten_run_name}.json"],
                    ),
                ]
            )

    steps.extend(
        [
            Step(
                name="phase2_compare_valid",
                description="Create the phase 2 validation comparison table.",
                command=[
                    python_bin,
                    "-m",
                    "src.evaluation.compare_runs",
                    "--baseline-clean",
                    "raw_valid_clean",
                    "--baseline-adv",
                    "raw_valid_adv",
                    "--candidate-clean",
                    "rewritten_valid_clean",
                    "--candidate-adv",
                    "rewritten_valid_adv",
                    "--comparison-name",
                    "phase2_valid",
                ],
                expected_outputs=[
                    REPO_ROOT / "results" / "comparisons" / "phase2_valid.json",
                    REPO_ROOT / "results" / "comparisons" / "phase2_valid.md",
                ],
            ),
            Step(
                name="phase2_compare_test",
                description="Create the phase 2 test comparison table.",
                command=[
                    python_bin,
                    "-m",
                    "src.evaluation.compare_runs",
                    "--baseline-clean",
                    "raw_test_clean",
                    "--baseline-adv",
                    "raw_test_adv",
                    "--candidate-clean",
                    "rewritten_test_clean",
                    "--candidate-adv",
                    "rewritten_test_adv",
                    "--comparison-name",
                    "phase2_test",
                ],
                expected_outputs=[
                    REPO_ROOT / "results" / "comparisons" / "phase2_test.json",
                    REPO_ROOT / "results" / "comparisons" / "phase2_test.md",
                ],
            ),
            Step(
                name="phase3_evolve_prompt",
                description="Run prompt evolution on the frozen dev subset.",
                command=[
                    python_bin,
                    "-m",
                    "src.evolution.evolve_prompt",
                    "--seed",
                    args.seed_prompt_version,
                    "--run-name",
                    args.phase3_run_name,
                    "--provider",
                    provider,
                    "--model",
                    model,
                    "--num-workers",
                    evolution_num_workers,
                    "--score-batch-size",
                    evolution_score_batch_size,
                    "--max-iterations",
                    str(args.phase3_max_iterations),
                    "--patience",
                    str(args.phase3_patience),
                    "--dev-size",
                    str(args.phase3_dev_size),
                    "--dev-seed",
                    str(args.phase3_dev_seed),
                ],
                expected_outputs=[
                    REPO_ROOT / "evolution_logs" / "summaries" / f"{args.phase3_run_name}_summary.json",
                    phase3_best_prompt_path,
                ],
            ),
        ]
    )

    for split in ("valid", "test"):
        for condition in ("clean", "adv"):
            evolved_rewrite_run_name = f"evolved_{split}_{condition}"
            evolved_bm25_run_name = f"evolved_bm25_{split}_{condition}"
            steps.extend(
                [
                    Step(
                        name=f"phase3_rewrite_{split}_{condition}",
                        description=f"Generate evolved-prompt rewrites for {split}/{condition}.",
                        command=[
                            python_bin,
                            "-m",
                            "src.rewriting.rewrite_queries",
                            "--split",
                            split,
                            "--condition",
                            condition,
                            "--run-name",
                            evolved_rewrite_run_name,
                            "--prompt",
                            phase3_best_prompt_name,
                            "--prompt-path",
                            str(phase3_best_prompt_path),
                            "--provider",
                            provider,
                            "--model",
                            model,
                            "--num-workers",
                            rewrite_num_workers,
                            "--batch-size",
                            rewrite_batch_size,
                        ]
                        + rewrite_overwrite_args,
                        expected_outputs=[
                            REPO_ROOT / "rewritten_queries" / condition / evolved_rewrite_run_name / "metadata.json",
                            REPO_ROOT / "rewritten_queries" / condition / evolved_rewrite_run_name / "rewrites.jsonl",
                        ],
                    ),
                    Step(
                        name=f"phase3_bm25_{split}_{condition}",
                        description=f"Run BM25 over evolved-prompt rewrites for {split}/{condition}.",
                        command=[
                            python_bin,
                            "-m",
                            "src.retrieval.run_bm25",
                            "--split",
                            split,
                            "--condition",
                            condition,
                            "--query-source",
                            "rewritten",
                            "--rewrite-run-name",
                            evolved_rewrite_run_name,
                            "--run-name",
                            evolved_bm25_run_name,
                        ],
                        expected_outputs=[
                            REPO_ROOT / "results" / "bm25" / evolved_bm25_run_name / "metadata.json",
                            REPO_ROOT / "results" / "bm25" / evolved_bm25_run_name / "predictions.jsonl",
                        ],
                    ),
                    Step(
                        name=f"phase3_eval_{split}_{condition}",
                        description=f"Evaluate evolved-prompt BM25 for {split}/{condition}.",
                        command=[python_bin, "-m", "src.evaluation.evaluate", "--run-name", evolved_bm25_run_name],
                        expected_outputs=[REPO_ROOT / "results" / "evaluations" / f"{evolved_bm25_run_name}.json"],
                    ),
                ]
            )

    steps.extend(
        [
            Step(
                name="phase3_compare_valid",
                description="Create the phase 3 validation comparison table.",
                command=[
                    python_bin,
                    "-m",
                    "src.evaluation.compare_runs",
                    "--baseline-clean",
                    "raw_valid_clean",
                    "--baseline-adv",
                    "raw_valid_adv",
                    "--candidate-clean",
                    "rewritten_valid_clean",
                    "--candidate-adv",
                    "rewritten_valid_adv",
                    "--evolved-clean",
                    "evolved_bm25_valid_clean",
                    "--evolved-adv",
                    "evolved_bm25_valid_adv",
                    "--comparison-name",
                    "phase3_valid",
                ],
                expected_outputs=[
                    REPO_ROOT / "results" / "comparisons" / "phase3_valid.json",
                    REPO_ROOT / "results" / "comparisons" / "phase3_valid.md",
                ],
            ),
            Step(
                name="phase3_compare_test",
                description="Create the phase 3 test comparison table.",
                command=[
                    python_bin,
                    "-m",
                    "src.evaluation.compare_runs",
                    "--baseline-clean",
                    "raw_test_clean",
                    "--baseline-adv",
                    "raw_test_adv",
                    "--candidate-clean",
                    "rewritten_test_clean",
                    "--candidate-adv",
                    "rewritten_test_adv",
                    "--evolved-clean",
                    "evolved_bm25_test_clean",
                    "--evolved-adv",
                    "evolved_bm25_test_adv",
                    "--comparison-name",
                    "phase3_test",
                ],
                expected_outputs=[
                    REPO_ROOT / "results" / "comparisons" / "phase3_test.json",
                    REPO_ROOT / "results" / "comparisons" / "phase3_test.md",
                ],
            ),
            Step(
                name="phase4_result_validator",
                description="Validate all final saved metrics.",
                command=[python_bin, "-m", "analysis.result_validator"],
                expected_outputs=[REPO_ROOT / "results" / "phase4" / "validation" / "validation_report.json"],
            ),
            Step(
                name="phase4_error_analysis",
                description="Generate per-query error analysis for test/adv.",
                command=[
                    python_bin,
                    "-m",
                    "analysis.error_analysis",
                    "--split",
                    args.phase4_error_split,
                    "--condition",
                    args.phase4_error_condition,
                ],
                expected_outputs=[REPO_ROOT / "results" / "phase4" / "error_analysis" / "error_summary.json"],
            ),
            Step(
                name="phase4_qualitative_review",
                description="Sample qualitative examples for manual review.",
                command=[
                    python_bin,
                    "-m",
                    "analysis.qualitative_review",
                    "--split",
                    args.phase4_error_split,
                    "--condition",
                    args.phase4_error_condition,
                    "--sample-size",
                    str(args.phase4_sample_size),
                    "--annotator",
                    args.phase4_annotator,
                    "--model",
                    model,
                ],
                expected_outputs=[REPO_ROOT / "results" / "phase4" / "qualitative_review" / "review_samples.csv"],
            ),
            Step(
                name="phase4_generate_tables",
                description="Generate tables and CSVs for reporting.",
                command=[python_bin, "-m", "analysis.generate_tables"],
                expected_outputs=[
                    REPO_ROOT / "reports" / "final_results.csv",
                    REPO_ROOT / "paper" / "final_tables.tex",
                ],
            ),
            Step(
                name="phase4_generate_figures",
                description="Generate paper-ready plots and overlap figure.",
                command=[python_bin, "-m", "analysis.generate_figures"],
                expected_outputs=[
                    REPO_ROOT / "figures" / "plots" / "test_mrr_comparison.png",
                    REPO_ROOT / "figures" / "venn" / "test_adv_improvement_overlap.png",
                ],
            ),
        ]
    )
    return steps


class PipelineRunner:
    def __init__(self, *, args: argparse.Namespace, python_bin: str, steps: list[Step]) -> None:
        self.args = args
        self.python_bin = python_bin
        self.steps = steps
        self.run_dir = Path(args.orchestration_root) / args.run_name
        self.logs_dir = self.run_dir / "logs"
        self.state_path = self.run_dir / "state.json"
        self.manifest_path = self.run_dir / "manifest.json"
        self.summary_path = self.run_dir / "summary.md"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.state = self._load_or_create_state()
        self._write_manifest()

    def _load_or_create_state(self) -> dict[str, Any]:
        if self.state_path.exists() and not self.args.restart:
            with self.state_path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                return payload
        state = {
            "run_name": self.args.run_name,
            "status": "initialized",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "python_bin": self.python_bin,
            "steps": {},
        }
        _write_json(self.state_path, state)
        return state

    def _write_manifest(self) -> None:
        manifest = {
            "run_name": self.args.run_name,
            "created_at": self.state["created_at"],
            "python_bin": self.python_bin,
            "cwd": str(REPO_ROOT),
            "provider": self.args.provider,
            "model": self.args.model,
            "phase3_run_name": self.args.phase3_run_name,
            "restart": self.args.restart,
            "steps": [
                {
                    "name": step.name,
                    "description": step.description,
                    "command": step.command,
                    "expected_outputs": [str(path) for path in step.expected_outputs],
                }
                for step in self.steps
            ],
        }
        _write_json(self.manifest_path, manifest)

    def _save_state(self) -> None:
        self.state["updated_at"] = _utc_now()
        _write_json(self.state_path, self.state)

    def _step_record(self, step: Step) -> dict[str, Any]:
        return self.state.setdefault("steps", {}).setdefault(step.name, {})

    def _step_completed(self, step: Step) -> bool:
        record = self._step_record(step)
        if record.get("status") == "completed" and _expected_exists(step.expected_outputs):
            return True
        if not self.args.restart and _expected_exists(step.expected_outputs):
            record.update(
                {
                    "status": "completed",
                    "completed_at": record.get("completed_at") or _utc_now(),
                    "log_path": str(self.logs_dir / f"{step.name}.log"),
                    "command": step.command,
                    "expected_outputs": [str(path) for path in step.expected_outputs],
                    "completion_source": "artifact_scan",
                }
            )
            self._save_state()
            return True
        return False

    def run(self) -> int:
        self.state["status"] = "running"
        self._save_state()
        for step in self.steps:
            if self._step_completed(step):
                print(f"[skip] {step.name}: outputs already exist")
                continue
            self._run_step(step)
        self.state["status"] = "completed"
        self.state["completed_at"] = _utc_now()
        self._save_state()
        self._write_summary()
        print(f"pipeline completed: {self.run_dir}")
        return 0

    def _run_step(self, step: Step) -> None:
        log_path = self.logs_dir / f"{step.name}.log"
        record = self._step_record(step)
        record.update(
            {
                "status": "running",
                "started_at": _utc_now(),
                "log_path": str(log_path),
                "command": step.command,
                "expected_outputs": [str(path) for path in step.expected_outputs],
                "description": step.description,
            }
        )
        self._save_state()

        header = (
            f"[{_utc_now()}] START {step.name}\n"
            f"description: {step.description}\n"
            f"command: {_shell_join(step.command)}\n\n"
        )
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(header)
            log_handle.flush()
            env = os.environ.copy()
            process = subprocess.Popen(
                step.command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                log_handle.write(line)
                log_handle.flush()
            return_code = process.wait()
            log_handle.write(f"\n[{_utc_now()}] END {step.name} return_code={return_code}\n")
            log_handle.flush()

        record["return_code"] = return_code
        if return_code != 0:
            record["status"] = "failed"
            record["failed_at"] = _utc_now()
            self.state["status"] = "failed"
            self.state["failed_step"] = step.name
            self._save_state()
            raise RuntimeError(f"Step failed: {step.name}")

        if not _expected_exists(step.expected_outputs):
            record["status"] = "failed"
            record["failed_at"] = _utc_now()
            record["failure_reason"] = "expected outputs missing after command completed"
            self.state["status"] = "failed"
            self.state["failed_step"] = step.name
            self._save_state()
            raise RuntimeError(f"Step completed but expected outputs are missing: {step.name}")

        record["status"] = "completed"
        record["completed_at"] = _utc_now()
        self._save_state()

    def _write_summary(self) -> None:
        lines = [
            f"# Pipeline Run `{self.args.run_name}`",
            "",
            f"- Status: `{self.state['status']}`",
            f"- Python: `{self.python_bin}`",
            f"- Provider: `{self.args.provider}`",
            f"- Model: `{self.args.model}`",
            f"- Phase 3 run name: `{self.args.phase3_run_name}`",
            f"- State file: `{self.state_path}`",
            "",
            "## Steps",
            "",
        ]
        for step in self.steps:
            record = self._step_record(step)
            lines.append(f"- `{step.name}`: `{record.get('status', 'unknown')}`")
        lines.append("")
        self.summary_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run phases 1-4 end to end with resume support.")
    parser.add_argument("--run-name", default="full_pipeline", help="Name for the orchestration run directory.")
    parser.add_argument(
        "--orchestration-root",
        default=str(DEFAULT_ORCHESTRATION_ROOT),
        help="Directory for pipeline state, manifest, and step logs.",
    )
    parser.add_argument("--python", default=None, help="Explicit Python interpreter to use.")
    parser.add_argument(
        "--provider",
        choices=["auto", "heuristic", "openai"],
        default="auto",
        help="Provider for phase 2 and phase 3 rewrite steps.",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="Model for OpenAI-backed rewrite and review steps.")
    parser.add_argument("--rewrite-num-workers", type=int, default=24, help="Parallel workers for rewrite_queries.")
    parser.add_argument("--rewrite-batch-size", type=int, default=12, help="Queries per OpenAI request for rewrite_queries.")
    parser.add_argument("--fixed-prompt-version", default="fixed_prompt_v1")
    parser.add_argument("--seed-prompt-version", default="seed_prompt_v1")
    parser.add_argument("--phase3-run-name", default="evolve_v1_openai_fast")
    parser.add_argument("--phase3-max-iterations", type=int, default=25)
    parser.add_argument("--phase3-patience", type=int, default=5)
    parser.add_argument("--phase3-dev-size", type=int, default=300)
    parser.add_argument("--phase3-dev-seed", type=int, default=42)
    parser.add_argument("--evolution-num-workers", type=int, default=24, help="Parallel workers for dev-set scoring.")
    parser.add_argument(
        "--evolution-score-batch-size",
        type=int,
        default=12,
        help="Queries per OpenAI request during evolution dev-set scoring.",
    )
    parser.add_argument("--phase4-error-split", choices=["valid", "test"], default="test")
    parser.add_argument("--phase4-error-condition", choices=["clean", "adv"], default="adv")
    parser.add_argument("--phase4-sample-size", type=int, default=24)
    parser.add_argument("--phase4-annotator", choices=["auto", "heuristic", "openai"], default="auto")
    parser.add_argument(
        "--restart",
        action="store_true",
        help=(
            "Ignore orchestration state and rerun completed steps. "
            "For a truly fresh experiment, also use a new --phase3-run-name."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    python_bin = _resolve_python(args.python)

    if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("run_all_phases failed: OPENAI_API_KEY is required when --provider openai.", file=sys.stderr)
        return 1

    steps = build_steps(args, python_bin)
    runner = PipelineRunner(args=args, python_bin=python_bin, steps=steps)
    try:
        return runner.run()
    except Exception as exc:
        print(f"run_all_phases failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
