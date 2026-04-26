from __future__ import annotations

import unittest
from argparse import Namespace

from src.pipeline.run_all_phases import build_steps


def _args(**overrides: object) -> Namespace:
    data = {
        "run_name": "full_pipeline",
        "orchestration_root": "results/orchestration",
        "python": None,
        "provider": "openai",
        "model": "gpt-4o-mini",
        "rewrite_num_workers": 24,
        "rewrite_batch_size": 12,
        "fixed_prompt_version": "fixed_prompt_v1",
        "seed_prompt_version": "seed_prompt_v1",
        "phase3_run_name": "evolve_v1_openai_fast",
        "phase3_max_iterations": 25,
        "phase3_patience": 5,
        "phase3_dev_size": 300,
        "phase3_dev_seed": 42,
        "evolution_num_workers": 24,
        "evolution_score_batch_size": 12,
        "phase4_error_split": "test",
        "phase4_error_condition": "adv",
        "phase4_sample_size": 24,
        "phase4_annotator": "auto",
        "restart": False,
    }
    data.update(overrides)
    return Namespace(**data)


class RunAllPhasesTests(unittest.TestCase):
    def test_build_steps_includes_pipeline_endpoints(self) -> None:
        steps = build_steps(_args(), "/tmp/python")
        self.assertEqual(steps[0].name, "phase1_prepare_dataset")
        self.assertEqual(steps[-1].name, "phase4_generate_figures")
        self.assertTrue(any(step.name == "phase3_evolve_prompt" for step in steps))
        self.assertTrue(any(step.name == "phase2_compare_test" for step in steps))

    def test_build_steps_uses_derived_best_prompt_path(self) -> None:
        steps = build_steps(_args(phase3_run_name="custom_run"), "/tmp/python")
        rewrite_steps = [step for step in steps if step.name == "phase3_rewrite_test_adv"]
        self.assertEqual(len(rewrite_steps), 1)
        command = rewrite_steps[0].command
        self.assertIn("best_prompt_custom_run", command)
        self.assertTrue(
            any("prompts/best_prompt/best_prompt_custom_run.txt" in part for part in command)
        )


if __name__ == "__main__":
    unittest.main()
