from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.data.loader import DatasetLoader
from src.data.schema import DatasetValidationError, SCHEMA_VERSION, validate_record_dict


def _record(example_id: str) -> dict:
    idx = int(example_id)
    return {
        "example_id": example_id,
        "query_text": f"query {example_id}",
        "code_text": f"code {example_id}",
        "query_tokens": ["query", example_id],
        "code_tokens": ["code", example_id],
        "language": "python",
        "split": "valid",
        "condition": "adv",
        "source_task": "unit_test",
        "metadata": {"idx": idx, "url": f"url-{example_id}"},
    }


class SchemaLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.processed_root = self.root / "processed"
        self.manifest_root = self.root / "manifests"
        (self.processed_root / "adv").mkdir(parents=True)
        self.manifest_root.mkdir(parents=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_split(self, records: list[dict], *, count_override: int | None = None) -> None:
        split_path = self.processed_root / "adv" / "valid.jsonl"
        with split_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record))
                handle.write("\n")

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "split": "valid",
            "condition": "adv",
            "source_task": "unit_test",
            "prepared_at": "2026-01-01T00:00:00Z",
            "record_count": len(records) if count_override is None else count_override,
            "record_path": str(split_path),
            "source_provenance": {"type": "fixture"},
            "file_sha256": None,
        }
        (self.manifest_root / "adv_valid.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

    def test_loader_returns_typed_records(self) -> None:
        self._write_split([_record("1"), _record("2")])
        loader = DatasetLoader(self.processed_root, self.manifest_root)
        records = loader.load_split("valid", "adv")
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].example_id, "1")
        self.assertEqual(records[1].metadata["idx"], 2)

    def test_loader_rejects_duplicate_ids(self) -> None:
        duplicate = _record("1")
        self._write_split([_record("1"), duplicate])
        loader = DatasetLoader(self.processed_root, self.manifest_root)
        with self.assertRaises(DatasetValidationError):
            loader.load_split("valid", "adv")

    def test_loader_rejects_manifest_count_mismatch(self) -> None:
        self._write_split([_record("1")], count_override=2)
        loader = DatasetLoader(self.processed_root, self.manifest_root)
        with self.assertRaises(DatasetValidationError):
            loader.load_split("valid", "adv")

    def test_validate_record_rejects_missing_field(self) -> None:
        record = _record("1")
        record.pop("query_tokens")
        with self.assertRaises(DatasetValidationError):
            validate_record_dict(record)

    def test_validate_record_rejects_wrong_field_type(self) -> None:
        record = _record("1")
        record["code_tokens"] = "not-a-list"
        with self.assertRaises(DatasetValidationError):
            validate_record_dict(record)

    def test_validate_record_rejects_bad_split(self) -> None:
        record = _record("1")
        record["split"] = "train"
        with self.assertRaises(DatasetValidationError):
            validate_record_dict(record)

    def test_validate_record_rejects_bad_condition(self) -> None:
        record = _record("1")
        record["condition"] = "dev"
        with self.assertRaises(DatasetValidationError):
            validate_record_dict(record)


if __name__ == "__main__":
    unittest.main()
