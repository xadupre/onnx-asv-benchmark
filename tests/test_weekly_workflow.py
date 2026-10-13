import json
import subprocess
import unittest
from pathlib import Path


WORKFLOW = (
    Path(__file__).resolve().parents[1] / ".github" / "workflows" / "weekly-benchmarks.yml"
).read_text(encoding="utf-8")


class TestWeeklyWorkflow(unittest.TestCase):
    def test_cache_migration_retries_concurrent_pushes(self):
        migration = WORKFLOW.split("  migrate-cache:", 1)[1].split("  benchmark:", 1)[0]
        self.assertIn("for attempt in 1 2 3; do", migration)
        self.assertIn("git -C cache-data pull --rebase origin main", migration)
        self.assertIn("if git -C cache-data push origin main; then", migration)
        self.assertIn('if [[ "$attempt" == 3 ]]; then', migration)

    def test_report_waits_for_jobs_and_reads_all_pages(self):
        report = WORKFLOW.split("  report-failures:", 1)[1]
        for job in ("prepare", "migrate-cache", "benchmark", "publish-data"):
            self.assertIn(f"      - {job}\n", report)
        self.assertIn("if: ${{ always() }}", report)
        self.assertIn("--paginate --slurp", report)

    def test_report_filters_failed_jobs(self):
        report = WORKFLOW.split("  report-failures:", 1)[1]
        query = report.split("              jq '[", 1)[1].split("]'", 1)[0]
        pages = [
            {"jobs": [
                {"name": "prepare", "conclusion": "success"},
                {"name": "migrate-cache", "conclusion": "failure"},
                {"name": "benchmark", "conclusion": "skipped"},
                {"name": "report-failures", "conclusion": "failure"},
            ]},
            {"jobs": [
                {"name": "benchmark (ops/math/add)", "conclusion": "cancelled"},
                {"name": "benchmark (ops/math/mul)", "conclusion": "success"},
                {"name": "publish-data / publish", "conclusion": "failure"},
            ]},
        ]
        result = subprocess.run(
            ["jq", "[" + query + "]"],
            input=json.dumps(pages),
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertEqual(
            [job["name"] for job in json.loads(result.stdout)],
            ["migrate-cache", "benchmark (ops/math/add)", "publish-data / publish"],
        )


if __name__ == "__main__":
    unittest.main()
