import contextlib
import io
import json
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.publish_results import publish_results
from tools.result_shards import merge_shards, write_shards


class TestPublishResults(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.remote = self.root / "remote.git"
        self.git(self.root, "init", "--bare", "--initial-branch=main", str(self.remote))
        self.publisher = self.make_checkout("publisher")
        self.git(self.publisher, "commit", "--allow-empty", "-m", "Initial")
        self.git(self.publisher, "push", "origin", "main")
        self.competitor = self.make_checkout("competitor")
        self.git(self.competitor, "fetch", "origin", "main")
        self.git(self.competitor, "reset", "--hard", "origin/main")

    @staticmethod
    def git(checkout, *arguments):
        repository = ["--git-dir", str(checkout)] if checkout.suffix == ".git" else []
        return subprocess.run(
            ["git", *repository, *arguments],
            cwd=checkout,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def make_checkout(self, name):
        checkout = self.root / name
        checkout.mkdir()
        self.git(checkout, "init", "--initial-branch=main")
        self.git(checkout, "config", "user.name", "Publisher test")
        self.git(checkout, "config", "user.email", "publisher@example.invalid")
        self.git(checkout, "remote", "add", "origin", str(self.remote))
        return checkout

    def make_results(self, name, results, durations=None):
        source = self.root / name
        machine = source / "worker"
        machine.mkdir(parents=True)
        (source / "benchmarks.json").write_text(
            json.dumps({"version": 2, **{key: {"type": "time"} for key in results}}),
            encoding="utf-8",
        )
        params = {"machine": "worker", "cpu": "Example CPU", "num_cpu": "4"}
        (machine / "machine.json").write_text(json.dumps(params), encoding="utf-8")
        (machine / "abc-env.json").write_text(
            json.dumps(
                {
                    "version": 2,
                    "commit_hash": "abc",
                    "date": 1,
                    "env_name": "existing",
                    "params": params,
                    "results": results,
                    "durations": durations or {},
                }
            ),
            encoding="utf-8",
        )
        return source

    def compete(self, first, second, expected_commits="3", shards=None):
        destination = self.competitor / "data"
        write_shards(first, destination / "shards", selected_shards=shards)
        self.git(self.competitor, "add", ".")
        self.git(self.competitor, "commit", "-m", "Competing publication")
        hook = self.publisher / ".git" / "hooks" / "pre-push"
        hook.write_text(
            "#!/bin/sh\n"
            'rm -- "$0"\n'
            f"git -C {shlex.quote(str(self.competitor))} push origin main\n",
            encoding="utf-8",
        )
        hook.chmod(0o755)
        publish_results(second, self.publisher, "data", shards=shards, retry_delay=0)
        self.assertEqual(
            self.git(self.remote, "rev-parse", "main"),
            self.git(self.publisher, "rev-parse", "HEAD"),
        )
        self.assertFalse((self.publisher / ".git" / "rebase-merge").exists())
        self.assertEqual(self.git(self.publisher, "status", "--porcelain"), "")
        self.assertEqual(
            self.git(self.remote, "rev-list", "--count", "main"), expected_commits
        )
        return self.publisher / "data"

    @staticmethod
    def load(path):
        return json.loads(path.read_text(encoding="utf-8"))

    def test_competing_same_path_combines_disjoint_benchmarks(self):
        first_name = "ops.math.div.Div.time_first"
        second_name = "ops.math.div.Div.time_second"
        first = self.make_results("first", {first_name: [1]})
        second = self.make_results("second", {second_name: [2]})
        destination = self.compete(first, second)
        result = destination / "shards/ops/math/div/Example CPU (4 vCPU)/abc-env.json"
        self.assertEqual(
            self.load(result)["results"], {first_name: [1], second_name: [2]}
        )
        self.assertFalse((destination / "publication-history").exists())

    def test_competing_cpu_div_batches_with_shard_selection(self):
        shard = "cpu_backend_cases/math/cases/div"
        first_name = "cpu_backend_cases.math.cases.DivFloat32Inputs2Shard1.time_div"
        second_name = "cpu_backend_cases.math.cases.DivUint8Inputs2Shard4.time_div"
        excluded_name = "ops.math.add.Add.time_run"
        first = self.make_results("first", {first_name: [1]}, {first_name: 10})
        second = self.make_results(
            "second",
            {second_name: [2], excluded_name: [3]},
            {second_name: 20, excluded_name: 30},
        )
        original = (second / "worker/abc-env.json").read_bytes()
        destination = self.compete(first, second, shards=[shard])
        result = destination / "shards" / shard / "Example CPU (4 vCPU)/abc-env.json"
        self.assertEqual(
            self.load(result)["results"], {first_name: [1], second_name: [2]}
        )
        self.assertEqual(
            self.load(result)["durations"], {first_name: 10, second_name: 20}
        )
        self.assertEqual(
            self.load(destination / "shards" / shard / "benchmarks.json"),
            {"version": 2, first_name: {"type": "time"}, second_name: {"type": "time"}},
        )
        self.assertFalse((destination / "shards/ops").exists())
        self.assertFalse((destination / "publication-history").exists())
        self.assertEqual((second / "worker/abc-env.json").read_bytes(), original)

    def test_competing_same_benchmark_preserves_previous_results(self):
        name = "ops.math.div.Div.time_run"
        first = self.make_results("first", {name: [1]}, {name: 10})
        second = self.make_results("second", {name: [2]}, {name: 20})
        original = (second / "worker/abc-env.json").read_bytes()
        destination = self.compete(first, second)
        result = destination / "shards/ops/math/div/Example CPU (4 vCPU)/abc-env.json"
        self.assertEqual(self.load(result)["results"], {name: [2]})
        archives = [
            path
            for path in (destination / "publication-history").rglob("*.json")
            if path.is_file()
        ]
        self.assertEqual(len(archives), 1)
        self.assertEqual(self.load(archives[0])["results"], {name: [1]})
        self.assertEqual(self.load(archives[0])["durations"], {name: 10})
        self.assertEqual((second / "worker/abc-env.json").read_bytes(), original)
        head = self.git(self.publisher, "rev-parse", "HEAD")
        publish_results(second, self.publisher, "data", retry_delay=0)
        self.assertEqual(self.git(self.publisher, "rev-parse", "HEAD"), head)
        merged = self.root / "merged"
        merge_shards(destination / "shards", merged)
        self.assertEqual(
            self.load(merged / "Example CPU (4 vCPU)/abc-env.json")["results"],
            {name: [2]},
        )

    def test_duration_only_conflict_is_preserved(self):
        name = "ops.math.div.Div.time_run"
        first = self.make_results("first", {name: [1]}, {name: 10})
        second = self.make_results("second", {name: [1]}, {name: 20})
        destination = self.compete(first, second)
        archives = [
            path
            for path in (destination / "publication-history").rglob("*.json")
            if path.is_file()
        ]
        self.assertEqual(len(archives), 1)
        self.assertEqual(self.load(archives[0])["durations"], {name: 10})

    def test_competing_independent_shards(self):
        first = self.make_results("first", {"ops.math.div.Div.time_run": [1]})
        second = self.make_results("second", {"ops.math.add.Add.time_run": [2]})
        destination = self.compete(first, second)
        for shard in ("div", "add"):
            self.assertTrue(
                (
                    destination
                    / f"shards/ops/math/{shard}/Example CPU (4 vCPU)/abc-env.json"
                ).is_file()
            )
        self.assertFalse((destination / "publication-history").exists())

    def test_identical_competing_results_are_noop(self):
        source = self.make_results("source", {"ops.math.div.Div.time_run": [1]})
        destination = self.compete(source, source, expected_commits="2")
        self.assertFalse((destination / "publication-history").exists())

    def test_push_retries_are_bounded(self):
        source = self.make_results("source", {"ops.math.div.Div.time_run": [1]})
        hook = self.publisher / ".git/hooks/pre-push"
        hook.write_text(
            "#!/bin/sh\n"
            f"echo attempt >> {shlex.quote(str(self.root / 'attempts'))}\n"
            "exit 1\n",
            encoding="utf-8",
        )
        hook.chmod(0o755)
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaisesRegex(
            RuntimeError, "after 5 push attempts.*Local results"
        ):
            publish_results(source, self.publisher, "data", retry_delay=0)
        self.assertEqual((self.root / "attempts").read_text().splitlines(), ["attempt"] * 5)
        self.assertEqual(output.getvalue().count("retrying in"), 4)
        self.assertEqual(self.git(self.remote, "rev-list", "--count", "main"), "1")
        self.assertTrue((source / "worker/abc-env.json").is_file())

    def test_transient_remote_rejections_are_retried_with_backoff(self):
        source = self.make_results("source", {"ops.tensor.grid_sample.Grid.time_run": [1]})
        original = (source / "worker/abc-env.json").read_bytes()
        attempts = self.root / "attempts"
        hook = self.remote / "hooks/update"
        hook.write_text(
            "#!/bin/sh\n"
            f"echo attempt >> {shlex.quote(str(attempts))}\n"
            f'if [ "$(wc -l < {shlex.quote(str(attempts))})" -le 3 ]; then\n'
            "  echo 'fatal error in commit_refs' >&2\n"
            "  exit 1\n"
            "fi\n",
            encoding="utf-8",
        )
        hook.chmod(0o755)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            publish_results(source, self.publisher, "data", retry_delay=0.01)
        self.assertEqual(attempts.read_text().splitlines(), ["attempt"] * 4)
        for delay in (0.01, 0.02, 0.04):
            self.assertIn(f"retrying in {delay}s", output.getvalue())
        self.assertEqual(
            self.git(self.remote, "rev-parse", "main"),
            self.git(self.publisher, "rev-parse", "HEAD"),
        )
        self.assertEqual(self.git(self.remote, "rev-list", "--count", "main"), "2")
        result = (
            self.publisher
            / "data/shards/ops/tensor/grid_sample/Example CPU (4 vCPU)/abc-env.json"
        )
        self.assertEqual(
            self.load(result)["results"], {"ops.tensor.grid_sample.Grid.time_run": [1]}
        )
        self.assertEqual((source / "worker/abc-env.json").read_bytes(), original)

    def test_incompatible_metadata_reports_result_path(self):
        name = "ops.math.div.Div.time_run"
        first = self.make_results("first", {name: [1]})
        second = self.make_results("second", {name: [2]})
        write_shards(first, self.competitor / "data/shards")
        self.git(self.competitor, "add", ".")
        self.git(self.competitor, "commit", "-m", "Existing results")
        self.git(self.competitor, "push", "origin", "main")
        path = second / "worker/abc-env.json"
        incoming = self.load(path)
        incoming["env_name"] = "incompatible"
        path.write_text(json.dumps(incoming), encoding="utf-8")
        with self.assertRaisesRegex(
            RuntimeError, "env_name.*abc-env.json.*Local results remain"
        ):
            publish_results(second, self.publisher, "data", retry_delay=0)
        self.assertEqual(self.git(self.remote, "rev-list", "--count", "main"), "2")


if __name__ == "__main__":
    unittest.main()
