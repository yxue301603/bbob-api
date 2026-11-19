"""Offline end-to-end test using a local HTTP server and a synthetic key."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTest(unittest.TestCase):
    def test_all_arms_and_resume(self):
        attempts = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                attempts.append((self.path, self.headers["Authorization"], body))
                if len(attempts) == 2:
                    self.send_error(429)
                    return
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({
                    "model": "fixture", "choices": [{"message": {"content": '{"pick":0}'}}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
                }).encode())

        with tempfile.TemporaryDirectory() as folder, ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            threading.Thread(target=server.serve_forever, daemon=True).start()
            out = Path(folder) / "results"
            cmd = [sys.executable, str(ROOT/"run.py"), "--model", "fixture", "--functions", "15",
                   "--instances", "18", "--rounds", "2", "--out", str(out),
                   "--base-url", f"http://127.0.0.1:{server.server_port}/v1"]
            env = dict(os.environ, BBO_API_KEY="synthetic-test-secret")
            def run(*extra):
                return subprocess.run(cmd+list(extra), env=env, capture_output=True, text=True)
            try:
                self.assertIn("429", run().stderr)
                self.assertEqual(len(json.loads((out/"f15-i18-history.json").read_bytes())["steps"]), 1)
                resumed = run()
                self.assertEqual(resumed.returncode, 0, resumed.stderr)
                self.assertEqual(run().returncode, 0)
                self.assertNotEqual(run("--model","different").returncode, 0)
                self.assertEqual(len(attempts), 15)
                for path, auth, body in attempts:
                    self.assertEqual((path, auth, body["model"]), ("/v1/chat/completions","Bearer synthetic-test-secret","fixture"))
                    state = json.loads(body["messages"][1]["content"].split("\n",1)[1])
                    self.assertIn(state["queries_left"], (1,2))
                    self.assertEqual(len(state["candidates_z"]),20)
                summary = json.loads((out/"summary.json").read_bytes())
                self.assertEqual(len(summary), 10)
                for field, expected in [("queries",220),("calls",14),("total_tokens",224)]:
                    self.assertEqual(sum(r[field] for r in summary.values()),expected)
                self.assertTrue((out/"results.csv").exists())
                for file in out.glob("*.json"):
                    self.assertNotIn(b"synthetic-test-secret",file.read_bytes())
            finally:
                server.shutdown()


if __name__ == "__main__":
    unittest.main()
