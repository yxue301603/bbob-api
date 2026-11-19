"""Check observed evaluation counts and recovery after an interrupted API run."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from zipfile import ZipFile


class CocoRunTest(unittest.TestCase):
    def test_interrupted_run(self):
        prompts = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                prompts.append(body["messages"][1]["content"])
                if len(prompts) == 2:
                    self.send_error(429)
                    return
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"choices":[{"message":{"content":"{\\"pick\\":0}"}}]}')

        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder, ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            threading.Thread(target=server.serve_forever, daemon=True).start()
            def run():
                return subprocess.run([sys.executable, str(root / "coco_run.py"),
                    "--arm", "l_formula", "--model", "fixture", "--dimensions", "2",
                    "--budget", "22", "--out", folder, "--base-url", f"http://127.0.0.1:{server.server_port}"],
                    env=dict(os.environ, BBO_API_KEY="fixture"), capture_output=True, text=True)
            try:
                self.assertIn("429", run().stderr)
                resumed = run()
                self.assertEqual(resumed.returncode, 0, resumed.stderr)
                self.assertEqual(len(prompts), 3)
                self.assertIn("z1..z2", prompts[0])
                self.assertIn("expressions_in_column_order", prompts[0])
                self.assertEqual(run().returncode, 0)
                self.assertEqual(len(prompts), 3)
                with ZipFile(Path(folder) / "l_formula-fixture.zip") as archive:
                    infos = [name for name in archive.namelist() if name.endswith(".info")]
                    self.assertEqual(len(infos), 1)
                    self.assertIn(b"1:22|", archive.read(infos[0]))
            finally:
                server.shutdown()


if __name__ == "__main__":
    unittest.main()
