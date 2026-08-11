import subprocess
import sys
import unittest
from pathlib import Path


class CLITest(unittest.TestCase):
  def test_bound_environment_launcher_does_not_exec_loop(self):
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, str(root / "tools/intel-qwen36-serve.py"),
         "--version"],
        cwd=root, text=True, capture_output=True, timeout=5, check=False)
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(result.stdout.strip(), "0.1.1")


if __name__ == "__main__":
  unittest.main()
