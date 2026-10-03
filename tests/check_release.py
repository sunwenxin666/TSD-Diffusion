"""Static release checks; do not execute GPU experiments or write bytecode."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class PublicReleaseTests(unittest.TestCase):
    def test_release_exists(self):
        self.assertTrue(ROOT.is_dir(), "The independent public release is missing")

    def test_required_files(self):
        for name in ("README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "requirements.txt",
                     "sd_autodiffusion_distill/distill/nas/searcher.py"):
            self.assertTrue((ROOT / name).is_file(), name)

    def test_no_data_results_or_weights(self):
        banned = {".png", ".jpg", ".jpeg", ".pdf", ".pt", ".pth", ".ckpt", ".npz",
                  ".npy", ".pkl", ".pickle", ".pb", ".safetensors", ".pyc", ".log", ".csv", ".tsv"}
        for path in ROOT.rglob("*"):
            self.assertNotIn(path.suffix.lower(), banned, str(path))
            self.assertNotEqual(path.name, "__pycache__", str(path))

    def test_python_syntax(self):
        paths = list(ROOT.rglob("*.py"))
        self.assertTrue(paths)
        for path in paths:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))

    def test_experiment_configurations_withheld(self):
        for path in ROOT.rglob("*"):
            if path.is_file():
                self.assertNotIn(path.suffix.lower(), {".json", ".yaml", ".yml"}, str(path))
                self.assertNotIn(path.name, {"config.py", "train_config.py"}, str(path))

    def test_partial_release_notice(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("This repository currently provides the core implementation.", text)
        self.assertIn("will be released upon acceptance.", text)
        self.assertIn("not directly runnable as distributed", text)

    def test_no_private_machine_paths(self):
        prefixes = ("/" + "sdd/", "/" + "home/xu/", "C:" + "\\Users\\")
        for path in ROOT.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".py", ".json", ".yaml", ".md"}:
                content = path.read_text(encoding="utf-8-sig")
                for prefix in prefixes:
                    self.assertNotIn(prefix, content, str(path))

if __name__ == "__main__":
    unittest.main()
