"""Notebook packaging checks; runnable without third-party dependencies."""
import ast
import json
from pathlib import Path
import unittest


class NotebookTests(unittest.TestCase):
    def test_standalone_sources_and_valid_cells(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "notebooks/OPRB_Kaggle_2GPU.ipynb").read_text(encoding="utf-8"))
        code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        for source in code:
            ast.parse(source)
        tree = ast.parse(code[0])
        payload = next(ast.literal_eval(n.value) for n in tree.body
                       if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FILES" for t in n.targets))
        for folder, pattern in (("src", "*.py"), ("requirements", "*.txt"), ("tests", "test_*.py")):
            for path in (root / folder).rglob(pattern):
                self.assertEqual(payload[path.relative_to(root).as_posix()], path.read_text(encoding="utf-8"))
        self.assertIn('"--gpus", "0,1"', "".join(code))


if __name__ == "__main__":
    unittest.main()
