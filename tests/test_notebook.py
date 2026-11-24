"""Notebook packaging checks; runnable without third-party dependencies."""
import ast
import json
from pathlib import Path
import unittest


class NotebookTests(unittest.TestCase):
    def test_five_primary_notebooks_use_shared_source(self):
        import importlib.util
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("generator", root / "generate_notebooks.py")
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        self.assertEqual(set(generator.NAMES), {"paddle", "granite", "got", "internvl", "qwen"})
        for key, filename in generator.NAMES.items():
            nb = json.loads((root / filename).read_text(encoding="utf-8"))
            for cell in nb["cells"]:
                if cell["cell_type"] == "code":
                    ast.parse("".join(cell["source"]))
            setup = "".join(nb["cells"][1]["source"])
            tree = ast.parse(setup)
            assignments = {node.targets[0].id: node.value for node in tree.body
                           if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
            self.assertEqual(ast.literal_eval(assignments["MODEL"]), key)
            payload = ast.literal_eval(assignments["FILES"])
            for path in (root / "src").rglob("*.py"):
                self.assertEqual(payload[path.relative_to(root).as_posix()], path.read_text(encoding="utf-8"))
        for filename in generator.LEGACY:
            self.assertFalse((root / filename).exists())
            self.assertTrue((root / "notebooks/legacy" / filename).exists())

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
        self.assertIn('GPUS = "0,1"', "".join(code))


if __name__ == "__main__":
    unittest.main()
