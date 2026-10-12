"""Check the frozen KL simulator against neural_only's canonical generator."""
import ast
import hashlib
from pathlib import Path

from pycode.neural_only import testdata2
from . import simulation


def verify_generator():
    """Require identical Python syntax, allowing only a module-docstring change.

    The completed KL study keeps its fingerprinted numerical sources intact.
    Fresh tuning must pass this check before using that frozen simulator.
    """
    records, signatures = {}, []
    for label, module in (("neural_only", testdata2), ("frozen_kl", simulation)):
        path = Path(module.__file__)
        source = path.read_bytes()
        tree = ast.parse(source.decode("utf-8"))
        if (tree.body and isinstance(tree.body[0], ast.Expr)
                and isinstance(tree.body[0].value, ast.Constant)
                and isinstance(tree.body[0].value.value, str)):
            tree.body.pop(0)
        signatures.append(ast.dump(tree, include_attributes=False))
        records[label] = {"file": str(path), "sha256": hashlib.sha256(source).hexdigest()}
    if signatures[0] != signatures[1]:
        raise RuntimeError(
            "KL simulation.py differs from neural_only/testdata2.py. "
            "Synchronize the generator for a NEW study/result directory; "
            "do not reuse the completed study's caches after changing numerical sources."
        )
    return {"passed": True, "comparison": "identical AST except module docstring", **records}
