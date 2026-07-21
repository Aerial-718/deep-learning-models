import ast
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_configs_have_shared_training_contract() -> None:
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["task"] in {"char_lm", "delayed_recall"}
        assert config["seeds"]
        assert int(config["model"]["embedding_size"]) > 0
        assert int(config["model"]["hidden_size"]) > 0
        assert int(config["training"]["steps"]) > 0
        assert float(config["training"]["gradient_clip"]) > 0


def test_notebooks_are_valid_nonempty_v4_documents() -> None:
    paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert len(paths) == 8
    assert {path.name for path in paths} >= {
        "05_vanilla_rnn_core.ipynb",
        "06_lstm_core.ipynb",
        "07_gru_core.ipynb",
    }
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert notebook["cells"]


def test_manual_torch_files_do_not_call_official_recurrent_modules() -> None:
    banned = {"RNN", "RNNCell", "LSTM", "LSTMCell", "GRU", "GRUCell"}
    violations: list[str] = []
    for relative in (
        "vanilla_rnn/torch_impl.py",
        "lstm/torch_impl.py",
        "gru/torch_impl.py",
    ):
        path = ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in banned:
                    violations.append(f"{relative}:{node.lineno}:{node.func.attr}")
    assert not violations, f"official recurrent modules are forbidden: {violations}"


def test_reference_modules_do_not_contain_unfinished_core_logic() -> None:
    for package in ("vanilla_rnn", "lstm", "gru"):
        for filename in ("numpy_impl.py", "torch_impl.py", "language_model.py"):
            source = (ROOT / package / filename).read_text(encoding="utf-8")
            assert "NotImplementedError" not in source, f"unfinished reference: {package}/{filename}"
