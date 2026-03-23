import importlib.util
import sys
import os

NOTEBOOK_SCRIPT = "notebook.py"

if not os.path.exists(NOTEBOOK_SCRIPT):
    raise FileNotFoundError(
        "O arquivo notebook.py não foi encontrado. "
        "O notebook foi convertido corretamente?"
    )

spec = importlib.util.spec_from_file_location("student_notebook", NOTEBOOK_SCRIPT)
nb = importlib.util.module_from_spec(spec)
sys.modules["student_notebook"] = nb
spec.loader.exec_module(nb)

def test_required_functions_exist():
    required_functions = [
        "load_data",
        "train_random_forest",
        "train_adaboost",
        "evaluate",
        "run_pipeline"
    ]

    for fn in required_functions:
        assert hasattr(nb, fn), f"A função {fn} não foi implementada."

def test_data_loading():
    X_train, X_test, y_train, y_test = nb.load_data()

    assert X_train is not None
    assert X_test is not None
    assert len(X_train) > 1000
    assert len(X_test) > 1000
    assert len(y_train) == len(X_train)

def test_random_forest_training():
    X_train, X_test, y_train, y_test = nb.load_data()
    model = nb.train_random_forest(X_train, y_train)

    assert model is not None

def test_adaboost_training():
    X_train, X_test, y_train, y_test = nb.load_data()
    model = nb.train_adaboost(X_train, y_train)

    assert model is not None

def test_model_performance():
    acc = nb.run_pipeline("rf", seed=42)

    assert acc > 0.75, (
        "A acurácia do modelo está muito baixa. "
        "Verifique seu pipeline."
    )

def test_reproducibility():
    acc1 = nb.run_pipeline("rf", seed=42)
    acc2 = nb.run_pipeline("rf", seed=42)

    assert abs(acc1 - acc2) < 1e-6, (
        "O pipeline não é reprodutível com a mesma seed."
    )

def test_models_are_different():
    acc_rf = nb.run_pipeline("rf", seed=42)
    acc_ab = nb.run_pipeline("ab", seed=42)

    assert abs(acc_rf - acc_ab) > 1e-4, (
        "Os dois modelos estão retornando exatamente o mesmo resultado."
    )