"""Small, repeatable symbolic-regression fits on already-standardized observations.

Backends: pyoperon 0.6.1 and gplearn 0.4.3. Both use one worker, 32
individuals, up to six generations, squared-error fitness and the same
arithmetic/trigonometric vocabulary. Their genetic operators, stopping rules
and coefficient optimization are backend-specific. Inputs are never rescaled.
"""
from operator import index

import numpy as np


OPERON_PARAMETERS = dict(
    allowed_symbols="add,sub,mul,sin,cos,constant,variable",
    objectives=["mse"], model_selection_criterion="mean_squared_error",
    optimizer="lm", optimizer_iterations=3, local_search_probability=1.0,
    population_size=32, pool_size=32, generations=6, max_evaluations=1800,
    max_length=15, max_depth=6, initialization_max_length=7,
    initialization_max_depth=4, n_threads=1,
)
GPLEARN_PARAMETERS = dict(
    function_set=("add", "sub", "mul", "sin", "cos"), metric="mse",
    population_size=32, generations=6, tournament_size=5, init_depth=(1, 3),
    parsimony_coefficient=0.001, low_memory=True, n_jobs=1,
)


def _gplearn_expression(program):
    # gplearn's default display rounds every constant to three decimal places.
    # Render the actual fitted tree with round-trip double-precision constants.
    nodes = iter(program.program)

    def visit():
        node = next(nodes)
        if hasattr(node, "arity"):
            return node.name + "(" + ", ".join(visit() for _ in range(node.arity)) + ")"
        if isinstance(node, (int, np.integer)):
            return f"x{node}"
        return repr(float(node))

    return visit()


def predict(method, X, y, Q, seed):
    """Return predictions on Q and the fitted expression. x0 is the first column."""
    X = np.asarray(X, dtype=np.float64, order="F")
    y = np.asarray(y, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64, order="F")
    if (X.ndim != 2 or X.shape[0] < 2 or X.shape[1] == 0 or y.shape != (len(X),)
            or Q.ndim != 2 or Q.shape[1] != X.shape[1]):
        raise ValueError("Expected X[n,d], y[n], Q[k,d] with n>=2 and d>=1")
    if not all(np.isfinite(a).all() for a in (X, y, Q)):
        raise ValueError("Nonfinite input to symbolic regression")
    seed = index(seed)
    if not 0 <= seed < 2**32:
        raise ValueError("Both backends take an unsigned 32-bit seed")
    names = [f"x{i}" for i in range(X.shape[1])]
    if method == "operon":
        from pyoperon.sklearn import SymbolicRegressor
        model = SymbolicRegressor(**OPERON_PARAMETERS, random_state=seed).fit(X, y)
        expression = model.get_model_string(model.model_, names=names, precision=12)
    elif method == "gplearn":
        from gplearn.genetic import SymbolicRegressor
        model = SymbolicRegressor(**GPLEARN_PARAMETERS, random_state=seed, feature_names=names).fit(X, y)
        expression = _gplearn_expression(model._program)
    else:
        raise ValueError(f"Unknown symbolic-regression backend: {method}")
    predictions = np.asarray(model.predict(Q), dtype=np.float64) if len(Q) else np.empty(0)
    if predictions.shape != (len(Q),) or not np.isfinite(predictions).all():
        raise FloatingPointError(f"{method} produced invalid candidate predictions")
    return predictions, expression
