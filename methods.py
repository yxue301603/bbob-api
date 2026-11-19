"""Build matched numeric and formula inputs or a deterministic rule decision."""
import json
import time
import numpy as np
import pilot
from backend import predict

def prompts(X, raw, Q, t, backend, budget):
    state = dict(queries_left=budget-t,
                 history_z_then_y=pilot.rounded(np.c_[X,(raw-np.median(raw))/max(np.quantile(raw,.75)-np.quantile(raw,.25),1e-9)]),
                 candidates_z=pilot.rounded(Q),
                 predictions_by_candidate=pilot.rounded(backend["predictions"]),
                 residual_std_by_candidate=pilot.rounded(backend["std"]),
                 sr_training_rmse=pilot.rounded(backend["training_rmse"]))
    instruction = (f"Minimize an unknown continuous function on [-1,1]^{X.shape[1]}. Return a single candidate index (zero based) for the next real query. "
        f"Goal: low best-so-far value during search and especially at the end. History rows are z1..z{X.shape[1]} followed by standardized y, lower is better. "
        "Candidate order is randomized. Three surrogate models provide predictions on the same standardized y scale. Each prediction combines a "
        "symbolic expression and a Gaussian-process residual correction learned from observed data. The reported standard deviation is conditional "
        "residual-model uncertainty and excludes uncertainty about the expression itself. Training RMSE is the raw expression's in-sample error, "
        "not a validation error. Weigh expected improvement and uncertainty using all supplied information.")
    numeric = instruction+"\n"+json.dumps(state,separators=(",",":"))
    state["expressions_in_column_order"] = backend["formulas"]
    formula = instruction+"\n"+json.dumps(state,separators=(",",":"))
    return numeric,formula


def prepare(X, raw, Q, t, arm, budget):
    start = time.perf_counter()
    if arm.startswith(("k_","l_")):
        fitted = predict(X,raw,Q,arm.startswith("l_"))
        numeric, formula = prompts(X,raw,Q,t,fitted,budget)
        prompt = formula if arm.endswith("formula") else numeric
        pick = fitted["ei_pick"]
    else:
        alternatives, pick, (P,F,E) = pilot.bundle(X,raw,Q,t,budget)
        name = arm.removeprefix("raw_")
        prompt = alternatives.get(name)
        fitted = dict(predictions=P.tolist(),formulas=F,errors=E.tolist())
    if arm.endswith(("rule","ei")):
        prompt = None
    return prompt,pick,fitted,time.perf_counter()-start


