"""CPU adaptations of residual Kriging and learned affine SR leaves."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import least_squares, minimize
from scipy.spatial.distance import cdist
from scipy.special import ndtr

import pilot

FAMILIES = (("lin", "sq"), ("lin", "sq", "cross", "cube"),
            ("lin", "sq", "cross", "wave"))


def affine_text(weights):
    return "(" + f"{weights[0]:.12g}" + "".join(
        f"{w:+.12g}*z{i+1}" for i, w in enumerate(weights[1:])) + ")"


def learn_leaves(X, y, wave):
    centered = X-X.mean(axis=0)
    correlations = np.abs(centered.T@(y-y.mean())) / (np.linalg.norm(centered,axis=0)+1e-12)
    leaves = []
    for axis in np.argsort(-correlations, kind="stable")[:2]:
        n_weights = X.shape[1] + 1
        start = np.zeros(n_weights)
        start[axis+1] = 1.
        u = X@start[1:]
        nonlinear = np.sin(2*np.pi*u) if wave else u*u
        outer = np.linalg.lstsq(np.column_stack([np.ones(len(X)),u,nonlinear]),y,rcond=1e-8)[0]
        def residual(parameters):
            weights, coefficients = parameters[:n_weights], parameters[n_weights:]
            z = weights[0]+X@weights[1:]
            transform = np.sin(2*np.pi*z) if wave else z*z
            estimate = coefficients[0]+coefficients[1]*z+coefficients[2]*transform
            return np.r_[(estimate-y)/np.sqrt(len(y)),
                         np.sqrt(.05)*(weights-start), np.sqrt(.001)*coefficients[1:]]
        bounds = (np.r_[np.full(n_weights,-3.),np.full(3,-np.inf)],
                  np.r_[np.full(n_weights,3.),np.full(3,np.inf)])
        result = least_squares(residual, np.r_[start,outer], bounds=bounds, max_nfev=40)
        leaves.append(result.x[:n_weights])
    return np.array(leaves)


def extended_design(X, weights):
    base, names, types = pilot.design(X)
    columns = list(base.T)
    names, types = list(names), list(types)
    u = weights[:,0]+X@weights[:,1:].T
    labels = [affine_text(w) for w in weights]
    for i, name in enumerate(labels):
        values = u[:,i]
        entries = [(values,name,"lin"),(values**2,name+"^2","sq"),
                   (values**3,name+"^3","cube"),
                   (np.sin(2*np.pi*values),"sin(2*pi*"+name+")","wave"),
                   (np.cos(2*np.pi*values),"cos(2*pi*"+name+")","wave")]
        for values, label, kind in entries:
            columns.append(values)
            names.append(label)
            types.append(kind)
    columns.append(u[:,0]*u[:,1])
    names.append(labels[0]+"*"+labels[1])
    types.append("cross")
    return np.array(columns).T, names, np.array(types)


def sparse_fit(V, y, candidates):
    used = []
    matrix = np.ones((len(y),1))
    beta = np.array([y.mean()])
    loss = max(np.mean((y-y.mean())**2),1e-16)
    for _ in range(min(13,len(y)//3)):
        available = np.setdiff1d(candidates,used)
        if len(available)==0:
            break
        residualized = V[:,available]-matrix@np.linalg.lstsq(matrix,V[:,available],rcond=1e-9)[0]
        scores = np.abs(residualized.T@(y-matrix@beta))/(np.linalg.norm(residualized,axis=0)+1e-10)
        chosen = int(available[np.argmax(scores)])
        trial = np.column_stack([matrix,V[:,chosen]])
        coefficients = np.linalg.lstsq(trial,y,rcond=1e-8)[0]
        new_loss = max(np.mean((y-trial@coefficients)**2),1e-16)
        if len(y)*np.log(new_loss/loss)+np.log(len(y))>=0:
            break
        matrix, beta, loss = trial, coefficients, new_loss
        used.append(chosen)
    return used, beta, matrix@beta


def sr_fit(X, y, Q, lcf):
    if not lcf:
        values, formulas, _ = pilot.fit(X,y,np.vstack([X,Q]))
        return values[:len(X)],values[len(X):],formulas,[]
    training, predictions, formulas, leaves = [], [], [], []
    for allowed in FAMILIES:
        weights = learn_leaves(X,y,"wave" in allowed)
        V, names, kinds = extended_design(X,weights)
        candidates = np.flatnonzero(np.isin(kinds,allowed))
        used, beta, fitted = sparse_fit(V,y,candidates)
        q = extended_design(Q,weights)[0]
        training.append(fitted)
        predictions.append(beta[0]+q[:,used]@beta[1:])
        formulas.append(f"{beta[0]:.12g}"+"".join(f"{b:+.12g}*{names[i]}" for b,i in zip(beta[1:],used)))
        leaves.append(weights.tolist())
    return np.array(training).T,np.array(predictions).T,formulas,leaves


def kernel_from_distance(distance, length):
    scaled = distance*np.sqrt(5)/length
    return (1+scaled+scaled**2/3)*np.exp(-scaled)


def residual_gp(X, residual, Q):
    distance = cdist(X,X)
    mean = residual.mean()
    variance = max(np.mean((residual-mean)**2),1e-8)
    standardized = (residual-mean)/np.sqrt(variance)
    def factor(parameters):
        length, nugget = np.exp(parameters)
        covariance = kernel_from_distance(distance,length)+(nugget+1e-9)*np.eye(len(X))
        return cho_factor(covariance,lower=True)
    def loss(parameters):
        cholesky = factor(parameters)
        return .5*standardized@cho_solve(cholesky,standardized)+np.log(np.diag(cholesky[0])).sum()
    optimized = minimize(loss,np.log([.65,.01]),method="L-BFGS-B",
                         bounds=[(np.log(.08),np.log(3.)),(np.log(1e-6),np.log(.1))],
                         options={"maxiter":35,"maxls":15})
    length, nugget = np.exp(optimized.x)
    cholesky = factor(optimized.x)
    cross = kernel_from_distance(cdist(Q,X),length)
    correction = mean+np.sqrt(variance)*cross@cho_solve(cholesky,standardized)
    uncertainty = np.sqrt(variance*np.maximum(1.-np.sum(cross*cho_solve(cholesky,cross.T).T,axis=1),1e-12))
    return correction,uncertainty,dict(length=float(length),nugget=float(nugget),
        residual_mean=float(mean),residual_variance=float(variance),converged=bool(optimized.success))


def predict(X, raw, Q, lcf):
    center = np.median(raw)
    scale = max(np.quantile(raw,.75)-np.quantile(raw,.25),1e-9)
    y = (raw-center)/scale
    training, base, formulas, leaves = sr_fit(X,y,Q,lcf)
    predictions, uncertainty, parameters = [], [], []
    for index in range(3):
        correction, sigma, params = residual_gp(X,y-training[:,index],Q)
        predictions.append(base[:,index]+correction)
        uncertainty.append(sigma)
        parameters.append(params)
    P, S = np.array(predictions).T,np.array(uncertainty).T
    improvement = y.min()-P
    z = improvement/np.maximum(S,1e-12)
    expected = improvement*ndtr(z)+S*np.exp(-.5*z*z)/np.sqrt(2*np.pi)
    return dict(predictions=P.tolist(),std=S.tolist(),formulas=formulas,
                training_rmse=np.sqrt(np.mean((y[:,None]-training)**2,axis=0)).tolist(),
                gp_parameters=parameters,leaves=leaves,ei_pick=int(np.argmax(expected.mean(axis=1))))
