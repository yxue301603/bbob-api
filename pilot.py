"""BBOB candidate generation, symbolic regression and original decision inputs."""
import json
import os
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from scipy.stats import qmc,rankdata
from scipy.special import ndtr

DIM=5

def seed(f,s,t): return np.random.default_rng(np.random.SeedSequence([260111,f,s,t]))
def initial(f,s):
    return qmc.LatinHypercube(DIM,seed=seed(f,s,0)).random(20)*2-1

def pool(X,y,rng):
    """Identical generator in all arms, independent of fitted models."""
    best=X[np.argmin(y)]
    elite=X[np.argsort(y)[:5]]
    v=[rng.uniform(-1,1,DIM) for _ in range(5)]
    for r in [.025,.075,.2,.5]:
        for _ in range(2): v.append(best+r*rng.normal(size=DIM))
    for _ in range(3):
        z=best.copy(); i=rng.integers(DIM); z[i]+=rng.normal(0,.3); v.append(z)
    for _ in range(4):
        a,b=rng.choice(len(X),2,replace=False)
        v.append(elite[rng.integers(5)]+rng.uniform(.15,.65)*(X[a]-X[b]))
    Q=np.clip(v,-1,1)
    for j in range(len(Q)):
        while np.min(cdist(Q[j:j+1],np.vstack([X,Q[:j]])))<1e-7:
            Q[j]=rng.uniform(-1,1,DIM)
    rng.shuffle(Q)
    return Q

def design(X):
    cols=[];names=[];families=[]
    for i in range(DIM):
        v=X[:,i]; n=f'z{i+1}'
        for feature,label,kind in [(v,n,'lin'),(v*v,n+'^2','sq'),(v**3,n+'^3','cube'),
                (np.cos(np.pi*2*v),'cos(2*pi*'+n+')','wave'),(np.sin(np.pi*2*v),'sin(2*pi*'+n+')','wave')]:
            names.append(label);cols.append(feature);families.append(kind)
    for i in range(DIM):
        for j in range(i):
            names.append(f'z{j+1}*z{i+1}');cols.append(X[:,j]*X[:,i]);families.append('cross')
    return np.array(cols).T,names,np.array(families)

def fit(X,y,Q):
    v,names,types=design(X); q=design(Q)[0]
    predictions=[]; formulas=[]; errors=[]
    for allowed in [('lin','sq'),('lin','sq','cross','cube'),('lin','sq','cross','wave')]:
        candidates=np.flatnonzero(np.isin(types,allowed))
        # Forward symbolic term selection with a BIC complexity penalty.
        used=[]; M=np.ones((len(v),1)); beta=np.array([y.mean()])
        loss=max(np.mean((y-y.mean())**2),1e-16)
        for _ in range(min(13,len(y)//3)):
            residual=y-M@beta; available=np.setdiff1d(candidates,used)
            if not len(available):break
            V=v[:,available]; Vr=V-M@np.linalg.lstsq(M,V,rcond=1e-9)[0]
            gain=np.abs(Vr.T@residual)/(np.linalg.norm(Vr,axis=0)+1e-10)
            k=int(available[np.argmax(gain)]); T=np.column_stack([M,v[:,k]])
            b=np.linalg.lstsq(T,y,rcond=1e-8)[0]
            new=max(np.mean((y-T@b)**2),1e-16)
            if len(y)*np.log(new/loss)+np.log(len(y))>=0:break
            M,beta,loss=T,b,new;used.append(k)
        predictions.append(beta[0]+q[:,used]@beta[1:])
        formulas.append(f'{beta[0]:.7g}'+''.join(f'{b:+.7g}*{names[i]}' for b,i in zip(beta[1:],used)))
        h=np.clip(np.diag(M@np.linalg.pinv(M)),0,.995)
        errors.append(float(np.sqrt(np.mean(((y-M@beta)/(1-h))**2))))
    return np.array(predictions).T,formulas,np.array(errors)

def gp_pick(X,y,Q):
    # Local matched GP reference, deterministic Matern-5/2 + expected improvement.
    def kernel(a,b):
        d=cdist(a,b)*np.sqrt(5)/.65
        return (1+d+d*d/3)*np.exp(-d)
    yc=y-y.mean()
    K=cho_factor(kernel(X,X)+1.e-6*np.eye(len(X)))
    Kx=kernel(Q,X); m=y.mean()+Kx@cho_solve(K,yc)
    sigma=max(y.std(),1e-6)*np.sqrt(np.maximum(1.-np.sum(Kx*cho_solve(K,Kx.T).T,axis=1),1e-12))
    gain=y.min()-m; z=gain/sigma
    return int(np.argmax(gain*ndtr(z)+sigma*np.exp(-.5*z*z)/np.sqrt(2*np.pi)))

def rounded(a):return np.asarray(a).round(6).tolist()
def bundle(X,raw,Q,t,budget=50):
    center=np.median(raw);scale=max(np.quantile(raw,.75)-np.quantile(raw,.25),1.e-9)
    y=(raw-center)/scale
    P,F,E=fit(X,y,Q)
    state=dict(queries_left=budget-t,history_z_then_y=rounded(np.c_[X,y]),
               candidates_z=rounded(Q))
    common="Minimize an unknown continuous function on [-1,1]^5. Return a single candidate index (zero based) for the next real query. Goal: low best-so-far value during search and especially at the end. History rows are z1..z5 followed by standardized y, lower is better. Candidate order is randomized."
    for_agents={"history":common+'\n'+json.dumps(state,separators=(',',':'))}
    state.update(predictions_by_candidate=rounded(P),model_LOO_RMSE=rounded(E))
    common+=" Columns in the prediction matrix are three fitted surrogate models in order. LOO RMSE is an approximate fit diagnostic on that same y scale. Weigh expected improvement and uncertainty using all supplied information."
    for_agents['numeric']=common+'\n'+json.dumps(state,separators=(',',':'))
    state['expressions_in_column_order']=F
    for_agents['formula']=common+'\n'+json.dumps(state,separators=(',',':'))
    quality=rankdata(-P.mean(axis=1));disagreement=rankdata(P.std(axis=1))
    return for_agents,int(np.argmax(.8*quality+.2*disagreement)), (P,F,E)

def write_atomic(path,obj):
    tmp=path.with_suffix('.partial')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    commit(tmp,path)

def commit(tmp,path):
    for i in range(12):
        try:
            os.replace(tmp,path)
            return
        except PermissionError:
            if i==11: raise
            time.sleep(min(.04*2**i,.5))

