# import bezier as bz
#%%
import jax.numpy as jnp 
import numpyro 
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return jax_comb(n, i) * ( t**i ) * (1 - t)**(n-i)

def bezier_curve(points, interp_loc):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
    """

    nPoints = len(points)
    xPoints = jnp.array([p[0] for p in points]) # <- fix 
    yPoints = jnp.array([p[1] for p in points]) # <- fix 

    t = interp_loc 
    polynomial_array = jnp.array([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)]) # <- fix 

    xvals = jnp.dot(xPoints, polynomial_array)
    yvals = jnp.dot(yPoints, polynomial_array)

    return xvals, yvals

def jax_comb(n, k):
    """ 
    Jax version to do n Choose k
    """
    numerator = jnp.prod(jnp.arange(n - k + 1, n + 1))
    denominator = jnp.prod(jnp.arange(1, k + 1))
    return numerator / denominator

def control_points(X, Y, y, BCtype = 'airfoil'):
    """ 
    Calculate control points based on the type of fitting 
    """
    num_ctrl_pts = 10#y.shape[0]+2
    if BCtype == 'airfoil':
            """ 
            Boundary conditions to draw airfoils
            Point 1: [0.0, 0.0] at LE, 
            Point 2: [0.0, y1 ] d
            Point N: [1.0, 0.0] at TE 
            """
            ctrl_x = jnp.linspace(0.0, 1.0, num_ctrl_pts-1)[:-1] # Control points to be optimized 
            ctrl_x = jnp.hstack((0.0, ctrl_x, 1.0))
            ctrl_y = jnp.hstack((0.0, y, Y[-1])) # 0.0, 
            ctrl_pts = jnp.hstack((ctrl_x[:,None], ctrl_y[:,None]))    
    elif BCtype == 'pressure':  
            """ 
            Boundary conditions to draw pressure
            Point 1: [0.0, first element of Cp] at LE, 
            Point 2: [0.0, y1 ] d
            Point N: [1.0, last element of Cp] at TE 
            """
            ctrl_x = jnp.linspace(0.0, 1.0, y.shape[0]+1)[:-1] * jnp.pi/2 # Control points to be optimized 
            ctrl_x = jnp.hstack((0.0, 1-jnp.cos(ctrl_x), 1.0))
            ctrl_y = jnp.hstack((Y[0], y, Y[-1])) # 0.0, 
            ctrl_pts = jnp.hstack((ctrl_x[:,None], ctrl_y[:,None]))    
    else: 
            raise ValueError('Incorrect operation type')
    return ctrl_pts

def split_data(ref_X, X, Y):
    sign_change = jnp.sign(jnp.diff(ref_X))
    split_idx = jnp.min(jnp.where(sign_change < 0.0, jnp.arange(sign_change.shape[0]), 
                jnp.max(jnp.where(sign_change < 0.0, jnp.arange(sign_change.shape[0]),0))))
    
    X1, Y1 = X[:split_idx], Y[:split_idx]
    X2, Y2 = X[split_idx:], Y[split_idx:]
    return X1, Y1, X2, Y2

# %%
import pandas as pd 
import matplotlib.pyplot as plt

path = os.getcwd()
os.chdir(path)

af = pd.read_csv('./data/temp/NACA64A410_coordinates.csv', header=None)
X1, Y1, X2, Y2 = split_data(af[0].values, af[0].values, af[1].values)
X1, Y1 = jnp.flip(X1), jnp.flip(Y1)

ctrl_pt = control_points(X1, Y1, jnp.ones((8)))
cur = bezier_curve(ctrl_pt, X1)
NOISE_VAR = 0.01
# %%
import numpyro
import numpyro.distributions as dist
import jax.random as random
from numpyro.infer import (SVI, Trace_ELBO, autoguide, init_to_median, NUTS, MCMC, TraceMeanField_ELBO)

def model(X, Y): 
    opt_y_locs = dist.Normal(loc = jnp.zeros(8), scale = jnp.ones(8))
    ctrl_pt = control_points(X, Y, opt_y_locs)
    # print
    new_curve = bezier_curve(ctrl_pt, X)
    D = dist.Normal("D", new_curve[1], NOISE_VAR, observed=Y, dims="obs")

svi = None
svi_result = None
# Train new SVI model
total_iterations = 0
svi_init_state = None
guide = autoguide.AutoDelta(model=model) #AutoLaplaceApproximation(model=model_bnn, hessian_fn=lambda f, x: jax.hessian(f)(x) + 1e-3 * jnp.eye(x.shape[0]))
optimizer = numpyro.optim.Adam(step_size = 0.001) 
svi = SVI(model, guide, optimizer, loss=TraceMeanField_ELBO())
rng_key = random.PRNGKey(0)
svi_result = svi.run(rng_key, 10000, X1, Y1, init_state=svi_init_state) 