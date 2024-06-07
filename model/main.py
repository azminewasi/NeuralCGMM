import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from math import *

import neuromancer.psl as psl
from neuromancer.system import Node, System
from neuromancer.modules import blocks
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
from neuromancer.plot import pltCL

from dataloader import get_data



# ground truth system model
sys = psl.systems['LinearSimpleSingleZone']()

# problem dimensions
nx = sys.nx           # number of states
nu = sys.nu           # number of control inputs
nd = sys.nD           # number of disturbances
nd_obs = sys.nD_obs   # number of observable disturbances
ny = sys.ny           # number of controlled outputs
nref = ny             # number of references

# control action bounds
umin = torch.tensor(sys.umin)
umax = torch.tensor(sys.umax)

# Data
nsteps = 100 # prediction horizon
n_samples = 2000    # number of sampled scenarios
batch_size = 64

# range for lower comfort bound
xmin_range = torch.distributions.Uniform(18., 20.)

train_loader, dev_loader = [
    get_data(sys, nsteps, n_samples, xmin_range, batch_size, name=name)
    for name in ("train", "dev")
]

# SSM
# extract exact state space model matrices:
A = torch.tensor(sys.A)
B = torch.tensor(sys.Beta)
C = torch.tensor(sys.C)
E = torch.tensor(sys.E)


# state-space model of the building dynamics:
#   x_k+1 =  A x_k + B u_k + E d_k
xnext = lambda x, u, d: x @ A.T + u @ B.T + d @ E.T
state_model = Node(xnext, ['x', 'u', 'd'], ['x'], name='SSM')

#   y_k = C x_k
ynext = lambda x: x @ C.T
output_model = Node(ynext, ['x'], ['y'], name='y=Cx')

# partially observable disturbance model
dist_model = lambda d: d[:, sys.d_idx]
patient_cond_change = Node(dist_model, ['d'], ['patient_obs'], name='patient_cond_change')

# neural net control policy
net = blocks.MLP_bounds(
    insize=ny + 2*nref + nd_obs,
    outsize=nu,
    hsizes=[32, 32],
    nonlin=nn.GELU,
    min=umin,
    max=umax,
)
policy = Node(net, ['y', 'ymin', 'ymax', 'patient_obs'], ['u'], name='policy')

# closed-loop system model
closed_loop_system = System([patient_cond_change, policy, state_model, output_model],
                    nsteps=nsteps,
                    name='closed_loop_system')
closed_loop_system.show()

# dpc
# variables
y = variable('y')
u = variable('u')
ymin = variable('ymin')
ymax = variable('ymax')

# objectives
action_loss = 0.01 * (u == 0.0)  # energy minimization
du_loss = 0.1 * (u[:,:-1,:] - u[:,1:,:] == 0.0)  # delta u minimization to prevent agressive changes in control actions
action_limit_loss = 0.02 * (abs(u[:, 1:, :] - u[:, :-1, :])==0.0)  # constraint loss for insulin delivery

# thermal comfort constraints
state_lower_bound_penalty = 50.*(y > ymin)
state_upper_bound_penalty = 50.*(y < ymax)

# objectives and constraints names for nicer plot
action_loss.name = 'control_loss'
du_loss.name = 'regularization_loss'
action_limit_loss.name = 'insulin_constraint_loss'
state_lower_bound_penalty.name = 'x_min'
state_upper_bound_penalty.name = 'x_max'

# list of constraints and objectives
objectives = [action_loss, du_loss, action_limit_loss]
constraints = [state_lower_bound_penalty, state_upper_bound_penalty]

# data (x_k, r_k) -> parameters (xi_k) -> policy (u_k) -> dynamics (x_k+1)
nodes = [closed_loop_system]
# create constrained optimization loss
loss = PenaltyLoss(objectives, constraints)
# construct constrained optimization problem
problem = Problem(nodes, loss)
# plot computational graph
problem.show()


optimizer = torch.optim.AdamW(problem.parameters(), lr=0.001)
#  Neuromancer trainer
trainer = Trainer(
    problem,
    train_loader,
    dev_loader,
    optimizer=optimizer,
    epochs=200,
    train_metric='train_loss',
    eval_metric='dev_loss',
    warmup=50,
)


# Train control policy
best_model = trainer.train()
# load best trained model
trainer.model.load_state_dict(best_model)


#RANDOM TEST

nsteps_test = 3000

# generate reference
np_refs = psl.signals.step(nsteps_test+1, 1, min=18, max=21, randsteps=8)
ymin_val = torch.tensor(np_refs, dtype=torch.float32).reshape(1, nsteps_test+1, 1)
ymax_val = ymin_val+6.0
# generate disturbance signal
torch_dist = torch.tensor(sys.get_D(nsteps_test+1)).unsqueeze(0)
# initial data for closed loop simulation
x0 = torch.tensor(sys.get_x0()).reshape(1, 1, nx)
data = {'x': x0,
        'y': x0[:, :, [-1]],
        'ymin': ymin_val,
        'ymax': ymax_val,
        'd': torch_dist}
closed_loop_system.nsteps = nsteps_test
# perform closed-loop simulation
trajectories = closed_loop_system(data)

# constraints bounds
Umin = umin * np.ones([nsteps_test, nu])
Umax = umax * np.ones([nsteps_test, nu])
Ymin = trajectories['ymin'].detach().reshape(nsteps_test+1, nref)
Ymax = trajectories['ymax'].detach().reshape(nsteps_test+1, nref)
# plot closed loop trajectories
pltCL(Y=trajectories['y'].detach().reshape(nsteps_test+1, ny),
        R=Ymax,
        X=trajectories['x'].detach().reshape(nsteps_test+1, nx),
        D=trajectories['d'].detach().reshape(nsteps_test+1, nd),
        U=trajectories['u'].detach().reshape(nsteps_test, nu),
        Umin=Umin, Umax=Umax, Ymin=Ymin, Ymax=Ymax)