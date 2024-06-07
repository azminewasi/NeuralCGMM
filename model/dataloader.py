import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import neuromancer.psl as psl
from neuromancer.system import Node, System
from neuromancer.modules import blocks
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
from neuromancer.plot import pltCL

def get_data(sys, nsteps, n_samples, xmin_range, batch_size, name="train"):
    #  sampled references for training the policy
    batched_xmin = xmin_range.sample((n_samples, 1, nref)).repeat(1, nsteps + 1, 1)
    batched_xmax = batched_xmin + 2.

    # sampled disturbance trajectories from the simulation model
    batched_dist = torch.stack([torch.tensor(sys.get_D(nsteps)) for _ in range(n_samples)])

    # sampled initial conditions
    batched_x0 = torch.stack([torch.tensor(sys.get_x0()).unsqueeze(0) for _ in range(n_samples)])

    data = DictDataset(
        {"x": batched_x0,
         "y": batched_x0[:,:,[-1]],
         "ymin": batched_xmin,
         "ymax": batched_xmax,
         "d": batched_dist},
        name=name,
    )

    return DataLoader(data, batch_size=batch_size, collate_fn=data.collate_fn, shuffle=False)