import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
import pdb

# Double precision is highly recommended for GPs.
# See https://github.com/pytorch/botorch/discussions/1444
train_X = torch.rand(10, 2, dtype=torch.double)
Y = 1 - (train_X - 0.5).norm(dim=-1, keepdim=True)  # explicit output dimension
Y += 0.1 * torch.rand_like(Y)
train_Y = (Y - Y.mean()) / Y.std()

X_init = torch.rand(20, 1, 2, dtype=torch.double)

#print(X_init)

gp = SingleTaskGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)

from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement

UCB = UpperConfidenceBound(gp, beta=0.1)
EI = ExpectedImprovement(gp, best_f=train_Y.max(), maximize=True)

from botorch.optim import optimize_acqf

bounds = torch.stack([torch.zeros(2), torch.ones(2)])
#print(bounds.shape)


candidate, acq_value = optimize_acqf(
    UCB, bounds=bounds, q=2, num_restarts=20, batch_initial_conditions=X_init
)

print(candidate)