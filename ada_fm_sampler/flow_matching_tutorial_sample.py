import math
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
import torchdyn
from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_moons

from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *
from torchcfm.optimal_transport import OTPlanSampler

savedir = "models/8gaussian-moons"
os.makedirs(savedir, exist_ok=True)

from sklearn.neighbors import KernelDensity

def compute_nll(samples, reference_samples, bandwidth=0.1):
    """
    Compute the negative log-likelihood of samples under a KDE fit on the reference samples.

    Parameters
    ----------
    samples : Tensor, shape (n, dim)
        The sampled points to evaluate.
    reference_samples : Tensor, shape (n, dim)
        The reference points to fit the KDE.
    bandwidth : float
        The bandwidth parameter for the KDE.

    Returns
    -------
    nll : float
        The average negative log-likelihood of the samples.
    """
    # Convert tensors to numpy arrays
    samples_np = samples.cpu().numpy()
    reference_samples_np = reference_samples.cpu().numpy()

    # Fit a KDE on the reference samples
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(reference_samples_np)

    # Compute log-likelihood of the samples
    log_probs = kde.score_samples(samples_np)

    # Return the negative log-likelihood
    return -np.mean(log_probs)

def compute_trajectory(model, x_start, t_span, num_steps=100, device='cpu'):
    """
    Compute the trajectory of samples over a given time span using the flow model.

    Parameters
    ----------
    model : nn.Module
        The trained flow model.
    x_start : Tensor, shape (bs, dim)
        The initial samples at t=0.
    t_span : Tensor, shape (num_steps,)
        The time steps to compute the trajectory, ranging from 0 to 1.
    num_steps : int
        Number of time steps for the trajectory.
    device : str
        The device for computation ('cpu' or 'cuda').

    Returns
    -------
    traj : Tensor, shape (num_steps, bs, dim)
        The computed trajectory of samples.
    """
    x = x_start.clone().to(device)  # Initial samples
    traj = [x]  # List to store trajectory points

    dt = (t_span[1] - t_span[0]).item()  # Time step size

    for i in range(1, len(t_span)):
        t = t_span[i - 1].unsqueeze(0).repeat(x.shape[0], 1).to(device)  # Current time step
        t_next = t_span[i].unsqueeze(0).repeat(x.shape[0], 1).to(device)  # Next time step

        # Compute the model's output vector field
        xt_input = torch.cat([x, t], dim=-1).to(device)
        v_pred = model(xt_input)

        # Euler update: x(t + dt) = x(t) + v_pred * dt
        x = x + v_pred * dt

        traj.append(x)

    # Stack trajectory points into a tensor
    traj = torch.stack(traj, dim=0)

    return traj


def plot_trajectories(traj, steps):
    """Plot trajectories of some selected samples."""
    n = 2000
    plt.figure(figsize=(6, 6))
    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black")
    plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c="olive")
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c="blue")
    plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"{savedir}/trajectories.png")
    
def sample_conditional_pt(x0, x1, t, sigma):
    """
    Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the target minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    xt : Tensor, shape (bs, *dim)

    References
    ----------
    [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
    """
    t = t.reshape(-1, *([1] * (x0.dim() - 1)))
    mu_t = t * x1 + (1 - t) * x0
    epsilon = torch.randn_like(x0)
    return mu_t + sigma * epsilon


def compute_conditional_vector_field(x0, x1):
    """
    Compute the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the target minibatch

    Returns
    -------
    ut : conditional vector field ut(x1|x0) = x1 - x0

    References
    ----------
    [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
    """
    return x1 - x0

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = "cpu"

ot_sampler = OTPlanSampler(method="exact")
sigma = 0.1
dim = 2
batch_size = 256
model = MLP(dim=dim, time_varying=True).to(device)
optimizer = torch.optim.Adam(model.parameters())
FM = ConditionalFlowMatcher(sigma=sigma)

start = time.time()
for k in tqdm(range(10000)):
    optimizer.zero_grad()

    x0 = sample_8gaussians(batch_size).to(device)
    x1 = sample_moons(batch_size).to(device)

    # Draw samples from OT plan
    x0, x1 = ot_sampler.sample_plan(x0, x1)

    t = torch.rand(x0.shape[0]).type_as(x0).to(device)
    xt = sample_conditional_pt(x0, x1, t, sigma=0.01).to(device)
    ut = compute_conditional_vector_field(x0, x1).to(device)

    vt = model(torch.cat([xt, t[:, None]], dim=-1).to(device))
    loss = torch.mean((vt - ut) ** 2)

    loss.backward()
    optimizer.step()

    if (k + 1) % 500 == 0:
        end = time.time()
        print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
        start = end

        with torch.no_grad():
            x_start = sample_8gaussians(1024).to(device)  # Initial samples
            t_span = torch.linspace(0, 1, 100).to(device)  # Time span for trajectory
            traj = compute_trajectory(model, x_start, t_span, num_steps=100, device=device)

            # Evaluate NLL of the final samples
            final_samples = traj[-1]
            reference_samples = sample_moons(1024).to(device)
            nll = compute_nll(final_samples, reference_samples, bandwidth=0.1)

            print(f"Step {k + 1}: Negative Log-Likelihood (NLL): {nll:.3f}")

            # Plot trajectories
            plot_trajectories(traj.cpu().numpy(), k)
