"""
Double-gyre on regular domain.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch

from mqgeometry.config import load_config
from mqgeometry.qgm import QGFV
from mqgeometry.specs import defaults

torch.backends.cudnn.deterministic = True

specs = defaults.get()
config = load_config(Path("configs/double_gyre.toml"))

n_ens = config["n_ens"]
nx = config["xv"].shape[0] - 1
ny = config["yv"].shape[0] - 1
dt = config["dt"]


output_dir = f"run_outputs/{nx}x{ny}_dt{dt}/"
os.makedirs(output_dir) if not os.path.isdir(output_dir) else None

yv = config["yv"]
Ly = yv[-1] - yv[0]

# forcing
yc = 0.5 * (yv[1:] + yv[:-1])  # cell centers
tau0 = config.pop("tau0")
curl_tau = -tau0 * 2 * torch.pi / Ly * torch.sin(2 * torch.pi * yc / Ly).tile((nx, 1))
curl_tau = curl_tau.unsqueeze(0).repeat(n_ens, 1, 1, 1)


qg = QGFV(config)
qg.set_wind_forcing(curl_tau)

# time params
t = 0
n_steps = int(50 * 365 * 24 * 3600 / dt) + 1
freq_log = 1000
n_steps_save = int(10 * 365 * 24 * 3600 / dt) + 1
freq_save = int(15 * 24 * 3600 / dt)
freq_plot = int(10 * 24 * 3600 / dt)

# surface vorticity plot
if freq_plot > 0:
    plt.ion()
    f, a = plt.subplots(1, 1, figsize=(20, 10))
    f.suptitle(f"Upper layer stream function, {t / (365 * 86400):.2f} yrs")


t0 = time.time()

# time integration
for n in range(1, n_steps + 1):
    qg.step()  # one RK3 integration step
    t += dt

    if n % 500 == 0 and torch.isnan(qg.psi).any():
        raise ValueError(f"Stopping, NAN number in p at iteration {n}.")

    if freq_plot > 0 and (n % freq_plot == 0 or n == n_steps):
        w_over_f0 = (qg.laplacian_h(qg.psi, qg.dx, qg.dy) / qg.f0 * qg.masks.psi).cpu()
        im = a.imshow(qg.psi[0, 0].cpu().T, cmap="bwr", origin="lower")
        f.colorbar(im) if n // freq_plot == 1 else None
        f.suptitle(f"Upper layer stream function, {t / (365 * 86400):.2f} yrs")
        plt.pause(0.01)

    if freq_log > 0 and n % freq_log == 0:
        print(
            f"{n=:06d}, t={t / (365 * 24 * 60**2):.2f} yr, "
            f"q: {qg.q.sum().cpu().item():+.5E}, "
            f"qabs: {qg.q.abs().sum().cpu().item():+.5E}"
        )

    if freq_save > 0 and n > n_steps_save and n % freq_save == 0:
        n_years, n_days = (
            int(t // (365 * 24 * 3600)),
            int(t % (365 * 24 * 3600) // (24 * 3600)),
        )
        fname = os.path.join(output_dir, f"psi_{n_years:03d}y_{n_days:03d}d.npy")
        np.save(fname, qg.psi.cpu().numpy().astype("float32"))
        print(f"saved psi to {fname}")
total_time = time.time() - t0
print(total_time)
print(f"{total_time // 3600}h {(total_time % 3600) // 60} min")
