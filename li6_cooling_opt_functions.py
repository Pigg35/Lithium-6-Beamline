######################################################################
# Imports and configuration for the lithium beam source cooling optimization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from dataclasses import dataclass, replace
from scipy.constants import k as k_B, h, c, atomic_mass
from scipy.special import erf
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

rng = np.random.default_rng(12345)

# Constants
h = 6.62607015e-34
hbar = h/(2*np.pi)
kB = 1.380649e-23
amu = 1.66053906660e-27

# NIST Antoine-like fit for Lithium
ANTOINE_A = 4.98831
ANTOINE_B = 7918.984
ANTOINE_C = -9.52
######################################################################



######################################################################
# Configuration dataclass for the lithium beam source simulation
@dataclass
class SourceConfig:
    # N atoms for Monte Carlo simulation
    N_atoms: int = 5000000

    # Isotope constants
    isotope: str = "Li-6"
    mass: float = 6.015 * amu

    # Lithium D2 transition constants
    lambda0: float = 670.977e-9
    k: float = 2 * np.pi / lambda0
    Gamma: float = 2 * np.pi * 5.87e6
    Isat_mW_cm2: float = 2.54
    Isat: float = Isat_mW_cm2 * 1e-3 / 1e-4

    # Oven/source parameters
    T_oven_C: float = 600.0
    T_oven_K: float = T_oven_C + 273.15
    oven_channel_radius: float = 0.02      # m, radius of the oven channel
    tube_length: float = 0.46803           # m
    tube_radius: float = 6.373e-3          # m

    # MOT parameters
    beta_mot: float = 0.0       # damping coefficient for MOT (kg/s)
    kappa_mot: float = 0.0      # spring constant for MOT (N/m)
    length_mot: float = 0.05    # m, length of MOT region

    # Mirrors / laser parameters
    N_bounce_x: int = 6          # number of bounces in x (horizontal) direction
    N_bounce_y: int = 6          # number of bounces in y (vertical) direction
    Rpass_x: float = 0.90        # reflectivity for p-polarized light
    Rpass_y: float = 0.90        # reflectivity for p-polarized light
    length_mirror: float = 0.2   # m, length of mirror region

    Px_mW: float = 80.0
    Py_mW: float = 80.0
    w_mm: float = 6.0

    delta_over_Gamma: float = -1.0   # detuning in units of Gamma (negative for red detuning)

    # Downstream geometry parameters (tube exit at z=0)
    z_cooling_start: float = 0.02                              # m, start of cooling region
    z_cooling_end: float = z_cooling_start + length_mirror     # m, end of cooling region
    z_mot_start: float = z_cooling_end + 0.01                  # m, start of MOT region
    z_mot_end: float = z_mot_start + length_mot                # m, end of MOT region
    z_orifice: float = 3.0                                     # m, position of the orifice plane
    orifice_radius: float = 5e-3                               # m (1 cm diameter)

    # Optional skimmer parameters
    use_skimmer: bool = False
    skimmer_radius: float = 5e-3    # m (1 cm diameter)

    # Monte Carlo simulation parameters
    dt: float = 2e-6
    include_diffusion: bool = True
    alpha_diff: float = 0.5
######################################################################



######################################################################
# Helper functions for basic calculations for source and cooling

# Standard vapor pressure formular (Antoine equation) for lithium, returns in Pa
def lithium_vapor_pressure_Pa(T_K):
    log10P_bar = ANTOINE_A - (ANTOINE_B / (T_K + ANTOINE_C))
    return (10**log10P_bar) * 1e5

# Mean speed from Maxwell-Boltzmann distribution
def mean_speed_MB(T_K, m):
    return np.sqrt(8*kB*T_K/(np.pi*m))

# Effusive flux through a hole, in atoms/s, given T, m, and hole diameter in mm
''' Note: Steffens et al. (1977) used a flux of 2e+14 atoms/s '''
def effusive_flux_atoms_per_s(T_K, m, hole_d):
    P = lithium_vapor_pressure_Pa(T_K)
    n = P/(kB*T_K)
    vbar = mean_speed_MB(T_K, m)
    A = np.pi*(hole_d/2)**2
    return 0.25*n*vbar*A, P, n, vbar, A

# After each reflection, beam intensity is reduced, so total beam intensity is sum of geometric series:
def gain_geometric(N, R):
    if abs(1-R) < 1e-12:
        return float(N)
    return (1 - R**N)/(1 - R)

# Standard function for intensity of a Gaussain beam, given Power and waist
def gaussian_I0(P_W, w_m):
    return 2*P_W/(np.pi*w_m*w_m)

# Scattering rate for a given effective detuning (delta_eff), saturation parameter, and natural linewidth, number of photons
# and atom absorbs and re-emits per second, in the two-level approximation
def gamma_sc(delta_eff, s, Gamma):
    return 0.5*Gamma * (s / (1.0 + s + (2.0*delta_eff/Gamma)**2))

# Local saturation parameter for a Gaussian beam, given effective saturation at the center (s0_eff), mirror waist (w_m),
# and position (x,y)
def s_local_gaussian(s0_eff, w_m, x, y):
    r2 = x*x + y*y
    return s0_eff*np.exp(-2.0*r2/(w_m*w_m))

# Model for the 2D molasses force, the two pairs of beams and mirror geometry
def force_2d_molasses(x, y, vx, vy, delta, Gamma, k, hbar, s0x_eff, s0y_eff, w_m):
    sx = s_local_gaussian(s0x_eff, w_m, x, y)
    sy = s_local_gaussian(s0y_eff, w_m, x, y)

    # Observed delta in atoms frame, used in scattering rate, delta here is the detuning defined by user, 
    # negative for red detuning
    dpx = delta - k*vx # Atom moving away from beam, red shifting
    dmx = delta + k*vx # Atom moving towards beam, blue shifting, brings red detuning closer to resonance
    dpy = delta - k*vy
    dmy = delta + k*vy

    # Calculated scattering rates for each beam, given local saturation and Doppler-shifted detuning, in units of photons/s
    gpx = gamma_sc(dpx, sx, Gamma)
    gmx = gamma_sc(dmx, sx, Gamma)
    gpy = gamma_sc(dpy, sy, Gamma)
    gmy = gamma_sc(dmy, sy, Gamma)

    # Net force, difference between plus and minus beams, using recoil force: F = hbar*k*gamma_sc
    Fx = hbar*k*(gpx - gmx)
    Fy = hbar*k*(gpy - gmy)

    # Total scattering rate, for reference, in photons/s
    gtot = gpx + gmx + gpy + gmy
    
    return Fx, Fy, gtot

# Simply a damped harmonic oscillator model, rather than the full Bloch-equation model for the MOT
def apply_2d_mot_step(x, y, vx, vy, beta, kappa, m, dt):
    """Linear spring+damping toy model for the 2D MOT transverse force."""
    ax = (-kappa*x - beta*vx)/m
    ay = (-kappa*y - beta*vy)/m
    vx = vx + ax*dt
    vy = vy + ay*dt
    return vx, vy

# Function to create a copy of the beam dictionary
def copy_beam(beam):
    return {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in beam.items()}

# Function to sample N points in a uniform disk
def sample_disk(radius, N, rng):
    r = radius * np.sqrt(rng.random(N)) # Uses CDF inversion to get uniform sampling in disk area
    phi = 2*np.pi*rng.random(N)
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    return x, y

# Samples speeds from the flux-weighted Maxwell-Boltzmann distribution for an effusive beam
def sample_flux_weighted_speed(T_K, mass, N, rng):
    # Sample from f(v) ∝ v^3 exp(-mv^2/2kT) using Gamma(k=2, theta=1/a) on v^2
    a = mass / (2*kB*T_K)
    y = rng.gamma(shape=2.0, scale=1.0/a, size=N)   # y = v^2
    # returns ensemble of speeds
    return np.sqrt(y)

# Creates a new key in beam dictionary indicating particles that pass through the final orifice
def apply_orifice_acceptance(beam, cfg):
    accepted = (beam["x"]**2 + beam["y"]**2) <= cfg.orifice_radius**2
    beam["accepted"] = accepted
    return beam
######################################################################



######################################################################
# Simulation functions for the lithium beam source and cooling optimization

# Create a dictionary of beamline parameters for easier management and passing to functions
def sample_effusive_beam(cfg):
    # Using long tube beam sampling function from li6_oven_opt_functions.py
    
    # Let N_entered be the total number of atoms that attempt to enter the capillary channel
    N_entered = int(cfg.N_atoms) 

    # --- ANGULAR SAMPLING (Lambertian Reservoir Source) ---
    u1 = rng.random(N_entered)
    theta_reservoir = np.arcsin(np.sqrt(u1))
    phi = 2 * np.pi * rng.random(N_entered)

    # --- CLAUSING GEOMETRIC FILTER ---
    x = (cfg.tube_length / (2 * cfg.tube_radius)) * np.tan(theta_reservoir)
    
    T_theta = np.zeros(N_entered)
    valid_mask = x < 1
    x_valid = x[valid_mask]
    T_theta[valid_mask] = (2 / np.pi) * (np.arccos(x_valid) - x_valid * np.sqrt(1 - x_valid**2))

    # Rejection sampling step
    u_accept = rng.random(N_entered)
    accepted = u_accept < T_theta
    
    # --- PHYSICAL METRICS TO ANSWER YOUR GOAL ---
    # The true simulated transmission fraction of the capillary geometry
    N_accepted = np.sum(accepted)
    simulated_transmission = N_accepted / N_entered if N_entered > 0 else 0.0
    
    # --- EXTRACT OUTGOING ATOMS ---
    theta_accepted = theta_reservoir[accepted]
    phi_accepted = phi[accepted]

    # --- POSITION AND VELOCITY ---
    x0, y0 = sample_disk(cfg.tube_radius, N_accepted, rng)
    v = sample_flux_weighted_speed(cfg.T_oven_K, cfg.mass, N_accepted, rng)
    z0 = np.zeros(N_accepted)

    nx = np.sin(theta_accepted) * np.cos(phi_accepted)
    ny = np.sin(theta_accepted) * np.sin(phi_accepted)
    nz = np.cos(theta_accepted)

    vx = v * nx
    vy = v * ny
    vz = v * nz


    return {
        "x": x0, "y": y0, "z": z0,
        "vx": vx, "vy": vy, "vz": vz,
        "transmission_probability": simulated_transmission 
    }

# Function for propagating beam through different regions using beam dictionary and source config
def propagate_beam(
    beam, cfg, z_stop, dt_step, mode='drift',
    record_idx=None, n_record=6, beta_mot=0.0, kappa_mot=0.0
):
    # Copy the beam dictionary to avoid modifying the original
    beam = copy_beam(beam)
    
    # Propagate the beam until it reaches z_stop for specified region
    z_stop = float(z_stop)
    snapshots = []

    # Setting up recording of selected trajectories
    if record_idx is not None and len(record_idx) > 0 and n_record > 0:
        record_idx = np.asarray(record_idx, dtype=int)
        nsel = len(record_idx)
        z_start_local = float(np.mean(beam["z"][record_idx]))
        z_targets = np.linspace(z_start_local, z_stop, n_record + 1)[1:]
        x_targets = [np.full(nsel, np.nan, dtype=float) for _ in z_targets]
        z_prev_sel = beam["z"][record_idx].copy()
        x_prev_sel = beam["x"][record_idx].copy()
    else:
        record_idx = None
        z_targets = None
        x_targets = None
        z_prev_sel = None
        x_prev_sel = None

    while True:
        active = beam["z"] < (z_stop - 1e-15)
        if not np.any(active):
            break

        # Uses defined dt_step, unless atoms are in region for less than dt_step time, then uses that time
        dt_local = np.minimum(dt_step, (z_stop - beam["z"][active]) / beam["vz"][active])

        # Calculating effective saturation parameters for the Gaussian beams in x and y directions
        w_m = cfg.w_mm*1e-3
        I0x = gaussian_I0(cfg.Px_mW*1e-3, w_m)
        I0y = gaussian_I0(cfg.Py_mW*1e-3, w_m)
        s0x = I0x/cfg.Isat
        s0y = I0y/cfg.Isat

        Gx = gain_geometric(cfg.N_bounce_x, cfg.Rpass_x)
        Gy = gain_geometric(cfg.N_bounce_y, cfg.Rpass_y)

        s0x_eff = Gx*s0x
        s0y_eff = Gy*s0y

        ### Selecting which propagation mode
        # Mirror mode, or 2D molasses
        if mode == "mirror":
            # Calculates forces for 2D molasses
            Fx, Fy, gtot = force_2d_molasses(
                beam["x"][active], beam["y"][active], beam["vx"][active], beam["vy"][active],
                cfg.delta_over_Gamma*cfg.Gamma, cfg.Gamma, cfg.k, hbar,
                s0x_eff, s0y_eff, w_m
            )

            # Updates velocities using standard Euler integration
            beam["vx"][active] += (Fx/cfg.mass)*dt_local
            beam["vy"][active] += (Fy/cfg.mass)*dt_local

            # Adds diffusion kicks from random photon emission
            if cfg.include_diffusion:
                sigma_v = np.sqrt(cfg.alpha_diff*(hbar*cfg.k)**2 * gtot * dt_local)/cfg.mass
                beam["vx"][active] += rng.normal(0.0, sigma_v)
                beam["vy"][active] += rng.normal(0.0, sigma_v)

        # 2D MOT mode, using damped SHO toy model
        elif mode == "mot":
            vx_new, vy_new = apply_2d_mot_step(
                beam["x"][active], beam["y"][active], beam["vx"][active], beam["vy"][active],
                beta_mot, kappa_mot, cfg.mass, dt_local
            )

            # Updates velocities, Euler integration already done in apply_2d_mot_step
            beam["vx"][active] = vx_new
            beam["vy"][active] = vy_new

        elif mode != "drift":
            raise ValueError(f"Unknown mode: {mode}")

        # Updates positions using standard Euler integration
        beam["x"][active] += beam["vx"][active] * dt_local
        beam["y"][active] += beam["vy"][active] * dt_local
        beam["z"][active] += beam["vz"][active] * dt_local

        if z_targets is not None:
            # grabs current z and x for selected trajectories
            z_sel = beam["z"][record_idx].copy()
            x_sel = beam["x"][record_idx].copy()

            for j, zt in enumerate(z_targets):
                # Checks which trajectory isn't recorded yet, equals True if empty
                pending = np.isnan(x_targets[j])
                if not np.any(pending):
                    continue

                # Checks which trajectories are between previous and current z location
                crossed = pending & (z_prev_sel <= zt) & (z_sel >= zt)
                if np.any(crossed):
                    denom = z_sel[crossed] - z_prev_sel[crossed]
                    frac = np.ones_like(denom)
                    nonzero = np.abs(denom) > 1e-20
                    frac[nonzero] = (zt - z_prev_sel[crossed][nonzero]) / denom[nonzero]
                    frac = np.clip(frac, 0.0, 1.0)
                    x_targets[j][crossed] = x_prev_sel[crossed] + frac * (x_sel[crossed] - x_prev_sel[crossed])

            z_prev_sel = z_sel
            x_prev_sel = x_sel

    beam["z"][:] = np.minimum(beam["z"], z_stop)

    if z_targets is not None:
        x_final_sel = beam["x"][record_idx].copy()
        for zt, xt in zip(z_targets, x_targets):
            if np.any(np.isnan(xt)):
                xt = xt.copy()
                xt[np.isnan(xt)] = x_final_sel[np.isnan(xt)]
            snapshots.append((np.full(len(record_idx), zt, dtype=float), xt))

    return beam, snapshots

# Function for propagating the full system using the beam dictionary and source config
def propagate_system_with_beam(beam, cfg, use_mot = True, mirror_cooling = True, record_traj=False, n_traj=200, n_record_region=6, orifice_reference="mirror_end"):
    # Create a beam copy to avoid modifying the original
    beam = copy_beam(beam)

    # Initialze z positions to z_start
    z =  np.full(len(beam["x"]), 0.0, dtype=float)

    if record_traj:
        # Selects random subset of trajectories to record
        idx = np.arange(len(beam["x"]))
        rng.shuffle(idx)
        sel = np.sort(idx[:min(n_traj, len(beam["x"]))])
        traj = {"z": [z[sel].copy()],
                "x": [beam["x"][sel].copy()]}
    else:
        sel = None
        traj = None 

    # propagate to mirror region
    b1, snaps_before_mirror = propagate_beam(
        beam, cfg, cfg.z_cooling_start, cfg.dt,
        mode="drift",
        record_idx=sel,
        n_record=n_record_region
    )

    if record_traj:
        for zz, xx in snaps_before_mirror:
            traj["z"].append(zz.copy())
            traj["x"].append(xx.copy())
        traj["z"].append(np.full(sel.size, cfg.z_cooling_start))
        traj["x"].append(b1["x"][sel].copy())

    # propagate through mirror region
    if mirror_cooling:
        b2, snaps_mirror = propagate_beam(
            b1, cfg, cfg.z_cooling_end, cfg.dt,
            mode="mirror" if cfg.N_bounce_x > 0 and cfg.N_bounce_y > 0 else "drift",
            record_idx=sel,
            n_record=n_record_region
        )
        if record_traj:
            for zz, xx in snaps_mirror:
                traj["z"].append(zz.copy())
                traj["x"].append(xx.copy())
            traj["z"].append(np.full(sel.size, cfg.z_cooling_end))
            traj["x"].append(b2["x"][sel].copy())
    else:
        b2, snaps_mirror = propagate_beam(
            b1, cfg, cfg.z_cooling_end, cfg.dt,
            mode="drift",
            record_idx=sel,
            n_record=n_record_region
        )

        if record_traj:
            for zz, xx in snaps_mirror:
                traj["z"].append(zz.copy())
                traj["x"].append(xx.copy())
            traj["z"].append(np.full(sel.size, cfg.z_cooling_end))
            traj["x"].append(b2["x"][sel].copy())

    # propagate to MOT region
    b3, snaps_to_mot = propagate_beam(
        b2, cfg, cfg.z_mot_start, cfg.dt,
        mode="drift",
        record_idx=sel,
        n_record=n_record_region
    )

    if record_traj:
        for zz, xx in snaps_to_mot:
            traj["z"].append(zz.copy())
            traj["x"].append(xx.copy())
        traj["z"].append(np.full(sel.size, cfg.z_mot_start))
        traj["x"].append(b3["x"][sel].copy())

    # propagate through MOT region
    if use_mot and (cfg.beta_mot > 0 or cfg.kappa_mot > 0):
        b4, snaps_mot = propagate_beam(
            b3, cfg, cfg.z_mot_end, cfg.dt,
            mode="mot" if cfg.N_bounce_x > 0 and cfg.N_bounce_y > 0 else "drift",
            record_idx=sel,
            n_record=n_record_region,
            beta_mot=cfg.beta_mot,
            kappa_mot=cfg.kappa_mot
        )
        if record_traj:
            for zz, xx in snaps_mot:
                traj["z"].append(zz.copy())
                traj["x"].append(xx.copy())
            traj["z"].append(np.full(sel.size, cfg.z_mot_end))
            traj["x"].append(b4["x"][sel].copy())
    else:
        b4, snaps_mot = propagate_beam(
            b3, cfg, cfg.z_mot_end, cfg.dt,
            mode="drift",
            record_idx=sel,
            n_record=n_record_region
        )

        if record_traj:
            for zz, xx in snaps_mot:
                traj["z"].append(zz.copy())
                traj["x"].append(xx.copy())
            traj["z"].append(np.full(sel.size, cfg.z_mot_end))
            traj["x"].append(b4["x"][sel].copy())

    # propagate to orifice region
    if cfg.z_orifice < cfg.z_mot_end - 1e-15:
        raise ValueError("z_orifice lies upstream of the MOT exit. Increase z_after_cm or shorten L_mot_cm.")

    b5, snaps_orifice = propagate_beam(
        b4, cfg, cfg.z_orifice, cfg.dt,
        mode="drift",
        record_idx=sel,
        n_record=n_record_region
    )

    if record_traj:
        for zz, xx in snaps_orifice:
            traj["z"].append(zz.copy())
            traj["x"].append(xx.copy())
        traj["z"].append(np.full(sel.size, cfg.z_orifice))
        traj["x"].append(b5["x"][sel].copy())

    # Checks which atoms pass through the orifice
    b6 = apply_orifice_acceptance(b5, cfg)
    frac_pass = b6["accepted"].mean()
    Ndot_orifice = len(beam['x']) * frac_pass

    return dict(
        frac_pass=frac_pass,
        Ndot_orifice=Ndot_orifice,
        traj=traj,
        orifice_reference=orifice_reference
    )
######################################################################



######################################################################
# Optimization functions for the lithium beam source cooling parameters

# Adjusting the optimize function to use the beam dictionary and source config
def optimize_2dmot_with_beam(
    beam, cfg,
    L_mot_list_cm=(0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0),
    beta_list=(0.0, 2e-24, 5e-24, 1e-23, 2e-23, 5e-23),
    kappa_list=(0.0, 2e-19, 5e-19, 1e-18, 2e-18, 5e-18),
    orifice_reference="mirror_end",
):
    rows = []
    best = None
    for Lmot in L_mot_list_cm:
        for beta in beta_list:
            for kappa in kappa_list:
                # Create local copy of cfg to avoid modifying the original configuration
                cfg_local = replace(cfg)

                # Update the configuration for the current parameters
                cfg_local.z_mot_end = cfg.z_mot_start + Lmot*1e-2
                cfg_local.beta_mot = float(beta)
                cfg_local.kappa_mot = float(kappa)
                cfg_local.use_mot = (Lmot > 0 and (beta > 0 or kappa > 0))

                # Propagate the system with the current configuration
                info = propagate_system_with_beam(
                    beam=beam,
                    cfg=cfg_local,
                    record_traj=False,
                    n_traj=20,
                    orifice_reference=orifice_reference
                )

                row = dict(
                    L_mot_cm=Lmot,
                    beta=beta,
                    kappa=kappa,
                    frac_pass=info["frac_pass"],
                    Ndot_orifice=info["Ndot_orifice"]
                )
                rows.append(row)
                if best is None or info["Ndot_orifice"] > best["Ndot_orifice"]:
                    best = row.copy()
    return pd.DataFrame(rows), best
######################################################################



######################################################################
# Functions for plotting and visualizing the results of the lithium beam source cooling optimization

# Generate 3 sideview plots for different configurations of the lithium beam source cooling optimization
def plot_sideview(case, cfg, title):
    traj = case["traj"]
    z_list = traj["z"]
    x_list = traj["x"]
    
    plt.figure()
    n = x_list[0].size
    for i in range(n):
        zz = [z_list[j][i] for j in range(len(z_list))]
        xx = [x_list[j][i] for j in range(len(x_list))]
        plt.plot(np.array(zz)*1e2, np.array(xx)*1e3, alpha=0.25)
    
    # region markers
    z_start = 0.0
    z_cooling_start = cfg.z_cooling_start*1e2
    z_end = cfg.z_cooling_end*1e2
    z_mot_end = cfg.z_mot_end*1e2
    z_or = cfg.z_orifice*1e2
    
    plt.axvline(z_start, linestyle="--")
    plt.axvline(z_cooling_start, linestyle="--", alpha=0.6)
    plt.axvline(z_end, linestyle="--")
    plt.axvline(z_mot_end, linestyle="--", alpha=0.6)
    plt.axvline(z_or, linestyle="--", alpha=0.8)

    # Orifice radius marker
    r_or = cfg.orifice_radius*1e3
    plt.axhline(r_or, linestyle="--", color="black")
    plt.axhline(-r_or, linestyle="--", color="black")
    
    plt.xlabel("z (cm)")
    plt.ylabel("x (mm)")
    plt.title(title)
    plt.grid(True)
    # plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
######################################################################