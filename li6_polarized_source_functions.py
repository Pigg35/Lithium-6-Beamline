######################################################################
# Imports and configuration for the lithium beam source simulation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from dataclasses import dataclass, replace
from scipy.constants import k as k_B, h, c, atomic_mass
from scipy.special import erf
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

# Constants
amu = atomic_mass
m_Li6 = 6.0151228874 * amu      # kg
I_nuclear = 1
hbar = h/(2*np.pi)

# Approximate D-line wavelength scale used for beam/laser geometry estimates
lambda_D = 670.977e-9           # m (representative Li D-line scale)
k_laser = 2*np.pi/lambda_D
nu_laser = c / lambda_D

# Natural scale; used only in the effective pumping model
Gamma_eff = 2*np.pi * 5.9e6     # rad/s, representative linewidth scale

# Magnetic moments (reduced design-model values)
gS = 2.0023
gI_Li6 = 0.0004476540
######################################################################



######################################################################
# Configuration dataclass for the lithium beam source simulation
@dataclass
class SourceConfig:
    # source geometry and parameters
    T_oven_C: float = 800.0
    oven_channel_radius: float = 0.02      # m, radius of the oven channel
    tube_length: float = 0.46803          # m
    tube_radius: float = 6.373e-3          # m
    z_orifice: float = 0.80              # m downstream of tube exit
    orifice_radius: float = 2.5e-3       # m
    N_atoms: int = 5000000

    # capillary array parameters
    capillary_outer_radius: float = 0.00021 / 2     # m, spacing between capillaries for material
    number_of_capillaries: int = 800
    array_radius: float = 0.0025
    array_shape: str = "hex"                        # "square" or "hex"

    # capillary geometry
    capillary_length: float = 0.0079318       # m
    capillary_radius: float = 0.00008579      # m

    # post-source beamline locations
    z_cooling_start: float = 0.02
    z_cooling_end: float   = 0.18
    z_mot_start: float     = 0.18
    z_mot_end: float       = 0.33
    z_pump_start: float    = 0.34
    z_pump_end: float      = 0.48
    z_rf_start: float      = 0.50
    z_rf_end: float        = 0.57

    # source density estimate at oven exit (effective design parameter)
    n_exit_m3: float = 2.0e18            # effective beam density, adjustable

    # reduced transverse cooling
    cooling_enabled: bool = True
    cooling_vt_scale: float = 0.45       # multiplicative damping on transverse v

    # reduced 2D-MOT
    mot_enabled: bool = True
    mot_capture_radius: float = 7.0e-3   # m
    mot_capture_vt: float = 55.0         # m/s
    mot_compression_factor: float = 0.35 # compress x/y
    mot_vt_factor: float = 0.28          # compress vx/vy

    # optical pumping
    pump_enabled: bool = True
    pump_power_W: float = 0.120
    pump_detuning_MHz: float = 0.0
    pump_shape: str = "elliptical_gaussian"   # "gaussian", "elliptical_gaussian", "top_hat"
    pump_wx: float = 4.0e-3
    pump_wy: float = 2.5e-3
    pump_polarization: str = "sigma0"         # "sigma+", "sigma-", "sigma0"
    pump_repump_fraction: float = 0.18
    pump_target: str = "mi_abs1"              # "mi_abs1" or "mi_0"
    pump_rate_scale: float = 1.0
    optical_thickness_enabled: bool = False
    sigma_abs_eff: float = 1e-15              # m^2, only for optional attenuation

    # RF block
    rf_enabled: bool = True
    B_RF_G: float = 100.0
    rf_B1_G: float = 0.45
    rf_detuning_MHz: float = 0.0
    rf_target: str = "mi_abs1"                # desired final nuclear projection class
    rf_transition_strength: float = 1.0
    rf_mix_width_G: float = 40.0              # reduced mixing scale

######################################################################



######################################################################
# Helper functions for basic calculations related to the lithium beam source

# Returns the most probable, mean, and root-mean-square speeds for a given temperature and particle mass
def thermal_speeds(T_K, mass):
    v_mp = np.sqrt(2*k_B*T_K/mass)
    v_mean = np.sqrt(8*k_B*T_K/(np.pi*mass))
    v_rms = np.sqrt(3*k_B*T_K/mass)
    return v_mp, v_mean, v_rms

# Samples speeds from the flux-weighted Maxwell-Boltzmann distribution for an effusive beam
def sample_flux_weighted_speed(T_K, mass, N, rng):
    # Sample from f(v) ∝ v^3 exp(-mv^2/2kT) using Gamma(k=2, theta=1/a) on v^2
    a = mass / (2*k_B*T_K)
    y = rng.gamma(shape=2.0, scale=1.0/a, size=N)   # y = v^2
    # returns ensemble of speeds
    return np.sqrt(y)

# Samples directions from a cosine-weighted forward hemisphere (Lambertian distribution)
def sample_lambert_directions(N, rng):
    # cosine-weighted forward hemisphere
    u1 = rng.random(N)
    u2 = rng.random(N)
    mu = np.sqrt(u1)             # mu = cos(theta)
    sin_t = np.sqrt(1 - mu**2)
    phi = 2*np.pi*u2 # uniform azimuth ( P(phi) = 1/2pi )
    nx = sin_t*np.cos(phi)
    ny = sin_t*np.sin(phi)
    nz = mu
    return nx, ny, nz

# Samples points uniformly from a disk of given radius; returns x,y arrays
def sample_disk(radius, N, rng):
    r = radius * np.sqrt(rng.random(N)) # Uses CDF inversion to get uniform sampling in disk area
    phi = 2*np.pi*rng.random(N)
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    return x, y

# Returns a list of tuples representing the possible internal state labels (ms, mi) for Li-6
def state_labels():
    labels = []
    for ms in (+0.5, -0.5):
        for mi in (-1, 0, +1):
            labels.append((ms, mi))
    return labels

# Returns the index of a given internal state (ms, mi) in the basis defined by state_labels()
def basis_index(ms, mi):
    STATE_LABELS = state_labels()
    return STATE_LABELS.index((ms, mi))

# Generates the positions of capillaries within an array
def generate_capillary_positions(number_of_capillaries, capillary_outer_radius, array_radius, capillary_radius, array_shape='square'):
    capillary_positions = []

    # Generate capillary positions based on the specified array shape
    if array_shape == 'square':
        grid_size = int(np.sqrt(number_of_capillaries))
        spacing = 2 * capillary_outer_radius
        for i in range(grid_size):
            for j in range(grid_size):
                x_pos = -array_radius + (i + 0.5) * spacing
                y_pos = -array_radius + (j + 0.5) * spacing
                capillary_positions.append((x_pos, y_pos))
    elif array_shape == "hex":
        # Hex packing = staggered rows (triangular lattice), clipped to a circle.
        pitch_x = 2 * capillary_outer_radius
        pitch_y = (np.sqrt(3) / 2) * pitch_x

        # Keep full capillary cross-section inside the array boundary.
        R = array_radius - capillary_outer_radius

        row = 0
        y = -R
        while y <= R + 1e-12:
            x_offset = 0.5 * pitch_x if (row % 2) else 0.0
            x = -R - pitch_x
            while x <= R + pitch_x:
                x_pos = x + x_offset
                if x_pos * x_pos + y * y <= R * R:
                    capillary_positions.append((x_pos, y))
                x += pitch_x
            y += pitch_y
            row += 1
    else:
        raise ValueError("Invalid array_shape. Must be 'square' or 'hex'.")
    
    # Check if any capillaries are outside the array radius for circular array
    if array_shape == 'circle':
        for x_pos, y_pos in capillary_positions:
            if np.sqrt(x_pos**2 + y_pos**2) > array_radius:
                print(f"Capillary at ({x_pos}, {y_pos}) is outside the array radius.")

    # Check if any capillary areas overlap
    for i in range(len(capillary_positions)):
        for j in range(i+1, len(capillary_positions)):
            x1, y1 = capillary_positions[i]
            x2, y2 = capillary_positions[j]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if distance < 2 * capillary_radius:  # Checks if capillary holes are overlapping
                print(f"Capillaries at ({x1}, {y1}) and ({x2}, {y2}) are overlapping.")

    return capillary_positions

# Propagate the beam freely to a new z position without any external forces
def propagate_free(beam, z_new):
    # Calculate the distance to propagate
    dz = z_new - beam["z"]

    # Update the positions based on the velocities and the distance to propagate
    x = beam["x"] + beam["vx"]/beam["vz"] * dz
    y = beam["y"] + beam["vy"]/beam["vz"] * dz

    # Update the z position to the new z position
    z = np.full_like(x, z_new)

    # Create a copy of the beam dictionary to avoid modifying the original beam
    out = {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in beam.items()}
    out["x"], out["y"], out["z"] = x, y, z

    return out

# Calculate the radial distance of the beam particles from the z-axis
def radial_distance(beam):
    return np.sqrt(beam["x"]**2 + beam["y"]**2)

# Function to create a copy of the beam dictionary
def copy_beam(beam):
    return {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in beam.items()}

# Function for calculating pump intensity profile at given x,y positions
def pump_intensity_profile(x, y, cfg):
    # Extract the pump power
    P = cfg.pump_power_W

    # Determine the shape of the pump beam and calculate the intensity profile accordingly
    if cfg.pump_shape == "gaussian":
        w = cfg.pump_wx
        I0 = 2*P/(np.pi*w**2)
        I = I0 * np.exp(-2*(x**2 + y**2)/w**2)

    elif cfg.pump_shape == "elliptical_gaussian":
        wx, wy = cfg.pump_wx, cfg.pump_wy
        I0 = 2*P/(np.pi*wx*wy)
        I = I0 * np.exp(-2*x**2/wx**2 - 2*y**2/wy**2)

    elif cfg.pump_shape == "top_hat":
        wx, wy = cfg.pump_wx, cfg.pump_wy
        area = np.pi * wx * wy
        I = np.where((x/wx)**2 + (y/wy)**2 <= 1.0, P/area, 0.0)

    else:
        raise ValueError("Unknown pump_shape")
    return I

# Function for computing the local density weight based on the spatial distribution of atoms in the pump plane
def compute_local_density_weight(beam, cfg, bins=72, pad=1.25):
    # Fast histogram-based density proxy in the pump plane.
    x = beam["x"]
    y = beam["y"]

    if x.size == 0:
        return np.array([], dtype=float)

    xmax = max(np.max(np.abs(x))*pad, 1.2*cfg.pump_wx, 2e-3)
    ymax = max(np.max(np.abs(y))*pad, 1.2*cfg.pump_wy, 2e-3)
    xedges = np.linspace(-xmax, xmax, bins+1)
    yedges = np.linspace(-ymax, ymax, bins+1)
    H, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
    ix = np.clip(np.digitize(x, xedges)-1, 0, bins-1)
    iy = np.clip(np.digitize(y, yedges)-1, 0, bins-1)
    rho = H[ix, iy].astype(float)
    rho /= max(rho.mean(), 1e-12)
    return rho

# Function for determing the target distribution for optical pumping based on the specified goal and polarization
def target_distribution_from_goal(goal, polarization="sigma0", repump_fraction=0.18):
    # Returns a 6-state target distribution.
    # Goal is expressed in terms of nuclear-spin categories relevant to your source.

    # Initialize an empty state distribution
    STATE_LABELS = state_labels()
    N_STATES = len(STATE_LABELS)

    p = np.zeros(N_STATES)

    '''
    There are four different |mI, mS> states the atoms in the beam can reside, depending on polarization of the pumping
    source, we can pump the atoms into the stretched states |mI=±1, mS=±1>. To represent incomplete pumping, for each
    LH and RH polarization, which should result in the left and right stretched states only, we favor the next two states
    in addition to the stretched states.
    '''

    # Goal is ±1 for the nuclear spin projection, mI
    if goal == "mi_abs1":
        # favor mI=±1, allow both electron-spin manifolds, bias depends on light polarization
        # Right hand polarization
        if polarization == "sigma+":
            favored = [(+0.5, +1), (+0.5, -1), (-0.5, +1)]
        # Left hand polarization
        elif polarization == "sigma-":
            favored = [(-0.5, +1), (-0.5, -1), (+0.5, -1)]
        # Linear polarization
        else:
            favored = [(+0.5, +1), (+0.5, -1), (-0.5, +1), (-0.5, -1)]

        # Normalize the distribution so that the sum of probabilities is 1
        for st in favored:
            p[basis_index(*st)] = 1.0
        p /= p.sum()

        # small repump leak into mI=0
        leak = np.zeros(N_STATES)
        leak[basis_index(+0.5,0)] = 0.5
        leak[basis_index(-0.5,0)] = 0.5
        p = (1-repump_fraction)*p + repump_fraction*leak

    # Goal is 0 for the nuclear spin projection, mI
    elif goal == "mi_0":
        p[basis_index(+0.5,0)] = 0.5
        p[basis_index(-0.5,0)] = 0.5
        leak = np.zeros(N_STATES)
        for st in [(+0.5,+1),(+0.5,-1),(-0.5,+1),(-0.5,-1)]:
            leak[basis_index(*st)] = 0.25
        p = (1-repump_fraction)*p + repump_fraction*leak

    else:
        raise ValueError("Unknown pumping target")
    return p

# Function to compute the effective optical pumping rate based on local intensity and configuration
def effective_pumping_rate(I_local, cfg):
    # reduced saturation-like model
    I_sat_eff = 25.0  # W/m^2, effective design value
    s = I_local / I_sat_eff
    delta = cfg.pump_detuning_MHz * 1e6
    gamma = 5.9e6
    lor = 1.0 / (1.0 + (2*delta/gamma)**2)
    R = cfg.pump_rate_scale * 0.5*gamma * (s/(1+s)) * lor
    return R

# Function used to calculate the RF transfer efficiency, or the probability of transitioning between mI projections
def rf_transfer_efficiency(vz, cfg):
    # Determine time in RF region
    L = cfg.z_rf_end - cfg.z_rf_start
    t_int = L / vz

    # Standard Rabi frequency for the RF transition
    gamma_rf = 2*np.pi * 1.4e6 * cfg.rf_B1_G   # effective rad/s
    delta = 2*np.pi * cfg.rf_detuning_MHz * 1e6
    omega = np.sqrt(gamma_rf**2 + delta**2)
    P = (gamma_rf**2 / omega**2) * np.sin(0.5 * omega * t_int)**2

    # reduced field-mixing factor, simulates Zeeman detuning by determining variation in RF field
    mix = np.exp(-0.5*((cfg.B_RF_G - 100.0)/cfg.rf_mix_width_G)**2)
    P *= mix * cfg.rf_transition_strength
    return np.clip(P, 0.0, 1.0)

# Builds target state vector distribution
def rf_target_distribution(goal):
    # Initializes state distribution vector
    STATE_LABELS = state_labels()
    N_STATES = len(STATE_LABELS)

    p = np.zeros(N_STATES)

    if goal == "mi_abs1":
        for st in [(+0.5,+1),(+0.5,-1),(-0.5,+1),(-0.5,-1)]:
            p[basis_index(*st)] = 0.25
    elif goal == "mi_0":
        p[basis_index(+0.5,0)] = 0.5
        p[basis_index(-0.5,0)] = 0.5
    else:
        raise ValueError("Unknown rf target")
    return p

# Function to extract nuclear spin populations from the state vector p
def nuclear_populations_from_p(p):
    Np1 = p[:, basis_index(+0.5,+1)] + p[:, basis_index(-0.5,+1)]
    N0  = p[:, basis_index(+0.5,0)]  + p[:, basis_index(-0.5,0)]
    Nm1 = p[:, basis_index(+0.5,-1)] + p[:, basis_index(-0.5,-1)]
    return Nm1, N0, Np1

# Function to calculate the ensemble-averaged nuclear spin polarization components from the state vector p
def ensemble_polarization(beam, weights=None):
    p = beam["p"]
    if weights is None:
        weights = np.ones(len(p))
    weights = np.asarray(weights)
    weights = weights / weights.sum()

    Nm1, N0, Np1 = nuclear_populations_from_p(p)
    Nm1_avg = np.sum(weights * Nm1)
    N0_avg  = np.sum(weights * N0)
    Np1_avg = np.sum(weights * Np1)

    Pz = Np1_avg - Nm1_avg
    Pzz = Np1_avg + Nm1_avg - 2*N0_avg

    return {
        "Nm1": Nm1_avg,
        "N0": N0_avg,
        "Np1": Np1_avg,
        "Pz": Pz,
        "Pzz": Pzz
    }

# Creates a new key in beam dictionary indicating particles that pass through the final orifice
def apply_orifice_acceptance(beam, cfg):
    b = propagate_free(beam, cfg.z_orifice)
    accepted = (b["x"]**2 + b["y"]**2) <= cfg.orifice_radius**2
    b["accepted"] = accepted
    return b

# Creates a new dictionary containing only the particles that are accepted through the final orifice
def accepted_beam(beam):
    mask = beam["accepted"]
    return {k: (v[mask] if isinstance(v, np.ndarray) and len(v)==len(mask) else v) for k,v in beam.items()}

# This is the emitted particle flux from the source inside the oven
def source_rate_effusive(cfg):
    T_K = cfg.T_oven_C + 273.15
    _, v_mean, _ = thermal_speeds(T_K, m_Li6)
    A = np.pi * cfg.oven_channel_radius**2  # Cross-sectional area of the oven channel before tube or capillaries

    # if there no oven flux estimate, use Antoine's law to estimate the vapor pressure and calculate the flux
    if cfg.n_exit_m3 is not None:
        return 0.25 * cfg.n_exit_m3 * v_mean * A
    else:
        # NIST Antoine-like fit (order-of-mag)
        ANTOINE_A = 4.98831
        ANTOINE_B = 7918.984
        ANTOINE_C = -9.52

        log10P_bar = ANTOINE_A - (ANTOINE_B / (T_K + ANTOINE_C))
        vapor_pressure = (10**log10P_bar) * 1e5

        n_vapor = vapor_pressure / (k_B * T_K)

        return 0.25 * n_vapor * v_mean * A

# Function for accepted flux through the orifice
def accepted_flux(beam_orifice, cfg, aperture_type="long_tube"):
    n_acc = np.count_nonzero(beam_orifice["accepted"])
    n_initial = cfg.N_atoms
    frac = n_acc / n_initial

    # Calculate the geometric acceptance (probability of an atom in the oven channel entering the tube or capillary array)
    if aperture_type == "long_tube":
        geometric_factor = (cfg.tube_radius / cfg.oven_channel_radius)**2
    elif aperture_type == "multi_capillary":
        geometric_factor = (cfg.array_radius / cfg.oven_channel_radius)**2
    else:
        raise ValueError("Unknown aperture_type. Must be 'long_tube' or 'multi_capillary'.")

    return source_rate_effusive(cfg) * frac * geometric_factor, frac * geometric_factor

# Function for calculating the flux exiting the oven
def oven_output_flux(beam, cfg, aperture_type="long_tube"):
    # Use transmission probability to estimate the flux exiting the oven
    transmission_probability = beam.get("transmission_probability", 0.0)

    # Applying geometric entrance factor
    if aperture_type == "long_tube":
        geometric_factor = (cfg.tube_radius / cfg.oven_channel_radius)**2
        transmission_probability *= geometric_factor

        return source_rate_effusive(cfg) * transmission_probability
    elif aperture_type == "multi_capillary":
        geometric_factor = (cfg.array_radius / cfg.oven_channel_radius)**2
        transmission_probability *= geometric_factor

        return source_rate_effusive(cfg) * transmission_probability
    else:
        raise ValueError("Unknown aperture_type. Must be 'long_tube' or 'multi_capillary'.")
######################################################################



######################################################################
# Functions for simulations of oven source

# Samples an effusive beam from the tube exit plane
def sample_effusive_beam(cfg, rng):
    T_K = cfg.T_oven_C + 273.15
    
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
    v = sample_flux_weighted_speed(T_K, m_Li6, N_accepted, rng)
    z0 = np.zeros(N_accepted)
    
    nx = np.sin(theta_accepted) * np.cos(phi_accepted)
    ny = np.sin(theta_accepted) * np.sin(phi_accepted)
    nz = np.cos(theta_accepted)

    vx = v * nx
    vy = v * ny
    vz = v * nz

    # start with equal populations in all 6 states
    STATE_LABELS = state_labels()
    N_STATES = len(STATE_LABELS)

    p = np.full((N_accepted, N_STATES), 1.0/N_STATES)

    return {
        "x": x0, "y": y0, "z": z0,
        "vx": vx, "vy": vy, "vz": vz,
        "p": p,
        # Metrics to answer your exact question:
        "transmission_probability": simulated_transmission 
    }

# Samples an effusive beam from the oven using a multi-capillary array exit
def rejection_sampling_multi_capillary(cfg, rng, print_metrics=True):
    T_K = cfg.T_oven_C + 273.15
    N_try = cfg.N_atoms # Number of atoms attempting to exit the oven reservoir
    
    capillary_L = cfg.capillary_length
    capillary_a = cfg.capillary_radius
    capillary_outer_radius = cfg.capillary_outer_radius
    array_radius = cfg.array_radius
    number_of_capillaries = cfg.number_of_capillaries # square grid
    
    # Generate Capillary Grid Positions
    capillary_positions = generate_capillary_positions(number_of_capillaries, capillary_outer_radius, array_radius, capillary_a, array_shape='hex')

    # Calculating geometrical parameters and transparency fraction
    capillary_positions = np.array(capillary_positions)
    total_array_area = np.pi * (array_radius**2)
    total_open_area = number_of_capillaries * (np.pi * (capillary_a**2))

    transparency_fraction = total_open_area / total_array_area

    
    # if transparency_fraction > 1.0:
    #     raise ValueError("Geometry error: Total capillary area exceeds macroscopic array area!", transparency_fraction)

    # Guard against transparency fraction greater than 1
    if transparency_fraction > 1.0:
        transparency_fraction = 0       # set to zero to for zero flux downstream

    # First Filter: Do the atoms hit an open capillary hole
    u_transparency = rng.random(N_try)  # Use N_try instead of N_reservoir_hits
    hit_hole_mask = u_transparency < transparency_fraction
    N_entered = np.sum(hit_hole_mask)

    # Sample Reservoir Flux (Prior to Capillary)
    # In the reservoir, the angular distribution hitting the aperture is standard Lambertian
    u1 = rng.random(N_entered)
    theta_reservoir = np.arcsin(np.sqrt(u1))  # Inverse transform for cos(theta)sin(theta)
    phi = 2 * np.pi * rng.random(N_entered)
    
    # Evaluate Transmission Probability (The Filter)
    # Calculate the Clausing parameter x for each attempted atom
    x = (capillary_L / (2 * capillary_a)) * np.tan(theta_reservoir)
    
    # Geometric transmission function T(theta)
    # If x >= 1, the atom physically cannot pass without hitting the wall
    T_theta = np.zeros(N_entered)
    valid_mask = x < 1
    x_valid = x[valid_mask]
    T_theta[valid_mask] = (2 / np.pi) * (np.arccos(x_valid) - x_valid * np.sqrt(1 - x_valid**2))
    
    # Rejection sampling: accept if a random number is below T(theta)
    u_accept = rng.random(N_entered)
    accepted = u_accept < T_theta
    
    # Extract Transmitted Atoms
    N_accepted = np.sum(accepted)
    theta_accepted = theta_reservoir[accepted]
    phi_accepted = phi[accepted]
    
    # Sample local spatial offsets inside the capillaries for accepted atoms
    dx, dy = sample_disk(capillary_a, N_accepted, rng)
    capillary_indices = rng.integers(0, len(capillary_positions), N_accepted)
    
    x0 = capillary_positions[capillary_indices, 0] + dx
    y0 = capillary_positions[capillary_indices, 1] + dy
    z0 = np.zeros(N_accepted)

    n = N_accepted
    
    # Sample speeds for accepted atoms
    v = sample_flux_weighted_speed(T_K, m_Li6, N_accepted, rng)
    
    # Directions
    nx = np.sin(theta_accepted) * np.cos(phi_accepted)
    ny = np.sin(theta_accepted) * np.sin(phi_accepted)
    nz = np.cos(theta_accepted)

    # start with equal populations in all 6 states
    STATE_LABELS = state_labels()
    N_STATES = len(STATE_LABELS)

    p = np.full((n, N_STATES), 1.0/N_STATES)
    
    # Metrics
    transmission_efficiency = N_accepted / N_try

    if print_metrics:
        print(f"Attempted: {N_try} | Transmitted: {N_accepted}")
        print(f"Total Array Transmission Efficiency: {transmission_efficiency * 100:.3f}%")
    
    return {
        "x": x0, "y": y0, "z": z0,
        "vx": v * nx, "vy": v * ny, "vz": v * nz,
        "transmission_probability": transmission_efficiency,
        "p": p
    }
######################################################################



# ####################################################################
# Functions for modeling different components and interactions of the lithium beam

# Function to apply transverse cooling to the beam
def apply_transverse_cooling(beam, cfg):
    # Create a copy of the beam dictionary
    b = copy_beam(beam)

    # Check if transverse cooling is enabled
    if not cfg.cooling_enabled:
        return propagate_free(b, cfg.z_cooling_end)
    
    # Propagate beam to the end of the cooling region
    b = propagate_free(b, cfg.z_cooling_end)

    # Simple model for transverse cooling, scale velocities by cooling factor
    b["vx"] *= cfg.cooling_vt_scale
    b["vy"] *= cfg.cooling_vt_scale

    return b

# Function for applying the 2D MOT cooling model
def apply_2d_mot(beam, cfg):
    # Create a copy of the beam dictionary
    b = copy_beam(beam)

    # Check to see if the 2D MOT is enabled
    if not cfg.mot_enabled:
        return propagate_free(b, cfg.z_mot_end)

    # Propagate beam through 2D MOT region
    b = propagate_free(b, cfg.z_mot_end)

    # Extract radial distance and transverse speed
    r = radial_distance(b)
    vt = np.sqrt(b["vx"]**2 + b["vy"]**2)

    # Determine which atoms are captured by the MOT
    captured = (r <= cfg.mot_capture_radius) & (vt <= cfg.mot_capture_vt)

    # Use Boolean indexing to apply simple model compression factors to only the captured atoms
    b["x"][captured] *= cfg.mot_compression_factor
    b["y"][captured] *= cfg.mot_compression_factor
    b["vx"][captured] *= cfg.mot_vt_factor
    b["vy"][captured] *= cfg.mot_vt_factor

    # Adds new key to dictionary to track which atoms were captured by the MOT
    b["captured_mot"] = captured
    
    return b

# Function to apply optical pumping to the beam in the pumping region
def apply_optical_pumping(beam, cfg):
    # Create a copy of the beam dictionary
    b = copy_beam(beam)

    # Propagate to the start of the pump region
    b = propagate_free(b, cfg.z_pump_start)

    # Guard for empty beam
    if len(b["x"]) == 0:
        b["I_pump"] = np.array([], dtype=float)
        b["rho_rel"] = np.array([], dtype=float)
        b["pump_eta"] = np.array([], dtype=float)
        return propagate_free(b, cfg.z_pump_end)

    # If no pumping, propagate freely to end of the pump region
    if not cfg.pump_enabled:
        return propagate_free(b, cfg.z_pump_end)

    # Compute pump region length, local pump intensity, and local density weight for attenuation
    L = cfg.z_pump_end - cfg.z_pump_start
    I_local = pump_intensity_profile(b["x"], b["y"], cfg)
    rho_rel = compute_local_density_weight(b, cfg)

    # Mimics optical thickness effects with a crude attenuation proxy based on local density and pump length
    if cfg.optical_thickness_enabled:
        # crude attenuation proxy
        tau = cfg.sigma_abs_eff * cfg.n_exit_m3 * L * rho_rel * 1e-3
        I_local = I_local * np.exp(-tau)

    # Computes effective pumping rate and interaction time
    R = effective_pumping_rate(I_local, cfg)
    t_int = L / b["vz"]
    Theta = R * t_int

    # Determines the target distribution based on the pumping goal and polarization
    target = target_distribution_from_goal(
        cfg.pump_target,
        polarization=cfg.pump_polarization,
        repump_fraction=cfg.pump_repump_fraction
    )

    eta = 1.0 - np.exp(-Theta)   # local approach-to-target fraction
    p0 = b["p"]
    b["p"] = (1 - eta[:,None]) * p0 + eta[:,None] * target[None,:]

    b["I_pump"] = I_local
    b["rho_rel"] = rho_rel
    b["pump_eta"] = eta
    
    # Propagate to end of pump region
    b = propagate_free(b, cfg.z_pump_end)
    
    return b

# Function to apply RF to the beam
def apply_rf_block(beam, cfg):
    b = copy_beam(beam)
    b = propagate_free(b, cfg.z_rf_start)
    if not cfg.rf_enabled:
        return propagate_free(b, cfg.z_rf_end)

    P = rf_transfer_efficiency(b["vz"], cfg)
    target = rf_target_distribution(cfg.rf_target)
    b["p"] = (1 - P[:,None])*b["p"] + P[:,None]*target[None,:]
    b["rf_eta"] = P
    b = propagate_free(b, cfg.z_rf_end)
    return b
######################################################################



######################################################################
# Simulation Functions

# Overall beam simulation function
def run_simulation(cfg, rng, aperture_type='long_tube'):
    # Collecting all beam dictionaries at each stage
    if aperture_type == 'long_tube':
        b0 = sample_effusive_beam(cfg, rng)
    elif aperture_type == 'multi_capillary':
        b0 = rejection_sampling_multi_capillary(cfg, rng)
    else:
        raise ValueError(f"Unknown aperture_type: {aperture_type}")
    b1 = apply_transverse_cooling(b0, cfg)
    b2 = apply_2d_mot(b1, cfg)
    b3 = apply_optical_pumping(b2, cfg)
    b4 = apply_rf_block(b3, cfg)
    b5 = apply_orifice_acceptance(b4, cfg)

    # Calculate the flux exiting the oven based on the transmission probability and geometric factors
    oven_flux = oven_output_flux(b0, cfg, aperture_type=aperture_type)

    # Calculate the accepted beam and flux metrics
    acc = accepted_beam(b5)
    flux_acc, frac_acc = accepted_flux(b5, cfg, aperture_type=aperture_type)

    pol_all = ensemble_polarization(b5)
    pol_acc = ensemble_polarization(acc) if len(acc["x"]) > 0 else {"Nm1":np.nan,"N0":np.nan,"Np1":np.nan,"Pz":np.nan,"Pzz":np.nan}

    fom_abs = flux_acc * abs(pol_acc["Pzz"]) if np.isfinite(pol_acc["Pzz"]) else np.nan

    results = {
        "beam0": b0, "beam1": b1, "beam2": b2, "beam3": b3, "beam4": b4, "beam5": b5,
        "accepted": acc,
        "flux_source": source_rate_effusive(cfg),
        "oven_exit_flux": oven_flux,
        "flux_acc": flux_acc,
        "frac_acc": frac_acc,
        "pol_all": pol_all,
        "pol_acc": pol_acc,
        "FOM_absPzz": fom_abs
    }
    return results

# Function to print a summary table of the simulation results
def summary_table(res):
    print("=== Beamline summary ===")
    print(f"Source flux estimate       : {res['flux_source']:.4e} atoms/s")
    print(f"Accepted fraction          : {res['frac_acc']:.4e}")
    print(f"Oven exit flux              : {res['oven_exit_flux']:.4e} atoms/s")
    print(f"Accepted flux              : {res['flux_acc']:.4e} atoms/s")
    print()
    print("All atoms at orifice plane:")
    for k, v in res["pol_all"].items():
        print(f"  {k:>4s} = {v: .5f}")
    print()
    print("Accepted atoms only:")
    for k, v in res["pol_acc"].items():
        print(f"  {k:>4s} = {v: .5f}")
    print()
    print(f"FOM = flux_acc * |Pzz|     : {res['FOM_absPzz']:.4e}")
    print("===================================")
######################################################################



######################################################################
# Functions for plotting distributions and beam profiles

# Plots longitudinal and transverse velocity distributions and beam profile
def plot_source_distributions(beam):
    # Calculates transverse speed from vx and vy
    vt = np.sqrt(beam["vx"]**2 + beam["vy"]**2)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    # Plots longitudinal (z) speed
    ax[0].hist(beam["vz"], bins=80)
    ax[0].set_xlabel("$v_z$ [m/s]")
    ax[0].set_ylabel("counts")
    ax[0].set_title("Longitudinal speed")

    # Plots transverse speed
    ax[1].hist(vt, bins=80)
    ax[1].set_xlabel("$v_t$ [m/s]")
    ax[1].set_title("Transverse speed")

    # Plots exit aperture positions
    ax[2].hist2d(beam["x"]*1e3, beam["y"]*1e3, bins=80)
    ax[2].set_xlabel("x [mm]")
    ax[2].set_ylabel("y [mm]")
    ax[2].set_title("Exit aperture positions")
    plt.tight_layout()
    plt.show()

# Plots beam profile and distribution after passing through entire setup
def plot_beamline_maps(res, cfg):
    b0, b2, b3, b5 = res["beam0"], res["beam2"], res["beam3"], res["beam5"]

    fig, ax = plt.subplots(2, 3, figsize=(14, 7))

    ax[0,0].hist2d(b0["x"]*1e3, b0["y"]*1e3, bins=70)
    ax[0,0].set_title("Source exit")
    ax[0,0].set_xlabel("x [mm]")
    ax[0,0].set_ylabel("y [mm]")

    ax[0,1].hist2d(b2["x"]*1e3, b2["y"]*1e3, bins=70)
    ax[0,1].set_title("After 2D-MOT block")
    ax[0,1].set_xlabel("x [mm]")
    ax[0,1].set_ylabel("y [mm]")

    im = ax[0,2].scatter(b3["x"]*1e3, b3["y"]*1e3, c=b3.get("pump_eta", np.zeros_like(b3["x"])),
                         s=2, alpha=0.45)
    ax[0,2].set_title("Pump plane: local pumping efficiency")
    ax[0,2].set_xlabel("x [mm]")
    ax[0,2].set_ylabel("y [mm]")
    plt.colorbar(im, ax=ax[0,2], label="$\eta_{pump}$")

    ax[1,0].hist(np.sqrt(b0["vx"]**2 + b0["vy"]**2), bins=80)
    ax[1,0].set_title("Source transverse speed")
    ax[1,0].set_xlabel("$v_t$ [m/s]")

    ax[1,1].hist(np.sqrt(b2["vx"]**2 + b2["vy"]**2), bins=80)
    ax[1,1].set_title("After 2D-MOT transverse speed")
    ax[1,1].set_xlabel("$v_t$ [m/s]")

    acc = b5["accepted"]
    ax[1,2].scatter(b5["x"][~acc]*1e3, b5["y"][~acc]*1e3, s=1, alpha=0.15, label="rejected")
    ax[1,2].scatter(b5["x"][acc]*1e3, b5["y"][acc]*1e3, s=2, alpha=0.45, label="accepted")
    th = np.linspace(0, 2*np.pi, 400)
    ax[1,2].plot(cfg.orifice_radius*1e3*np.cos(th), cfg.orifice_radius*1e3*np.sin(th))
    ax[1,2].set_title("At orifice plane")
    ax[1,2].set_xlabel("x [mm]")
    ax[1,2].set_ylabel("y [mm]")
    ax[1,2].legend()

    plt.tight_layout()
    plt.show()

# Plots the beam profiles at different stages of the cooling process
def plot_cooling_cross_sections(res):
    b0, b1, b2 = res["beam0"], res["beam1"], res["beam2"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # plotting source exit cross-section as 2D histogram
    axes[0].hist2d(b0["x"]*1e3, b0["y"]*1e3, bins=80)
    axes[0].set_title("Source Exit")
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")

    # plotting after transverse cooling cross-section as scatter plot
    axes[1].scatter(b1["x"]*1e3, b1["y"]*1e3, s=2)
    axes[1].set_title("Transverse Cooling Exit")
    axes[1].set_xlabel("x [mm]")
    axes[1].set_ylabel("y [mm]")

    # plotting after 2D MOT cross-section as scatter plot
    axes[2].scatter(b2["x"]*1e3, b2["y"]*1e3, s=2)
    axes[2].set_title("2D MOT Exit")
    axes[2].set_xlabel("x [mm]")
    axes[2].set_ylabel("y [mm]")

    plt.tight_layout()
    plt.show()
######################################################################



######################################################################
# Optimization functions using Scikit-Optimize

# Imports for optimization
from dataclasses import replace

import numpy as np
import warnings
from skopt import gp_minimize
from skopt.space import Real

# Function for calculating theta distribution for the optimizer
def calculate_theta(b0):
    if len(b0['x']) == 0:
        return np.nan
    
    # Small angle approximation: theta ≈ sqrt(vx^2 + vy^2) / vz
    theta = np.sqrt((b0['vx']**2 + b0['vy']**2)) / b0['vz']

    return theta

# Runs simulation of atomic beam, returning only a single parameter to optimize with Scikit
def simulation_for_optimizer(cfg, rng, aperture_type='long_tube', parameter_to_optimize='flux_acc'):
    if aperture_type == 'long_tube':
        b0 = sample_effusive_beam(cfg, rng)
    elif aperture_type == 'multi_capillary':
        b0 = rejection_sampling_multi_capillary(cfg, rng, print_metrics=False)
    else:
        raise ValueError(f"Unknown aperture_type: {aperture_type}")
    
    # Calculate Theta distribution after oven exit
    theta = calculate_theta(b0)

    # Only keep the 95% of particle with small theta values to reduce the impact of large-angle outliers
    theta_to_minimize = np.percentile(theta,95)

    b1 = apply_transverse_cooling(b0, cfg)
    b2 = apply_2d_mot(b1, cfg)
    b3 = apply_optical_pumping(b2, cfg)
    b4 = apply_rf_block(b3, cfg)
    b5 = apply_orifice_acceptance(b4, cfg)

    acc = accepted_beam(b5)
    flux_acc, frac_acc = accepted_flux(b5, cfg, aperture_type=aperture_type)

    pol_acc = ensemble_polarization(acc) if len(acc["x"]) > 0 else {"Nm1":np.nan,"N0":np.nan,"Np1":np.nan,"Pz":np.nan,"Pzz":np.nan}

    fom_abs = flux_acc * abs(pol_acc["Pzz"]) if np.isfinite(pol_acc["Pzz"]) else np.nan

    # Always return accepted flux, since we have a minimum requirement of 10e13
    if parameter_to_optimize == 'frac_acc':
        return frac_acc, flux_acc
    elif parameter_to_optimize == 'FOM_absPzz':
        return fom_abs, flux_acc
    elif parameter_to_optimize == 'theta':
        return theta_to_minimize, flux_acc
    else:
        raise ValueError(f"Unknown parameter_to_optimize: {parameter_to_optimize}")
    
# Objective function for the optimization
def objective(x, cfg_base, aperture_type, max_theta=None, min_flux=10e13):
    # Work on a copy so each skopt call evaluates an independent configuration.
    if aperture_type == 'long_tube':
        tube_length_mm, tube_radius_mm = x
        cfg_local = replace(cfg_base)
        cfg_local.tube_length = tube_length_mm * 1e-3
        cfg_local.tube_radius = tube_radius_mm * 1e-3
    elif aperture_type == 'multi_capillary':
        capillary_length_mm, capillary_radius_mm = x
        cfg_local = replace(cfg_base)
        cfg_local.capillary_length = capillary_length_mm * 1e-3
        cfg_local.capillary_radius = capillary_radius_mm * 1e-3
    else:
        raise ValueError(f"Unknown aperture_type: {aperture_type}")

    metric, flux_acc = simulation_for_optimizer(
        cfg_local,
        rng=np.random.default_rng(42),
        aperture_type=aperture_type,
        parameter_to_optimize='theta'
    )

    # gp_minimize minimizes
    # penalty if the metric is not finite
    if not np.isfinite(metric):
        return 1e9
    # Penalty if flux does not meet the minimum requirement
    if flux_acc < min_flux:
        return 1e9
    # Penalty for large theta values, if desired
    if max_theta is not None and metric > max_theta:
        return 1e9
    
    # If both conditions are met, return the metric value
    return float(metric)

# Function for plotting heat map of objective function for visualization of optimizer
def plot_objective_heatmap(radius_array, length_array, cfg_base, aperture_type='long_tube', min_flux=10e13, punishment_value=1e9):
    cfg_local = replace(cfg_base)

    length_grid, radius_grid = np.meshgrid(length_array, radius_array)
    metric_values = np.zeros_like(length_grid)
    flux_values = np.zeros_like(length_grid)

    for i, r in enumerate(radius_array):
        for j, l in enumerate(length_array):
            if aperture_type == 'long_tube':
                cfg_local.tube_radius = r * 1e-3
                cfg_local.tube_length = l * 1e-3
            elif aperture_type == 'multi_capillary':
                cfg_local.capillary_radius = r * 1e-3
                cfg_local.capillary_length = l * 1e-3

            metric, flux = simulation_for_optimizer(
                cfg_local,
                rng=np.random.default_rng(42),
                aperture_type=aperture_type,
                parameter_to_optimize='theta'
            )

            # Checks to mimic punishment in the real optimization function
            if not np.isfinite(metric):
                metric = punishment_value
            if flux < min_flux:
                metric = punishment_value

            metric_values[i, j] = metric
            flux_values[i, j] = flux

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # max_abs = np.nanmax(np.abs(metric_values))
    # norm0 = colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    im0 = axes[0].imshow(
        metric_values,
        extent=[length_array[0], length_array[-1], radius_array[0], radius_array[-1]],
        origin='lower',
        aspect='auto',
        cmap='inferno',
        # norm=norm0
    )
    fig.colorbar(im0, ax=axes[0], label='Theta95')
    axes[0].set_xlabel('Length (mm)')
    axes[0].set_ylabel('Radius (mm)')
    axes[0].set_title('Objective Landscape: theta95')

    # Subplot 2: flux heatmap (log scale helps readability)
    im1 = axes[1].imshow(
        flux_values,
        extent=[length_array[0], length_array[-1], radius_array[0], radius_array[-1]],
        origin='lower',
        aspect='auto',
        cmap='magma',
        norm=colors.LogNorm(vmin=max(np.nanmin(flux_values), 1e10), vmax=np.nanmax(flux_values))
    )
    fig.colorbar(im1, ax=axes[1], label='Accepted Flux (atoms/s)')
    axes[1].set_xlabel('Length (mm)')
    axes[1].set_ylabel('Radius (mm)')
    axes[1].set_title('Flux Landscape')

    plt.tight_layout()
    plt.show()
######################################################################