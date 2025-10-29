# Copyright (C) 2025 xARPES Developers
# This program is free software under the terms of the GNU GPLv3 license.

# Physical constants
kilo = 1e3 # 1000 [-]
uncr = 1.95996398 # Standard deviation to 95 % confidence [-]
k_B = 8.617333e-5 # Boltzmann constant [eV / K]
pref = 3.80998211616 # hbar^2 / (2 m_e) [eV Angstrom^2]
dtor = 0.01745329252 # Degrees to radians - pi / 180 [rad / deg]
fwhm_to_std = 2.35482004503 # Convert FWHM to std, sqrt[8 \times ln(2)] [-]
sigma_extend = 5 # Extend data range by "5 sigma"
stdv = 1.959963984 # "2 sigma" Gaussian STD - scipy.stats.norm.ppf(0.975)