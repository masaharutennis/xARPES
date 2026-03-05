#!/usr/bin/env python3

# # Verification example
# In this example, we investigate the verification example from the xARPES manuscript examples section.

# The notebook also contains an execution of the Bayesian loop with a set of parameters that is "far from" the optimal solution, similar to the supplemental section on the example.

# In the future, functionality will be added to xARPES for users to generate their own mock example, allowing for testing of desired hypotheses. 

import matplotlib as mpl
mpl.use('Qt5Agg')

# Necessary packages
import xarpes
import numpy as np
import matplotlib.pyplot as plt
import os

# Default plot configuration from xarpes.plotting.py
xarpes.plot_settings('default')

script_dir = xarpes.set_script_dir()

dfld = 'data_sets'           # Folder containing the data
flnm = 'csro_fixed' # Name of the file
extn = '.ibw'         # Extension of the file

data_file_path = os.path.join(script_dir, dfld, flnm + extn)

# angl = np.load(os.path.join(script_dir, dfld, "verification_angles.npy"))
# ekns = np.load(os.path.join(script_dir, dfld, "verification_kinergies.npy"))
# intn = np.load(os.path.join(script_dir, dfld, "verification_intensities.npy"))

bmap = xarpes.BandMap.from_ibw_file(data_file_path, energy_resolution=0.001, 
        angle_resolution=0.1, temperature=10)


bmap = xarpes.BandMap.from_ibw_file(data_file_path, energy_resolution=0.01, 
        angle_resolution=0.1, temperature=50)

fig = plt.figure(figsize=(8, 5)); ax = fig.gca()

fig = bmap.plot(abscissa='angle', ordinate='kinetic_energy', ax=ax)


fig = plt.figure(figsize=(6, 5)); ax = fig.gca()

fig = bmap.fit_fermi_edge(hnuminPhi_guess=52.25, background_guess=1e4,
                          integrated_weight_guess=3e5, angle_min=-20,
                          angle_max=20, ekin_min=52.20, ekin_max=52.30,
                          ax=ax, show=True, fig_close=True,
                          title='Fermi edge fit')

print('The optimised hnu - Phi=' + f'{bmap.hnuminPhi:.4f}' + ' +/- '
      + f'{1.96 * bmap.hnuminPhi_std:.5f}' + ' eV.')


angle_min = 1.25
angle_max = 1.75

energy_range = [-0.20, 0.0001]
energy_value = -0.10

k_0 = 0

mdcs = xarpes.MDCs(*bmap.mdc_set(angle_min, angle_max, energy_range=energy_range))

guess_dists = xarpes.CreateDistributions([
xarpes.Constant(offset=850),
xarpes.SpectralQuadratic(amplitude=0.2, peak=1.55, broadening=0.00009,
            center_wavevector=k_0, name='Right_branch', index='1')
])

fig = plt.figure(figsize=(8, 6)); ax = fig.gca()

fig = mdcs.visualize_guess(distributions=guess_dists, energy_value=energy_value, ax=ax)

# **Note on interactive figures**
# - The interactive figure might not work inside the Jupyter notebooks, despite our best efforts to ensure stability.
# - As a fallback, the user may switch from "%matplotlib widget" to "%matplotlib qt", after which the figure should pop up in an external window.
# - For some package versions, a static version of the interactive widget may spuriously show up inside other cells. In that case, uncomment the #get_ipython()... line in the first cell for your notebooks.


fig = plt.figure(figsize=(8, 6)); ax = fig.gca()

fig = mdcs.fit_selection(distributions=guess_dists, ax=ax)


fig = plt.figure(figsize=(6, 5)); ax = fig.gca()

self_energy = xarpes.SelfEnergy(*mdcs.expose_parameters(select_label='Right_branch_1', 
                                bare_mass=-0.01557611009, fermi_wavevector=0.09500534275, side='right'))

fig = self_energy.plot_both(ax=ax, scale='meV')

plt.show()


self_energies = xarpes.CreateSelfEnergies([self_energy])

fig = plt.figure(figsize=(8, 5)); ax = fig.gca()

fig = bmap.plot(abscissa='momentum', ordinate='electron_energy', 
                plot_dispersions='domain', 
                self_energies=self_energies, ax=ax)

# In the following cell, we extract the Eliashberg function from the self-energy.
# The result of the chi2kink fit is plotted during the extraction. Setting
# `show=False` and `fig_close=True` will prevent the figure from being displayed.
# Afterwards, we plot the Eliashberg function and model function with the
# appropriate self-energy methods.

# To model the characteristic kink-like behavior observed in the dependence of
# the goodness-of-fit metric on the control parameter $x$, we employ a smooth logistic function, 
# hereafter referred to as the $\chi^2$-kink model. The fitted function is defined as

# $$
# \phi(x; g, b, c, d)
# =
# g + \frac{b}{1 + e^{-d(x - c)}},
# $$

# where $g$ denotes the asymptotic baseline value of the metric for small $x$,
# $b$ sets the amplitude of the step, $c$ determines the position of the kink
# along the $x$-axis, and $d$ controls the sharpness of the transition. In the
# limit $d \to \infty$, the function approaches a piecewise-constant form with a
# sharp crossover at $x = c$, whereas smaller values of $d$ correspond to a more
# gradual evolution.


fig, spectrum, model, omega_range, aval_select = self_energy.extract_a2f(
    parts="real",
    omega_min=1.0, omega_max=200, omega_num=500, omega_I=10, omega_M=190, 
    aval_min=0, aval_num=30, aval_max=9.5, lambda_el=1.892698978e-05,
    impurity_magnitude=58.73509854, h_n=0.4738482189, 
    f_chi_squared=2.5, # default
    g_guess=2.0, b_guess=1.5, c_guess=3.0, d_guess=1.5,
    show=True, fig_close=False)

plt.show()


fig = plt.figure(figsize=(7, 5)); ax = fig.gca()

fig = self_energy.plot_spectra(ax=ax)

plt.show()

# The following plots all of the extracted quantities in a single figure. The default plotting range is taken from the second plotting statement.
# By default, The Eliashberg function is extracted while removing the self-energies for binding energies smaller than the energy resolution. In that case, it is transparent to also eliminate these self-energies from the displayed result.


fig = plt.figure(figsize=(10, 8)); ax1 = fig.add_subplot(111); ax2 = ax1.twinx()

# ax1.set_ylim([0, 0.5]); ax2.set_ylim([0, 40])

self_energy.plot_spectra( ax=ax1, abscissa="reversed", show=False, fig_close=False)
self_energy.plot_both(ax=ax2, scale="meV", resolution_range='applied', show=False, fig_close=False)

# --- Change colours for spectra
a2f_line, model_line = ax1.get_lines()[-2:]
a2f_line.set_color("mediumvioletred")
model_line.set_color("darkgoldenrod"); model_line.set_linestyle("--")

# --- Change colours for self-energy lines
real_line, imag_line = ax2.get_lines()[-2:]
real_line.set_color("tab:blue"); imag_line.set_color("tab:orange")

# Change colours for error bars
real_err, imag_err = ax2.collections[-2:]
real_err.set_color(real_line.get_color()); imag_err.set_color(imag_line.get_color())

# --- Overwrite the legend with a custom legend
for ax in (ax1, ax2): ax.get_legend() and ax.get_legend().remove()
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, ncol=2)

plt.show()

# In the following cell, we start from the optimal solution. Unsurprisingly, the optimal solution is obtained after just a couple of iterations.

with xarpes.trim_notebook_output(print_lines=10):
    spectrum, model, omega_range, aval_select, cost, params = self_energy.bayesian_loop(
            parts="real",
            omega_min=1.0, omega_max=200, omega_num=500, omega_I=10, omega_M=190,
            aval_min=1.0, aval_max=10,bare_mass=-0.02,
            fermi_wavevector=0.094, h_n=0.5,
            impurity_magnitude=70, lambda_el=0,
            vary=("impurity_magnitude", "lambda_el", "fermi_wavevector",
                  "bare_mass", "h_n"),
            scale_mb=0.005, scale_imp=10, scale_kF=0.001,
            scale_lambda_el=1.0e-5, scale_hn=0.1,
        )

# Following the recommended procedure, we perform a final optimisation with very tight criteria, for the purpose of further narrowing down the solution.

# With the tested combination of packages, the result is a tiny bit closer to the true solution for bare_mass, impurity_magnitude, and lambda_el.

with xarpes.trim_notebook_output(print_lines=10):
    spectrum, model, omega_range, aval_select, cost, params = self_energy.bayesian_loop(
                omega_min=1.0, omega_max=200, omega_num=500, omega_I=10, omega_M=190,
                aval_min=1.0, aval_max=10, sigma_svd=1e-4,
                bare_mass=-0.01022522839, fermi_wavevector=0.09275671279, h_n=1.698533472, 
                impurity_magnitude=0.1016326705, lambda_el=0.013228474,
                vary=("impurity_magnitude", "lambda_el", "fermi_wavevector", "bare_mass", "h_n"), 
                converge_iters=100, tole=1e-8, scale_mb=0.1, scale_imp=0.1, scale_kF=0.01,
                scale_lambda_el=0.1, scale_hn=0.1, print_lines=10)
