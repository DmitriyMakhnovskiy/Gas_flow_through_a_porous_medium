#
# Calculating deltaP(G) for a flat porous sample
#
# Dmitriy Makhnovskiy, FINGO R&D, Moscow, 03.04.2026
#

import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# ENTERING PARAMETERS IN TEXT
# ==========================================

# Mass flow range, kg/s
G_min = 0.0
G_max = 8.0e-4
n_points = 300

# Geometry of a flat sample
D_mm = 7.0 # sample diameter, mm
L_mm = 3.0 # sample thickness, mm

# Parameters of a porous medium
por = 0.420 # porosity
k = 5.02e-12 # m^2
beta = 0.608 / np.sqrt(k)

# Gas temperature in the porous medium
T_s_C = 30.0 # degrees Celsius

#Outlet pressure
P_atm = 101325.0 # Pa (absolute)

# Gas: dry air
mu_g = 1.872e-5 # Pa*s, dynamic viscosity of dry air at ~30 C
R = 287.05 # J/(kg*K), specific gas constant of air

# =========================
# UNIT CONVERSION
# =========================

D = D_mm * 1e-3
L = L_mm * 1e-3
T_s = T_s_C + 273.15

# Area of The flat sample
A = np.pi * D**2 / 4.0

# Mass flow rate
G = np.linspace(G_min, G_max, n_points)

# Dry air density at outlet conditions
rho_g = P_atm / (R * T_s)

# =================================
# FORMULA 1: WITH COMPRESSIBILITY
# ΔP_flat(G) = sqrt( P_atm^2
# + 2L[(mu_g R T_s)/(kA) * G + (beta R T_s)/(A^2) * G^2] ) - P_atm
# =================================

term1 = (mu_g * R * T_s / (k * A * por)) * G
term2 = (beta * R * T_s / ((A * por) **2)) * G**2

DeltaP_flat_sqrt = np.sqrt(P_atm**2 + 2.0 * L * (term1 + term2)) - P_atm # Pa

# =============================================================================
# FORMULA 2: WITHOUT CONSIDERING COMPRESSIBILITY
# Using q = (G / rho_g) / A
# ΔP_flat = L[(mu_g/k) q + beta rho_g q^2]
# =============================================================================

Q_out = G / rho_g # m^3/s at outlet conditions
q = Q_out / (A * por) # m/s

DeltaP_flat_nosqrt = L * (
(mu_g/k)*q
+ beta * rho_g * q**2
) #Pa

# =========================
#CONVERSION TO bar
# =========================

DeltaP_flat_sqrt_bar = DeltaP_flat_sqrt / 1e5
DeltaP_flat_nosqrt_bar = DeltaP_flat_nosqrt / 1e5

# =========================
# OUTPUT OF CONTROL VALUES
# =========================

print("Calculation parameters:")
print(f"D = {D:.6f} m")
print(f"L = {L:.6f} m")
print(f"A = {A:.6e} m^2")
print(f"k = {k:.6e} m^2")
print(f"beta = {beta:.6e} 1/m")
print(f"T_s = {T_s:.2f} K")
print(f"mu_g = {mu_g:.6e} Pa·s")
print(f"R = {R:.2f} J/(kg K)")
print(f"rho_g = {rho_g:.6f} kg/m^3")

#Checkpoint
G_test = 1.0e-4
idx = np.argmin(np.abs(G - G_test))

print("\nCheckpoint:")
print(f"G = {G[idx]:.6e} kg/s")
print(f"ΔP (with root) = {DeltaP_flat_sqrt[idx]:.3f} Pa = {DeltaP_flat_sqrt_bar[idx]:.6f} bar")
print(f"ΔP (without compressibility) = {DeltaP_flat_nosqrt[idx]:.3f} Pa = {DeltaP_flat_nosqrt_bar[idx]:.6f} bar")

# =========================
# Plotting graphs
# =========================

plt.figure(figsize=(9, 6))
plt.plot(G, DeltaP_flat_sqrt_bar, label='Taking into account compressibility', linewidth=2)
plt.plot(G, DeltaP_flat_nosqrt_bar, label='Without taking into account compressibility', linewidth=2, linestyle='--')

plt.xlabel('G, kg/s')
plt.ylabel('ΔP, bar')
plt.title('Pressure drop across a flat porous sample')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# EXTRA: CHART WITH X-AXIS IN UNITS 10^-4 kg/s
# ============================================================

plt.figure(figsize=(9, 6))
plt.plot(G / 1e-4, DeltaP_flat_sqrt_bar, label='Taking into account compressibility', linewidth=2)
plt.plot(G / 1e-4, DeltaP_flat_nosqrt_bar, label='Without taking into account compressibility', linewidth=2, linestyle='--')

plt.xlabel(r'G, $10^{-4}$ kg/s')
plt.ylabel('ΔP, bar')
plt.title('Pressure drop across a flat porous sample')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()