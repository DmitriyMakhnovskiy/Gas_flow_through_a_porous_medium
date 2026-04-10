#
# Calculating deltaP(G) for a flat porous sample
# in the measurement rig where P1 = const and P2 changes
#
# Dmitriy Makhnovskiy, FINGO R&D, Moscow, 03.04.2026
#

import numpy as np
import matplotlib.pyplot as plt

# ===========================
# ENTERING PARAMETERS IN TEXT
# ===========================

# Mass Flow Rate Range, kg/s
G_min = 0.0
G_max = 3.0e-4
n_points = 300

# Flat Sample Geometry
D_mm = 4.0 # Sample Diameter, mm
L_mm = 2.0 # Sample Thickness, mm

# Porous Medium Parameters
por = 0.397 # Porosity
k = 2.15e-12
beta = 0.512 / np.sqrt(k)

# Gas Temperature in Porous Medium
T_s_C = 30.0

# Fixed Inlet Pressure
P1_bar_abs = 5.849 # bar(abs), for example as in the article by Zhong et al.
P1 = P1_bar_abs * 1e5 # Pa(abs)

# Gas: Dry air
mu_g = 1.872e-5
R = 287.05

# ============================
# UNIT CONVERSION
# ============================

D = D_mm * 1e-3
L = L_mm * 1e-3
T_s = T_s_C + 273.15

# Area of ​​a flat sample
A = np.pi * D**2 / 4.0

# Mass flow rate
G = np.linspace(G_min, G_max, n_points)

# ===========================
# COMPRESSIBLE MODEL CALCULATION
# P1^2 - P2^2 = 2L[(mu_g R T_s)/(kA) * G + (beta R T_s)/(A^2) * G^2]
# ===========================

rhs = 2.0 * L * (
(mu_g * R * T_s / (k * A * por)) * G
+ (beta * R * T_s / ((A * por)**2)) * G**2
)

# Express P2 through P1 = const
P2_sq = P1**2 - rhs
P2_sq = np.maximum(P2_sq, 0.0) # protection against negative values ​​under the square root
P2 = np.sqrt(P2_sq)

# Then the differential pressure
DeltaP = P1 - P2 # Pa

# ============================
# FOR COMPARISON: INCOMPRESSIBLE MODEL
# ΔP_flat = L[(mu_g/k) q + beta rho q^2]
# ============================

rho_ref = P1 / (R * T_s) # density at inlet conditions
Q_ref = G / rho_ref # equivalent volume flow rate
q_ref = Q_ref / (A * por)

DeltaP_nosqrt = L * (
(mu_g / k) * q_ref
+ beta * rho_ref * q_ref**2
)

# ============================
# CONVERSION TO CONVENIENT UNITS
# ===========================

P2_bar = P2 / 1e5
DeltaP_bar = DeltaP / 1e5
DeltaP_nosqrt_bar = DeltaP_nosqrt / 1e5

# ===========================
# OUTPUT OF CONTROL VALUES
# ============================

print("Calculation parameters:")
print(f"D = {D:.6f} m")
print(f"L = {L:.6f} m")
print(f"A = {A:.6e} m^2")
print(f"k = {k:.6e} m^2")
print(f"beta = {beta:.6e} 1/m")
print(f"T_s = {T_s:.2f} K")
print(f"mu_g = {mu_g:.6e} Pa·s")
print(f"R = {R:.2f} J/(kg K)")
print(f"P1 = {P1:.1f} Pa = {P1_bar_abs:.3f} bar(abs)")
print(f"rho_in = {rho_ref:.6f} kg/m^3")

# Checkpoint
G_test = 1.0e-4
idx = np.argmin(np.abs(G - G_test))

print("\nCheckpoint:")
print(f"G = {G[idx]:.6e} kg/s")
print(f"P2 = {P2[idx]:.3f} Pa = {P2_bar[idx]:.6f} bar(abs)")
print(f"ΔP (compressible model) = {DeltaP[idx]:.3f} Pa = {DeltaP_bar[idx]:.6f} bar")
print(f"ΔP (incompressible model) = {DeltaP_nosqrt[idx]:.3f} Pa = {DeltaP_nosqrt_bar[idx]:.6f} bar")

# ==============================
# PLOTTING GRAPHS
# ==============================

plt.figure(figsize=(9, 6))
plt.plot(G / 1e-4, DeltaP_bar, label='Considering compressibility: ΔP = P1 - P2(G)', linewidth=2)
plt.plot(G / 1e-4, DeltaP_nosqrt_bar, label='Considering compressibility', linewidth=2, linestyle='--')

plt.xlabel(r'G, $10^{-4}$ kg/s')
plt.ylabel('ΔP, bar')
plt.title('Pressure drop across a flat porous sample at P1 = const')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ==============================
# ADDITIONAL GRAPH P2(G)
# ===========================

plt.figure(figsize=(9, 6))
plt.plot(G / 1e-4, P2_bar, linewidth=2, color='darkgreen')

plt.xlabel(r'G, $10^{-4}$ kg/s')
plt.ylabel('P2, bar(abs)')
plt.title('Output pressure P2 on a flat porous sample at P1 = const')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()