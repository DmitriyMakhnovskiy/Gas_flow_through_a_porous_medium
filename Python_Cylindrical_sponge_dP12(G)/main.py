#
# Calculating deltaP(G) for a cylindrical bubbler
# in the measurement rig where P1 = const and P2 changes
#
# Dmitriy Makhnovskiy, FINGO R&D, Moscow, 03.04.2026
#

import numpy as np
import matplotlib.pyplot as plt

# ============================
# ENTERING PARAMETERS IN TEXT
# ============================

# Mass Flow Rate Range, kg/s
G_min = 0.0
G_max = 3.0e-4
n_points = 300

# Bubbler Geometry
R_out_mm = 12.5 # mm
R_in_mm = 5.0 # mm
L_c_mm = 100.0 # mm

# Porous Medium Parameters
por = 0.395 # Porosity
k = 2.15e-12
beta = 0.512 / np.sqrt(k)

# Gas Temperature in Sponge
T_s_C = 30.0

# Fixed Inlet Pressure
P1_bar_abs = 2.5
P1 = P1_bar_abs * 1e5 # Pa(abs)

# Gas: Dry Air
mu_g = 1.872e-5
R = 287.05

# ==============================
# UNIT CONVERSION
# =========================

R_out = R_out_mm * 1e-3
R_in  = R_in_mm  * 1e-3
L_c   = L_c_mm   * 1e-3
T_s   = T_s_C + 273.15

# Mass Flow Rate
G = np.linspace(G_min, G_max, n_points)

# ============================
# COMPRESSIBLE MODEL CALCULATION
# P1^2 - P2^2 = a*G + b*G^2
# ============================

term1 = (mu_g * R * T_s / (2.0 * np.pi * k * L_c)) * (G / por) * np.log(R_out / R_in)
term2 = (beta * R * T_s / (2.0 * np.pi**2 * L_c**2)) * (G / por)**2 * (1.0 / R_in - 1.0 / R_out)

rhs = term1 + term2

# First, express P2 using P1 = const
P2_sq = P1**2 - rhs
P2_sq = np.maximum(P2_sq, 0.0) # protection against negative radical expressions
P2 = np.sqrt(P2_sq)

# Then calculate the pressure drop
DeltaP = P1 - P2 # Pa

# =============================
# FOR COMPARISON: INCOMPRESSIBLE MODEL
# Use q_out = (G / rho_ref) / A_out
# ΔP_cyl = (mu_g/k) q_out R_out ln(R_out/R_in)
# + beta rho_ref q_out^2 R_out^2 (1/R_in - 1/R_out)
# =========================

rho_ref = P1 / (R * T_s) # density at inlet conditions
A_out = 2.0 * np.pi * R_out * L_c
Q_ref = G / rho_ref # equivalent volumetric flow rate
q_out = Q_ref / (A_out * por)

DeltaP_nosqrt = (
(mu_g / k) * q_out * R_out * np.log(R_out / R_in)
+ beta * rho_ref * q_out**2 * R_out**2 * (1.0 / R_in - 1.0 / R_out)
)

# =============================
# CONVERSION TO BAR
# =============================

P2_bar = P2 / 1e5
DeltaP_bar = DeltaP / 1e5
DeltaP_nosqrt_bar = DeltaP_nosqrt / 1e5

# =============================
# OUTPUT OF CONTROL VALUES
# ===========================

print("Calculation parameters:")
print(f"R_out = {R_out:.6f} m")
print(f"R_in = {R_in:.6f} m")
print(f"L_c = {L_c:.6f} m")
print(f"k = {k:.6e} m^2")
print(f"beta = {beta:.6e} 1/m")
print(f"T_s = {T_s:.2f} K")
print(f"mu_g = {mu_g:.6e} Pa·s")
print(f"R = {R:.2f} J/(kg K)")
print(f"P1 = {P1:.1f} Pa = {P1_bar_abs:.3f} bar(abs)")
print(f"rho_in = {rho_ref:.6f} kg/m^3")
print(f"A_out = {A_out:.6e} m^2")

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
plt.title('Pressure drop across a cylindrical bubbler at P1 = const')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ==============================
# ADDITIONAL PLOTS OF P2(G)
# ===========================

plt.figure(figsize=(9, 6))
plt.plot(G / 1e-4, P2_bar, linewidth=2, color='darkgreen')

plt.xlabel(r'G, $10^{-4}$ kg/s')
plt.ylabel('P2, bar(abs)')
plt.title('Output pressure P2 on a cylindrical bubbler at P1 = const')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
