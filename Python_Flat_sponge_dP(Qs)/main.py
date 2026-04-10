#
# Calculating deltaP(Q_s) for a flat porous sample
#
# Dmitriy Makhnovskiy, FINGO R&D, Moscow, 03.04.2026
#

import numpy as np
import matplotlib.pyplot as plt

# ===========================
# ENTERING PARAMETERS IN TEXT
# ===========================

# Free flow range after sample, l/min
Q_min_lpm = 0.0
Q_max_lpm = 10.0
n_points = 300

# Flat sample geometry
D_mm = 11.0 # sample diameter, mm
L_mm = 3.0 # sample thickness, mm

# Porous medium parameters
por = 0.395 # porosity
k = 2.15e-12
beta = 0.512 / np.sqrt(k)

# Gas temperature in the porous medium
T_s_C = 30.0

# Outlet pressure
P_atm = 101325.0 # Pa (absolute)

# Gas: Dry air
mu_g = 1.872e-5 # Pa*s
R = 287.05 # J/(kg*K)

# ============================
# UNIT CONVERSION
# ============================

D = D_mm * 1e-3
L = L_mm * 1e-3
T_s = T_s_C + 273.15

# Area of ​​flat sample
A = np.pi * D**2 / 4.0

# Flow rate Q_s: from l/min to m^3/s
# Left similar to your cylindrical code: divide by eps
Q_s_lpm = np.linspace(Q_min_lpm, Q_max_lpm, n_points)
Q_s = Q_s_lpm * 1e-3 / 60.0

# Specific flow rate over the full sample area
q = Q_s / (A * por)

# Dry air density at outlet
rho_g = P_atm / (R * T_s)

# =============================
# FORMULA 1: WITH COMPRESSIBILITY
# ΔP_flat(Q_s) = sqrt(P_atm^2 + 2L[(mu_g*P_atm)/(kA)*Q_s + (beta*P_atm^2)/(R*T_s*A^2)*Q_s^2]) - P_atm
# ============================

term1 = (mu_g * P_atm / (k * A * por)) * Q_s
term2 = (beta * P_atm**2 / (R * T_s * (A * por)**2)) * Q_s**2

DeltaP_flat_sqrt = np.sqrt(P_atm**2 + 2.0 * L * (term1 + term2)) - P_atm # Pa

# =============================
# FORMULA 2: WITHOUT COMPRESSIBILITY
# ΔP_flat = L[(mu_g/k) q + beta rho_g q^2]
# ============================

DeltaP_flat_nosqrt = L * (
(mu_g / k) * q
+ beta * rho_g * q**2
) # Pa

# ===========================
# CONVERSION TO BAR
# ============================

DeltaP_flat_sqrt_bar = DeltaP_flat_sqrt / 1e5
DeltaP_flat_nosqrt_bar = DeltaP_flat_nosqrt / 1e5

# ============================
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

# Value at 10 l/min
Q_test_lpm = 10.0
idx = np.argmin(np.abs(Q_s_lpm - Q_test_lpm))

print("\nCheckpoint:")
print(f"Q_s = {Q_s_lpm[idx]:.3f} L/min")
print(f"ΔP (with square root) = {DeltaP_flat_sqrt[idx]:.3f} Pa = {DeltaP_flat_sqrt_bar[idx]:.6f} bar")
print(f"ΔP (without compressibility) = {DeltaP_flat_nosqrt[idx]:.3f} Pa = {DeltaP_flat_nosqrt_bar[idx]:.6f} bar")

# ==============================
# PLOTTING GRAPHS
# ===========================

plt.figure(figsize=(9, 6))
plt.plot(Q_s_lpm, DeltaP_flat_sqrt_bar, label='Considering compressibility', linewidth=2)
plt.plot(Q_s_lpm, DeltaP_flat_nosqrt_bar, label='Disregarding compressibility', linewidth=2, linestyle='--')

plt.xlabel('Q_s, l/min')
plt.ylabel('ΔP, bar')
plt.title('Pressure drop across a flat porous sample')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()