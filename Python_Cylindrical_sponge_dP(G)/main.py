#
# Calculating deltaP(G) for a cylindrical bubbler
#
# Dmitriy Makhnovskiy, FINGO R&D, Moscow, 03.04.2026
#

import numpy as np
import matplotlib.pyplot as plt

# ==========================
# ENTERING PARAMETERS IN TEXT
# ==========================

# Mass flow range, kg/s
G_min = 0.0
G_max = 3.0e-4
n_points = 300

# Bubbler geometry
R_out_mm = 12.5     # mm
R_in_mm  = 5.0      # mm
L_c_mm   = 100.0    # mm

# Parameters of a porous medium
por = 0.395     # porosity
k = 2.15e-12        # m^2
beta = 0.512 / np.sqrt(k)

# Gas temperature in the sponge
T_s_C = 30.0        # degrees Celsius

# Outlet pressure
P_atm = 101325.0    # Pa (absolute)

# Gas: dry air
mu_g = 1.872e-5     # Pa*s, dynamic viscosity of dry air at ~30 C
R = 287.05          # J/(kg*K), specific gas constant of air

# =========================
# UNIT CONVERSION
# =========================

R_out = R_out_mm * 1e-3
R_in  = R_in_mm  * 1e-3
L_c   = L_c_mm   * 1e-3
T_s   = T_s_C + 273.15

# Mass flow rate
G = np.linspace(G_min, G_max, n_points)

# The outer surface of the cylinder
A_out = 2.0 * np.pi * R_out * L_c

# Dry air density at outlet conditions
rho_g = P_atm / (R * T_s)

# =========================
# FORMULA 1: WITH COMPRESSIBILITY
# ΔP_cyl(G) = sqrt( P_atm^2
#                 + (mu_g R T_s)/(pi k L_c) * G * ln(R_out/R_in)
#                 + (beta R T_s)/(2 pi^2 L_c^2) * G^2 * (1/R_in - 1/R_out) ) - P_atm
# =========================

term1 = (mu_g * R * T_s / (np.pi * k * L_c)) * (G / por) * np.log(R_out / R_in)
term2 = (beta * R * T_s / (2.0 * np.pi**2 * L_c**2)) * (G / por)**2 * (1.0 / R_in - 1.0 / R_out)

DeltaP_cyl_sqrt = np.sqrt(P_atm**2 + term1 + term2) - P_atm   # Па

# =========================
# FORMULA 2: WITHOUT CONSIDERING COMPRESSIBILITY
# Using q_out = (G / rho_g) / A_out
# ΔP_cyl = (mu_g/k) q_out R_out ln(R_out/R_in)
#          + beta rho_g q_out^2 R_out^2 (1/R_in - 1/R_out)
# =========================

Q_out = G / rho_g                  # m^3/s at outlet conditions
q_out = Q_out / (A_out * por)              # m/s

DeltaP_cyl_nosqrt = (
    (mu_g / k) * q_out * R_out * np.log(R_out / R_in)
    + beta * rho_g * q_out**2 * R_out**2 * (1.0 / R_in - 1.0 / R_out)
)  # Pa

# =========================
# CONVERSION TO bar
# =========================

DeltaP_cyl_sqrt_bar   = DeltaP_cyl_sqrt / 1e5
DeltaP_cyl_nosqrt_bar = DeltaP_cyl_nosqrt / 1e5

# =========================
# OUTPUT OF CONTROL VALUES
# =========================

print("Calculation parameters:")
print(f"R_out = {R_out:.6f} m")
print(f"R_in  = {R_in:.6f} m")
print(f"L_c   = {L_c:.6f} m")
print(f"k     = {k:.6e} m^2")
print(f"beta  = {beta:.6e} 1/m")
print(f"T_s   = {T_s:.2f} K")
print(f"mu_g  = {mu_g:.6e} Pa·s")
print(f"R     = {R:.2f} J/(kg·K)")
print(f"rho_g = {rho_g:.6f} kg/m^3")
print(f"A_out = {A_out:.6e} m^2")

# Checkpoint
G_test = 1.0e-4
idx = np.argmin(np.abs(G - G_test))

print("\nCheckpoint:")
print(f"G = {G[idx]:.6e} kg/s")
print(f"ΔP (with root)        = {DeltaP_cyl_sqrt[idx]:.3f} Pa = {DeltaP_cyl_sqrt_bar[idx]:.6f} bar")
print(f"ΔP (without compressibility) = {DeltaP_cyl_nosqrt[idx]:.3f} Pa = {DeltaP_cyl_nosqrt_bar[idx]:.6f} bar")

# =========================
# Plotting graphs
# =========================

plt.figure(figsize=(9, 6))
plt.plot(G, DeltaP_cyl_sqrt_bar, label='Taking into account compressibility', linewidth=2)
plt.plot(G, DeltaP_cyl_nosqrt_bar, label='Without taking into account compressibility', linewidth=2, linestyle='--')

plt.xlabel('G, kg/s')
plt.ylabel('ΔP, bar')
plt.title('Pressure drop across a cylindrical bubbler')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# ADDITIONAL: GRAPH WITH X-AXIS IN UNITS OF 10^-4 kg/s
# =========================

plt.figure(figsize=(9, 6))
plt.plot(G / 1e-4, DeltaP_cyl_sqrt_bar, label='Taking into account compressibility', linewidth=2)
plt.plot(G / 1e-4, DeltaP_cyl_nosqrt_bar, label='Without taking into account compressibility', linewidth=2, linestyle='--')

plt.xlabel(r'G, $10^{-4}$ kg/s')
plt.ylabel('ΔP, bar')
plt.title('Pressure drop across a cylindrical bubbler')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()