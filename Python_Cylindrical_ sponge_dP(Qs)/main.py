#
# Calculating deltaP(Q_s) for a cylindrical bubbler
#
# Dmitriy Makhnovskiy, FINGO R&D, 03.04.2026
#

import numpy as np
import matplotlib.pyplot as plt

# ===========================
# ENTERING PARAMETERS IN TEXT
# ===========================

# Free flow range after bubbler, l/min
Q_min_lpm = 0.0
Q_max_lpm = 10.0
n_points = 300

# Bubbler geometry
R_out_mm = 12.5     # mm
R_in_mm = 5.0      # mm
L_c_mm = 100.0      # mm

# Parameters of the porous medium
por = 0.395     # porosity
k = 2.15e-12        # m^2
beta = 0.512 / np.sqrt(k)

# Gas temperature in the sponge
T_s_C = 30.0        # degrees Celsius

# Outlet pressure
P_atm = 101325.0    # Pa (absolute)

# Газ: сухой воздух
mu_g = 1.872e-5     # Pa*s, dynamic viscosity of dry air at ~30 C
R = 287.05          # J/(kg*K), specific gas constant of air

# =========================
# UNIT CONVERSION
# =========================

R_out = R_out_mm * 1e-3
R_in = R_in_mm * 1e-3
L_c = L_c_mm * 1e-3
T_s = T_s_C + 273.15

# Flow rate Q_s: from l/min to m^3/s
Q_s_lpm = np.linspace(Q_min_lpm, Q_max_lpm, n_points)
Q_s = Q_s_lpm * 1e-3 / 60.0

# The outer surface of the cylinder
A_out = 2.0 * np.pi * R_out * L_c

# Specific consumption on the outer surface
q_out = Q_s / (A_out * por)

# Density of dry air at outlet, if needed for the second formula
rho_g = P_atm / (R * T_s)

# ===============================
# FORMULA 1: WITH COMPRESSIBILITY
# ΔP_cyl = sqrt(...) - P_atm
# ===============================

term1 = (mu_g / (np.pi * k * L_c)) * ((Q_s / por) * P_atm) * np.log(R_out / R_in)
term2 = (beta / (2.0 * np.pi**2 * L_c**2 * R * T_s)) * ((Q_s / por) * P_atm)**2 * (1.0 / R_in - 1.0 / R_out)

DeltaP_cyl_sqrt = np.sqrt(P_atm**2 + term1 + term2) - P_atm   # Па

# =========================================================
# FORMULA 2: WITHOUT COMPRESSIBILITY
# ΔP_cyl = (mu_g/k) q_out R_out ln(R_out/R_in)
#          + beta rho_g q_out^2 R_out^2 (1/R_in - 1/R_out)
# ==========================================================

DeltaP_cyl_nosqrt = (
    (mu_g / k) * q_out * R_out * np.log(R_out / R_in)
    + beta * rho_g * q_out**2 * R_out**2 * (1.0 / R_in - 1.0 / R_out)
)  # Па

# =========================
# CONVERSION TO bar
# =========================

DeltaP_cyl_sqrt_bar = DeltaP_cyl_sqrt / 1e5
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

# Value at 10 l/min
Q_test_lpm = 10.0
idx = np.argmin(np.abs(Q_s_lpm - Q_test_lpm))

print("\nCheckpoint:")
print(f"Q_s = {Q_s_lpm[idx]:.3f} L/min")
print(f"ΔP (with root)        = {DeltaP_cyl_sqrt[idx]:.3f} Pa = {DeltaP_cyl_sqrt_bar[idx]:.6f} bar")
print(f"ΔP (without compressibility) = {DeltaP_cyl_nosqrt[idx]:.3f} Pa = {DeltaP_cyl_nosqrt_bar[idx]:.6f} bar")

# =========================
# Plotting graphs
# =========================

plt.figure(figsize=(9, 6))
plt.plot(Q_s_lpm, DeltaP_cyl_sqrt_bar, label='Taking into account compressibility', linewidth=2)
plt.plot(Q_s_lpm, DeltaP_cyl_nosqrt_bar, label='Without taking into account compressibility', linewidth=2, linestyle='--')

plt.xlabel('Q_s, l/min')
plt.ylabel('ΔP, bar')
plt.title('Pressure drop across a cylindrical bubbler')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()