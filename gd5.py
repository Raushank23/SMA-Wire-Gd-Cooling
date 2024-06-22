import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
sigma = 5.67e-8  # Stefan-Boltzmann constant in W/m^2/K^4
epsilon_eff = 0.1607
A = 1.602e-4  # Surface area in m^2

# Masses and specific heat capacities
M_wire = 1.317735e-4  # kg
C_wire = 836.8  # J/kg·K
M_tube = 6.887958e-6  # kg
C_tube = 236  # J/kg·K

# Initial temperatures
T_wire_initial = 363.15  # K
T_tube_initial = 298.15  # K

# Equilibrium temperature
T_eq = 362.18  # K

# Time span for the simulation
t_span = (0, 1000)  # seconds
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# Define the system of differential equations
def heat_transfer_system(t, T):
    T_wire, T_tube = T
    dT_wire_dt = -sigma * epsilon_eff * A * (T_wire**4 - T_tube**4) / (M_wire * C_wire)
    dT_tube_dt = sigma * epsilon_eff * A * (T_wire**4 - T_tube**4) / (M_tube * C_tube)
    return [dT_wire_dt, dT_tube_dt]

# Initial conditions
T_initial = [T_wire_initial, T_tube_initial]

# Solve the differential equations
solution = solve_ivp(heat_transfer_system, t_span, T_initial, t_eval=t_eval, method='RK45')

# Extract results
time = solution.t
T_wire = solution.y[0]
T_tube = solution.y[1]

# Find the time to reach equilibrium temperature within 1% tolerance
T_eq_tol = 0.01 * T_eq
equilibrium_reached = np.where((np.abs(T_wire - T_eq) < T_eq_tol) & (np.abs(T_tube - T_eq) < T_eq_tol))[0]

if equilibrium_reached.size > 0:
    time_to_eq = time[equilibrium_reached[0]]
else:
    time_to_eq = None

print("Time to reach equilibrium:", time_to_eq, "seconds")

# Plot the results
plt.plot(time, T_wire, label='SMA Wire Temperature (K)')
plt.plot(time, T_tube, label='Gadolinium Tube Temperature (K)')
plt.axhline(y=T_eq, color='r', linestyle='--', label='Equilibrium Temperature (K)')
plt.xlabel('Time (seconds)')
plt.ylabel('Temperature (K)')
plt.legend()
plt.title('Temperature vs. Time')
plt.show()
