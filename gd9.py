import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Given constants
sigma = 5.67e-8  # Stefan-Boltzmann constant in W/m^2K^4

# Properties of SMA wire
rho_SMA = 6450  # kg/m^3
c_SMA = 836.8  # J/kgK
epsilon_SMA = 0.45
diameter_SMA = 0.51e-3  # m
length_SMA = 0.1  # m
T_SMA_initial = 90 + 273.15  # initial temperature in K

# Properties of Gadolinium tube
rho_Gadolinium = 7900  # kg/m^3
c_Gadolinium = 236  # J/kgK
epsilon_Gadolinium = 0.2
outer_diameter_Gadolinium = 0.60e-3  # m
inner_diameter_Gadolinium = 0.55e-3  # m
length_Gadolinium = 0.1  # m
T_Gadolinium_initial = 25 + 273.15  # initial temperature in K

# Effective emissivity
epsilon_eff = 1 / (1/epsilon_SMA + 1/epsilon_Gadolinium - 1)

# Surface area of the SMA wire (inner surface of the tube)
A_SMA = np.pi * diameter_SMA * length_SMA

# Mass of SMA wire
A_SMA_cross_section = np.pi * (diameter_SMA / 2) ** 2
m_SMA = rho_SMA * A_SMA_cross_section * length_SMA

# Mass of Gadolinium tube
A_Gadolinium_cross_section = np.pi * ((outer_diameter_Gadolinium / 2) ** 2 - (inner_diameter_Gadolinium / 2) ** 2)
m_Gadolinium = rho_Gadolinium * A_Gadolinium_cross_section * length_Gadolinium

# Equilibrium temperature calculation
T_eq = (m_SMA * c_SMA * T_SMA_initial + m_Gadolinium * c_Gadolinium * T_Gadolinium_initial) / (m_SMA * c_SMA + m_Gadolinium * c_Gadolinium)

# Time to reach equilibrium (using numerical approach)
def dTdt(t, T):
    T_SMA, T_Gadolinium = T
    dQdt = sigma * epsilon_eff * A_SMA * (T_SMA**4 - T_Gadolinium**4)
    dT_SMA_dt = -dQdt / (m_SMA * c_SMA)
    dT_Gadolinium_dt = dQdt / (m_Gadolinium * c_Gadolinium)
    return [dT_SMA_dt, dT_Gadolinium_dt]

# Initial temperatures
T_initial = [T_SMA_initial, T_Gadolinium_initial]

# Time span for the solution
t_span = (0, 250)

# Solve the system of ODEs
sol = solve_ivp(dTdt, t_span, T_initial, method='RK45', rtol=1e-6, atol=1e-8)

# Extract equilibrium time when temperatures are within a small tolerance of T_eq
t_eq_index = np.argmax(np.abs(sol.y[0] - T_eq) < 0.01)
time_to_equilibrium = sol.t[t_eq_index]

# Output results
print(f"Equilibrium Temperature: {T_eq:.2f} K")
print(f"Time to Equilibrium: {time_to_equilibrium:.2f} seconds")

# Animation
fig, ax = plt.subplots()
ax.set_xlim(0, 250)
ax.set_ylim(20, 100)
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Temperature (°C)')

line1, = ax.plot([], [], lw=2, label='SMA Wire')
line2, = ax.plot([], [], lw=2, label='Gadolinium Tube')
eq_line = ax.axhline(y=T_eq - 273.15, color='r', linestyle='--', label='Equilibrium Temperature')
ax.legend()

temp_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    temp_text.set_text('')
    return line1, line2, temp_text

def update(frame):
    line1.set_data(sol.t[:frame], sol.y[0][:frame] - 273.15)
    line2.set_data(sol.t[:frame], sol.y[1][:frame] - 273.15)
    temp_text.set_text(f'Time: {sol.t[frame]:.1f} s\nWire Temp: {sol.y[0][frame] - 273.15:.2f} °C\nTube Temp: {sol.y[1][frame] - 273.15:.2f} °C')
    return line1, line2, temp_text

aani = FuncAnimation(fig, update, frames=len(sol.t), init_func=init, blit=True, interval=1000)
plt.show()
