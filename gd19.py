import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Constants
rho_sma = 6450  # Density of SMA wire (kg/m^3)
c_sma = 836.8  # Specific heat capacity of SMA wire (J/kgK)
e_sma = 0.45  # Emissivity of SMA wire
sigma = 5.67e-8  # Stefan-Boltzmann constant (W/m^2K^4)

rho_gd = 7900  # Density of Gadolinium tube (kg/m^3)
c_gd = 236  # Specific heat capacity of Gadolinium tube (J/kgK)
e_gd = 0.2  # Emissivity of Gadolinium tube

k_sma = 18  # Thermal conductivity of SMA wire (W/mK)
k_gd = 8.8  # Thermal conductivity of Gadolinium tube (W/mK)

h_sma = 70  # Heat transfer coefficient of SMA wire (W/m^2K)
h_gd = 10  # Heat transfer coefficient of Gadolinium tube (W/m^2K)

# Initial conditions
T_sma_0 = 90  # Initial temperature of SMA wire (°C)
T_gd_0 = 25  # Initial temperature of Gadolinium tube (°C)

# Functions
def q_rad(Ts, Tg):
    return e_sma * sigma * (Ts + 273.15) ** 4 - e_gd * sigma * (Tg + 273.15) ** 4

def q_conv(Ts, Tg):
    return h_sma * (Ts - Tg)

def q_cond(Ts, Tg):
    return k_sma * (Ts - Tg) / (0.0005)  # Assuming a thickness of 0.5 mm for the wire

def ode_system(t, y):
    Ts, Tg = y
    dTsdt = (q_cond(Ts, Tg) + q_conv(Ts, Tg) + q_rad(Ts, Tg)) / (rho_sma * c_sma * 0.0005)  # Length = 100 mm
    dTgdt = (q_cond(Ts, Tg) + q_conv(Ts, Tg) - q_rad(Ts, Tg)) / (rho_gd * c_gd * 0.0005)  # Length = 100 mm
    return [dTsdt, dTgdt]

# Solve the differential equation system
sol = solve_ivp(ode_system, [0, 10], [T_sma_0, T_gd_0], t_eval=np.linspace(0, 10, 1000))

# Create animation
fig, ax = plt.subplots()
ax.set_xlabel('Time (s)')
ax.set_ylabel('Temperature (°C)')
ax.set_title('Temperature vs. Time')

line_sma, = ax.plot(sol.t, sol.y[0], label='SMA Wire')
line_gd, = ax.plot(sol.t, sol.y[1], label='Gadolinium Tube')

ax.legend()

print("Shape of sol.t:", sol.t.shape)
print("Shape of sol.y[0]:", sol.y[0].shape)
print("Shape of sol.y[1]:", sol.y[1].shape)

def update(frame):
    line_sma.set_ydata(sol.y[0][:frame])
    line_gd.set_ydata(sol.y[1][:frame])
    return line_sma, line_gd

ani = FuncAnimation(fig, update, frames=len(sol.t), blit=True)

# Display the animation
HTML(ani.to_html5_video())

