from vpython import *
import tkinter as tk

# --- Tkinter GUI for Input Control ---
root = tk.Tk()
root.title("Basic Rate Gyro Control")

input_rate = tk.DoubleVar(value=0.0)

def update_slider(val):
    input_rate.set(float(val))

slider = tk.Scale(root, from_=-50, to=50, resolution=1, orient=tk.HORIZONTAL,
                  label="Rotate Case (deg/sec)", command=update_slider, length=400)
slider.pack()

# --- VPython Scene ---
scene = canvas(title="Basic Rate Gyro", width=800, height=600, background=color.white)

# Gyro frame base (represents the aircraft or mount)
case = box(pos=vector(0, -0.3, 0), size=vector(4, 0.1, 2), color=color.gray(0.6))

# Gimbal ring (1 DOF: output axis Y-Y)
gimbal = ring(pos=vector(0, 0, 0), axis=vector(0, 1, 0), radius=0.8, thickness=0.05, color=color.blue)

# Rotor spins inside the gimbal on Z-axis
rotor = cylinder(pos=vector(0, 0, -0.2), axis=vector(0, 0, 0.4), radius=0.2, color=color.red)

# Pointer arm shows precession (output)
pointer = arrow(pos=vector(0, 0, 0), axis=vector(1, 0, 0), shaftwidth=0.05, color=color.orange)

# Restraining spring (visual only)
spring = helix(pos=pointer.pos + pointer.axis, axis=vector(0.3, 0, 0), radius=0.05, thickness=0.01, coils=10, color=color.green)

# Status label
info = label(pos=vector(0,1.2,0), text='', box=False, height=14, color=color.black)

# --- Physical Constants ---
gyro_momentum = 1.0
spring_k = 1.0
arm_length = 1.0
damping = 0.95
current_deflection = 0.0
rotor_angle = 0.0

# --- Simulation Loop ---
def run_sim():
    global current_deflection, rotor_angle
    while True:
        rate(60)

        # Input torque = rotation of the gyro case
        rate_input = input_rate.get()
        torque = rate_input * gyro_momentum

        # Spring force opposes precession
        spring_force = spring_k * arm_length * current_deflection
        net = torque - spring_force

        # Integrate deflection with damping
        current_deflection += net * 0.01
        current_deflection *= damping

        # Rotor spinning
        rotor_angle += 0.3
        rotor.axis = vector(0.4 * cos(rotor_angle), 0, 0.4 * sin(rotor_angle))

        # Gimbal rotates around Y axis — the precession
        gimbal.axis = vector(cos(current_deflection), 0, sin(current_deflection))
        pointer.axis = gimbal.axis * arm_length
        spring.pos = pointer.pos + pointer.axis
        spring.axis = pointer.axis.norm() * 0.3

        # Label update
        info.text = f"Input Rate: {rate_input:.1f} deg/s\nPrecession: {degrees(current_deflection):.1f}°"



