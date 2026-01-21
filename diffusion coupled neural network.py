import numpy as np
import matplotlib.pyplot as plt

#parametry modelu
a = 0.7
b = 0.8
epsilon = 0.08
I = 0.5
D = 1.0 #WSPOLCZYNNIK DYFUZJI

#parametry siatki
Nx, Ny = 50, 50
dx = 1.0
dt = 0.01
steps = 3000

v = -1 * np.ones((Nx, Ny))
w =  1 * np.ones((Nx, Ny))

#lokalne xrodlo impulsu
v[Nx//2, Ny//2] = 1.0

def laplacian(Z):
    return (
        np.roll(Z, 1, axis=0) +
        np.roll(Z, -1, axis=0) +
        np.roll(Z, 1, axis=1) +
        np.roll(Z, -1, axis=1) -
        4 * Z
    ) / dx**2

for step in range(steps):
    dv = v - (v**3)/3 - w + I + D * laplacian(v)
    dw = epsilon * (v + a - b*w)

    v += dt * dv
    w += dt * dw

    if step % 500 == 0:
        plt.figure(figsize=(4, 4))
        plt.imshow(v, cmap="viridis")
        plt.colorbar(label="v")
        plt.title(f"aktywność sieci – krok {step}")
        plt.axis("off")
        plt.show()


