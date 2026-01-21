import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#parametry modelu fitzhugh–nagumo 0.7 0.8 0.08 0.5
a = 0.7
b = 0.8
epsilon = 0.08
I = 0.5

#definicja ukladu rownan
def fhn(t, y):
    v, w = y
    dvdt = v - (v**3)/3 - w + I
    dwdt = epsilon * (v + a - b*w)
    return [dvdt, dwdt]

#przedział czasu i punkty czasowe
t_span = (0, 200)
t_eval = np.linspace(t_span[0], t_span[1], 5000)

#warunki poczatkowe
y0 = [-1.0, 1.0]

#rozwiazanie numeryczne
solution = solve_ivp(fhn, t_span, y0, t_eval=t_eval)

#wykres przebiegu czasowego potencjalu
plt.figure(figsize=(8, 4))
plt.plot(solution.t, solution.y[0])
plt.xlabel("czas t")
plt.ylabel("potencjał v(t)")
plt.title("przebieg czasowy potencjału neuronu – model fitzhugh–nagumo")
plt.grid(True)
plt.show()

#portret fazowy
plt.figure(figsize=(5, 5))
plt.plot(solution.y[0], solution.y[1])
plt.xlabel("v")
plt.ylabel("w")
plt.title("portret fazowy modelu fitzhugh–nagumo")
plt.grid(True)
plt.show()