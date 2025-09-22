import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


import scipy


d = np.array([
    10,
    11.5,
    11.5,
    11.5,
    11.5,
    11.5,
    11,
    12.5,
    12.5,
    12.5,
    12.5,
    12.5,
    12.5,
    12,
    12,
    12,
    12,
    12,
    13,
    8
    ], dtype=float)   # distances
y = np.array([
    4017.319336,
    3247.293701,
    2906.342285,
    2954.206299,
    3767.125,
    2767.629883,
    3350.699707,
    2062.477295,
    2430.64209,
    2809.628418,
    2204.137451,
    1602.113647,
    2868.698975,
    2631.836426,
    2507.609375,
    2364.415771,
    3000.119873,
    2479.798096,
    2249.380615,
    6452.161621], dtype=float)  # max RMS values


idx = np.argsort(d)
d_sorted = d[idx]
y_sorted = y[idx]


def curve(d, A, B):
    return A/d + B


popt, pcov = curve_fit(curve, d_sorted, y_sorted, p0=[100000, 0], bounds=(-np.inf, np.inf))
# pcov useless for now


A, B = popt
print(f"A = {A:.2f}, B = {B:.2f}")



plt.scatter(d, y)

curve_xs = np.linspace(min(d), max(d), 100)
curve_ys = curve(curve_xs, A, B)

plt.plot(curve_xs, curve_ys, label=f"Fitted Curve: A={A:.1f}, B={B:.1f}") # RMS = A/dist + B
plt.xlabel("Distance")
plt.ylabel("Max RMS")
plt.legend()
plt.show()

bh = [
2572,
2877,
2886,
2846,
2049,
2514,
2638,
]

mean_bh = np.mean(bh)
std_bh = np.std(bh)
print("Mean BH:", mean_bh)
print("Std BH:", std_bh)

def reverse_curve(rms, A, B):
    return A / (rms - B)

estimated_distances = reverse_curve(np.array(bh), A, B)
mean_estimated_distance = np.mean(estimated_distances)
std_estimated_distance = np.std(estimated_distances)
print("Estimated Distances from BH:", estimated_distances)
print("Mean Estimated Distance from BH:", mean_estimated_distance)
print("Std Estimated Distance from BH:", std_estimated_distance)
