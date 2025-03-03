#!/usr/bin/env python3


import numpy as np
from pysr import PySRRegressor
from my_functions import read_in, delta_t, mean_t50, delta_t_naiv_ln

events = read_in("data_data/events6t5.txt")
delta_T_values, n_values, t50_values, tVar_values, distance_values, zenith_values, signal_values, group_values, counter, delta_s_values = delta_t_naiv_ln(events, 300)            

# a, b, d = 177, 1.5, 64

# def V_t0(a, b, d, T_50, n, distance):
#     return a + b * ((T_50 + d) / (n + 1))**2 * (n / (n + 2))

# def compute_rms(delta_T_values, n_values, t50_values, a, b, d, V_t0, distance):
#     print("Vt0:  ", V_t0)
#     def V_Delta_T(a, b, d, T_50_i, T_50_j, n_i, n_j, distance_i, distance_j):
#         return V_t0(a, b, d, T_50_i, n_i, distance_i) + V_t0(a, b, d, T_50_j, n_j, distance_j)
    
#     normalized_values = []
#     for i in range(len(delta_T_values)):
#         delta_T_i = delta_T_values[i]
#         T_50_i, T_50_j = t50_values[i]
#         n_i, n_j = n_values[i]
#         distance_i, distance_j = distance[i]
#         V_delta_T_i = V_Delta_T(a, b, d, T_50_i, T_50_j, n_i, n_j, distance_i, distance_j)
#         if V_delta_T_i > 0:
#             normalized_value = delta_T_i / np.sqrt(V_delta_T_i)
#             normalized_values.append(normalized_value)
#     rms = np.sqrt(np.mean(np.square(normalized_values)))
#     return rms

##### mean only 3 no input

# target_rms = compute_rms(delta_T_values, n_values, t50_values, a, b, d, V_t0, distance_values)
# print("Initial RMS:", target_rms)
t50_flat = np.ravel(t50_values)
n_flat = np.ravel(n_values)
distance_flat = np.ravel(distance_values)
signal_flat = np.ravel(signal_values)
delta_s = np.repeat(np.array(delta_s_values), 2)
zenith_flat = np.repeat(np.array(zenith_values), 2)
# X = np.column_stack((t50_flat, n_flat, distance_flat, signal_flat, zenith_flat))
X = np.column_stack((distance_flat, signal_flat, zenith_flat))
y = delta_s
# X, y = X[:10000], y[:10000]


# target_rms = compute_rms(delta_T_values, n_values, t50_values, a, b, d, V_t0, distance_values)
# print("Initial RMS:", target_rms)
# t50_flat = np.ravel(t50_values)
# n_flat = np.ravel(n_values)
# distance_flat = np.ravel(distance_values)
# signal_flat = np.ravel(signal_values)
# zenith_flat = np.repeat(np.array(zenith_values), 2)
# delta_t = np.repeat(np.array(delta_T_values), 2)
# X = np.column_stack((t50_flat, n_flat, distance_flat, signal_flat, zenith_flat))
# y = delta_t
# # X, y = X[:10000], y[:10000]


function_likeli = """
function negative_log_likelihood(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return L(Inf)
    end
    ε = 1e-8
    prediction = max.(prediction, ε)  # Avoid log of zero or negative values
    nll = sum(
        (dataset.y .^ 2) ./ (2 .* prediction .+ ε) .+ 
        log.(4 .* π .* prediction .+ ε)
    )
    return nll
end
"""

print("hmm")
start_equation = "177 + 1.5 * ((x0 + 65) / (x1 + 1))**2 * (x1 / (x1 + 2))"
model = PySRRegressor(
    model_selection="accuracy",  # Nutze den besten Fit basierend auf der Loss-Funktion
    loss_function=function_likeli,
    niterations=500,
    maxsize=20, 
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=["sqrt", "log", "exp", "sin", "cos"],
    extra_sympy_mappings={},  # Falls zusätzliche Sympy-Funktionen benötigt werden
)
model.fit(X, y)
print(model)
new_formula = model.sympy()
# print("new", new_formula)
# def V_t0_new(a, b, d, T_50, n, distance):
#     return eval(str(new_formula).replace('x0', 'T_50').replace('x1', 'n').replace('x2', 'distance').replace('sin', 'np.sin').replace('cos', 'np.cos').replace('sqrt', 'np.sqrt'))

# new_rms = compute_rms(delta_T_values, n_values, t50_values, a, b, d, V_t0_new, distance_values)
# old_rms = compute_rms(delta_T_values, n_values, t50_values, a, b, d, V_t0, distance_values)

# print("New RMS:", new_rms)
# print("Old RMS: ", old_rms)

print("fase2")
