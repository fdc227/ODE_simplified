from sympy import *
from sympy.parsing.sympy_parser import parse_expr
from pathos.multiprocessing import ProcessingPool as Pool
from sympylist_to_txt import sympylist_to_txt
import pickle

t = symbols('t')
x, w, L, theta_0 = symbols('x, w, L, theta_0')
M, m, x, y, z, g, h, E, I, G, J, x_f, c, s, K = symbols('M, m, x, y, z, g, h, E, I, G, J, x_f, c, s, K')
rho, V, a_w, gamma, M_thetadot, e = symbols('rho, V, a_w, gamma, M_thetadot, e')
beta, P, Q, R = symbols('beta, P, Q, R')
W_x, W_y, W_z = symbols('W_x, W_y, W_z')
P_s, gamma_alpha = symbols('P_s, gamma_alpha')

theta = symbols('theta')
phi = symbols('phi')
psi = symbols('psi')
X = symbols('X')
Y = symbols('Y')
Z = symbols('Z')
short_var_list = [theta, phi, psi, X, Y, Z]

theta_dt = symbols('theta_dt')
phi_dt = symbols('phi_dt')
psi_dt = symbols('psi_dt')
X_dt = symbols('X_dt')
Y_dt = symbols('Y_dt')
Z_dt = symbols('Z_dt')
short_var_list_dt = [theta_dt, phi_dt, psi_dt, X_dt, Y_dt, Z_dt]

theta_dt_dt = symbols('theta_dt_dt')
phi_dt_dt = symbols('phi_dt_dt')
psi_dt_dt = symbols('psi_dt_dt')
X_dt_dt = symbols('X_dt_dt')
Y_dt_dt = symbols('Y_dt_dt')
Z_dt_dt = symbols('Z_dt_dt')
short_var_list_dt_dt = [theta_dt_dt, phi_dt_dt, psi_dt_dt, X_dt_dt, Y_dt_dt, Z_dt_dt]

var_q_bending = []
for i in range(3,0,-1):
    globals()[f'p{i}_b'] = symbols(f'p{i}_b')
    var_q_bending.append(globals()[f'p{i}_b'])
for i in range(1,4):
    globals()[f'q{i}_b'] = symbols(f'q{i}_b')
    var_q_bending.append(globals()[f'q{i}_b'])

var_q_bending_dot = []
for i in range(3,0,-1):
    globals()[f'p{i}_b_dot'] = symbols(f'p{i}_b_dot')
    var_q_bending_dot.append(globals()[f'p{i}_b_dot'])
for i in range(1,4):
    globals()[f'q{i}_b_dot'] = symbols(f'q{i}_b_dot')
    var_q_bending_dot.append(globals()[f'q{i}_b_dot'])

var_q_torsion = []
for i in range(3,0,-1):
    globals()[f'p{i}_t'] = symbols(f'p{i}_t')
    var_q_torsion.append(globals()[f'p{i}_t'])
for i in range(1,4):
    globals()[f'q{i}_t'] = symbols(f'q{i}_t')
    var_q_torsion.append(globals()[f'q{i}_t'])

var_q_inplane = []
for i in range(3,0,-1):
    globals()[f'p{i}_i'] = symbols(f'p{i}_i')
    var_q_inplane.append(globals()[f'p{i}_i'])
for i in range(1,4):
    globals()[f'q{i}_i'] = symbols(f'q{i}_i')
    var_q_inplane.append(globals()[f'q{i}_i'])

var_q_inplane_dot = []
for i in range(3,0,-1):
    globals()[f'p{i}_i_dot'] = symbols(f'p{i}_i_dot')
    var_q_inplane_dot.append(globals()[f'p{i}_i_dot'])
for i in range(1,4):
    globals()[f'q{i}_i_dot'] = symbols(f'q{i}_i_dot')
    var_q_inplane_dot.append(globals()[f'q{i}_i_dot'])


var_q_list = [*var_q_bending, *var_q_bending_dot, *var_q_torsion, *var_q_inplane, *var_q_inplane_dot]


var_q_bending_dt = []
for i in range(3, 0, -1):
    globals()[f'p{i}_b_dt'] = symbols(f'p{i}_b_dt')
    var_q_bending_dt.append(globals()[f'p{i}_b_dt'])
for i in range(1, 4):
    globals()[f'q{i}_b_dt'] = symbols(f'q{i}_b_dt')
    var_q_bending_dt.append(globals()[f'q{i}_b_dt'])

var_q_bending_dot_dt = []
for i in range(3, 0, -1):
    globals()[f'p{i}_b_dot_dt'] = symbols(f'p{i}_b_dot_dt')
    var_q_bending_dot_dt.append(globals()[f'p{i}_b_dot_dt'])
for i in range(1, 4):
    globals()[f'q{i}_b_dot_dt'] = symbols(f'q{i}_b_dot_dt')
    var_q_bending_dot_dt.append(globals()[f'q{i}_b_dot_dt'])

var_q_torsion_dt = []
for i in range(3, 0, -1):
    globals()[f'p{i}_t_dt'] = symbols(f'p{i}_t_dt')
    var_q_torsion_dt.append(globals()[f'p{i}_t_dt'])
for i in range(1, 4):
    globals()[f'q{i}_t_dt'] = symbols(f'q{i}_t_dt')
    var_q_torsion_dt.append(globals()[f'q{i}_t_dt'])

var_q_inplane_dt = []
for i in range(3, 0, -1):
    globals()[f'p{i}_i_dt'] = symbols(f'p{i}_i_dt')
    var_q_inplane_dt.append(globals()[f'p{i}_i_dt'])
for i in range(1, 4):
    globals()[f'q{i}_i_dt'] = symbols(f'q{i}_i_dt')
    var_q_inplane_dt.append(globals()[f'q{i}_i_dt'])

var_q_inplane_dot_dt = []
for i in range(3, 0, -1):
    globals()[f'p{i}_i_dot_dt'] = symbols(f'p{i}_i_dot_dt')
    var_q_inplane_dot_dt.append(globals()[f'p{i}_i_dot_dt'])
for i in range(1, 4):
    globals()[f'q{i}_i_dot_dt'] = symbols(f'q{i}_i_dot_dt')
    var_q_inplane_dot_dt.append(globals()[f'q{i}_i_dot_dt'])


var_q_list_dt = [*var_q_bending_dt, *var_q_bending_dot_dt, *var_q_torsion_dt, *var_q_inplane_dt, *var_q_inplane_dot_dt]


var_q_bending_dt_dt = []
for i in range(3, 0, -1):
    globals()[f'p{i}_b_dt_dt'] = symbols(f'p{i}_b_dt_dt')
    var_q_bending_dt_dt.append(globals()[f'p{i}_b_dt_dt'])
for i in range(1, 4):
    globals()[f'q{i}_b_dt_dt'] = symbols(f'q{i}_b_dt_dt')
    var_q_bending_dt_dt.append(globals()[f'q{i}_b_dt_dt'])

var_q_bending_dot_dt_dt = []
for i in range(3, 0, -1):
    globals()[f'p{i}_b_dot_dt_dt'] = symbols(f'p{i}_b_dot_dt_dt')
    var_q_bending_dot_dt_dt.append(globals()[f'p{i}_b_dot_dt_dt'])
for i in range(1, 4):
    globals()[f'q{i}_b_dot_dt_dt'] = symbols(f'q{i}_b_dot_dt_dt')
    var_q_bending_dot_dt_dt.append(globals()[f'q{i}_b_dot_dt_dt'])

var_q_torsion_dt_dt = []
for i in range(3, 0, -1):
    globals()[f'p{i}_t_dt_dt'] = symbols(f'p{i}_t_dt_dt')
    var_q_torsion_dt_dt.append(globals()[f'p{i}_t_dt_dt'])
for i in range(1, 4):
    globals()[f'q{i}_t_dt_dt'] = symbols(f'q{i}_t_dt_dt')
    var_q_torsion_dt_dt.append(globals()[f'q{i}_t_dt_dt'])

var_q_inplane_dt_dt = []
for i in range(3, 0, -1):
    globals()[f'p{i}_i_dt_dt'] = symbols(f'p{i}_i_dt_dt')
    var_q_inplane_dt_dt.append(globals()[f'p{i}_i_dt_dt'])
for i in range(1, 4):
    globals()[f'q{i}_i_dt_dt'] = symbols(f'q{i}_i_dt_dt')
    var_q_inplane_dt_dt.append(globals()[f'q{i}_i_dt_dt'])

var_q_inplane_dot_dt_dt = []
for i in range(3, 0, -1):
    globals()[f'p{i}_i_dot_dt_dt'] = symbols(f'p{i}_i_dot_dt_dt')
    var_q_inplane_dot_dt_dt.append(globals()[f'p{i}_i_dot_dt_dt'])
for i in range(1, 4):
    globals()[f'q{i}_i_dot_dt_dt'] = symbols(f'q{i}_i_dot_dt_dt')
    var_q_inplane_dot_dt_dt.append(globals()[f'q{i}_i_dot_dt_dt'])


var_q_list_dt_dt = [*var_q_bending_dt_dt, *var_q_bending_dot_dt_dt, *var_q_torsion_dt_dt, *var_q_inplane_dt_dt, *var_q_inplane_dot_dt_dt]


q_list = [*short_var_list, *var_q_list]
q_list_dt = [*short_var_list_dt, *var_q_list_dt]
q_list_dt_dt = [*short_var_list_dt_dt, *var_q_list_dt_dt]

###############################################################
################### Actual Program ############################
###############################################################

T_raw = open('T_terms_manipulated.pkl', 'rb')
T = pickle.load(T_raw)
# T = T_.tolist()
# sympylist_to_txt(T, 'T_final.txt')

print(len(T))

T_expanded = []
for row in T:
    T_expanded.append(row.expand())

A_, b_ = linear_eq_to_matrix(T_expanded, q_list_dt_dt)
A = A_.tolist()
b = b_.tolist()
print('finished seperation of variables')
sympylist_to_txt(A, 'T_dt_dt.txt')
sympylist_to_txt(b, 'T_others.txt')
T_dt_dt_raw = open('T_dt_dt.pkl', 'wb')
pickle.dump(A, T_dt_dt_raw)
T_others = open('T_others.pkl', 'wb')
pickle.dump(b, T_others)