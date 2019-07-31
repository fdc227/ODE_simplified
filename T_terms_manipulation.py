from sympy import *
from sympy.parsing.sympy_parser import parse_expr
from pathos.multiprocessing import ProcessingPool as Pool
from sympylist_to_txt import sympylist_to_txt
import pickle
import sys
sys.setrecursionlimit(50000)
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

theta_dt_dt = symbols('theta_dt_dt')
phi_dt_dt = symbols('phi_dt_dt')
psi_dt_dt = symbols('psi_dt_dt')
X_dt_dt = symbols('X_dt_dt')
Y_dt_dt = symbols('Y_dt_dt')
Z_dt_dt = symbols('Z_dt_dt')

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


# fn = 'T_terms_differentiated_final.txt'
# raw = open(fn)
# data = raw.read()
# rows = data.split('\n')
# str_list = []
# for row in rows:
#     str_list.append(row.split(','))
    
# m = len(str_list)

# def list_parse(row_num):
#     local_list = []
#     n = len(str_list[row_num])
#     for j in range(n):
#         local_list.append(parse_expr(str_list[row_num][j]))
#         print(f'finished converting {j+1}/{n} on row {row_num+1}/{m}')
#     return local_list

# p = Pool(12)
# R = [i for i in range(m)]
# T_terms_differentiated = p.map(list_parse, R)
# for i in R:
#     T_terms_differentiated.append(list_parse(i))

T_raw = open('T_terms_differentiated_simplified.pkl', 'rb')
T_terms_differentiated = pickle.load(T_raw)

def term_sum(j):
    print(f'Now finishing summing {j+1} / 36 terms')
    local_sum = []
    for i in range(8):
        local_sum.append(T_terms_differentiated[i][j])
    return nsimplify(sum(local_sum))

# p = Pool(8)
R = [i for i in range(36)]
T_terms_manipulated = []
for i in R:
    T_terms_manipulated.append(term_sum(i))

T_manipulated_raw = open('T_terms_manipulated.pkl', 'wb')
pickle.dump(T_terms_manipulated, T_manipulated_raw)
