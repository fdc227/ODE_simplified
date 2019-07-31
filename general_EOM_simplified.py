import sys
from sympy import *
from shape_gen import shape_gen
from torsion_shape_gen import torsion_shape_gen
from cross_product import Cross
from dot_product import Dot
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
from sympylist_to_txt import sympylist_to_txt
import pickle

#########################################
######### Define Variables ##############
#########################################

t = symbols('t')
x, w, L, theta_0 = symbols('x, w, L, theta_0')
M, m, x, y, z, g, h, E, I, G, J, x_f, c, s, K, A = symbols('M, m, x, y, z, g, h, E, I, G, J, x_f, c, s, K, A')
rho, V, a_w, gamma, M_thetadot, e = symbols('rho, V, a_w, gamma, M_thetadot, e')
beta, P, Q, R = symbols('beta, P, Q, R')
W_x, W_y, W_z = symbols('W_x, W_y, W_z')
P_s, gamma_alpha = symbols('P_s, gamma_alpha')
MOI = symbols('MOI')

######## Define FUnctions ################
theta = Function('theta')(t)
phi = Function('phi')(t)
psi = Function('psi')(t)
X = Function('X')(t)
Y = Function('Y')(t)
Z = Function('Z')(t)
short_var_list = [theta, phi, psi, X, Y, Z]

short_func_to_sym = {}
for func in short_var_list:
    short_func_to_sym[func] = symbols(str(func)[0:-3])

######### Define Symbols ###################

theta_dt = symbols('theta_dt')
phi_dt = symbols('phi_dt')
psi_dt = symbols('psi_dt')
X_dt = symbols('X_dt')
Y_dt = symbols('Y_dt')
Z_dt = symbols('Z_dt')
short_var_list_dt = [theta_dt, phi_dt, psi_dt, X_dt, Y_dt, Z_dt]
short_var_list_dt_raw = [diff(i, t) for i in short_var_list]

theta_dt_dt = symbols('theta_dt_dt')
phi_dt_dt = symbols('phi_dt_dt')
psi_dt_dt = symbols('psi_dt_dt')
X_dt_dt = symbols('X_dt_dt')
Y_dt_dt = symbols('Y_dt_dt')
Z_dt_dt = symbols('Z_dt_dt')
short_var_list_dt_dt = [theta_dt_dt, phi_dt_dt, psi_dt_dt, X_dt_dt, Y_dt_dt, Z_dt_dt]

short_subs_dict = {}
for i in range(len(short_var_list)):
    short_subs_dict[diff(short_var_list[i], t)] = short_var_list_dt[i]
    short_subs_dict[diff(short_var_list_dt_raw[i], t)] = short_var_list_dt_dt[i]

#########################################

shape_func = shape_gen(4)
shape_func = [i.subs({x:y}) for i in shape_func]
S1, S2, S3, S4 = shape_func[0], shape_func[1], shape_func[2], shape_func[3]
S5_tilde, S6_tilde = torsion_shape_gen()
torsion_shape = [S5_tilde, S6_tilde]
S5 = S5_tilde * (x - x_f)
S6 = S6_tilde * (x - x_f)

# print(shape_func, S5, S6)

##########################################
## Injecting q and q_dot into globals() ##
##########################################

var_q_bending = []
for i in range(3,0,-1):
    globals()[f'p{i}_b'] = Function(f'p{i}_b')(t)
    var_q_bending.append(globals()[f'p{i}_b'])
for i in range(1,4):
    globals()[f'q{i}_b'] = Function(f'q{i}_b')(t)
    var_q_bending.append(globals()[f'q{i}_b'])

var_q_bending_dot = []
for i in range(3,0,-1):
    globals()[f'p{i}_b_dot'] = Function(f'p{i}_b_dot')(t)
    var_q_bending_dot.append(globals()[f'p{i}_b_dot'])
for i in range(1,4):
    globals()[f'q{i}_b_dot'] = Function(f'q{i}_b_dot')(t)
    var_q_bending_dot.append(globals()[f'q{i}_b_dot'])

var_q_torsion = []
for i in range(3,0,-1):
    globals()[f'p{i}_t'] = Function(f'p{i}_t')(t)
    var_q_torsion.append(globals()[f'p{i}_t'])
for i in range(1,4):
    globals()[f'q{i}_t'] = Function(f'q{i}_t')(t)
    var_q_torsion.append(globals()[f'q{i}_t'])

var_q_inplane = []
for i in range(3,0,-1):
    globals()[f'p{i}_i'] = Function(f'p{i}_i')(t)
    var_q_inplane.append(globals()[f'p{i}_i'])
for i in range(1,4):
    globals()[f'q{i}_i'] = Function(f'q{i}_i')(t)
    var_q_inplane.append(globals()[f'q{i}_i'])

var_q_inplane_dot = []
for i in range(3,0,-1):
    globals()[f'p{i}_i_dot'] = Function(f'p{i}_i_dot')(t)
    var_q_inplane_dot.append(globals()[f'p{i}_i_dot'])
for i in range(1,4):
    globals()[f'q{i}_i_dot'] = Function(f'q{i}_i_dot')(t)
    var_q_inplane_dot.append(globals()[f'q{i}_i_dot'])

var_q_list = [*var_q_bending, *var_q_bending_dot, *var_q_torsion, *var_q_inplane, *var_q_inplane_dot]
var_q_list_to_sym = {}
for term in var_q_list:
    var_q_list_to_sym[term] = symbols(str(term)[0:-3])

###########################################

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
var_q_list_dt_raw = [diff(i, t) for i in var_q_list]

###########################################

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

q_sub_dict = {}
for i in range(len(var_q_list)):
    q_sub_dict[diff(var_q_list[i], t)] = var_q_list_dt[i]
    q_sub_dict[diff(diff(var_q_list[i], t), t)] = var_q_list_dt_dt[i]


###########################################
###### Combine var with shape_func ########
###########################################

var_q_bending.insert(3, 0)
var_q_bending_dot.insert(3, 0)
var_q_torsion.insert(3, 0)
var_q_inplane.insert(3, 0)
var_q_inplane_dot.insert(3, 0)

bending_shape_func = []
for i in range(6):
    output = S1 * var_q_bending[i] + S2 * var_q_bending_dot[i] + S3 * var_q_bending[i+1] + S4 * var_q_bending_dot[i+1]
    bending_shape_func.append(output)
print(f'bening shape func is {bending_shape_func}')

torsion_shape_func = []
for i in range(6):
    output = S5 * var_q_torsion[i] + S6 * var_q_torsion[i+1]
    torsion_shape_func.append(output)
print(f'torsion_shape_func is {torsion_shape_func}')

torsion_shape_tilde_func = []
for i in range(6):
    output = S5_tilde * var_q_torsion[i] + S6_tilde * var_q_torsion[i+1]
    torsion_shape_tilde_func.append(output)

inplane_shape_func = []
for i in range(6):
    output = S1 * var_q_inplane[i] + S2 * var_q_inplane_dot[i] + S3 * var_q_inplane[i+1] + S4 * var_q_inplane_dot[i+1]
    inplane_shape_func.append(output)

#############################################

bending_shape_func_dt = []
for i in range(6):
    output = S1 * diff(var_q_bending[i], t) + S2 * diff(var_q_bending_dot[i], t) + S3 * diff(var_q_bending[i+1], t) + S4 * diff(var_q_bending_dot[i+1], t)
    bending_shape_func_dt.append(output)

torsion_shape_func_dt = []
for i in range(6):
    output = S5 * diff(var_q_torsion[i], t) + S6 * diff(var_q_torsion[i+1], t)
    torsion_shape_func_dt.append(output)

torsion_shape_tilde_func_dt = []
for i in range(6):
    output = S5_tilde * diff(var_q_torsion[i], t) + S6_tilde * diff(var_q_torsion[i+1], t)
    torsion_shape_tilde_func_dt.append(output)

inplane_shape_func_dt = []
for i in range(6):
    output = S1 * diff(var_q_inplane[i], t) + S2 * diff(var_q_inplane_dot[i], t) + S3 * diff(var_q_inplane[i+1], t) + S4 * diff(var_q_inplane_dot[i+1], t)
    inplane_shape_func_dt.append(output)
print(torsion_shape_func_dt)
# print(torsion_shape_tilde_func_dt)

##############################################

T_variables = [*short_var_list_dt_raw, *var_q_list_dt_raw]
W_variables = [*short_var_list, *var_q_list]

subs_dict = {**short_subs_dict, **q_sub_dict, **short_func_to_sym, **var_q_list_to_sym}

###############################################

# T_terms = [[]]*20 # contains lists of dT/dq for each wing section
A = Matrix([[cos(theta)*cos(psi), cos(psi)*sin(theta)*sin(phi)-cos(phi)*sin(psi),cos(phi)*cos(psi)*sin(theta)+sin(phi)*sin(psi)],
            [cos(theta)*sin(psi), cos(phi)*cos(psi)+sin(theta)*sin(phi)*sin(psi), -cos(psi)*sin(phi)+cos(phi)*sin(theta)*sin(psi)],
            [-sin(theta), cos(theta)*sin(phi), cos(theta)*cos(phi)]])

A_omega = Matrix([[-sin(theta), 0, 1],
                [cos(theta)*sin(phi), cos(phi), 0],
                [cos(theta)*cos(phi), -sin(phi), 0]])

Omega = A_omega * Matrix([[diff(psi, t)],[diff(theta, t)],[diff(phi, t)]])

def integral_gen(i):
    bs = bending_shape_func[i]
    bs_dt = bending_shape_func_dt[i]
    ts = torsion_shape_func[i]
    ts_dt = torsion_shape_func_dt[i]
    tst = torsion_shape_tilde_func[i]
    tst_dt = torsion_shape_func_dt[i]
    ips = inplane_shape_func[i]
    ips_dt = inplane_shape_func_dt[i]

    P_b = Matrix([[ips],[y + (i - 10) * L],[bs]])
    V_M = Matrix([[diff(X, t)],[diff(Y, t)],[diff(Z, t)]])

    P1 = Matrix([[0],[0],[bs_dt]])
    P2 = Matrix([[0],[0],[tst_dt]])
    P3 = Matrix([[ips_dt],[0],[0]])
    P4 = A**(-1) * V_M
    P5 = Omega.cross(P_b)

    S1 = Dot(P1, P1)
    S2 = 2 * Dot(P1, P2)
    S3 = 2 * Dot(P1, P3)
    S4 = 2 * Dot(P1, P4)
    S5 = 2 * Dot(P1, P5)
    S6 = Dot(P2, P2)
    S7 = 2 * Dot(P2, P3)
    S8 = 2 * Dot(P2, P4)
    S9 = 2 * Dot(P2, P5)
    S10 = Dot(P3, P3)
    S11 = 2 * Dot(P3, P4)
    S12 = 2 * Dot(P3, P5)
    S13 = Dot(P4, P4)
    S14= 2 * Dot(P4, P5)
    S15= Dot(P5, P5)

    S_list = [S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15]
    # print(S_list)

    S_integrated = []
    for k in range(len(S_list)):
        # if k not in [15]:
        #     print(f'Now integrating {k+1}/{len(S_list)} in S_list for {i+1}th wing')
        #     S_integrated.append(nsimplify(Rational(1/2) * m * integrate(S_list[k], (x, 0, c), (y, 0, L))))
        # else:
        terms = Add.make_args(S_list[k].expand())
        local_integral = []
        for l in range(len(terms)):
            integrand = Rational(1/2) * m * integrate(terms[l], (x, 0, c), (y, 0, L))
            print(f'now integrating {l+1}/{len(terms)} of {k+1}/{15} on {i+1}th wing')
            local_integral.append(integrand)
        summed = sum(local_integral)
        S_integrated.append(nsimplify(summed))
    print(S_integrated)


    # T_list = []
    # var_count = 0
    # for j in T_variables:
    #     integrand = 2 * Dot(diff(diff(expr, j), t), expr) + 2 * Dot(diff(expr, j), diff(expr, t))
    #     T_list.append(integrand.xreplace(subs_dict))
    #     print(f'Now finishing {var_count + 1}/{len(T_variables)} terms of the {i+1}th wing')
    #     var_count += 1
    
    return nsimplify(sum(S_integrated))


    # print(lvar_T_rdt)
R = [i for i in range(6)]
# p = Pool(12)
# T_terms = p.map(integral_gen, R)
T_terms_integral = []
for i in R:
    T_terms_integral.append(integral_gen(i))


T_linear = Rational(1/2) * M * (diff(X, t)**2 + diff(Y, t)**2 + diff(Z, t)**2)
T_angular = Rational(1/2) * MOI * Omega.dot(Omega)
T_terms_integral.append(T_linear)
T_terms_integral.append(T_angular)

def T_differentiate(i, T_terms_integral=T_terms_integral):
    count = 0
    T_local_differentiate = []
    for j in T_variables:
        print(f'Now differentiating {count+1}/{len(T_variables)} of {i+1}/{22}th T_term')
        diffed = diff(diff(T_terms_integral[i], j), t)
        T_local_differentiate.append(diffed.xreplace(subs_dict))
        count += 1
    return T_local_differentiate

T_terms_differentiated = []
for i in range(8):
    T_terms_differentiated.append(T_differentiate(i))

fn = 'T_terms_differentiated_simplified.txt'
try:
    file_raw = open('T_terms_differentiated_simplified.pkl', 'wb')
    pickle.dump(T_terms_differentiated, file_raw)
except:
    sympylist_to_txt(T_terms_differentiated, fn)

# T_terms = []
# for i in R:
#     T_terms.append(T_term_gen(i))

# def T_terms_simplify(T_terms):
#     m = len(T_terms[0])
#     n = len(T_terms)
#     T_terms_simplified = [[]]*n
#     for j in range(n):
#         for i in range(m):
#             T_terms_simplified[j].append(nsimplify(T_terms[j][i]))
#             print(f'Now simplifying {i+1}/{m} of {j+1}th term')
#     return T_terms_simplified

# T_terms_simplified = T_terms_simplify(T_terms)




    # T_array = []
    # for i in range(len(T_variables)):
    #     sum_term = 0
    #     for j in R:
    #         sum_term += j[i]
    #     T_array.append(sum_term)

        
    # file1 = open('dT_dq.txt', '+w')
    # len_T = len(T_array) 
    # for i in range(len_T):
    #     if i != len_T - 1:
    #         file1.write(str(T_array[i]))
    #         file1.write('\n')
    #     else:
    #         file1.write(str(T_array[i]))
    # file1.close()

'''
for i in range(0, 20):
    bs = bending_shape_func[i]
    bs_dt = bending_shape_func_dt[i]
    ts = torsion_shape_func[i]
    ts_dt = torsion_shape_func_dt[i]
    tst = torsion_shape_tilde_func[i]
    tst_dt = torsion_shape_func_dt[i]
    ips = inplane_shape_func[i]
    ips_dt = inplane_shape_func_dt[i]
    beta = torsion_shape_tilde_func[i]

    A = Matrix([[cos(theta)*cos(psi), cos(psi)*sin(theta)*sin(phi)-cos(phi)*sin(psi),cos(phi)*cos(psi)*sin(theta)+sin(phi)*sin(psi)],
                [cos(theta)*sin(psi), cos(phi)*cos(psi)+sin(theta)*sin(phi)*sin(psi), -cos(psi)*sin(phi)+cos(phi)*sin(theta)*sin(psi)],
                [-sin(theta), cos(theta)*sin(phi), cos(theta)*cos(phi)]])

    A_omega = Matrix([[-sin(theta), 0, 1],
                    [cos(theta)*sin(phi), cos(phi), 0],
                    [cos(theta)*cos(phi), -sin(phi), 0]])

    Omega = A_omega * Matrix([[psi_dot],[theta_dot],[phi_dot]])
    P_b = Matrix([[ips - (x-x_f)*sin(beta)*tan(beta)],[y + (i - 10) * L],[bs + (x-x_f)*sin(beta)]])
    V_M = Matrix([[V_x],[V_y],[V_z]])

    P1 = Matrix([[0],[0],[bs_dt]])
    P2 = Matrix([[tst_dt * (-1) * sin(beta)],[0],[tst_dt * cos(beta)]])
    P3 = Matrix([[ips_dt],[0],[0]])
    P4 = A**(-1) * V_M
    P5 = Cross(Omega, P_b)

    expr = P1 + P2 + P3 + P4 + P5
    integrand = Dot(expr, expr)
    terms = integrand.expand().args
    n = len(terms)
    m = len(T_variables)

    arg_list = []
    term_count = 0
    for term in terms:
        local_arg_list = []
        for j in T_variables:
            local_arg_list.append(diff(term, j))
        arg_list.append(local_arg_list)
        print(f'the {term_count+1}/{n} term of {i+1}th wing section has been differentiated')
        term_count += 1
    
    var_count = 0
    dT_dq = []
    for k in range(len(T_variables)):
        arg_sum = 0
        for j in arg_list:
            arg_sum += j[k]
        dT_dq.append(arg_sum)
        print(f'the {var_count+1}/{m} term of _varialbe of {i+1}th wing section has been created')
        var_count += 1
    T_terms[i] = dT_dq
'''

# print(nsimplify(integrand))
# full_arg_list = [q1_b, q1_b_dt, ]

# integral_list = []
# count = 0
# for i in arguments:
#     print(f'Now integrating {count + 1}th term')
#     integral_list.append(integrate(integrate(i, (x, 0, c)),(y, 0, L)))
#     count += 1
 

# wing_KE = nsimplify(integrate(nsimplify(integrate(integrand, (x, 0 ,c))), (y, 0, L)))
# print(wing_KE)

# file_KE = open('kinetic_energy_wing.txt', 'w+')
# file_KE.write(f'{wing_KE}')
# file_KE.close()

##############################################
############ Strain Terms ####################
##############################################

def strain_terms(i):
    bs = bending_shape_func[i]
    tst = torsion_shape_tilde_func[i]
    ips = inplane_shape_func[i]

    V_bs = integrate(Rational(1/2) * E * A * (diff(diff(bs, y), y)) ** 2, (y, 0, L))
    V_tst = integrate(Rational(1/2) * G * J * (diff(tst, y)) ** 2, (y, 0, L))
    V_ips = integrate(Rational(1/2) * E * A * (diff(diff(ips, y), y)) ** 2, (y, 0, L))

    strain_var_each = []

    for j in W_variables:
        strain_var_each.append(diff(V_bs, j) + diff(V_tst, j) + diff(V_ips, j))
    
    return strain_var_each

# R = [i for i in range(20)]
# p = Pool(12)
# strain_arrays = p.map(strain_terms, R)


'''
################################################
############# Aerodynamics #####################
################################################

V_wind = Matrix([[W_z],[W_y],[W_z]])
P_s = Matrix([[ips],[y + K * L],[bs]])

V_strip = Matrix([[ips_dt],[0],[bs]]) + A**(-1) * V_M + Cross(Omega, P_s)

V_aero = V_strip - V_wind

dL = Rational(1/2) * rho * V**2 * c * a_w * (gamma_alpha + tst)  ## replaced V_aero[2]/V_aero[0] with gamma_alpha for efficiency

dM = Rational(1/2) * rho * V**2 * c**2 * (e * a_w * (bs_dt + tst) + M_thetadot * c * Rational(1/4) * tst_dt / V)

Q_qb_list = []
for i in shape_func:
    output = integrate(dL * i, (y, 0, L))
    print(f'finished integrateing with respect to variable {i}')
    Q_qb_list.append(output)

Q_A_list = []
A_Vm = (A**(-1)*V_M)[2]
A_var_list = [V_x, V_y, V_z, psi, theta, phi]
for i in A_var_list:
    output = integrate(diff(A_Vm, i) * dL, (y, 0, L))        ## Slow wrt psi, theta and phi
    print(f'finished integrating with respect to variable {i}')
    Q_A_list.append(output)

Q_Omega_list = []
Omega_P = Cross(Omega, P_s)[2]
Omega_var_list = [var_q_bending[11], var_q_bending_dot[11], var_q_bending[12], var_q_bending_dot[12], y, R]
for i in Omega_var_list:
    output = integrate(diff(Omega_P, i) * dL, (y, 0, L))
    print(f'finished integrating with respect to variable {i}')
    Q_Omega_list.append(output)

Q_qt_list = []
for i in torsion_shape:
    output = integrate(dM * i, (y, 0, L))
    print(f'finished integrating with respect to variable {i}')
    Q_qt_list.append(output)

'''
