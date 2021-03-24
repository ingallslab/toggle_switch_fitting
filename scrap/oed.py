import numpy as np
import casadi as cs


def find_points(atc,iptg,param):
    #laci=rfp=high atc
    #tetr=gfp=high iptg
    #param = alpha_1, beta_1, K_1, kappa_1, alpha_2, beta_2, K_2, kappa_2

    rfp_root_func = lambda rfp: param[0] + param[1]/(1+((
        (param[4] + param[5]/(1+((rfp/param[6])*(1/(1+(iptg/param[7])**2.0)))**2.0))
        /param[2])*(1/(1+(atc/param[3])**2.0)))**2.0)-rfp

    min_rfp = param[0]
    max_rfp = param[0] + param[1]

    tol=(max_rfp-min_rfp)*.01

    grid = np.array([min_rfp, max_rfp])
    sign = np.array([np.sign(rfp_root_func(min_rfp)),np.sign(rfp_root_func(max_rfp))])
    sgn_change_count=1
    grid_spacing = max_rfp - min_rfp
    while sgn_change_count==1 and grid_spacing>tol:
        bisect_points = (grid[:-1]+grid[1:])/2
        bisect_signs = np.sign(rfp_root_func(bisect_points))

        new_grid = np.empty((grid.size + bisect_points.size,), dtype=grid.dtype)
        new_grid[0::2] = grid
        new_grid[1::2] = bisect_points

        new_sign = np.empty((sign.size + bisect_signs.size,), dtype=sign.dtype)
        new_sign[0::2] = sign
        new_sign[1::2] = bisect_signs

        grid = new_grid
        sign = new_sign

        sign_change_bool= abs(np.diff(sign)/2)
        sgn_change_count = sum(sign_change_bool)
        grid_spacing = grid[1] - grid[0]

    search_intervals = [(grid[i], grid[i+1]) 
                            for i in range(len(sign_change_bool))
                                if sign_change_bool[i]]
    
    stable_points=[]
    for interval in search_intervals:
        stable_point = root_scalar(rfp_root_func,method='brentq',bracket=interval).root
        stable_points.append(stable_point)

    return stable_points


alpha_1 = 13.609
alpha_2 = 60.882   
beta_1 = 3529.923   
beta_2 = 1053.916   
K_1 = 30.0    
K_2 = 31.94 
kappa_1 = 11.65  
kappa_2 = 0.0906   

param=[alpha_1, beta_1, K_1, kappa_1, alpha_2, beta_2, K_2, kappa_2]

iptg_min=0
itpg_max=0.45

atc_min=0
atc_max=45

num=250
iptg_vals = np.linspace(itpg_max,iptg_min,num)
atc_vals = np.linspace(atc_min,atc_max,num)

alpha_1 = 13.609
alpha_2 = 60.882
beta_1 = 3529.923  
beta_2 = 1053.916  
K_1 = 31.94
K_2 = 30.0
n_1  = 2.00
n_2 = 2.00
kappa_1 = 0.0906
kappa_2 = 11.65
m_1= 2.00
m_2 = 2.00

theta=[alpha_1, alpha_2, beta_1, beta_2, K_1, K_2, n_1, n_2, kappa_1, kappa_2, m_1, m_2]

gamma=[.5, 150]

pars=[theta, gamma]

Omega=.01

phi_1=[0.45, 0]
phi_2=[0, 45]

Phi=[phi_1 phi_2]

# Casadi Stuff
x1_sym = cs.SX.sym('x1_sym')
x2_sym = cs.SX.sym('x2_sym')
x_sym=[x1_sym, x2_sym]

Omega_sym = cs.SX.sym('Omega_sym')
u_cntrl_sym = cs.SX.sym('u_cntrl_sym')
u_val_sym=(phi_2-phi_1)*u_cntrl_sym+phi_1

alpha1_sym = cs.SX.sym('alpha1_sym')
alpha2_sym = cs.SX.sym('alpha2_sym')
beta1_sym = cs.SX.sym('beta1_sym')
beta2_sym = cs.SX.sym('beta2_sym')     
K1_sym = cs.SX.sym('K1_sym')    
K2_sym = cs.SX.sym('K2_sym')   
n1_sym  = cs.SX.sym('n1_sym')     
n2_sym = cs.SX.sym('n2_sym')    
kappa1_sym = cs.SX.sym('kappa1_sym')  
kappa2_sym = cs.SX.sym('kappa2_sym')   
m1_sym = cs.SX.sym('m1_sym')     
m2_sym = cs.SX.sym('m2_sym')

theta1_sym=[alpha1_sym, beta1_sym, K1_sym, n1_sym, kappa1_sym, m1_sym]
theta2_sym=[alpha2_sym, beta2_sym, K2_sym, n2_sym, kappa2_sym, m2_sym]
theta_sym=[alpha1_sym, alpha2_sym, beta1_sym, beta2_sym, K1_sym, K2_sym, n1_sym, n2_sym, kappa1_sym, kappa2_sym, m1_sym, m2_sym]

F1 = alpha1_sym + beta1_sym./(1+(((alpha2_sym + beta2_sym./(1+((x1_sym./K1_sym)*(1./(1+(u_val_sym(1)./kappa1_sym).^m1_sym))).^n2_sym))./K2_sym)*(1./(1+(u_val_sym(2)./kappa2_sym).^m2_sym))).^n1_sym)-x1_sym
g1 = alpha1_sym + beta1_sym./(1+((x2_sym./K2_sym)*(1./(1+(u_val_sym(2)./kappa2_sym).^m2_sym))).^n1_sym)-x1_sym
g2 = alpha2_sym + beta2_sym./(1+((x1_sym./K1_sym)*(1./(1+(u_val_sym(1)./kappa1_sym).^m1_sym))).^n2_sym)-x2_sym
g=[g1 g2]

F1_theta=jacobian(F1,theta_sym)
F1_x1=jacobian(F1,x1_sym)
x1_theta=-F1_theta./F1_x1

g2_x1=jacobian(g2,x1_sym)
g2_theta=jacobian(g2,theta_sym)
x2_theta=g2_theta+g2_x1*x1_theta

x_theta=[x1_theta, x2_theta]
%x_th_func = Function('x_th_func', {x_sym,u1_sym,u2_sym,theta_sym}, {x_theta})

b1 = alpha1_sym + beta1_sym./(1+((x2_sym./K2_sym)*(1./(1+(u_val_sym(2)./kappa2_sym).^m2_sym))).^n1_sym)+x1_sym
b2 = alpha2_sym + beta2_sym./(1+((x1_sym./K1_sym)*(1./(1+(u_val_sym(1)./kappa1_sym).^m1_sym))).^n2_sym)+x2_sym
B = [b1 0 0 b2]
A=jacobian(g,x_sym)
A_func = Function('A_func',{x_sym,u_cntrl_sym,theta_sym},{A})

C_vec=inv(kron(eye(size(A)),A)+kron(A,eye(size(A))))*vec(-B)
C=reshape(C_vec,size(A))
C_func = Function('C_func', {x_sym,u_cntrl_sym,theta_sym}, {C})