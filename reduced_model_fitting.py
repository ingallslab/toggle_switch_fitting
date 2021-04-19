import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.stats as st
import scipy.linalg as ln
import matplotlib.pyplot as plt

def promoter_model(fp,inducer,param):
    '''
    This is a helped function to compute the equation describing a promoter, either gfp or rfp.
    The input fp is the other gene's expression (i.e. gfp if you modelling the rfp promoter and 
    rfp if you are modelling the gfp promoter), inducer is atc for rfp promoter, iptg for gfp promoter
    and param are the relevant rho, omega and kappa for the given promoter.
    '''
    #name the parameters for clarity
    rho = param[0]
    omega = param[1]
    kappa = param[2]

    #compute the effect of the inducer
    ind_effect = 1/(1+(inducer/kappa)**2)

    #return the expect fp expression
    return rho + (1-rho)/(1 + (fp*ind_effect/omega)**2)


def rfp_hat_roots(atc,iptg,param):
    '''
    This functin accepts the inducer levels (single set) and parameter values for the model and 
    returns the rfp values of the fixed points in phase space for the given inputs/parameters. 
    Gfp levels can be computed from the rfp values found here.
    
    Note this function should return either one or three rfp values. If it returns three values, they
    should be in increasing order with the middle value corresponding to an unstable fixed point and
    the lower and higher values corresponds to stable fixed points. This is due to the type of 
    bifurcation the model undergoes. If one rfp value is returned it should always be stable. Stable
    means it is an rfp value the system can settle down to after being left to equilibrate over a
    long time (i.e. 6-8 hours as in experiments)

    Mathematically this is a root finding problem. The function we need the roots of is an algebraic
    combination of the model's steady state equations for the rfp and gfp levels. However it is more
    complex then just running default root finding because for any given input/parameter set,
    we don't know if there is 1 or 3 roots and the search interval can change. The approach used here
    is to use a bisecting grid search to find intervals in which the roots can be found and then to
    run a normal root finding algorihtm (a variant of bisection search) on each interval.

    Parameters here are expected in their natural scaling (i.e. no log transform)

    '''
    #NOTE:
    #laci=rfp=high atc
    #tetr=gfp=high iptg
    #param = rho_r, omega_g, kappa_a, rho_g, omega_r, kappa_i
    rfp_par = param[0:3]
    gfp_par = param[3:6]

    # model consists of two equations rfp=g1(gfp,inputs) and gfp=g2(rfp,inputs)
    # this function implements 0=g1(g2(rfp,inputs),inputs)-rfp
    # the lower and upper roots of this function for rfp correspond to model branches
    # to get gfp we take the rfp root and sub it in g2, i.e. gfp=g2(rfp_root,inputs)
    rfp_root_func = lambda rfp: promoter_model(promoter_model(rfp,iptg,gfp_par),atc,rfp_par) - rfp

    #extract the min and max possible rfp values from the model parameters, roots will occur in this
    #intervale
    min_rfp = rho_r
    max_rfp = 1

    #set the search tolerance over the rfp range, below this tolerance 3 nearby roots and a single
    #root will look the same to the algorithm
    tol = (max_rfp - min_rfp)*1e-5

    #create a grid of rfp points spanning the feasible range
    grid = np.array([min_rfp, max_rfp])
    #evaluet the signe of g1(g2(rfp,inputs),inputs)-rfp, on the end points of the interval
    #these two points should have opposite signs as they are on opposite sides of a root
    sign = np.array([np.sign(rfp_root_func(min_rfp)),np.sign(rfp_root_func(max_rfp))])
    #initialize the sign change count, telling us the current known number of sign changes occuring 
    # over the grid for the function g1(g2(rfp,inputs),inputs)-rfp
    sgn_change_count=1
    #set the current grid space to the spacing of the endpoints in grid array
    grid_spacing = max_rfp - min_rfp
    # loop until we either find that these inputs have 3 roots or we have subdivide grid into 
    # spacing smaller than the search tolerence
    while sgn_change_count==1 and grid_spacing>tol:
        #compute the rfp values that bisect all the current grid points
        bisect_points = (grid[:-1]+grid[1:])/2
        #evaluate the sign of the current grid points passed to root funciton: g1(g2(rfp,inputs),inputs)-rfp
        bisect_signs = np.sign(rfp_root_func(bisect_points))

        #create a new grid 
        new_grid = np.empty((grid.size + bisect_points.size,), dtype=grid.dtype)
        #add the old grid points
        new_grid[0::2] = grid
        #add the bisection points, basically doubles size/fineness of the grid
        new_grid[1::2] = bisect_points

        #create a new sign array
        new_sign = np.empty((sign.size + bisect_signs.size,), dtype=sign.dtype)
        #add old sign values for root function on old grid points
        new_sign[0::2] = sign
        #dd new sign values for root function on new bisected grid points
        new_sign[1::2] = bisect_signs

        #overight the original grid and sign arrays for next loop
        grid = new_grid
        sign = new_sign

        #create a boolean array indicating if a sign change in the root function occurs on each 
        #interval of the grid, if so there is either 1 or 3 roots in that grid interval
        sign_change_bool= abs(np.diff(sign)/2)
        #count the number of signs changes, and thus roots that we know about
        sgn_change_count = sum(sign_change_bool)
        #refine the grid spacing
        grid_spacing = grid[1] - grid[0]

    #after loop end we have either found 3 roots or know there is 1 root up to the given tolerence
    #we know take the grid and pull out the rfp grid intervals bracketing the known roots
    search_intervals = [(grid[i], grid[i+1]) 
                            for i in range(len(sign_change_bool))
                                if sign_change_bool[i]]
    
    #create an empy list to store the rfp roots we will return
    fixed_points=[]
    #loop over each rfp grid interval where there was a sign change in the root function
    for interval in search_intervals:
        #for each interval run the root finding function to find the rfp root value accurately 
        fixed_point = opt.root_scalar(rfp_root_func,method='brentq',bracket=interval).root
        # add the root to the list
        fixed_points.append(fixed_point)

    #return the list of roots i.e. fixed points
    return fixed_points

def log_ratio_points(atc,iptg,par_all):
    '''
    This function accepts the inducer levels and the parameter values and returns a list of
    rfp and gfp points corresponding to the models stable fixed points at these inputs.

    This function calls the rfp_roots function. This function only returns stable fixed points and
    returns both rfp and gfp corrdinates in a list of tuples.

    Parameters here are expected to be log transformed (i.e. we exponentiate the values passed
     to get the numerical values from the paper/writeup). This is because this function will be
     called by the logelikelihood/squared errror function for fitting and fitting should happen
     on the log transformed valued of the natural parameters so that the natural parameter value
     is always positive.
    '''
    #exponeniate the parameters to get them on the natural scale
    param = np.exp(par_all)

    #un-transforming the parameters so they are  on their natural scale
    rho_r = 1/(1+np.exp(-par_all[0]))
    rho_g = 1/(1+np.exp(-par_all[1]))
    omega_g = np.exp(par_all[2])
    omega_r = np.exp(par_all[3])
    kappa_a = np.exp(par_all[4])
    kappa_i = np.exp(par_all[5])
    theta = np.exp(par_all[6])

    #group natural parameters into gfp and rfp block
    rfp_par = [rho_r, omega_g, kappa_a]
    gfp_par = [rho_g, omega_r, kappa_i]
    #creat a full list of natural parameters for the rfp root finding 
    param = rfp_par + gfp_par

    #call the rfp_roots function to get the stable rfp values
    rfp = rfp_hat_roots(atc,iptg,param)

    #check if 1 or 3 rfp roots returned
    if len(rfp)==1:
        #compute gfp level for a single root, compute ratio*theta and return
        gfp = promoter_model(rfp[0],iptg,gfp_par)
        return [np.log(theta*rfp[0]/gfp)]
    elif len(rfp)==3:
        #compute gfp level for smallest and largest roots, compute ratio*theta and return
        gfp_low = promoter_model(rfp[0],iptg,gfp_par)
        gfp_high = promoter_model(rfp[2],iptg,gfp_par)
        return [np.log(theta*rfp[0]/gfp_low), np.log(theta*rfp[2]/gfp_high)]
    else:
        #should never get here
        return 0

def simulate_model(u_list, endpoints, param):
    '''
    This function accepts inputs like generate data but it doesn't add any noise. It just simulates
    the model's predictions for the inputs and parameters passed in the given experiment/dataset.
    Useful for plotting and comparing the model on different parameters

    u_list is a list of tuples, one for each input level used in the experiment. Each tuple
    has three entries:
    u[0] is the % of master atc media in the 
    u[1] is a boolean, true if ATC overnight was used, false if IPTG was used
    u[2] is the number of cells measured in the given input condition
    And example u_list for 3 input levels of 0%, 50% and 100% ATC, with both ATC and IPTG overnights
    in each condition and 200 cells segmented from each would be:
    [(0.0,True,200),(0.5,True,200),(1.0,True,200),(0.0,False,200),(0.5,False,200),(1.0,False,200)]

    Endpoints is list with two tuples in it, the first tuple is the point in ATC x IPTG space with
    the greater ATC value (i.e. high atc or pure atc master media mix), the second is the point 
    in input inducer space with the low/zero ATC level.

    '''
    #NOTE:
    #branch label is true if prepped in ATC overnight, false if IPTG

    #create lists for input percentage atc, input values (atc, iptg), observations (rfp,gfp),
    #branch_lables (True/False Atc overnight), and number of cells
    #values are accumulated in these lists and merged into a dataframe at the end
    input_perc = []
    ratio_list = []
    branch_labels = []
    #loop over the input list
    for u in u_list:
        #compute the input values (ng/ml, mM) from the atc perc. and the endpoints
        inputs = u[0]*endpoints[0] + (1-u[0])*endpoints[1]
        #compute the stable rfp,gfp pairs for the given inputs (and parameters)
        points = log_ratio_points(inputs[0],inputs[1],param)

        #check if 1 or 2 stable points
        if len(points)==1:
            ratio = points[0]
        elif len(points)==2:
            #if 2 points, check which overnight was used
            if not u[1]:
                ratio = points[0]
            else:
                ratio = points[1]

        #append the inputs/observations for the current input_list condition to the storage lists
        input_perc.append(u[0])
        ratio_list.append(ratio)
        branch_labels.append(u[1])

    #create the pandas dataframe for the data
    dataset = df = pd.DataFrame(data={'perc':input_perc,
                                      'ratio':ratio_list,
                                      'branch':branch_labels})

    return dataset

##################################################################################################
# Example script
##################################################################################################

#Set some parameter values (taken from overleaf, from original paper)
alpha_1 = 13.609
alpha_2 = 60.882   
beta_1 = 3529.923   
beta_2 = 1053.916   
K_1 = 30.0    
K_2 = 31.94 
kappa_1 = 11.65  
kappa_2 = 0.0906  

#Map them to the new parameter's from the overleaf, we don't need to start with above but
#I did this to show the process for the original parameters, we can start rigth from here from 
#now on
rho_r = alpha_1/(alpha_1+beta_1)
rho_g = alpha_2/(alpha_2+beta_2)
omega_g = K_1/(alpha_2+beta_2)
omega_r = K_2/(alpha_1+beta_1)
kappa_1 = 11.65  
kappa_2 = 0.0906   
theta = beta_1/beta_2

#Transform them for fitting
eps_rho_r = np.log(rho_r/(1-rho_r))
eps_rho_g = np.log(rho_g/(1-rho_g))
eps_omega_g = np.log(omega_g)
eps_omega_r = np.log(omega_r)
eps_kappa_a = np.log(kappa_1)
eps_kappa_i = np.log(kappa_2)
eps_theta = np.log(theta)

#log the natural parameter values so we can fit log-transformed values
#this ensures the fit parameters are positive on the natural scale
par_all = [eps_rho_r, eps_rho_g, eps_omega_g, eps_omega_r, eps_kappa_a, eps_kappa_i, eps_theta]

#to generate some sample data:
#set number of cells to observe
num = 100
#set atc % to observe
u_vals = [.1, .2, .3, .4, .42,.44,.46,.48, .5, .52,.54,.56,.58, .6, .7, .8, .9]
#bind atc % and num cells into a u_list with booleans indicating atc overnight
u_list = [(u,True) for u in u_vals]+[(u,False) for u in u_vals]

#set the atc and iptg endpoints in input inducer space (same as in original paper)
iptg_min=0
iptg_max=0.45
atc_min=0
atc_max=45
endpoints =np.array([[atc_max,iptg_min],[atc_min,iptg_max]])

#test log ratio solver
lr=log_ratio_points(22.5,.225,par_all)

#simulate the model
data = simulate_model(u_list, endpoints, par_all)

#change the thete parameter by 2x
eps_theta = np.log(theta*2)
par_all2 = [eps_rho_r, eps_rho_g, eps_omega_g, eps_omega_r, eps_kappa_a, eps_kappa_i, eps_theta]
data2 = simulate_model(u_list, endpoints, par_all2)

#change the rho_r parameter by 20x
eps_rho_r = np.log(rho_r*20/(1-rho_r*20))
par_all3 = [eps_rho_r, eps_rho_g, eps_omega_g, eps_omega_r, eps_kappa_a, eps_kappa_i, eps_theta]
data3 = simulate_model(u_list, endpoints, par_all3)

#example plotting of rfp/gfp ratio
plt.plot(data['perc'],data['ratio'],'r+')
plt.plot(data2['perc'],data2['ratio'],'b+')
plt.plot(data3['perc'],data3['ratio'],'g+')
plt.show()