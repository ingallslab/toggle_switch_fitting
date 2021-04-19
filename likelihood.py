import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.stats as st
import scipy.linalg as ln
import matplotlib.pyplot as plt

def rfp_roots(atc,iptg,param):
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
    #param = alpha_1, beta_1, K_1, kappa_1, alpha_2, beta_2, K_2, kappa_2

    # model consists of two equations rfp=g1(gfp,inputs) and gfp=g2(rfp,inputs)
    # this function implements 0=g1(g2(rfp,inputs),inputs)-rfp
    # the lower and upper roots of this function for rfp correspond to model branches
    # to get gfp we take the rfp root and sub it in g2, i.e. gfp=g2(rfp_root,inputs)
    rfp_root_func = lambda rfp: param[0] + param[1]/(1+((
        (param[4] + param[5]/(1+((rfp/param[6])*(1/(1+(iptg/param[7])**2.0)))**2.0))
        /param[2])*(1/(1+(atc/param[3])**2.0)))**2.0)-rfp

    #extract the min and max possible rfp values from the model parameters, roots will occur in this
    #intervale
    min_rfp = param[0]
    max_rfp = param[0] + param[1]

    #set the search tolerance over the rfp range, below this tolerance 3 nearby roots and a single
    #root will look the same to the algorithm
    tol = (max_rfp - min_rfp)*.01

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


def stable_points(atc,iptg,par):
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
    param = np.exp(par)

    #implement the gfp steady-state function, mapping rfp, inputs and parameters to gfp
    gfp_func = lambda rfp,iptg: param[4] + param[5]/(1+((rfp/param[6])*(1/(1+(iptg/param[7])**2.0)))**2.0)

    #call the rfp_roots function to get the stable rfp values
    rfp = rfp_roots(atc,iptg,param)

    #check if 1 or 3 rfp roots returned
    if len(rfp)==1:
        #compute gfp level for a single root, tuple with rfp and return
        gfp = gfp_func(rfp[0],iptg)
        return [np.array([rfp[0],gfp])]
    elif len(rfp)==3:
        #compute gfp level for smallest and largest roots, tuple with rfps and return
        gfp_low = gfp_func(rfp[0],iptg)
        gfp_high = gfp_func(rfp[2],iptg)
        return [np.array([rfp[0],gfp_low]),np.array([rfp[2],gfp_high])]
    else:
        #should never get here
        return 0


def lna_covariance(rfp,gfp,atc,iptg,par,lamb,size):
    '''
    This function estimates the covariance matrix of the rfp and gfp measurments taken at a specific
    point using the LNA approximation.

    This covariance matrix is not scaled by the system size by default, the user must do this to
    the result i.e. cov/size

    This function computes the covariance of a single celled observation. If many single-celled 
    observations are taken at the same input point, the covariance matrix entries will be smaller 
    for the mean of the observations than for the individual values (law of large numbers).
    The covariance matrix for a given single celled observation (as computed here) can be scaled by the
     number of observations taken to get the covariance matrix for the mean.
    '''
    #define the steady-state functions for each species
    rfp_func = lambda gfp,atc: param[0] + param[1]/(1+((gfp/param[2])*(1/(1+(atc/param[3])**2.0)))**2.0)
    gfp_func = lambda rfp,iptg: param[4] + param[5]/(1+((rfp/param[6])*(1/(1+(iptg/param[7])**2.0)))**2.0)
    #define the derivatives of the steady state-functions w.r.t. the opposite species
    rfp_d_gfp = lambda gfp,atc: -2*param[1]*gfp*((1/param[2])*(1/(1+(atc/param[3])**2.0)))**2.0 \
                                    /(1+(gfp**2)*((1/param[2])*(1/(1+(atc/param[3])**2.0)))**2.0)
    gfp_d_rfp = lambda rfp,iptg: -2*param[5]*rfp*((1/param[6])*(1/(1+(iptg/param[7])**2.0)))**2.0 \
                                    /(1+(rfp**2)*((1/param[6])*(1/(1+(iptg/param[7])**2.0)))**2.0)

    #compute the A (damping) matrix
    A = np.array([[-lamb, lamb*rfp_d_gfp(gfp,atc)],[lamb*gfp_d_rfp(rfp,iptg),-lamb]])
    #compute the B (fluctuation) matrix
    B = np.array([[lamb*(rfp_func(gfp,atc)+rfp),0],[0,lamb*(gfp_d_rfp(rfp,iptg)+gfp)]])

    #solve the sylvester/lyapunov equation (this may be ill-conditioned near where a branch 
    # disappears i.e. a bifurcation point
    C = ln.solve_continuous_lyapunov(A,-B)

    return C
    

def generate_data(u_list, endpoints, param, batch=True, lna=False,lamb=1,size=1):
    '''
    This function generates simulated data for the model. This can be used to evaluate fitting and
    check how we expect the fitting algorithm to perform.

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

    It is assumed all experiments measure the system along some linear combination of the ATC and IPTG
    mixtures specified by the end points. 

    This function returns a dataframe with the simulated data in it. The columns in the dataframe are:
    perc,atc,iptg,rfp,gfp,branch,num
    perc - is the ATC master percentage from 0-1
    atc - is the actual atc concentration in ng/ml
    iptg - is the actual iptg concentration in mM
    rfp - is the observed rfp value (mean if batch=True, single cell if batch=False)
    gfp - is the observed gfp value
    branch - is a boolean indicating if ATC overnight (True) or IPTG overnight (False) was used
    num - (optional, only if batch=True) is the number of segmented cells observed in that condition
                                        used to compute rfp and gfp mean observations

    The returned data by default is in batch mode (batch=True) in which case single cell observations
    of the same input condition are mered into a single row, the mean rfp and gfp values are returned
    and the numberd of cell involved in the mean is given in the num column. If batch=False is used
    the function returns a dataframe where each row corresponds to a single cell. This can be useful
    for plotting or more complex likelihood functions, if we ever get to them.
    '''
    #NOTE:
    #branch label is true if prepped in ATC overnight, false if IPTG

    #create lists for input percentage atc, input values (atc, iptg), observations (rfp,gfp),
    #branch_lables (True/False Atc overnight), and number of cells
    #values are accumulated in these lists and merged into a dataframe at the end
    input_perc = []
    input_values = []
    observations = []
    branch_labels = []
    num_cells = []
    #loop over the input list
    for u in u_list:
        #compute the input values (ng/ml, mM) from the atc perc. and the endpoints
        inputs = u[0]*endpoints[0] + (1-u[0])*endpoints[1]
        #compute the stable rfp,gfp pairs for the given inputs (and parameters)
        points = stable_points(inputs[0],inputs[1],param)

        #check if 1 or 2 stable points
        if len(points)==1:
            if lna:
                cov = lna_covariance(points[0][0],points[0][1],inputs[0],inputs[1],param,lamb,size)/size
            else:
                cov = np.diagflat((0.1*points[0])**2)
            #if 1 point,  generate the approriate number of rfp,gfp observations
            obs = np.random.multivariate_normal(points[0],cov,u[2])
        elif len(points)==2:
            #if 2 points, check which overnight was used
            if not u[1]:
                if lna:
                    cov = lna_covariance(points[0][0],points[0][1],inputs[0],inputs[1],param,lamb,size)/size
                else:
                    cov = np.diagflat((0.1*points[0])**2)
                #generated observations arround lower rfp branch if IPTG overnight was used
                obs = np.random.multivariate_normal(points[0],cov,u[2])
            else:
                if lna:
                    cov = lna_covariance(points[1][0],points[1][1],inputs[0],inputs[1],param,lamb,size)/size
                else:
                    cov = np.diagflat((0.1*points[1])**2)
                #generated observations arround upper rfp branch if ATC overnight was used
                obs = np.random.multivariate_normal(points[1],cov,u[2])

        #check if data is batched or single cell
        if batch:
            #if batched, set reps to 1 (numbers of rows for each input list condition)
            reps = 1
            #add the number of cells to be observed in the given condition to the num_cells list
            num_cells.append(u[2])
            #compute the mean of the rfp and gfp values observed for each cell (we could be more efficient here)
            data = np.mean(obs,axis=0)
        else:
            #if single cell data is desired, set reps to number of cells (one row per cell)
            reps = u[2]
            #add all of the single cell observations to the data variable
            data = obs

        #append the inputs/observations for the current input_list condition to the storage lists
        input_perc.append(np.repeat(u[0],reps))
        input_values.append(np.tile(inputs,(reps,1)))
        observations.append(data)
        branch_labels.append(np.full(reps,u[1]))

    #merge the storage lists into numpy arrays
    input_perc = np.concatenate(input_perc)
    input_values = np.vstack(input_values)
    observations = np.vstack(observations)
    branch_labels = np.concatenate(branch_labels)

    #create the pandas dataframe for the data
    dataset = df = pd.DataFrame(data={'perc':input_perc,
                                      'atc':input_values[:,0],
                                      'iptg':input_values[:,1],
                                      'rfp':observations[:,0],
                                      'gfp':observations[:,1],
                                      'branch':branch_labels})

    #if data is batched, add the number of cells column to the dataframe
    if batch:
        num_cells = np.array(num_cells)
        dataset['num'] = num_cells

    return dataset


def loglike(param_all, dataset, lamb):
    '''
    This function computes the loglikelihood for the model with the given parameter values on the
    passed dataset (formated as a pandas array, like in the data generating function)

    This function assumes the dataset is in batched form, this makes things more efficient

    This function can be used, in conjunction with an numpy optimization function, to fit the model
    '''
    
    param = param_all[:-1]
    size = np.exp(param_all[-1])

    #create a accumulator variable for the loglikelihood and set the value to zero
    ll=0

    #loop over rows in the dataframe
    for i,row in dataset.iterrows():

        #for the current row's input conditions, compute the expected stable points (rfp,gfp) with
        #passed parameters
        means = stable_points(row['atc'],row['iptg'],param)

        #check if current input points, with passed parameters has 1 or 2 stable points
        if len(means)==1:
            #compute the LNA covariance matrix around the single mean point
            covar = lna_covariance(means[0][0],means[0][1],row['atc'],row['iptg'],param,lamb,size)
            #if a single stable points add the log pdf of the data to the loglikelihood accumulation
            #function
            ll = ll + np.log(row['num']*size) - 0.5*np.log(np.linalg.det(covar)) \
                 - row['num']*size*(row[['rfp','gfp']].to_numpy()-means[0])@np.linalg.inv(covar)@(row[['rfp','gfp']].to_numpy()-means[0]).T
            #np.log(st.multivariate_normal.pdf([row['rfp'],row['gfp']], means[0], np.diagflat((0.1*means[0])**2))/row['num'])
        else:
            #if there are 2 stable points, check overnight condition
            if not row['branch']:
                #compute the LNA covariance matrix around the mean on the lower branch
                covar = lna_covariance(means[0][0],means[0][1],row['atc'],row['iptg'],param,lamb,size)
                #if iptg overnight, compute log-pdf value of data around lower rfp stable point
                ll = ll + np.log(row['num']*size) - 0.5*np.log(np.linalg.det(covar)) \
                 - row['num']*size*(row[['rfp','gfp']].to_numpy()-means[0])@np.linalg.inv(covar)@(row[['rfp','gfp']].to_numpy()-means[0]).T
                #np.log(st.multivariate_normal.pdf([row['rfp'],row['gfp']], means[0], np.diagflat((0.1*means[0])**2))/row['num'])
            else:
                #compute the LNA covariance matrix around the mean on the upper branch
                covar = lna_covariance(means[1][0],means[1][1],row['atc'],row['iptg'],param,lamb,size)
                #if atc overnight, compute log-pdf value of data around upper rfp stable point
                ll = ll + np.log(row['num']*size) - 0.5*np.log(np.linalg.det(covar)) \
                 - row['num']*size*(row[['rfp','gfp']].to_numpy()-means[1])@np.linalg.inv(covar)@(row[['rfp','gfp']].to_numpy()-means[1]).T
                #np.log(st.multivariate_normal.pdf([row['rfp'],row['gfp']], means[1], np.diagflat((0.1*means[1])**2))/row['num'])

    #return the accumulated loglikelihood, this value needs to be maximized
    return ll

def squared_error(param, dataset):
    '''
    This function computes the sum of squared error for the model with the given parameter values on the
    passed dataset (formated as a pandas array, like in the data generating function)

    This function assumes the dataset is in batched form, this makes things more efficient

    This function can be used, in conjunction with an numpy optimization function, to fit the model
    '''
    #create an accumulator variable for the sum of squared error, and set it to zero
    sse=0

    #loop over the rows of the dataset
    for i,row in dataset.iterrows():
        
        #for the current row's input conditions, compute the expected stable points (rfp,gfp) with
        #passed parameters
        points = stable_points(row['atc'],row['iptg'],param)

        #check if current input points, with passed parameters has 1 or 2 stable points
        if len(points)==1:
            #if a single stable points add sum of squared error between the observation and the stable
            #point
            sse = sse + row['num']*(row['rfp'] - points[0][0])**2\
                        + row['num']*(row['gfp'] - points[0][1])**2
        else:
            #if there are 2 stable points, check overnight condition
            if not row['branch']:
                #if iptg overnight, compute log-pdf value of data around lower rfp stable point
                sse = sse + row['num']*(row['rfp'] - points[0][0])**2 \
                            + row['num']*(row['gfp'] - points[0][1])**2
            else:
                #if atc overnight, compute log-pdf value of data around upper rfp stable point
                sse = sse + row['num']*(row['rfp'] - points[1][0])**2 \
                             + row['num']*(row['gfp'] - points[1][1])**2

            #Code above assumes all observations have the same variance/covariance, commented below is
            #part of a weighted least squares computation that accounts for changeing variance
            #haven't tested much
            # if not row['branch']:
            #     sse = sse + row['num']*(row['rfp'] - points[0][0])**2/points[0][0]**2 \
            #                 + row['num']*(row['gfp'] - points[0][1])**2/points[0][1]**2
            # else:
            #     sse = sse + row['num']*(row['rfp'] - points[1][0])**2/points[1][0]**2 \
            #                  + row['num']*(row['gfp'] - points[1][1])**2/points[1][1]**2

    #return accumulated sum of squared error, this value should be minimized
    return sse

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

#log the natural parameter values so we can fit log-transformed values
#this ensures the fit parameters are positive on the natural scale
param= np.log([alpha_1, beta_1, K_1, kappa_1, alpha_2, beta_2, K_2, kappa_2])

#to generate some sample data:
#set number of cells to observe
num = 100
#set atc % to observe
u_vals = [.1, .2, .3, .4, .42,.44,.46,.48, .5, .52,.54,.56,.58, .6, .7, .8, .9]
#bind atc % and num cells into a u_list with booleans indicating atc overnight
u_list = [(u,True,num) for u in u_vals]+[(u,False,num) for u in u_vals]

#set the atc and iptg endpoints in input inducer space (same as in original paper)
iptg_min=0
iptg_max=0.45
atc_min=0
atc_max=45
endpoints =np.array([[atc_max,iptg_min],[atc_min,iptg_max]])

# generate a batched dataset
dataset_batch = generate_data(u_list, endpoints, param, batch =True, lna=True)
# generate a single cell dataset
dataset_cell = generate_data(u_list, endpoints, param, batch =False,lna=True)

#example plotting of rfp/gfp ratio
plt.plot(dataset_batch['perc'], np.log(dataset_batch['rfp']/dataset_batch['gfp']),'r+')
plt.show()

#load in the real data (batched), rescale atc % to 0-1
real_batch_data = pd.read_csv('DataFrame_2_March_23',index_col=0)
real_batch_data['perc']=real_batch_data['perc']/100
#load in the real data (single celled), rescale atc % to 0-1
real_cell_data = pd.read_csv('DataFrame_1_March_23',index_col=0)
real_cell_data['perc']=real_cell_data['perc']/100

#add the system size to the parameters
param_all = np.append(param,[0])
# test out the loglikelihood and sum of squared error functions, just to test, use lamb=1
ll = loglike(param_all, dataset_batch, 1)
sse = squared_error(param, dataset_batch)

# #create an anonymous function for the sum of squared error of the simulated dataset
# s_error = lambda p: squared_error(p,dataset_batch)
# #choose some initial parameters, the most basic test is to set the to the "true" values used
# # to generate the data. Fitting should converge for this condition, with fitting error only due to 
# #observation randomness. Next step is add some random perturbations (commented out)
# param_start = param #+ np.random.multivariate_normal(param,np.diagflat((0.01*param)**2))

# #use the optimization function in scipy to generate an estimate
# #i use nelder-mead because it doesn't need any derivatives and works reasonably on non-smooth functions
# #our current loglik/sse is non-smooth because of the bifurcation points
# est2=opt.minimize(s_error, param_start, method='Nelder-Mead', tol=1e-6)

#create an anonymous function for the negative loglikelihood of the simulated dataset
#(negative because we will minimize instead of maximizing)
neg_log_lik = lambda p: -loglike(p,dataset_batch,1)
#set initial parameter value, from which optimization starts (setting it to truth is pretty much
# cheating, only good for numerical testing)
param_start = param_all #+ np.random.multivariate_normal(param,np.diagflat((0.01*param)**2))

#use the optimization function in scipy to generate an estimate
#this is currently giving some numerical problems, sse may be more stable
est1=opt.minimize(neg_log_lik, param_start, method='Nelder-Mead', tol=1e-6)


