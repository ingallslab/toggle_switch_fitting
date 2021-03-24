import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.stats as st
import matplotlib.pyplot as plt


def rfp_roots(atc,iptg,param):
    #laci=rfp=high atc
    #tetr=gfp=high iptg
    #param = alpha_1, beta_1, K_1, kappa_1, alpha_2, beta_2, K_2, kappa_2

    rfp_root_func = lambda rfp: param[0] + param[1]/(1+((
        (param[4] + param[5]/(1+((rfp/param[6])*(1/(1+(iptg/param[7])**2.0)))**2.0))
        /param[2])*(1/(1+(atc/param[3])**2.0)))**2.0)-rfp

    min_rfp = param[0]
    max_rfp = param[0] + param[1]

    tol = (max_rfp - min_rfp)*.01

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
        stable_point = opt.root_scalar(rfp_root_func,method='brentq',bracket=interval).root
        stable_points.append(stable_point)

    return stable_points


def stable_points(atc,iptg,par):

    param = np.exp(par)

    gfp_func = lambda rfp,iptg: param[4] + param[5]/(1+((rfp/param[6])*(1/(1+(iptg/param[7])**2.0)))**2.0)

    rfp = rfp_roots(atc,iptg,param)

    if len(rfp)==1:
        gfp = gfp_func(rfp[0],iptg)
        return [np.array([rfp[0],gfp])]
    elif len(rfp)==3:
        gfp_low = gfp_func(rfp[0],iptg)
        gfp_high = gfp_func(rfp[2],iptg)
        return [np.array([rfp[0],gfp_low]),np.array([rfp[2],gfp_high])]
    else:
        return 0


def generate_data(u_list, endpoints, param_all, num):

    input_perc = []
    input_values = []
    observations = []
    labels = []
    for u in u_list:
        inputs = u[0]*endpoints[0] + (1-u[0])*endpoints[1]
        points = stable_points(inputs[0],inputs[1],param_all[:-2])

    #least squares in here, u is tuple, second entry is branch label
        if len(points)==1:
            data =np.random.multivariate_normal(points[0],np.diagflat((0.1*points[0])**2),num)
            bimodal_label = np.full(num,False)
        elif len(points)==2:
            upper_branch_probability = 1/(1+np.exp(-param_all[-1]*(u-param_all[-2])))
            upper_samples = np.random.binomial(num, upper_branch_probability)

            lower_data = np.random.multivariate_normal(points[0],np.diagflat((0.1*points[0])**2),num-upper_samples)
            upper_data =np.random.multivariate_normal(points[1],np.diagflat((0.1*points[1])**2),upper_samples)

            data = np.concatenate((lower_data,upper_data))
            bimodal_label = np.full(num,True)

        input_perc.append(np.repeat(u,num))
        input_values.append(np.tile(inputs,(num,1)))
        observations.append(data)
        labels.append(bimodal_label)

    input_perc = np.concatenate(input_perc)
    input_values = np.concatenate(input_values)
    observations = np.concatenate(observations)
    labels = np.concatenate(labels)

    dataset = df = pd.DataFrame(data={'atc%':input_perc,
                                      'atc':input_values[:,0],
                                      'iptg':input_values[:,1],
                                      'rfp':observations[:,0],
                                      'gfp':observations[:,1],
                                      'bimodal':labels})

    return dataset


def loglike(param_all, dataset):

    ll=0

    print(param_all)

    for i,row in dataset.iterrows():

        points = stable_points(row['atc'],row['iptg'],param_all[:-2])

        if len(points)==1:
            ll = ll + np.log(st.multivariate_normal.pdf([row['rfp'],row['gfp']], points[0], np.diagflat((0.1*points[0])**2)))
        else:
            upper_branch_probability = 1/(1+np.exp(-param_all[-1]*(row['atc%']-aux[-2])))
            lower_ll = st.multivariate_normal.pdf([row['rfp'],row['gfp']], points[0], np.diagflat((0.1*points[0])**2))
            upper_ll2 = st.multivariate_normal.pdf([row['rfp'],row['gfp']], points[1], np.diagflat((0.1*points[1])**2))

            ll = ll + np.log((1-upper_branch_probability)*lower_ll + upper_branch_probability*upper_ll2)

        # if row['bimodal']:
        #     if len(points)!=2:
        #         ll = ll - 1e9
        #     else:
        #         upper_branch_probability = 1/(1+np.exp(-param_all[-1]*(row['atc%']-aux[-2])))
        #         lower_ll = st.multivariate_normal.pdf([row['rfp'],row['gfp']], points[0], np.diagflat((0.1*points[0])**2))
        #         upper_ll2 = st.multivariate_normal.pdf([row['rfp'],row['gfp']], points[1], np.diagflat((0.1*points[1])**2))

        #         ll = ll + np.log((1-upper_branch_probability)*lower_ll + upper_branch_probability*upper_ll2)

        # elif not row['bimodal']:
        #     if len(points)!=1:
        #         ll = ll - 1e9
        #     else:
        #         ll = ll + np.log(st.multivariate_normal.pdf([row['rfp'],row['gfp']], points[0], np.diagflat((0.1*points[0])**2)))
        
    return ll
        

alpha_1 = 13.609
alpha_2 = 60.882   
beta_1 = 3529.923   
beta_2 = 1053.916   
K_1 = 30.0    
K_2 = 31.94 
kappa_1 = 11.65  
kappa_2 = 0.0906   

param= [alpha_1, beta_1, K_1, kappa_1, alpha_2, beta_2, K_2, kappa_2]
aux=[0.5, 10]
param_all = np.log(param+aux)

u_list = [.1, .2, .3, .4, .42,.44,.46,.48, .5, .52,.54,.56,.58, .6, .7, .8, .9]

iptg_min=0
iptg_max=0.45
atc_min=0
atc_max=45
endpoints =np.array([[atc_max,iptg_min],[atc_min,iptg_max]])

num =10

dataset = generate_data(u_list, endpoints, param_all, num)

ll=loglike(param_all, dataset)

neg_log_lik = lambda p: -loglike(p,dataset)
param_start = param_all #+ np.random.multivariate_normal(param_all,np.diagflat((0.01*param_all)**2))

est=opt.minimize(neg_log_lik, param_start, method='Nelder-Mead', tol=1e-6)

plt.plot(dataset['atc%'], np.log(dataset['rfp']/dataset['gfp']),'r+')
plt.show()


i=0
