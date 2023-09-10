#!/usr/bin/env python3

'''
Deterministic numerical solver for ODE systems
Pablo Cardenas R.

used for
Zhang et al., 2023
Coded by github.com/pablocarderam based on original model by An-Ni Zhang
'''


### Imports ###
import numpy as np # handle arrays
import pandas as pd
from scipy import integrate # numerical integration
import joblib as jl
import itertools as it

import seaborn as sns # for plots
import matplotlib.pyplot as plt
import matplotlib.colors as mlc

sns.set_style("white") # make pwetty plots
cb_palette = ["#999999", "#E69F00", "#56B4E9", "#009E73",
              "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
    # http://www.cookbook-r.com/Graphs/Colors_(ggplot2)/#a-colorblind-friendly-palette
    # http://jfly.iam.u-tokyo.ac.jp/color/



### Methods ###
# User-defined methods #
def params():

    """
    Returns default values constant values for the model in a dictionary.
    """

    gen_t = 24

    # Sorry! parameter naming convention is a bit different here than in manuscript

    params = {
        't_0':0/gen_t,          # h - Initial time value
        't_f':24*100/gen_t,      # h - Final time value
        't_den':0.1/gen_t,      # h - Size of time step to evaluate with

        'd':1*gen_t,       # 1/generation - infected bacteria death rate
        'b':1e-9*gen_t,     # 1/(PFU/mL * h) - phage-bacteria infection rate
        'r':np.log(2)/(0.5),  # CFU/mL - max bacteria growth rate
        'l':0,#1e-5*gen_t          # 1/generation - spacer loss rate 
        'h':0,#1e-9*gen_t,       # 1/(CFU/mL * h) - HGT rate of spacer
        'c':1e-3*gen_t,          # 1/generation - spacer acquisition rate
        'g':0*gen_t,          # 1/generation - phage degradation rate
        'n':10,          # PFU/mL - phage burst size
        'K':1e78,        # CFU/mL - maximum bacterial population size

        'd_t':24/gen_t,        # h – time between dilution events
        'd_f':1.5,        # no dimension – dilution factor
    }

    return params


def initCond():

    '''
    Return initial conditions values for the model in a dictionary.
    '''

    y0 = [
        # Initial concentrations in [M] (order of state variables matters)
        params()['K'],  # CFU/mL - susceptible, uninfected bacteria S
        0,              # CFU/mL - infected bacteria I
        0,              # CFU/mL - Resistant bacteria with spacer R
        10,             # PFU/mL - Free phage P
        ]

    return y0


def odeFun(t,y,**kwargs):

    """
    Contains system of differential equations.

    Arguments:
        t        : current time variable value
        y        : current state variable values (order matters)
        **kwargs : constant parameter values, interpolanting functions, etc.
    Returns:
        Dictionary containing dY/dt for the given state and parameter values
    """

    S,I,R,P = y # unpack state variables
    # (state variable order matters for numerical solver)

    # Unpack variables passed through kwargs (see I thought I could avoid this
    # and I just made it messier)
    d,b,r,l,h,c,g,n,K = \
        kwargs['d'],kwargs['b'],kwargs['r'],kwargs['l'], \
        kwargs['h'],kwargs['c'],kwargs['g'],kwargs['n'],kwargs['K']

    # ODEs
    dS = r * ( 1 - ( ( S + I + R ) / K ) ) * S + l * R - h * S * R - b * S * P
    dI = b * S * P - d * I - c * I
    dR = r * ( 1 - ( ( S + I + R ) / K ) ) * R + c * I + h * S * R - l * R
    dP = d * n * I - g * P - b * S * P

    # Gather differential values in list (state variable order matters for
    # numerical solver)
    dy = [dS,dI,dR,dP]

    return dy


def figTSeries(sol,f_name='ODE_tseries.png'):

    """
    This function makes a plot for Figure 1 by taking all the solution objects
    as arguments, and prints out the plot to a file.
    Arguments:
        sol : solution object taken from solver output
    """

    t = sol.t[:] # get time values

    plt.figure(figsize=(6, 4), dpi=200) # make new figure

    ax = plt.subplot(2, 1, 1) # Fig A
    plt.plot(t, sol.y[0,:], label=r'$S$', color=cb_palette[2])
    plt.plot(t, sol.y[1,:], label=r'$I$', color=cb_palette[4])
    plt.plot(t, sol.y[2,:], label=r'$R$', color=cb_palette[6])
    plt.plot(t, sol.y[3,:], label=r'$P$', color=cb_palette[1])
    plt.yscale('log')
    plt.ylim(1,params()['K']*params()['n']*10)
    plt.xlabel('Time (generations)')
    plt.ylabel('Viable individuals/mL')
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels)

    ax = plt.subplot(2, 1, 2) # Fig A
    plt.plot(t, sol.y[2,:]/(sol.y[2,:]+sol.y[0,:]), color=cb_palette[3])
    # plt.yscale('log')
    plt.ylim(1e-4,1.1)
    plt.xlabel('Time (generations)')
    plt.ylabel('Fraction of bacterial \npopulation with spacer')
    # handles, labels = ax.get_legend_handles_labels()
    # plt.legend(handles, labels)

    plt.savefig(f_name, bbox_inches='tight')

# Pre-defined methods #
# These shouldn't have to be modified for different models
def odeSolver(func,t,y0,p,solver='LSODA',rtol=1e-8,atol=1e-8,**kwargs):

    """
    Numerically solves ODE system.

    Arguments:
        func     : function with system ODEs
        t        : array with time span over which to solve
        y0       : array with initial state variables
        p        : dictionary with system constant values
        solver   : algorithm used for numerical integration of system ('LSODA'
                   is a good default, use 'Radau' for very stiff problems)
        rtol     : relative tolerance of solver (1e-8)
        atol     : absolute tolerance of solver (1e-8)
        **kwargs : additional parameters to be used by ODE function (i.e.,
                   interpolation)
    Outputs:
        y : array with state value variables for every element in t
    """

    # default settings for the solver
    options = { 'RelTol':10.**-8,'AbsTol':10.**-8 }
    min_state_var = 1e-4

    # takes any keyword arguments, and updates options
    options.update(kwargs)

    current_t = 0
    next_t = current_t + p['d_t']
    y0_chunk = y0
    solution = np.vstack(y0)

    while current_t < t[-1]:
        next_t = min(current_t + p['d_t'], t[-1])
        t_chunk = t[ (t >= current_t) & (t < next_t) ]
        # runs scipy's new ode solver
        # print(current_t,next_t,t_chunk,p['d_t'],p['K'])
        if len(t_chunk)>1:
            # print([t_chunk[0],t_chunk[-1]])
            y = integrate.solve_ivp(
                    lambda t_var,y: func(t_var,y,**p,**kwargs), # use a lambda function
                        # to sub in all parameters using the ** double indexing operator
                        # for dictionaries, pass any additional arguments through
                        # **kwargs as well
                    [t_chunk[0],t_chunk[-1]], # initial and final time values
                    y0_chunk, # initial conditions
                    method=solver, # solver method
                    t_eval=t_chunk, # time point vector at which to evaluate
                    # rtol=rtol, # relative tolerance value
                    # atol=atol # absolute tolerance value
                )

        y0_chunk = y.y[:,-1] / p['d_f']
        y0_chunk[y0_chunk < min_state_var] = 0
        current_t = next_t
        solution = np.concatenate( (solution,y.y),axis=1 )

    y.y = solution
    y.t = t

    return y


# Solving model
# To generate data, uncomment the following...
# Single timecourse
def solveModel():

    '''
    Main method containing single solver and plotter calls.
    Writes figures to file.
    '''

    # Set up model conditions
    p = params() # get parameter values, store in dictionary p
    y_0 = initCond() # get initial conditions
    t = np.linspace(p['t_0'],p['t_f'],int((p['t_f']-p['t_0'])/p['t_den']) + 1)
        # time vector based on minimum, maximum, and time step values

    # Solve model
    sol = odeSolver(odeFun,t,y_0,p,solver="LSODA");

    # Call plotting of figure 1
    figTSeries(sol,f_name='ODE_tseries_crispr.png')

    plt.close()

solveModel()

# Heatmaps!
# First heatmap is K and d_t
samples = 200 # heatmap width in pixels

# These are the values of parameters used in sweeps (have to fill in axis ticks manually)
K_drops = np.power( 10, np.linspace( 6, 10, samples, endpoint=True ) )
print(K_drops)
print(np.power( 10, np.linspace( 6, 10, 5, endpoint=True ) ))

dilution_times = 2 * np.power( 10, np.linspace( 1, -1, samples, endpoint=True ) )
print(dilution_times)
print(np.linspace( 2, 0, 5, endpoint=True ))
print(2 * np.power( 10, np.linspace( 1, -1, 5, endpoint=True ) ))
# These times are then shown as frequencies

# stores parameters and values to be used in sweep
param_sweep_dic = { 'd_t':dilution_times,'K':K_drops }

# generate dataframe with all combinations
params_list = param_sweep_dic.keys()
value_lists = [ param_sweep_dic[param] for param in params_list ]
combinations = list( it.product( *value_lists ) )
param_df = pd.DataFrame(combinations)
param_df.columns = params_list

results = {} # store results

# This runs a single pixel
def run(param_values):
    # Set up model conditions
    p = params() # get parameter values, store in dictionary p
    y_0 = initCond() # get initial conditions
    t = np.linspace(p['t_0'],p['t_f'],int((p['t_f']-p['t_0'])/p['t_den']) + 1)
        # time vector based on minimum, maximum, and time step values

    for i,param_name in enumerate(params_list):
        p[param_name] = param_values[i]

    # Solve model
    sol = odeSolver(odeFun,t,y_0,p,solver="LSODA");

    res_frac = sol.y[2,:]/(sol.y[2,:]+sol.y[0,:])
    t_10 = np.argmax(res_frac>0.1) * params()['t_den']

    return [res_frac[-1],t_10]

# Parallelize running all pixels in heatmap
n_cores = jl.cpu_count()

res = jl.Parallel(n_jobs=n_cores, verbose=10) (
    jl.delayed( run ) (param_values) for param_values in combinations
     )

dat = param_df

dat['res_frac'] = np.array(res)[:,0]
dat['t_10'] = np.array(res)[:,1]
dat.to_csv('crispr_heatmaps_Kd.csv')
# ...until here

dat = pd.read_csv('crispr_heatmaps_Kd.csv')

# Reformat data for heatmaps
dat['d_t'] = 1/dat['d_t']
print('Rates: ')
print(list(dat['d_t'].unique()))
dat_frac_res = dat.pivot(index='K',columns='d_t',values='res_frac')
dat_frac_t10 = dat.pivot(index='K',columns='d_t',values='t_10')

# Plot heatmaps
plt.rcParams.update({'font.size': 12})

def plotHeatmap(
        dat,cmap_lab,file_name,cmap, vmin=None, vmax=None, show_labels=False):
    plt.figure(figsize=(8,8), dpi=200)
    ax = plt.subplot(1, 1, 1)
    if cmap == 'magma':
        ax = sns.heatmap(
            dat, linewidth = 0 , annot = False, cmap=cmap,
            cbar_kws={'label': cmap_lab}, vmin=vmin, vmax=vmax,
            xticklabels=show_labels, yticklabels=show_labels, norm=mlc.LogNorm()
            )
    else:
        ax = sns.heatmap(
            dat, linewidth = 0 , annot = False, cmap=cmap,
            cbar_kws={'label': cmap_lab}, vmin=vmin, vmax=vmax,
            xticklabels=show_labels, yticklabels=show_labels, #norm=mlc.LogNorm()
            )
    ax.figure.axes[-1].yaxis.label.set_size(15)
    ax.invert_yaxis()
    spacing = '\n\n\n'# if show_labels else ''
    plt.ylabel('Maximum bacterial density (CFU/mL)'+spacing,fontsize=15)
    plt.xlabel(spacing+'Dilution rate (1/generation)',fontsize=15)
    plt.savefig(file_name, bbox_inches='tight')

plotHeatmap(
    dat_frac_res,'Fraction of bacterial population with spacer',
    'crispr_heatmap_deterministic_Kd_frac.png','viridis_r', vmin=0, vmax=1#, show_labels=True
    )

plotHeatmap(
    dat_frac_t10,'Time at which resistant bacteria with spacer reach 10% (generations)',
    'crispr_heatmap_deterministic_Kd_t10.png','magma',  vmin=1e-1, vmax=1e2,#, show_labels=True
    )


# Second heatmap is h and c

samples = 200 # heatmap width in pixels

h_drops = (params()['b']/0.24) * np.power( 10, np.linspace( -2, 2, samples, endpoint=True ) )
    # fitness costs tested in Opqua stochastic model
print(h_drops)
print((params()['b']/0.24) * np.power( 10, np.linspace( -2, 2, 5, endpoint=True ) ))
# contact rates tested in Opqua stochastic model
c_drops = (params()['c']/0.24) * np.power( 10, np.linspace( -2, 2, samples, endpoint=True ) )*10
# print( np.linspace( 3, 1, samples, endpoint=True ) )
# print(np.power( 5, -np.linspace( 3, 1, samples, endpoint=True ) ))
print(c_drops)
print((params()['c']/0.24) * np.power( 10, np.linspace( -2, 2, 5, endpoint=True ) )*10)

# stores parameters and values to be used in sweep
param_sweep_dic = { 'c':c_drops,'h':h_drops }

# generate dataframe with all combinations
params_list = param_sweep_dic.keys()
value_lists = [ param_sweep_dic[param] for param in params_list ]
combinations = list( it.product( *value_lists ) )
param_df = pd.DataFrame(combinations)
param_df.columns = params_list

results = {} # store results

# Parallelize running all pixels in heatmap
n_cores = jl.cpu_count()

res = jl.Parallel(n_jobs=n_cores, verbose=10) (
    jl.delayed( run ) (param_values) for param_values in combinations
     )

dat = param_df

dat['res_frac'] = np.array(res)[:,0]
dat['t_10'] = np.array(res)[:,1]
dat.to_csv('crispr_heatmaps_HGT.csv')
# ...until here

dat = pd.read_csv('crispr_heatmaps_HGT.csv')

# Reformat data for heatmaps
dat_frac_res = dat.pivot(index='c',columns='h',values='res_frac')
dat_frac_t10 = dat.pivot(index='c',columns='h',values='t_10')

# Plot heatmaps
plt.rcParams.update({'font.size': 12})

def plotHeatmap(
        dat,cmap_lab,file_name,cmap, vmin=None, vmax=None, show_labels=False):
    plt.figure(figsize=(8,8), dpi=200)
    ax = plt.subplot(1, 1, 1)
    if cmap == 'magma':
        ax = sns.heatmap(
            dat, linewidth = 0 , annot = False, cmap=cmap,
            cbar_kws={'label': cmap_lab}, vmin=vmin, vmax=vmax,
            xticklabels=show_labels, yticklabels=show_labels, norm=mlc.LogNorm()
            )
    else:
        ax = sns.heatmap(
            dat, linewidth = 0 , annot = False, cmap=cmap,
            cbar_kws={'label': cmap_lab}, vmin=vmin, vmax=vmax,
            xticklabels=show_labels, yticklabels=show_labels, #norm=mlc.LogNorm()
            )
    ax.figure.axes[-1].yaxis.label.set_size(15)
    ax.invert_yaxis()
    spacing = '\n\n\n'# if show_labels else ''
    plt.xlabel(spacing+'Spacer horizontal transfer rate (1/(CFU/mL * gen.))',fontsize=15)
    plt.ylabel('Spacer direct acquisition rate (1/generation)'+spacing,fontsize=15)
    plt.savefig(file_name, bbox_inches='tight')

plotHeatmap(
    dat_frac_res,'Fraction of bacterial population with spacer',
    'crispr_heatmap_deterministic_HGT_frac.png','viridis_r'#, show_labels=True
    )

plotHeatmap(
    dat_frac_t10,'Time at which resistant bacteria with spacer reach 10% (generations)',
    'crispr_heatmap_deterministic_HGT_t10.png','magma', vmin=1e-1, vmax=1e2,#, show_labels=True
    )

# Last one spacer loss

samples = 200 # heatmap width in pixels

K_drops = np.power( 10, np.linspace( 6, 10, samples, endpoint=True ) )
    # fitness costs tested in Opqua stochastic model
print(K_drops)
print(np.power( 10, np.linspace( 6, 10, 5, endpoint=True ) ))
# contact rates tested in Opqua stochastic model
loss_rates = 1e-4 * np.power( 10, np.linspace( -2, 2, samples, endpoint=True ) )
# print( np.linspace( 3, 1, samples, endpoint=True ) )
# print(np.power( 5, -np.linspace( 3, 1, samples, endpoint=True ) ))
print(loss_rates)
print(1e-4 * np.power( 10, np.linspace( -2, 2, 5, endpoint=True ) ))

# stores parameters and values to be used in sweep
param_sweep_dic = { 'l':loss_rates,'K':K_drops }

# generate dataframe with all combinations
params_list = param_sweep_dic.keys()
value_lists = [ param_sweep_dic[param] for param in params_list ]
combinations = list( it.product( *value_lists ) )
param_df = pd.DataFrame(combinations)
param_df.columns = params_list

results = {} # store results

# # Parallelize running all pixels in heatmap
# n_cores = jl.cpu_count()
#
# res = jl.Parallel(n_jobs=n_cores, verbose=10) (
#     jl.delayed( run ) (param_values) for param_values in combinations
#      )
#
# # for param_values in combinations:
#     # run(param_values)
#
# dat = param_df
#
# dat['res_frac'] = np.array(res)[:,0]
# dat['t_10'] = np.array(res)[:,1]
# dat.to_csv('crispr_heatmaps_loss.csv')
# ...until here

dat = pd.read_csv('crispr_heatmaps_loss.csv')

# Reformat data for heatmaps
dat_frac_res = dat.pivot(index='K',columns='l',values='res_frac')
dat_frac_t10 = dat.pivot(index='K',columns='l',values='t_10')

# Plot heatmaps
plt.rcParams.update({'font.size': 12})

def plotHeatmap(
        dat,cmap_lab,file_name,cmap, vmin=None, vmax=None, show_labels=False):
    plt.figure(figsize=(8,8), dpi=200)
    ax = plt.subplot(1, 1, 1)
    if cmap == 'magma':
        ax = sns.heatmap(
            dat, linewidth = 0 , annot = False, cmap=cmap,
            cbar_kws={'label': cmap_lab}, vmin=vmin, vmax=vmax,
            xticklabels=show_labels, yticklabels=show_labels, norm=mlc.LogNorm()
            )
    else:
        ax = sns.heatmap(
            dat, linewidth = 0 , annot = False, cmap=cmap,
            cbar_kws={'label': cmap_lab}, vmin=vmin, vmax=vmax,
            xticklabels=show_labels, yticklabels=show_labels, #norm=mlc.LogNorm()
            )
    ax.figure.axes[-1].yaxis.label.set_size(15)
    ax.invert_yaxis()
    spacing = '\n\n\n'# if show_labels else ''
    plt.ylabel('Maximum bacterial density (CFU/mL)'+spacing,fontsize=15)
    plt.xlabel(spacing+'Spacer loss rate (1/generation)',fontsize=15)
    plt.savefig(file_name, bbox_inches='tight')

plotHeatmap(
    dat_frac_res,'Fraction of bacterial population with spacer',
    'crispr_heatmap_deterministic_loss_frac.png','viridis_r'#, show_labels=True
    )

plotHeatmap(
    dat_frac_t10,'Time at which resistant bacteria with spacer reach 10% (generations)',
    'crispr_heatmap_deterministic_loss_t10.png','magma'#, show_labels=True
    )
