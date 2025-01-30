
import pdb
import sys
import time
import pickle

import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

def progressbar(it, prefix="", size=30, out=sys.stdout):

    ## accessed from https://stackoverflow.com/questions/3160699/python-progress-bar on Feb 4 2024
    ## imbr's answer

    count = len(it)
    start = time.time()
    def show(j):
        x = int(size*j/count)
        remaining = ((time.time() - start) / j) * (count - j)
        
        mins, sec = divmod(remaining, 60)
        time_str = f"{int(mins):02}:{sec:05.2f}"
        
        print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count} Est wait {time_str}", end='\r', file=out, flush=True)
        
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)


rng = np.random.default_rng(2011)

## Symmetriations

def sym_dataset( data, transformations ):

    points = data[0]
    new_points = points
    new_Y = data[1]

    for tau in transformations:
        applied_points = np.mod( points + tau, 1 )
        new_points = np.concatenate( (new_points, applied_points), axis = 0 )
        new_Y = np.concatenate( (new_Y, data[1] ) )

    return( new_points, new_Y )


def symmetrisation( data_train, data_test, transformations, orbit_dim, model, func_name,  sym_group = "I", bandwidth = 0):
    # param transformation: list of n numpy arrays of shape (d, d)
    # param model: a function with inputs (data_train,  data_test, *args)

    n = data_train[0].shape[0]

    new_data_test = sym_dataset( data_test, transformations )
    MSPE, risk, y_pred = model( data_train, new_data_test, func_name, orbit_dim, sym_group, 5, bandwidth = bandwidth )
    y_pred = y_pred.reshape( (len(transformations) + 1, data_test[0].shape[0]) )

    sym_y_pred = np.mean( y_pred, axis = 0,  where = (y_pred**2 > 0) )
    sym_MSPE = np.mean( (sym_y_pred - data_test[1] )**2 )
    sym_risk = np.mean( (sym_y_pred - reg_function( data_test[0], func_name ) )**2 ) 

    return( sym_MSPE, sym_risk, sym_y_pred )


## T2 Subgroup Symmetrisation

def sample_T2( n = 1, d = 2 ):

    return( rng.uniform( 0, 1, (n,d) ) )


def sample_T1( angle, n = 1, d = 2 ):

    tau_1 = rng.uniform( 0, 1, (n,1) )
    tau_2 = np.mod( tau_1 * np.tan(angle), 1 )

    return( np.concatenate( (tau_1, tau_2), axis = 1 ) )


def T2_angle_grid( delta ):
    ## This generates a set of axis u_i \in S^1 in the first quadrant such that 
    ##      arccos( \langle u, u_i \rangle ) < \delta / sqrt(2)
    ## Returns a numpy array of shape m \times 3 for some m
    max_denom = int( np.ceil( 1 / delta ) )

    theta_grid = np.array([])

    for steps in range(1, max_denom + 1):
        theta_grid = np.append( theta_grid, 2 * np.pi * np.append( np.arange( 0, 1, 1 / steps ), 1 )  )

    return( np.unique( theta_grid ) )


def torus_dists( X, Y ):
    ## X and Y are np arrays of shape (n_X, 2) and (n_Y, 2) respectively

    output = np.zeros( (X.shape[0], Y.shape[0]) )

    for i in range( Y.shape[0] ):

        d1_dists = np.min( np.abs( np.array( [(X[:,0] - Y[i,0]), (X[:,0] - Y[i,0] - 1), (X[:,0] - Y[i,0] + 1)] ) ), axis = 0 )
        d2_dists = np.min( np.abs( np.array( [(X[:,1] - Y[i,1]), (X[:,1] - Y[i,1] - 1), (X[:,1] - Y[i,1] + 1)] ) ), axis = 0 )

        dists_squared_to_Y = d1_dists ** 2 + d2_dists ** 2

        output[:,i] = np.sqrt( dists_squared_to_Y )

    return( output )


## Data Generation

def generate_data( sample_size, dimension, sigma_eps, func_name ):

    X = rng.uniform( 0, 1, ( sample_size, dimension ) )

    eps = rng.normal( 0, sigma_eps , sample_size )

    Y = reg_function( X, func_name ) + eps

    return( X, Y )


def reg_function( X, func_name ):

    ## T2 Invariant function
    if func_name == "f1" or func_name == "f1_2":
        return( np.ones( X.shape[0] ) )

    ## T^1_0 Invariant function
    if func_name == "f2" or func_name == "f2_2": 
        return( np.sin( 2 * np.pi * X[:,1] ) )

    ## T^1_pi/4 invariant function
    if func_name == "f3" or func_name == "f3_2":
        return( np.cos(  2 * np.pi * ( X[:,0] - X[:,1] ) ) )

    ## T^1_pi/2 Invariant function -- unused in paper
    if func_name == "f4" or func_name == "f4_2": 
        return( np.sin( 2 * np.pi * X[:,0] ) )




## Local Constant Modelling

def fit_LCE(data_train, data_test, bandwidth, func_name, sym_group = "I", kernel = "rect" ):
    X_train = data_train[0]
    Y_train = data_train[1]

    X_test = data_test[0]
    Y_test = data_test[1]

    dists = torus_dists( X_train, X_test )

    if kernel == "rect":
        flags = np.array( (dists < bandwidth) * 1 )
        vals = (flags.T * Y_train).T

    if kernel == "triangle":
        flags = np.array( (dists < bandwidth) * 1 ) * dists
        vals = (flags.T * Y_train).T

    y_pred = np.zeros( vals.shape[1] )
    mean_val = Y_train.mean()

    for i in range( vals.shape[1] ):
        if np.count_nonzero( vals[:,i] ):
            y_pred[i] = np.sum( vals[:,i] ) / np.count_nonzero( vals[:,i] )
        else: 
            y_pred[i] = mean_val

    MSPE = ((y_pred - Y_test)**2 ).mean()
    risk = np.mean( (y_pred - reg_function(X_test, func_name ) )**2 )

    return( MSPE, risk, y_pred )


def fit_bandwidth_CV(data, folds, func_name, sym_group = "I", bandwidth_numbers = 600 ):

    bandwidth_grid = np.linspace(0.03, 0.3, bandwidth_numbers)
    errors = np.zeros( bandwidth_grid.shape )

    n = data[0].shape[0]
    per_fold = int( n / folds )

    for j in progressbar( range( bandwidth_numbers ) ):

        h = bandwidth_grid[j]
        this_error = np.zeros( folds )

        for i in range(folds):
            test_inds = list( range( i * per_fold,  (i + 1) * per_fold, 1 ) )
            train_inds = list( range(n) )
            for elem in test_inds: train_inds.remove(elem) 
            data_test = ( data[0][test_inds, ], data[1][test_inds, ] )
            data_train = ( data[0][train_inds, ], data[1][train_inds, ] )
            this_error[i] = fit_LCE( data_train, data_test, func_name, h, sym_group )[0]

        errors[j] = this_error.mean()

    plt.plot( bandwidth_grid, errors )
    plt.show()

    return( bandwidth_grid[ np.argmin( errors ) ] )


def fit_bandwidth_holder( n, beta, dim, scale = 1/np.sqrt(2) ):
    return( scale * ( n**(-1 / (2 * beta + dim )) ) )


def fit_LCE_CV(data_train, data_test, func_name, orbit_dim = 0, sym_group = "I",  folds = 5, bandwidth = 0, kernel = "rect" ):

    n = data_test[0].shape[0]
    d = data_test[0].shape[1]

    if bandwidth == 0:
        bandwidth = fit_bandwidth_CV( data_train, folds, func_name, sym_group )

    MSPE, risk, preds = fit_LCE(data_train, data_test, bandwidth, func_name = func_name, sym_group = sym_group, kernel = kernel)

    return(  MSPE, risk, preds )



## M Estimation of Symmetries

def find_delta( n, beta, dimension, max_orbit_dim, scale_cst = 1 ):
    return( scale_cst * n ** ( - beta / ( 2 * beta + dimension - max_orbit_dim ) ) )


def fit_S_G_f_n( data_train, data_test, data_validation, delta, beta, func_name, sym_scale_cst = 1, verbose = False):

    n = data_train[0].shape[0]
    dimension = 2

    models = []

    #### Model fitting
    ## K_0
    bandwidth_I = fit_bandwidth_holder( n, beta, dimension )
    models.append( fit_LCE_CV( data_train, data_test, func_name, bandwidth = bandwidth_I ) )

    ## K_1
    angle_grid = T2_angle_grid( delta )
    bandwidth_T1 = fit_bandwidth_holder( n, beta, dimension - 1 )

    for angle in angle_grid:
        models.append( symmetrisation( data_train, 
                                       data_test, 
                                       sample_T1( angle, int(n * sym_scale_cst) ), 
                                       orbit_dim = 1, 
                                       model = fit_LCE_CV, 
                                       func_name = func_name, 
                                       sym_group = "T1", 
                                       bandwidth = bandwidth_T1 ) )


    ## K_2
    bandwidth_T2 = fit_bandwidth_holder( n, beta, dimension - 2 )
    models.append( symmetrisation( data_train, 
                                   data_test, 
                                   sample_T2( int(n * sym_scale_cst) ), 
                                   orbit_dim = 2, 
                                   model = fit_LCE_CV, 
                                   func_name = func_name, 
                                   sym_group = "T2", 
                                   bandwidth = bandwidth_T2 ) )

    #### Model Selection
    best_index = 0
    for i, m in enumerate( models ):
        if m[0] < models[best_index][0]:
            best_index = i

    if best_index == 0:
        if verbose: print( f"Best Symmetric Group: I\n")
        output = list( fit_LCE_CV( data_train, data_validation, func_name, bandwidth = bandwidth_I ) ) + ["I"]
    elif best_index == len( models ) - 1:
        if verbose: print( f"Best Symmetric Group: T^2\n")
        output = list( symmetrisation( data_train, 
                                         data_validation, 
                                         sample_T2( int(n * sym_scale_cst) ), 
                                         orbit_dim = 2, 
                                         model = fit_LCE_CV, 
                                         func_name = func_name, 
                                         sym_group = "T2", 
                                         bandwidth = bandwidth_T2 ) ) + ["T2"]
    else:
        angle = angle_grid[best_index - 1]
        if verbose: print( f"Best Symmetric Group: T^1_{angle})\n")
        output = list( symmetrisation( data_train, 
                                         data_validation, 
                                         sample_T1( angle, int(n * sym_scale_cst) ), 
                                         orbit_dim = 1, 
                                         model = fit_LCE_CV, 
                                         func_name = func_name, 
                                         sym_group = "T1", 
                                         bandwidth = bandwidth_T1 ) ) + [f"T^1_{angle}"]


    return( tuple( output ) )



## Main Code

def run_sims(num_sims, sample_sizes, func_name, verbose = False):

    print( f"Numder of Simulations: {num_sims}" )
    print( f"Sample Sizes: {sample_sizes}" )
    print( f"Regression Function: {func_name}" )

    val_sample_size = 200
    dimension = 2
    max_orbit_dim = 2
    sigma_eps = 0.1
    beta = 1
    sym_scale_cst = 0.2

    test_errors_LCE = np.zeros( (num_sims, len(sample_sizes)) )
    test_errors_S_G_HAT_LCE = np.zeros( (num_sims, len(sample_sizes)) )

    best_groups = []

    base_bandwidths = np.zeros( len(sample_sizes) )

    for j in range( len(sample_sizes) ):
        sample_size = sample_sizes[j]
        print( f"\nSample Size:     {sample_size}" )
        if verbose:
            print( f"Delta:     {find_delta( sample_size, beta, dimension, max_orbit_dim, 1 )}")
            print( f"Angle Grid:    {T2_angle_grid(find_delta( sample_size, beta, dimension, max_orbit_dim, 1 ))}")

        base_bandwidths[j] = fit_bandwidth_holder( sample_size, beta, dimension )
        this_best_groups =[]

        for i in progressbar( range(num_sims) ):
            data_train = generate_data( sample_size, dimension, sigma_eps, func_name )
            data_test = generate_data( sample_size, dimension, sigma_eps, func_name )
            data_validation = generate_data( val_sample_size, dimension, sigma_eps, func_name )
            
            data_all = ( np.concatenate( (data_train[0], data_test[0]), axis = 0 ), 
                         np.concatenate( (data_train[1], data_test[1]), axis = 0 ) )

            base_start = time.time()

            lce_error = fit_LCE_CV( data_all, 
                                    data_validation, 
                                    func_name, 
                                    folds = 5, 
                                    bandwidth = base_bandwidths[j] ) 
            test_errors_LCE[i,j] = lce_error[1]

            base_end = time.time()

            s_g_hat_lce_error = fit_S_G_f_n( data_train, 
                                             data_test, 
                                             data_validation,
                                             delta = find_delta( sample_size, beta, dimension, max_orbit_dim, 1 ), 
                                             beta = beta, 
                                             func_name = func_name,
                                             sym_scale_cst = sym_scale_cst,
                                             verbose = False )
            test_errors_S_G_HAT_LCE[i,j] = s_g_hat_lce_error[1]

            sym_end = time.time()

            if verbose:
                print( f"Base Estimator time:\t{base_end - base_start:0.5f}" )
                print( f"Sym Estimator time:\t{sym_end - base_end:0.5f}" )

            this_best_groups.append( s_g_hat_lce_error[3] )


        print( f"LCE Risk: \t {np.mean( test_errors_LCE[:,j] ):e} +/- {np.std( test_errors_LCE[:,j] ):e}" )
        ratio = np.mean( test_errors_S_G_HAT_LCE[:,j] ) / np.mean( test_errors_LCE[:,j] )
        print( f"Sym_LCE Risk: \t {np.mean( test_errors_S_G_HAT_LCE[:,j] ):e} +/- {np.std( test_errors_S_G_HAT_LCE[:,j] ):e}, \t {ratio:%}" )
        print( "A Sample of 20 Estimated groups: ")
        for x in rng.choice( this_best_groups, 20, replace = False ):
            print( "\t", x )
        best_groups.append(this_best_groups)

    X = np.vstack([ np.log( sample_sizes ), np.ones(len(sample_sizes)) ]).T
    base_rate, base_int = np.linalg.lstsq( X, np.log( test_errors_LCE.mean(axis = 0) ), rcond = None )[0]
    sym_rate, sym_int = np.linalg.lstsq( X, np.log( test_errors_S_G_HAT_LCE.mean(axis = 0) ), rcond = None )[0]

    print(f"Base log/log slope:  {base_rate}, expected {- 2*beta / (2*beta + 2)}")
    print(f"Base log/log slope:  {sym_rate}, expected {- 2*beta / (2*beta + 0)}")

    with open(func_name + "_lce_results.pkl", "wb") as f:
        pickle.dump(test_errors_LCE, f)
    with open(func_name + "_sym_lce_results.pkl", "wb") as f:
        pickle.dump(test_errors_S_G_HAT_LCE, f)
    with open(func_name + "_est_groups.pkl", "wb") as f:
        pickle.dump(best_groups, f)

    return(0)


def plot_results( num_sims, sample_sizes, func_name ):


    with open(func_name + "_lce_results.pkl", "rb") as f:
        test_errors_LCE = pickle.load(f)
    with open(func_name + "_sym_lce_results.pkl", "rb") as f:
        test_errors_S_G_HAT_LCE = pickle.load(f)

    plt.plot( np.log( sample_sizes ),
              np.log( test_errors_LCE.mean(axis = 0) ), 
              label = "Baseline" )
    plt.plot( np.log( sample_sizes ),
              np.log( test_errors_S_G_HAT_LCE.mean(axis = 0) ),
              label = "Symmetrised" )

    plt.xlabel("Log Sample Size (log n)")
    plt.ylabel("Log Risk")
    plt.title("Generalisation Risk: " + func_name )
    plt.legend(loc="upper right")


    plt.fill_between(np.log( sample_sizes ), 
                     np.log( test_errors_LCE.mean(axis = 0) - 1.96 * test_errors_LCE.std(axis = 0)/ np.sqrt( num_sims ) ), 
                     np.log( test_errors_LCE.mean(axis = 0) + 1.96 * test_errors_LCE.std(axis = 0) / np.sqrt( num_sims ) ),
                     color='blue', alpha=0.2)
    plt.fill_between(np.log( sample_sizes ), 
                     np.log( test_errors_S_G_HAT_LCE.mean(axis = 0) - 1.96 * test_errors_S_G_HAT_LCE.std(axis = 0) / np.sqrt( num_sims ) ), 
                     np.log( test_errors_S_G_HAT_LCE.mean(axis = 0) + 1.96 * test_errors_S_G_HAT_LCE.std(axis = 0) / np.sqrt( num_sims ) ),
                     color='orange', alpha=0.2)

    plt.show()

    return(0)


def main():
    
    num_sims = 30
    sample_sizes = [30, 50, 75, 100, 150, 200, 300] 
    func_name = "f3"

    run_sims( num_sims, sample_sizes, func_name, verbose = False )
    plot_results( num_sims, sample_sizes, func_name )

    return(0)


if __name__ == "__main__":
    main()




