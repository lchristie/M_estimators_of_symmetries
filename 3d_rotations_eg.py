
import pdb
import sys
import time
import pickle

import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt


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
        applied_points = np.matmul( tau, points.T ).T 
        new_points = np.concatenate( (new_points, applied_points), axis = 0 )
        new_Y = np.concatenate( (new_Y, data[1] ) )

    return( new_points, new_Y )


def find_small_R_inds( data, sym_group, bandwidth, axis, sample_size ):

    if sym_group == "SO3":
        R_x = np.sqrt( 2 * np.sum( data**2, axis = 1 ) )
        limit = 2 * bandwidth * ( - 2 * np.log( sample_size**( - 2 / (2 + 1) ) ) )**(1/2)
        return( R_x < limit )

    if sym_group == "S1":
        X_proj = data - np.outer( data @ axis, axis )
        R_x = np.sqrt( np.sum( X_proj**2, axis = 1 ) )
        limit = 2 * bandwidth * ( - 2 * np.log( sample_size**( - 2 / (2 + 2) ) ) )
        print( limit )
        return( R_x < limit )

    return( np.array([]) )


def symmetrisation( data_train, data_test, transformations, orbit_dim, model, func_name,  
                    sym_group = "I", bandwidth = 0, base_bandwidth = 0, axis = 0):
    # param transformations: list of n numpy arrays of shape (d, d)
    # param model: a function with inputs (data_train,  data_test, *args)

    n = data_train[0].shape[0]

    new_data_test = sym_dataset( (data_test[0], data_test[1]), transformations )

    MSPE, risk, y_pred = model( data_train, new_data_test, func_name, orbit_dim, sym_group, 5, bandwidth = bandwidth )

    non_zero_inds = np.nonzero(y_pred)
    pre_syms_preds = y_pred.reshape( (len(transformations) + 1, data_test[0].shape[0]) )

    sym_y_pred = np.mean( pre_syms_preds, axis = 0, where = (pre_syms_preds**2 > 0) )
    sym_MSPE = np.mean( (sym_y_pred - data_test[1] )**2 )
    sym_risk = np.mean( (sym_y_pred - reg_function( data_test[0], func_name ) )**2 ) 

    return( sym_MSPE, sym_risk, sym_y_pred )


## SO(3) Subgroup Symmetrisation

def rotation_matrix( angle, axis ):
    u = axis / np.sqrt( np.sum( axis**2 ) )
    phi = angle
    R = np.zeros( (3,3) )
    R[0,0] = np.cos(phi) + u[0]**2 * (1 - np.cos(phi))
    R[0,1] = u[0] * u[1] * (1 - np.cos(phi)) - u[2] * np.sin(phi)
    R[0,2] = u[0] * u[2] * (1 - np.cos(phi)) + u[1] * np.sin(phi)
    R[1,0] = u[0] * u[1] * (1 - np.cos(phi)) + u[2] * np.sin(phi)
    R[1,1] = np.cos(phi) + u[1]**2 * (1 - np.cos(phi))
    R[1,2] = u[1] * u[2] * (1 - np.cos(phi)) - u[0] * np.sin(phi)
    R[2,0] = u[2] * u[0] * (1 - np.cos(phi)) - u[1] * np.sin(phi)
    R[2,1] = u[1] * u[2] * (1 - np.cos(phi)) + u[0] * np.sin(phi)
    R[2,2] = np.cos(phi) + u[2]**2 * (1 - np.cos(phi))
    return( R )


def sample_SO3(n = 1, d = 3): 

    angles = rng.random( n )
    axes = rng.normal( 0, 1, (n, d) )
    axes = ( axes.T / np.sqrt( np.sum( axes**2, axis = 1 ) ) ).T
    output = [ np.array( rotation_matrix( angles[0], axes[0] ) ) ]

    for i in range(1,n):
        output.append( rotation_matrix( angles[i], axes[i] ) )

    return( output )


def sample_S1(axis, n = 1):

    angles = rng.random( n ) * 2 * np.pi
    output = [ np.array( rotation_matrix( angles[0], axis ) ) ]

    for i in range(1,n):
        output.append( rotation_matrix( angles[i], axis ) )

    return( output )


def S2_axis_grid( delta ): 
    ## This generates a set of axis u_i \in S^2 such that 
    ##      arccos( \langle u, u_i \rangle ) < \delta 
    ## Returns a numpy array of shape m \times 3 for some m

    equator_angle_grid = 2 * np.pi * np.arange( 0, 1, delta/(4 * np.pi) )

    merid_angle_grid = np.pi * np.arange( 0, 1, delta/(4 * np.pi) )
    
    m_axes = np.array([[0,0,1], [0,0,-1]])            ## Polar axes

    for phi in equator_angle_grid:
        for theta in merid_angle_grid[1:]:
            x = np.sin( theta ) * np.cos( phi )
            y = np.sin( theta ) * np.sin( phi )
            z = np.cos( theta )
            m_axis = np.atleast_2d( np.array( [x,y,z] ) )
            m_axes = np.concatenate( (m_axes, m_axis), axis = 0 )

    return(m_axes)


def angle_dists( arr_1, arr_2 ):

    output = np.zeros( (arr_1.shape[0], arr_2.shape[1]) )

    for i in range(arr_1.shape[0]):
        for j in range(arr_2.shape[0]):
            output[i,j] = np.arccos( np.dot( arr_1[i,], arr_2[j,] ) )

    return( output )


## Data Generation

def generate_data( sample_size, dimension, sigma_eps, func_name ):

    X = rng.normal( 0, 2, ( sample_size, dimension ) )
    X_dirs = ( X.T / np.sqrt( np.sum( X**2, axis = 1 ) ) )
    X = np.transpose( X_dirs * ( rng.random( sample_size ) ** (1/3) ) ) ## Generates points uniforms on B_{R^3}(0,1)

    eps = rng.normal( 0, sigma_eps , sample_size )

    Y = reg_function( X, func_name ) + eps

    return( X, Y )


def reg_function( X, func_name ):

    ## SO(3) Invariant functions
    if func_name == "f1" or func_name == "f1_2":
        return( np.cos( 2 * np.pi * np.sum( X**2, axis = 1 ) ) )

    ## S^1_x Invariant functions
    if func_name == "f2" or func_name == "f2_2": 
        return( np.cos( 2 * np.pi * np.sqrt(  X[:,1] ** 2 + X[:,2] ** 2 ) ) )

    ## I invariant functions
    if func_name == "f3" or func_name == "f3_2":
        return(  X[:,0] ** 2 + X[:,1] - 0.6 * X[:,2] )



## Local Constant Modelling

def fit_LCE(data_train, data_test, bandwidth, func_name, sym_group = "I", kernel = "rect" ):
    X_train = data_train[0]
    Y_train = data_train[1]

    X_test = data_test[0]
    Y_test = data_test[1]

    dists = sp.distance_matrix( X_train, X_test )
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

    bandwidth_grid = np.exp( np.linspace(-3.0, 0.5, bandwidth_numbers) )
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

            if sym_group == "I":
                this_error[i] = fit_LCE( data_train, data_test, h, func_name , sym_group )[0]
            if sym_group == "SO3":
                this_error[i] = symmetrisation( data_train, 
                                                data_test, 
                                                sample_SO3( int(n) ), 
                                                orbit_dim = 2, 
                                                model = fit_LCE_CV, 
                                                func_name = func_name, 
                                                sym_group = "SO3", 
                                                bandwidth = h,
                                                base_bandwidth = h )[0]


        errors[j] = this_error.mean()

    plt.plot( bandwidth_grid, errors )
    plt.show()

    return( bandwidth_grid[ np.argmin( errors ) ] )


def fit_bandwidth_holder( n, beta, dim, scale = 1 ):
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
    dimension = 3

    models = []

    #### Model fitting
    ## K_0
    bandwidth_I = fit_bandwidth_holder( n, beta, dimension )
    models.append( fit_LCE_CV( data_train, data_test, func_name, bandwidth = bandwidth_I ) )

    ## K_1
    axial_grid = S2_axis_grid( delta )
    bandwidth_S1 = fit_bandwidth_holder( n, beta, dimension - 1 )

    for axis in axial_grid:
        models.append( symmetrisation( data_train, 
                                       data_test, 
                                       sample_S1( axis, int(n * sym_scale_cst) ), 
                                       orbit_dim = 1, 
                                       model = fit_LCE_CV, 
                                       func_name = func_name, 
                                       sym_group = "S1", 
                                       bandwidth = bandwidth_S1,
                                       base_bandwidth = bandwidth_I, 
                                       axis = axis ) )


    ## K_2
    bandwidth_SO_3 = fit_bandwidth_holder( n, beta, dimension - 2 )
    models.append( symmetrisation( data_train, 
                                   data_test, 
                                   sample_SO3( int(n * sym_scale_cst) ), 
                                   orbit_dim = 2, 
                                   model = fit_LCE_CV, 
                                   func_name = func_name, 
                                   sym_group = "SO3", 
                                   bandwidth = bandwidth_SO_3,
                                   base_bandwidth = bandwidth_I ) )

    #### Model Selection
    best_index = 0
    for i, m in enumerate( models ):
        if m[0] < models[best_index][0]:
            best_index = i

    if best_index == 0:
        if verbose: print( f"Best Symmetric Group: I\n")
        output = list( fit_LCE_CV( data_train, 
                                   data_validation, 
                                   func_name, 
                                   bandwidth = bandwidth_I) ) + ["I"]
    elif best_index == len( models ) - 1:
        if verbose: print( f"Best Symmetric Group: SO(3)\n")
        output = list( symmetrisation( data_train, 
                                       data_validation, 
                                       sample_SO3( int(n * sym_scale_cst) ), 
                                       orbit_dim = 2, 
                                       model = fit_LCE_CV, 
                                       func_name = func_name, 
                                       sym_group = "SO3", 
                                       bandwidth = bandwidth_SO_3,
                                       base_bandwidth = bandwidth_I ) ) + ["SO3"]
    else: 
        axis = axial_grid[best_index - 1]
        if verbose: 
            print( f"Best Symmetric Group: S^1_({axis[0]:.2f},{axis[1]:.2f},{axis[2]:.2f})")
            print( f"SO3 MSPE {models[-1][0]:.2f} vs S1 MSPE {models[best_index][0]:.2f}\n"  )
        output = list( symmetrisation( data_train, 
                                       data_validation, 
                                       sample_S1( axis, int(n * sym_scale_cst) ), 
                                       orbit_dim = 1, 
                                       model = fit_LCE_CV, 
                                       func_name = func_name, 
                                       sym_group = "S1", 
                                       bandwidth = bandwidth_S1,
                                       base_bandwidth = bandwidth_I ) ) + [f"S^1_({axis[0]:.2f},{axis[1]:.2f},{axis[2]:.2f})"]

    if verbose: print(f"Number of fitted models:\t{len(models)}")

    return( tuple( output ) )



## Main Code

def run_sims(num_sims, sample_sizes, func_name, verbose = False):

    print( f"Numder of Simulations: {num_sims}" )
    print( f"Sample Sizes: {sample_sizes}" )
    print( f"Regression Function: {func_name}")

    val_sample_size = 200
    dimension = 3
    max_orbit_dim = 2
    sigma_eps = 0.1
    beta = 1
    sym_scale_cst = 1

    test_errors_LCE = np.zeros( (num_sims, len(sample_sizes)) )
    test_errors_S_G_HAT_LCE = np.zeros( (num_sims, len(sample_sizes)) )

    best_groups = []

    base_bandwidths = np.zeros( len(sample_sizes) )

    for j in range( len(sample_sizes) ):
        sample_size = sample_sizes[j]
        print( f"\nSample Size:     {sample_size}" )

        base_bandwidths[j] = fit_bandwidth_holder( sample_size, beta, 3, scale = 1 )
        this_best_groups = []

        for i in progressbar( range(num_sims) ):

            data_train = generate_data( sample_size, dimension, sigma_eps, func_name )
            data_test = generate_data( sample_size, dimension, sigma_eps, func_name )
            data_validation = generate_data( val_sample_size, dimension, sigma_eps, func_name )
            
            data_train_full = ( np.concatenate( (data_train[0], data_test[0]), axis = 0),
                                np.concatenate( (data_train[1], data_test[1]), axis = 0) )


            base_start = time.time() 

            lce_error = fit_LCE_CV( data_train_full, 
                                    data_validation, 
                                    func_name, 
                                    folds = 5, 
                                    bandwidth = fit_bandwidth_holder( 2 * sample_size, beta, 3, scale = 1 ) ) 
            test_errors_LCE[i,j] = lce_error[1]

            base_end = time.time()

            s_g_hat_lce_error = fit_S_G_f_n( data_train, 
                                             data_test, 
                                             data_validation,
                                             delta = find_delta( sample_size, beta, dimension, max_orbit_dim, 2*np.pi ), 
                                             beta = beta, 
                                             func_name = func_name,
                                             sym_scale_cst = sym_scale_cst,
                                             verbose = verbose )
            test_errors_S_G_HAT_LCE[i,j] = s_g_hat_lce_error[1]

            sym_end = time.time()

            if verbose: 
                print( f"Base time:\t{base_end - base_start:.5f}s")
                print( f"Sym time:\t{sym_end - base_end:.5f}s" )

            this_best_groups.append( s_g_hat_lce_error[3] )



        print( f"LCE Risk: \t {np.mean( test_errors_LCE[:,j] ):e} +/- {np.std( test_errors_LCE[:,j] ):e}" )
        ratio = np.mean( test_errors_S_G_HAT_LCE[:,j] ) / np.mean( test_errors_LCE[:,j] )
        print( f"Sym_LCE Risk: \t {np.mean( test_errors_S_G_HAT_LCE[:,j] ):e} +/- {np.std( test_errors_S_G_HAT_LCE[:,j] ):e}, \t {ratio:%}" )
        print( "Estimated groups: ")
        for x in this_best_groups:
            print( "\t", x )
        best_groups.append(this_best_groups)

    X = np.vstack([ np.log( sample_sizes ), np.ones(len(sample_sizes)) ]).T
    base_rate, base_int = np.linalg.lstsq( X, np.log( test_errors_LCE.mean(axis = 0) ), rcond = None )[0]
    sym_rate, sym_int = np.linalg.lstsq( X, np.log( test_errors_S_G_HAT_LCE.mean(axis = 0) ), rcond = None )[0]

    print(f"Base log/log slope:  {base_rate}, expected {- 2*beta / (2*beta + 3)}")
    print(f"Sym log/log slope:  {sym_rate}, expected {- 2*beta / (2*beta + 2)}")

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






























