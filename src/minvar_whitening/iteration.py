import minvar_whitening.poly as mvp
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from scipy.linalg import qr
import seaborn as sns
import time 

def moment(sigma, i):
    """
    Find the i-th moment of a matrix.

    Input: 
     - sigma: the matrix
     - i: the moment to be found, int

    Output:
     - the i-th moment of the matrix sigma.
    """
    return np.trace(np.linalg.matrix_power(sigma, i))

def moment_condition(sigma):
    """
    Moment condition to detect if the eigenvalues of a matrix are distributed at only two points.

    Input:
     - sigma: the matrix

    Output:
      - the moment condition will be close to zero if the eigenvalues are distributed at only two points. 
    """

    m1, m2  = (moment(sigma, i) for i in range(1, 3))
    d = sigma.shape[0]

    return (m2/(m1**2)) - (1/d)

def heatmap_plots(sigma_, k=0, iteration=0, save_heatmap_filepath=None, heatmap_title=True):
    """
    Plot the heatmap of the covariance matrix sigma_.

    Input: 
     - sigma_: the covariance matrix to be plotted
     - k: give a value of k to include the information in the title
     - iteration: give the iteration count to include the information in the title
     - save_heatmap_filepath: the filepath for the heatmap to be saved to 

    Output: 
     - plot a heatmap
    """
    fig, ax = plt.subplots()
    ax = sns.heatmap(sigma_, cmap='twilight_shifted', center=0)
    # if heatmap_title == True:
    #     ax.set_title('k = %d, iter = %d' % (k, iteration+1))
    if save_heatmap_filepath != None:
        fig.savefig('%s/k%d_iter%d.png' % (save_heatmap_filepath, k, iteration+1), bbox_inches='tight')
        
    return None

def polynomial_plots(sigma, k, iteration, save_poly_filepath=None, x_lim=None, y_lim=None):
    """
    Plot the polynomial of the whitening matrix A.

    Input: 
     - sigma: the covariance matrix
     - k: k-1 is the degree of the polynomial
     - iteration: give the iteration count to include the information in the title
     - save_poly_filepath: the filepath for the plot to be saved to 
     - x_lim: a tuple specifying the limits for the x-axis
     - y_lim: a tuple specifying the limits for the y-axis

    Output: 
     - plot the eigenvalues of sigma against the inverse sqrt eigenvalues, and the whitening polynomial
    """
    t = sym.Symbol('t')
    Apol = mvp.A_matrix_poly(sigma, k, t)
    c = mvp.analytical_c(sigma, Apol)
    Apol = sym.lambdify(t, c*Apol)
    
    fig, ax = plt.subplots()
    eigs = np.linalg.eigvals(sigma)
    ax.scatter(eigs, 1/np.sqrt(eigs))
    ts = np.linspace(0, max(eigs) + 1)
    ax.plot(ts, Apol(ts))
    if x_lim != None:
        ax.set_xlim(x_lim)
    if y_lim != None:
        ax.set_ylim(y_lim)

    # ax.set_title('k = %d, iter = %d' % (k, iteration))
    
    if save_poly_filepath != None:
        fig.savefig('%s/k%d_iter%d.png' % (save_poly_filepath, k, iteration+1))
    
    return None

def repeated_k_whitening(sigma, 
                         X, 
                         k, 
                         adjust_at_end=True,
                         weight_power=1,
                         max_iter=100,
                         stopping_method='zero',
                         prev_moment=None,
                         prev_A=None,
                         plot_heatmap=False,
                         save_heatmap_filepath=None,
                         heatmap_title=True,
                         plot_polynomial=False,
                         save_poly_filepath=None,
                         poly_lims=None,
                         timing_overall=False,
                         timing_each_loop=False,
                         dtype=np.float64,
                         store_X=True):
    """
    Perform iterative whitening for a given value of k. 

    Input:
     - sigma: covariance matrix of data
     - X: the data to be whitened
     - k: k-1 is the degree of the polynomial to be found
     - adjust_at_end: use analytical_c to adjust the polynomial at the end to find the best fit
     - weight_power: the power of the eigenvalues used as weights 
     - max_iter: the maximum number of whitening iterations to be performed
     - stopping_method: how to stop. If 'zero', stop when moment condition is zero. If 'increase', stop when moment condition increases
     - prev_moment: if this is not the starting point and some moments have already been calculated, feed the previous moment in
     - prev_A: feed the previous value of A in 
     - plot_heatmap: plot the heatmaps of the covariance matrix for each iteration
     - save_heatmap_filepath: the filepath to save the heatmaps to
     - plot_polynomial: plot the eigenvalues, inv sqrt eigenvalues and the minimal variance polynomial for each iteration
     - save_poly_filepath: the filepath to save the polynomial plots to
     - poly_lims: feed a tuple of tuples giving the x and y axis limits

    Outputs:
     - Xs: the dataset at each iteration
     - As: the whitening matrix A at each iteration 
     - sigmas: the covariance matrix of the dataset at each iteration
     - moments_list: the list of the moments for each iteration

    """
    init_sigma, init_X = sigma.copy(), X.copy()
        
    if prev_moment == None:
        moments_list = []
        if store_X == True:
            Xs, As, sigmas = [], [], []
        else:
            As, sigmas = [], []
    else:
        moments_list = [prev_moment]
        if store_X == True:
            Xs = [X]
        assert (prev_A).all() != None, "If giving prev_moment we must also know prev_A"
        As = [prev_A]
        sigmas = [sigma]
        
    if timing_each_loop == True:
        loop_time_list = []

    if timing_overall == True:
        start_overall = time.time()
    for iteration in range(max_iter):
        print('Iteration ', (iteration + 1))

        if timing_each_loop == True:
            start = time.time()

        X = np.array(X, dtype=dtype)
        sigma = np.array(sigma, dtype=dtype)
        if iteration != 0:
            A = np.array(A, dtype=dtype)

        t = sym.Symbol('t')
        print(mvp.A_matrix_poly(sigma, k, t))
        X, A, sigma = mvp.apply_whitening(sigma, X, k, dtype=dtype)
        
        if timing_each_loop == True:
            loop_time_list.append(time.time() - start)

        moment_value = moment_condition(sigma)
        
        if plot_heatmap == True:
            heatmap_plots(sigma, k, iteration, save_heatmap_filepath=save_heatmap_filepath, heatmap_title=heatmap_title)

        if stopping_method == 'zero':
            stop_condition = moment_value < 1e-12
        elif stopping_method == 'increase':
            stop_condition = moment_value > moments_list[-1]

        if len(moments_list) >= 1 and stop_condition:
            break
        else:
            
            if plot_polynomial == True:
                    
                if len(sigmas) == 0: 
                    sig = sigma
                else:
                    sig = sigmas[-1]
                    
                if type(poly_lims) == tuple:
                    x_lim, y_lim = poly_lims
                else:
                    x_lim, y_lim = None, None
                    
                polynomial_plots(sig, 
                                 k, 
                                 iteration, 
                                 save_poly_filepath=save_poly_filepath,
                                 x_lim=x_lim,
                                 y_lim=y_lim)
            
            moments_list.append(moment_value)
            if store_X == True:
                Xs.append(X)
            else: 
                old_X = X
            As.append(A)
            sigmas.append(sigma)
            
    if adjust_at_end == True and len(sigmas) > 1: #only adjust if we've actually made an improvement
        prev_sigma = sigmas[-2]
        if store_X == True:
            prev_X = Xs[-2]
        else:
            prev_X = old_X
        c = mvp.adjustment_value(prev_sigma, k)
        As[-1] = c*As[-1]
        X = As[-1] @ prev_X
        sigmas[-1] = np.cov(X)
        if store_X == True:
            Xs[-1] = X 

    if store_X == True:
        if timing_overall == True and timing_each_loop == False:
            overall_time = time.time() - start_overall
            return Xs, As, sigmas, moments_list, overall_time
        elif timing_overall == False and timing_each_loop == True:
            return Xs, As, sigmas, moments_list, loop_time_list
        elif timing_overall == True and timing_each_loop == True:
            overall_time = time.time() - start_overall
            return Xs, As, sigmas, moments_list, overall_time, loop_time_list
        else:
            return Xs, As, sigmas, moments_list
    else:
        if timing_overall == True and timing_each_loop == False:
            overall_time = time.time() - start_overall
            return X, As, sigmas, moments_list, overall_time
        elif timing_overall == False and timing_each_loop == True:
            return X, As, sigmas, moments_list, loop_time_list
        elif timing_overall == True and timing_each_loop == True:
            overall_time = time.time() - start_overall
            return X, As, sigmas, moments_list, overall_time, loop_time_list
        else:
            return X, As, sigmas, moments_list


def iterative_whitening(sigma, 
                         X, 
                         start_k,
                         end_k,
                         adjust_at_end=True,
                         weight_power=1,
                         max_iter=100,
                         prev_moment=None,
                         prev_A=None, 
                         plot_heatmap=False,
                         save_heatmap_filepath=None,
                         heatmap_title=True,
                         plot_polynomial=False,
                         save_poly_filepath=None,
                         poly_lims=None,
                         dtype=np.float64, 
                         store_X=True):
    
    """
    Perform iterative whitening for multiple values of k. Reach the best whitening possible for start_k, then move on to start_k plus/minus 1, continue until end_k. 

    Input:
     - sigma: covariance matrix of data
     - X: the data to be whitened
     - start_k: start with this value of k
     - end_k: end with this value of k
     - adjust_at_end: use analytical_c to adjust the polynomial at the end to find the best fit
     - weight_power: the power of the eigenvalues used as weights 
     - max_iter: the maximum number of whitening iterations to be performed for each value of k
     - prev_moment: if this is not the starting point and some moments have already been calculated, feed the previous moment in
     - prev_A: feed the previous value of A in 
     - plot_heatmap: plot the heatmaps of the covariance matrix for each iteration
     - save_heatmap_filepath: the filepath to save the heatmaps to
     - plot_polynomial: plot the eigenvalues, inv sqrt eigenvalues and the minimal variance polynomial for each iteration
     - save_poly_filepath: the filepath to save the polynomial plots to
     - poly_lims: feed a tuple of tuples giving the x and y axis limits

    Outputs:
     - Xs: the dataset at each iteration (split into a dictionary with values of k as keys)
     - As: the whitening matrix A at each iteration 
     - sigmas: the covariance matrix of the dataset at each iteration
     - moments_list: the list of the moments for each iteration

    """

    Xs, As, sigmas, moments = {}, {}, {}, {}
    
    if start_k >= end_k:
        ks = np.arange(end_k, start_k + 1)[::-1]
    else:
        ks = np.arange(start_k, end_k + 1)
        
    for k_count, k in enumerate(ks):
        if type(max_iter) == list:
            k_max_iter = max_iter[k_count]
        else:
            k_max_iter = max_iter
        
        if len(As) == 0:
            Xs[k], As[k], sigmas[k], moments[k] = repeated_k_whitening(sigma, X, k,
                                                                       adjust_at_end=adjust_at_end,
                                                                       weight_power=weight_power,
                                                                       max_iter=k_max_iter,
                                                                       prev_A = prev_A,
                                                                       prev_moment=prev_moment,
                                                                       plot_heatmap=plot_heatmap,
                                                                       heatmap_title=heatmap_title,
                                                                       save_heatmap_filepath=save_heatmap_filepath,
                                                                       plot_polynomial=plot_polynomial,
                                                                       save_poly_filepath=save_poly_filepath,
                                                                       poly_lims=poly_lims,
                                                                       dtype=dtype,
                                                                       store_X=store_X)
        else:
            prev_k = ks[k_count-1]
            if store_X == False:
                prev_X = Xs[prev_k]
            else:
                prev_X = Xs[prev_k][-1]
            Xs[k], As[k], sigmas[k], moments[k] = repeated_k_whitening(
                                                                 sigmas[prev_k][-1],
                                                                 prev_X,
                                                                 k, 
                                                                 adjust_at_end=adjust_at_end,
                                                                 weight_power=weight_power,
                                                                 max_iter=k_max_iter,
                                                                 prev_moment=moments[prev_k][-1], 
                                                                 prev_A=As[prev_k][-1], 
                                                                 plot_heatmap=plot_heatmap,
                                                                 save_heatmap_filepath=save_heatmap_filepath,
                                                                 heatmap_title=True,
                                                                 plot_polynomial=plot_polynomial,
                                                                 save_poly_filepath=save_poly_filepath,
                                                                 poly_lims=poly_lims,
                                                                 dtype=dtype,
                                                                 store_X=store_X)
    return Xs, As, sigmas, moments


def eigvals_from_dict_of_sigmas(sigmas, round_val = 6):
    """
    Given a dictionary of matrices, return the eigenvalues of each matrix.

    Input:
     - sigmas: a dictionary of matrices
     - round_val: round to this number of decimal places

    Output:
     - eigvals: dictionary with same keys as sigmas, but with eigenvalues rather than the matrices.
    """
    eigvals = {}
    
    for k in sigmas.keys():
        eigvals[k] = [np.sort(np.round(np.linalg.eigvals(sigma), round_val))[::-1] for sigma in sigmas[k]]
        
    return eigvals
