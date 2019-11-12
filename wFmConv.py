# Necessary architecture to implement calculation of the weighted
# Frechet mean.

import numpy as np
import doctest

def enforce_bounds(theta):
    '''Ensures that theta remains within the interval [-pi,pi].

    >>> enforce_bounds(-4)
    2.2831853071795862
    >>> enforce_bounds(-np.pi)
    -3.141592653589793
    >>> enforce_bounds(np.pi)
    3.141592653589793
    >>> enforce_bounds(3*np.pi)
    3.141592653589793
    >>> enforce_bounds(0)
    0
    >>> enforce_bounds(np.sqrt(2))
    1.4142135623730951
    '''
    while theta < -np.pi:
        theta += 2*np.pi
    while theta > np.pi:
        theta -= 2*np.pi
    return theta

def identify(re, im):
    '''Identifies re + j*im with r*exp(-j*theta) in the relevant
    manifold, the complex numbers.

    >>> identify(1/np.sqrt(2), 1/np.sqrt(2))
    (0.9999999999999999, 0.7853981633974483)
    '''
    r = np.sqrt(re**2 + im**2)
    theta = np.arctan2(im,re)
    #print('Original theta: ', theta)
    theta = enforce_bounds(theta)
    #print('Bounded theta: ', theta)
    return r, theta

def get_manifold_distance(z1, z2):
    '''Computes the distance between complex numbers z1 and z2
    in the manifold defined on R^+ x SO(2) = {(r,R(theta))}.

    >>> get_manifold_distance(np.complex(1,0), 0.01)
    4.605170185988091
    >>> get_manifold_distance([1,0], [0.01,0])
    4.605170185988091
    >>> get_manifold_distance([1,0], np.complex(0,1))
    2.221441469079183
    '''
    #print('Original z1: ', z1)
    if type(z1) not in [list,  np.ndarray, tuple]:
        z1 = [np.real(z1), np.imag(z1)]
        # print('New z1: ', z1)
    r1, theta1 = identify(z1[0], z1[1])
    # print('r1, theta1: ', r1, theta1)

    #print('Original z2: ', z2)
    if type(z2) not in [list,  np.ndarray, tuple]:
        z2 = [np.real(z2), np.imag(z2)]
        # print('New z1: ', z1)
    r2, theta2 = identify(z2[0], z2[1])
    # print('r1, theta1: ', r1, theta1)

    theta_diff = theta2 - theta1
    theta_diff = enforce_bounds(theta_diff)
    dman = np.sqrt( (np.log(r2/r1))**2 + 2*(theta_diff**2) )
    return dman

def pick_three_rand(m, realmin, realmax, imagmin, imagmax):
    '''Generates three unique 1x2 arrays of random floats from the 
    intervals [realmin, realmax] and [imagmin, imagmax], respectively.
    '''

    real_diff = realmax-realmin
    imag_diff = imagmax-imagmin

    a = np.array(\
            [ real_diff*np.random.random_sample() + realmin, \
            imag_diff*np.random.random_sample() + imagmin] )
    while np.complex(a[0], a[1]) == np.complex(m[0], m[1]):
        a = np.array(\
            [ real_diff*np.random.random_sample() + realmin, \
            imag_diff*np.random.random_sample() + imagmin] )
    b = np.array(\
            [ real_diff*np.random.random_sample() + realmin, \
            imag_diff*np.random.random_sample() + imagmin] )
    while np.complex(b[0],b[1]) == np.complex(m[0], m[1]) or\
            np.complex(b[0],b[1]) == np.complex(a[0],a[1]):
        b = np.array(\
            [ real_diff*np.random.random_sample() + realmin, \
            imag_diff*np.random.random_sample() + imagmin] )
    c = np.array(\
            [ real_diff*np.random.random_sample() + realmin, \
            imag_diff*np.random.random_sample() + imagmin] )
    while np.complex(c[0],c[1]) == np.complex(m[0],m[1]) or \
            np.complex(c[0],c[1]) == np.complex(a[0],a[1]) or \
            np.complex(c[0],c[1]) == np.complex(b[0], b[1]):
        c = np.array(\
            [ real_diff*np.random.random_sample() + realmin, \
            imag_diff*np.random.random_sample() + imagmin] )
    return a,b,c  

def get_weighted_sum(point_list, weight_list, m):
    '''Computes the weighted sum
    Sum_i=1^K( weight_list[i] * (dman([point_list[:][i]], m))**2 )

    >>> get_weighted_sum([[1],[1]], [0.5], [1,0])
    0.67690690180786
    '''
    point_list = np.array(point_list)
    weight_list = np.array(weight_list)
    if point_list.shape[0] != 2:
        point_list = point_list.reshape(2,-1)

    s = [weight_list[i] * \
            (get_manifold_distance([point_list[0][i], point_list[1][i]],m))**2 for i in range(weight_list.shape[0]) ]
    S = np.sum(s)
    return S

def calc_wfm(point_list, weight_list, num_iters=500, crossover_prob=0.75,\
        F = 0.25, verbose=0):
    '''Computes the weighted Frechet mean of point_list with filter 
    weights weight_list.
    Uses differential evolution to minimize the weighted variance.
    '''
    realmin, realmax = min(point_list[0]), max(point_list[0])
    imagmin, imagmax = min(point_list[1]), max(point_list[1])
    iternum = 0
    start_point = \
            (np.mean(point_list[0]), np.mean(point_list[1]))
    m = start_point
    if verbose: print('m: ', m)
        
    while iternum < num_iters:
        iternum += 1
        f_m = get_weighted_sum(point_list, weight_list, m)
        # Implement differential evolution.
        # For each agent m (1 agent), pick three agents a,b,c
        # from the possible parameter space.
        a,b,c = pick_three_rand(m, realmin, realmax, imagmin, imagmax)

        # Pick random number to determine crossover.
        r = np.random.rand(1)
        if r < crossover_prob:
            y = a + F*(b - c)
            f_y = get_weighted_sum(point_list, weight_list, y)
            if verbose: print('f_y: ', f_y)
            if f_y <= f_m:
                m = y
                if verbose:
                    print('New min: ', m)
            else:
                if verbose:
                    print('No new min')
        else:
            if verbose:
                print('No crossover')
        return m

    
#### Example usage ####
def main():
    point_list = [[1,2,3],[4,5,6]]
    weight_list = [1, 0.5, 0.33]
    m = [1,0]
    return calc_wfm(point_list, weight_list, verbose=1)
