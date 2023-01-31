
#Library imports 
import numpy as np 
import min_var as mv
import min_var.poly as mvp

#ideas for hypothesis:
# s_j(sigma, 0) = d
# X_matrix(sigma, k)[:, 0] = np.ones(d), X_matrix(sigma, k)[:, 0] = eigenvalues

sigma = np.diag([3, 2, 1])

def test_V_matrix():
    assert (mvp.V_matrix(sigma, 2) == np.array([[3, 9], [2, 4], [1, 1]])).all()
    assert (mvp.V_matrix(sigma, 3) == np.array([[3, 9, 27], [2, 4, 8], [1, 1, 1]])).all()

def test_s_j():
    assert mvp.s_j(sigma, 0) == 3
    assert mvp.s_j(sigma, 1) == 6
    assert mvp.s_j(sigma, 2) == 14
    assert mvp.s_j(sigma, 3) == 36
    assert mvp.s_j(sigma, 1/2) == np.sqrt(3) + np.sqrt(2) + 1

def test_s_jk():
    assert (mvp.s_jk(sigma, 1, 2) == np.array([mvp.s_j(sigma, 1), mvp.s_j(sigma, 2)])).all()
    assert (mvp.s_jk(sigma, 1, 2) == np.array([6, 14])).all()

    assert (mvp.s_jk(sigma, 1, 3) == np.array([mvp.s_j(sigma, 1), mvp.s_j(sigma, 2), mvp.s_j(sigma, 3)])).all()
    assert (mvp.s_jk(sigma, 1, 3) == np.array([6, 14, 36])).all()

    assert np.isclose(mvp.s_jk(sigma, 1/2, 2), np.array([4.14626437, 9.02457955])).all()
    assert np.isclose(mvp.s_jk(sigma, 1/2, 2), np.array([mvp.s_j(sigma, 1/2), mvp.s_j(sigma, 3/2)])).all()

    assert np.isclose(mvp.s_jk(sigma, 1/2, 3), np.array([ 4.14626437,  9.02457955, 22.24531152])).all()
    assert np.isclose(mvp.s_jk(sigma, 1/2, 3), np.array([mvp.s_j(sigma, 1/2), mvp.s_j(sigma, 3/2), mvp.s_j(sigma, 5/2)])).all()
    
def test_sigma_jk():
    assert (mvp.sigma_jk(sigma, 0, 2) == np.array([np.identity(3), sigma], dtype = int)).all()
    assert (mvp.sigma_jk(sigma, 0, 3) == np.array([np.identity(3), sigma, np.linalg.matrix_power(sigma, 2)], dtype = int)).all()
    assert (mvp.sigma_jk(sigma, 1, 3) == np.array([sigma, np.linalg.matrix_power(sigma, 2)], dtype = int)).all()

def test_inv():
    assert (mvp.inv(np.identity(3)) == np.identity(3)).all()
    assert (mvp.inv(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]), pinv = True) == np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])).all()

    assert np.isclose(mvp.inv(mvp.V_matrix(sigma, 3).T @ mvp.V_matrix(sigma, 3)), np.array([[ 11.36111111, -10.66666667,   2.30555556],
       [-10.66666667,  10.5       ,  -2.33333333],
       [  2.30555556,  -2.33333333,   0.52777778]])).all()

def test_theta():
    assert np.isclose(mvp.theta_alpha(sigma, 1, 3), np.array([ 1.83333333, -1.        ,  0.16666667])).all()
    assert np.isclose(mvp.theta_alpha(sigma, 1, 2), np.array([ 1.125     , -0.26785714])).all()

    assert np.isclose(mvp.theta_alpha(sigma, 1/2, 2, power = -1/2), np.array([ 1.14912843, -0.19553157])).all()
    assert np.isclose(mvp.theta_alpha(sigma, 1/2, 3, power = -1/2), np.array([ 1.45602993, -0.53759828,  0.08156835])).all()

def test_A_matrix():
    assert np.isclose(mvp.A_matrix(sigma, 1, 2), np.array([[0.32142857, 0.        , 0.        ],
       [0.        , 0.58928571, 0.        ],
       [0.        , 0.        , 0.85714286]])).all()
    assert np.isclose(mvp.A_matrix(sigma, 1, 3), np.diag([1/3, 1/2, 1])).all()
    
    assert np.isclose(mvp.A_matrix(sigma, 1/2, 2, power = -1/2), np.array([[0.5625337 , 0.        , 0.        ],
       [0.        , 0.75806528, 0.        ],
       [0.        , 0.        , 0.95359685]])).all()
    assert np.isclose(mvp.A_matrix(sigma, 1/2, 3, power = -1/2), np.array([[0.57735027, 0.        , 0.        ],
       [0.        , 0.70710678, 0.        ],
       [0.        , 0.        , 1.        ]])).all()

def test_X_matrix():
    assert np.isclose(mvp.X_matrix(sigma, 3), np.array([[1, 3, 9], [1, 2, 4], [1, 1, 1]], dtype = int)).all()
    assert np.isclose(mvp.X_matrix(sigma, 2), np.array([[1, 3], [1, 2], [1, 1]], dtype = int)).all()

def test_W_inv():
    assert (mvp.W_inv(sigma) == np.diag([9, 4, 1])).all()
    assert (mvp.W_inv(sigma, weightpower=1) == sigma).all()
    
def test_XWX():
    X = mvp.X_matrix(sigma, k = 3)
    Winv = mvp.W_inv(sigma, weightpower = 1)
    assert (X.T @ Winv @ X == np.array([[6, 14, 36], [14, 36, 98], [36, 98, 276]])).all()
    X = mvp.X_matrix(sigma, k = 2)
    Winv = mvp.W_inv(sigma, weightpower = 1)
    assert (X.T @ Winv @ X == np.array([[6, 14], [14, 36]])).all() 

def test_omega_alpha():
    assert type(mvp.omega_alpha(sigma, 1, 3)) == np.float64
    assert np.isclose(mvp.omega_alpha(sigma, 1, 2), 1.0178571428571435)
    assert np.isclose(mvp.omega_alpha(sigma, 1, 3), 0.9999999999999964)
    assert np.isclose(mvp.omega_alpha(sigma, 1/2, 3, power = -1/2), 0.9999999999999519)
    assert np.isclose(mvp.omega_alpha(sigma, 1/2, 2, power = -1/2), 1.002668460240467)

# def test_reg_Y_matrix():
#     assert (mvp.reg_Y_matrix(sigma, 1) == np.array([1/3, 1/2, 1])).all()
#     assert (mvp.reg_Y_matrix(sigma, 2) == np.array([1, 1, 1])).all()


# def test_theta_w():
#     assert (mvp.theta_w(sigma, 1, 2) == mvp.theta_hat(sigma, 1, 2)).all()
#     assert (mvp.theta_w(sigma, 1, 3) == mvp.theta_hat(sigma, 1, 3)).all()

#     assert np.isclose(mvp.theta_w(sigma, 1, 3), np.array([ 1.83333333, -1.        ,  0.16666667])).all()
#     assert np.isclose(mvp.theta_w(sigma, 1, 2), np.array([ 1.125     , -0.26785714])).all()


from hypothesis import given
from hypothesis.extra.numpy import arrays, array_shapes
# from hypothesis.strategies import floats, integers, lists
import hypothesis.strategies as st
# @given(st.lists(st.floats))
@given(arrays(dtype=np.float, shape=10, elements=st.floats(0, 1000)))
def test_VTV_inv(eigs):
    eigs = np.array(eigs)
    print(eigs)
    d = len(eigs)
    sigma = np.diag(eigs)
    unique = np.sort(np.unique(eigs))
    no_unique = len(unique)
    for k in range(1, no_unique+1):
        V = mvp.V_matrix(sigma, k)
        VTV = V.T @ V
        # if (eigs == np.int).all():
        if (eigs == eigs[0]).all():
            break
        if (eigs == 0).all():
            break
        if no_unique == 2:
            if np.isclose(unique, 0).any():
                break
        # if np.isclose(VTV, np.zeros([k, k]), atol = 1e-7).all():
        #     break
        # if (eigs == 2).all():
        #     break
        # if (eigs == 3).all():
        #     break
        if np.isclose(np.linalg.det(VTV), 0):
            det0 = True
        else:
            det0 = False
        assert det0 == False

# @given
# dims = array_shapes(min_dims = 2, max_dims = 2)
# arrays(np.float, [], )



