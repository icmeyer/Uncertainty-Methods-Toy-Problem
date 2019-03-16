import numpy as np

## The vector space of functions we are dealing with: Generic rational fractions with an offset and an entire part.
# > The entire part is specified by the set of coefficients.
# > If no coefficients are provided (the empty set), the rational function is build with a simple offset, the default value of which is zero.
def rational_function(z, poles, residues, offset=0, poly_coeff=()):
    if poly_coeff == ():
        poly_order = 0
    ## print("The rational_function function was not given an entire polynomial part to build.")
    else:
        poly_order = poly_coeff.shape[0]
    return  sum(residues[n]/(z-poles[n]) for n in range(poles.size)) + offset + sum(poly_coeff[n]*z**(n+1) for n in range(poly_order))



# Vector Fitting Algorithm
# The simplest barycentric SK iterations: no relaxation, no QR, no orthogonal basis, no real numbers
# The space of functions we are learning in is that of proper rational fraction of degree zero (poles + offset)

## Create the Y_vector for the LS problem :: Here, we weight it with the sqrt of the rho_k for the system
def vectorize_Y_for_LS(z_train, Y_train, dim_residues):
    number_train_points = Y_train.shape[0]
    sqrt_weights_rho = build_square_weights_rho_k(z_train, Y_train)
    Y_LS_vector = np.zeros([dim_residues*number_train_points],  dtype=np.complex)
    for d in range(dim_residues):
        for k in range(number_train_points):
            Y_LS_vector[d*number_train_points + k] =  sqrt_weights_rho[k]*Y_train[k,d]
    return Y_LS_vector

## define the Cauchy matrix
def Cauchy_matrix(z,p):
    C = np.zeros([z.size , p.size],  dtype=np.complex)
    for k in range(z.size):
        for j in range(p.size):
            C[k,j] = 1/(z[k] - p[j])
    return C

## define the Vandermonde matrix without offset
def Vandermonde_matrix(z,poly_order):
    V = np.zeros([z.size , poly_order],  dtype=np.complex)
    for k in range(z.size):
        for n in range(poly_order):
            V[k,n] = z[k]**(n+1)
    return V

## build the square root of the rho_k weights for the LS system :: linear heteroscedastic case
def build_square_weights_rho_k(z_train, Y_train):
    number_train_points = z_train.size
    sqrt_weights_rho = np.zeros([number_train_points], dtype=np.complex)
    for k in range(number_train_points):
        sqrt_weights_rho[k] =  1.0/(np.sqrt(number_train_points*(np.linalg.norm(Y_train[k]))))   ## For heteroskedastic (np.sqrt(np.linalg.norm(Y_train[k])))))  ## * (np.linalg.norm(Y_train[k])) ))) ## for linear heteroscedastic case, add: *np.linalg.norm(Y_train[k]))
    return sqrt_weights_rho

## build the matrix of the barycentric LS system:
def build_barycentric_LS_matrix(z_train, Y_train, learn_poles, poly_order=0):
  dim_residues = Y_train[0].size
  number_train_points = z_train.size
  number_poles_learn = learn_poles.size
  sqrt_weights_rho = build_square_weights_rho_k(z_train, Y_train) 
  C = Cauchy_matrix(z_train, learn_poles)
  V = Vandermonde_matrix(z_train, poly_order)
  LS_matrix = np.zeros([dim_residues*number_train_points, dim_residues*(number_poles_learn+1+poly_order) + number_poles_learn],  dtype=np.complex)
  for d in range(dim_residues):
      for k in range(number_train_points):
          for p in range(number_poles_learn):
              LS_matrix[d*number_train_points + k, d*number_poles_learn + p] = sqrt_weights_rho[k]*C[k,p]
              for n in range(poly_order):
                  LS_matrix[d*number_train_points + k, dim_residues*number_poles_learn + d*poly_order + n] = sqrt_weights_rho[k]*V[k,n] 
              LS_matrix[d*number_train_points + k, dim_residues*(number_poles_learn+poly_order) + d] = sqrt_weights_rho[k]*1 
              LS_matrix[d*number_train_points + k, dim_residues*(number_poles_learn+poly_order+1) + p] = -sqrt_weights_rho[k]*Y_train[k,d]*C[k,p]
  return LS_matrix

## build the matrix of the simple LS system:
def build_LS_matrix(z_train, Y_train, learn_poles, poly_order=0):
  dim_residues = Y_train[0].size
  number_train_points = z_train.size
  number_poles_learn = learn_poles.size
  sqrt_weights_rho = build_square_weights_rho_k(z_train, Y_train) 
  C = Cauchy_matrix(z_train, learn_poles)
  V = Vandermonde_matrix(z_train,poly_order)
  LS_matrix = np.zeros([dim_residues*number_train_points, dim_residues*(number_poles_learn+1+poly_order) ],  dtype=np.complex)
  for d in range(dim_residues):
      for k in range(number_train_points):
          for p in range(number_poles_learn):
              LS_matrix[d*number_train_points + k, d*number_poles_learn + p] = sqrt_weights_rho[k]*C[k,p]  
              for n in range(poly_order):
                  LS_matrix[d*number_train_points + k, dim_residues*number_poles_learn + d*poly_order + n] = sqrt_weights_rho[k]*V[k,n] 
              LS_matrix[d*number_train_points + k, dim_residues*(number_poles_learn+poly_order) + d] = sqrt_weights_rho[k]*1 
  return LS_matrix

## Function that vectorizes the residues and offset for the problem
def build_LS_vector(residues, offset, poly_coeff=()):
    if type(poly_coeff) == tuple:
        poly_order = 0
        #print("The build_LS_vector function was not given an entire polynomial part to build.") 
    else:
        poly_order = poly_coeff.shape[0]
    number_poles = residues.shape[0]
    dim_residues = residues.shape[1]
    LS_vector    = np.zeros([dim_residues*(number_poles+poly_order+1)], dtype=np.complex)
    for d in range(dim_residues):
        for p in range(number_poles):
            LS_vector[d*number_poles + p] = residues[p][d]
        for n in range(poly_order):
            LS_vector[dim_residues*number_poles + d*poly_order + n] = poly_coeff[n][d]
    LS_vector[dim_residues*(number_poles+poly_order):LS_vector.size] = offset
    return LS_vector

## Function that takes the vectorized solution and spits out the different elements
def extract_from_LS_vector(LS_vector, number_poles, dim_residues , poly_order=0):
    residues = np.zeros([number_poles, dim_residues], dtype=np.complex)
    poly_coeff = np.zeros([poly_order, dim_residues], dtype=np.complex)
    offset   = np.zeros([dim_residues], dtype = np.complex)
    for d in range(dim_residues): 
        for p in range(number_poles):
            residues[p][d] = LS_vector[d*number_poles + p]
        for n in range(poly_order):
            poly_coeff[n][d] = LS_vector[dim_residues*number_poles + d*poly_order + n]
    offset = LS_vector[dim_residues*(number_poles+poly_order) : LS_vector.size]
    if poly_order == 0:
        return residues, offset
    else:
        return residues, poly_coeff, offset


def VF_algorithm(z_train, Y_train, number_VF_iteration, poly_order = 0 , *arguments ):
    """
    *The Vector Fitting algorithm: *

    The algorithm takes as input a training set {z_k, Y_k}, composed of training vectors {Y_k}, matched on a grid {z_k}.

    A standard number of VF iterations is 10, which usually suffices to converge the poles.

    Our implementation does not require the input of an initial guess. If you do not give one, the program will come up with a linearly spaced complex diagonal along the rectangle of the mesh of training points on the complex plane.
    """

    ## build the Y vector to solve for:
    dim_residues = Y_train[0].size
    number_train_points = z_train.size
    Y_LS_vector = vectorize_Y_for_LS(z_train, Y_train, dim_residues)
    ## Initialize the poles
    if arguments == ():
        raise AssertionError("The VF_algorithm function must be given either a number of poles, or an array of initial poles guess") 
    for arg in arguments:
        if type(arg) == np.ndarray: ## was given an initial guess as argument
            print(" The VF_algorithm was provided an initial guess for the poles")
            learn_poles = arg
            number_poles = learn_poles.size
        elif type(arg) == int: ## was given a number of poles without any initial guess
            print("The VF_algorithm was provided a number of poles to learn and is generating an initial guess")
            number_poles = arg
            if (np.amax(np.imag(z_train)) - np.amin(np.imag(z_train))) == 0: ## only real training data
                print("The training points are only along the real axis, and the initial guesses are generated accordingly with a shift")
                learn_poles = np.linspace(np.amin(np.real(z_train))+1/(10*(np.amax(np.real(z_train)) - np.amin(np.real(z_train)))),np.amax(np.real(z_train)) + 1/(10*(np.amin(np.real(z_train))-np.amax(np.real(z_train)))) , number_poles) + 1j*np.linspace(np.amin(np.imag(z_train)),np.amax(np.imag(z_train)), number_poles)
            elif (np.amax(np.real(z_train)) - np.amin(np.real(z_train)) ) == 0:
                print("The training points are exactly along the imaginary axis, and the initial guesses are generated accordingly with a shift")
                learn_poles = np.linspace(np.amin(np.real(z_train)),np.amax(np.real(z_train)) , number_poles) + 1j*np.linspace(np.amin(np.imag(z_train)) + 1/(10*(np.amax(np.imag(z_train)) - np.amin(np.imag(z_train)))) ,np.amax(np.imag(z_train)) - 1/(10*(np.amax(np.imag(z_train)) - np.amin(np.imag(z_train)))), number_poles)
            else:
                learn_poles = np.linspace(np.amin(np.real(z_train))+1/(10*(np.amax(np.real(z_train))-np.amin(np.real(z_train)))),np.amax(np.real(z_train)) - 1/(10*(np.amax(np.real(z_train)) - np.amin(np.real(z_train)))) , number_poles) + 1j*np.linspace(np.amin(np.imag(z_train)) + 1/(10*(np.amax(np.imag(z_train)) - np.amin(np.imag(z_train)))), np.amax(np.imag(z_train)) -  1/(10*(np.amax(np.imag(z_train)) - np.amin(np.imag(z_train)))) , number_poles)
    ## POLE CONVERGENCE: Run the VF iterations 
    for i in range(number_VF_iteration):
        ## build the barycentric L2 system
        barycentric_LS_matrix = build_barycentric_LS_matrix(z_train, Y_train, learn_poles, poly_order)
        ## solve the barycentric L2 system
        barycentric_LS_vector , barycentric_LS_residual , barycentric_LS_rank , barycentric_LS_singular_values = np.linalg.lstsq(barycentric_LS_matrix, Y_LS_vector, poly_order)
        ## extract the barycentric residues
        barycentric_residues = barycentric_LS_vector[dim_residues*(learn_poles.size + poly_order +1):barycentric_LS_vector.size]
        ## Build the matrix the spectrum of which will be the recolated poles
        P = np.diag(learn_poles) - np.tensordot(barycentric_residues,np.ones([barycentric_residues.size], dtype=np.complex),0)
        ## Solve the spectral problem & relocate poles
        learn_poles , eigenvectors = np.linalg.eig(P)
        ## convergence criteria
    ## RESIDUES EXTRACTION: Solve the LS system
    ## build the quadratic system
    LS_matrix = build_LS_matrix(z_train, Y_train, learn_poles, poly_order)
    ## solve the quadratic L2 system
    LS_vector, LS_residual, LS_rank , LS_singular_values = np.linalg.lstsq(LS_matrix, Y_LS_vector) ## np.linalg.solve(A.T.dot(A) + lamb * np.identity(n_col), A.T.dot(y)) For Tichonov
    ## extract the residues, polynomial coefficients and offset
    if poly_order == 0:
        VF_poly_coeff= np.array([ 0.0 for n in range(poly_order)])
        VF_residues , VF_offset = extract_from_LS_vector(LS_vector, learn_poles.size, dim_residues)
        return learn_poles, VF_residues, VF_poly_coeff, VF_offset, LS_residual, barycentric_residues ## Artificially added an emplty set of VF_poly_coeff for homogeneity
    else:
        VF_residues , VF_poly_coeff, VF_offset = extract_from_LS_vector(LS_vector, learn_poles.size, dim_residues, poly_order)
        return learn_poles, VF_residues, VF_poly_coeff, VF_offset, LS_residual, barycentric_residues
