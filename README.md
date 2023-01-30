# minvar_whitening
A package to use minimal variance whitening.

## Background
Data whitening is a transformation of a dataset intended to decorrelate and standardize its variables. This results in a new dataset with the identity covariance matrix. The most common method of data whitening is Mahalanobis whitening. For a $d$-dimensional dataset $X \in \mathbb{R}^{d \times N}$ with empirical mean $\mu$ and covariance matrix $\Sigma$, the whitened dataset would be found as:
$$Y = \Sigma^{-1/2}(X - \mu).$$

If the covariance matrix $\Sigma$ is singular (or close to singularity) the inverse square root of the covariance matrix is not available (or very unstable). As such, Mahalanobis whitening is not available. 

## Minimal Variance Whitening
Minimal variance whitening finds a whitening matrix to be used in place of $\Sigma^{-1/2}$. This whitening matrix is found to be a $k$-degree polynomial in $\Sigma$, where $k$ is a user-defined parameter. The minimal variance whitening polynomial is typically represented as:
$$A_{k} = \theta_{0}I + \theta_{1}\Sigma + \dots + \theta_{k-1}\Sigma^{k-1},$$
where $I$ is the $d$-dimensional identity matrix. The coefficients $\theta_{0}, \theta_{1}, \dots, \theta_{k-1}$ are calculated to fulfil the following optimization criterion: we wish to minimize the total variation of the transformed data subject to the constraint $\text{trace}(A_{k}\Sigma) = d$. Minimizing the total variation is equivalent to minimizing the diagonal of the covariance matrix of the transformed dataset. This combined with the above constraint ensure that the matrix $A_{k}$ behaves similarly to $\Sigma^{-1/2}$, when the latter matrix exists. 

For more information on the minimal variance whitening method, its calculation and examples of applications, please see the following publication: Gillard, J., O’Riordan, E. & Zhigljavsky, A. Polynomial whitening for high-dimensional data. Comput Stat (2022). https://doi.org/10.1007/s00180-022-01277-6
