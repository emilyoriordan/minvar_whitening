# minvar_whitening
A package to use minimal variance whitening.

Data whitening is a transformation of a dataset intended to decorrelate and standardize its variables. This results in a new dataset with the identity covariance matrix. The most common method of data whitening is Mahalanobis whitening. For a \(d\)-dimensional dataset \(X \in \mathbb{R}^{d \times N}\) with empirical mean \(\mu\) and covariance matrix \(\Sigma\), the whitened dataset would be found as:
\[Y = \Sigma^{-1/2}(X - \mu).\]

If the covariance matrix \(\Sigma\) is singular (or close to singularity) the inverse square root of the covariance matrix is not available (or very unstable). As such, Mahalanobis whitening is not available. 

