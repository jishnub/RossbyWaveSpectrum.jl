# Theory

## Equations

We solve the linearized Navier-Stokes equation in the anelastic approximation in a tracking frame that rotates at $\Omega_0$. We represent the unperturbed zero-order profiles of background density by $\bar{\rho}$, pressure by $\bar{p}$, temperature by $\bar{T}$, gravity by $\mathbf{g}$ and entropy by $S_0$. We denote perturbatations to these quantities using primes as superscripts.
We define the velocity $\mathbf{u}=\mathbf{u}_f + \mathbf{u}_\Omega$, where $\mathbf{u}_f$ is the velocity associated with
the waves, and $\mathbf{u}_\Omega$ is that for the differentially rotating background. We also represent the perturbation in the background entropy by $S^\prime$.

This leads to the following equations:
```math
\begin{align*}
\nabla\cdot\mathbf{u}&=0\\
\partial_{t}\mathbf{u}&=\mathbf{u}\times\boldsymbol{\nabla}\times\mathbf{u}-2\boldsymbol{\Omega}\times\mathbf{u}-\boldsymbol{\nabla}\left(\frac{p^{\prime}}{\bar{\rho}}+\frac{1}{2}\mathbf{u}^{2}\right)-\frac{S^{\prime}}{c_{p}}\mathbf{g}+\frac{1}{\bar{\rho}}\mathbf{F}^{\nu},\\
\partial_{t}S^{\prime}&=-im\Delta\Omega S^{\prime}-\left(\mathbf{u}_f\cdot\boldsymbol{\nabla}\right)S_{0}+\kappa\frac{1}{\bar{\rho}\bar{T}}\boldsymbol{\nabla}\cdot\left(\bar{\rho}\bar{T}\boldsymbol{\nabla} S^{\prime}\right)
\end{align*}
```

Here, $c_{p}$ is the specific heat at constant pressure, and $\mathbf{F}^{\nu}$ is the viscous force.

We may expand the velocity in terms of stream functions as
```math
\begin{align*}
\bar{\rho}\mathbf{u}_f &= \boldsymbol{\nabla}\times\boldsymbol{\nabla}\times\left(\bar{\rho}\,W\left(\mathbf{x}\right)\hat{r}\right)+\boldsymbol{\nabla}\times\left(\bar{\rho}\,V\left(\mathbf{x}\right)\hat{r}\right).
\end{align*}
```
This form of the velocity automatically satisfies the mass-conservation constraint. We expand the individual stream functions in a Chebyshev-spherical harmonic basis as
```math
\begin{align*}V\left(\mathbf{x}\right) & =\sum_{n\ell m}V_{n\ell m}T_{n}\left(\hat{r}\right)\hat{P}_{\ell m}\left(\cos\theta\right)\exp\left(im\phi\right),\\
W\left(\mathbf{x}\right) & =\sum_{n\ell m}W_{n\ell m}T_{n}\left(\hat{r}\right)\hat{P}_{\ell m}\left(\cos\theta\right)\exp\left(im\phi\right),\\
S^{\prime}\left(\mathbf{x}\right) & =\sum_{n\ell m}S_{n\ell m}^{\prime}T_{n}\left(\hat{r}\right)\hat{P}_{\ell m}\left(\cos\theta\right)\exp\left(im\phi\right),
\end{align*}
```
where $\hat{r}$ is a scaled radial coordinate that maps the radial domain to $[-1,1]$, and $\hat{P}_{\ell m}\left(\cos\theta\right)$ represents the normalized associated Legendre polynomial of degree $\ell$ and azimuthal order $m$. We may collectively express the coefficients $V_{n\ell m}$ as $\boldsymbol{V}$, and similarly for the other fields.

Substituting, and transforming to temporal frequency space, we may rewrite the equations in the form
```math
\left(\begin{array}{ccc}
A_{VV} & A_{VW} & A_{VS}\\
A_{WV} & A_{WW} & A_{WS}\\
A_{SV} & A_{SW} & A_{SS}
\end{array}\right)
\left(\begin{array}{c}
\boldsymbol{V}\\
\boldsymbol{W}\\
\boldsymbol{S}
\end{array}\right)=
\frac{\omega}{\Omega_0}
\left(\begin{array}{ccc}
B_{VV} & 0 & 0\\
0 & B_{WW} & 0\\
0 & 0 & B_{SS}
\end{array}\right)
\left(\begin{array}{c}
\boldsymbol{V}\\
\boldsymbol{W}\\
\boldsymbol{S}
\end{array}\right)
```
where each matrix element represents an operator -- potentially a differential one. We may express this in a condensed form as
```math
A \mathbf{v} = \frac{\omega}{\Omega_0} B \mathbf{v},
```
where $\mathbf{v}$ now collects all the coefficients, and represents the eigenfunction that we want to solve for.

We impose the boundary conditions
```math
 u_{r} = \partial_{r}\left(\frac{u_{\theta}}{r}\right)=\partial_{r}\left(\frac{u_{\phi}}{r}\right)=\partial_r S^\prime=0,\quad\text{on}\;r=r_{i},\quad\text{and}\;r=r_{o}.
```
and use the standard spherical-harmonic boundary conditions at the poles. These conditions may be translated to the spectral coefficients to obtain sets of equations of the form
```math
\left(\begin{array}{ccc}
C_{VV} & 0 & 0\\
0 & C_{WW} & 0\\
0 & 0 & C_{SS}
\end{array}\right)\left(\begin{array}{c}
\boldsymbol{V}\\
\boldsymbol{W}\\
\boldsymbol{S}
\end{array}\right)	=\left(\begin{array}{c}
\boldsymbol{0}\\
\boldsymbol{0}\\
\boldsymbol{0}
\end{array}\right)
```
We may express this in a condensed form as
```math
C \mathbf{v} = 0.
```
This tells us that the eigenfunction $\mathbf{v}$ lies in the null-space of $C$. We transform from the radial Chebyshev basis to a different one that satisfies the radial boundary conditions. If we collect the basis elements as the columns of
a matrix $Z$, we obtain $CZ=0$ by construction. We may therefore express the eigenfunction as
```math
\mathbf{v} = Z \mathbf{w}.
```
Substituting this into the eigenvalue problem, and multiplying both sides by $Z^T$, we obtain
```math
\left(Z^T A Z\right) \mathbf{w} = \frac{\omega}{\Omega_0} \left(Z^T B Z\right) \mathbf{w},
```
which is now unconstrained in $\mathbf{w}$. We solve the eigenvalue problem to obtain solutions $(\omega, \mathbf{w})$, and subsequently project the eigenfunctions back to the Chebyshev basis to obtain $(\omega, \mathbf{v}=Z\mathbf{w})$.

We note that solutions to $A \mathbf{v} = (\omega/\Omega_0) B \mathbf{v}$ must necessarily satisfy $\left(Z^T A Z\right) \mathbf{w} = (\omega/\Omega_0) \left(Z^T B Z\right) \mathbf{w}$, but the projected problem may produce solutions that do not satisfy the original system. We therefore include a post-processing step to filter out such solutions. We also filter out solutions that are not smooth, as well as exponentially growing ones.

## Reference

Bhattacharya, J., & Hanasoge, S. M. 2023, ApJS, 264, 21, doi: [10.3847/1538-4365/aca09a](https://iopscience.iop.org/article/10.3847/1538-4365/aca09a). Preprint available at the [Arxiv link](https://arxiv.org/abs/2211.03323)
