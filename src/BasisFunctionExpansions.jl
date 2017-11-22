module BasisFunctionExpansions
using Clustering
export BasisFunctionExpansion, UniformRBFE, MultiUniformRBFE, MultiDiagonalRBFE, MultiRBFE, BasisFunctionApproximation
export get_centers, get_centers_multi, get_centers_automatic, quadform, γ2σ, σ2γ
export toeplitz, getARregressor, getARXregressor, LPVSS, predict, output_variance

## Types
abstract type BasisFunctionExpansion{N} end

# function Base.show{N}(io::IO,b::BasisFunctionExpansion{N})
#     s = string(typeof(b),"\n")
#     for fn in fieldnames(b)
#         s *= string(fn) * ": " * string(getfield(b,fn)) * "\n"
#     end
#     print(io,s)
# end

struct BasisFunctionApproximation
    bfe::BasisFunctionExpansion
    linear_combination::Vector{Float64}
end

"""
    BasisFunctionApproximation(y::Vector, v, bfe::BasisFunctionExpansion, λ = 0)

Perform parameter identification to identify the Function `y = ϕ(v)`, where `ϕ` is a Basis Function Expansion of type `bfe`.
`λ` is an optional regularization parameter (L² regularization).
"""
function BasisFunctionApproximation(y::AbstractVector,v,bfe::BasisFunctionExpansion, λ = 0)
    A = bfe(v)
    p = size(A,2)
    if λ == 0
        x = A\y
    else
        x = [A; λ*eye(p)]\[y;zeros(p)]
    end
    BasisFunctionApproximation(bfe,x)
end

function (bfa::BasisFunctionApproximation)(v)
    A = bfa.bfe(v)
    return A*bfa.linear_combination
end

### UniformRBFE ================================================================
"""
A Uniform RBFE has the same variance for all basis functions
"""
struct UniformRBFE <: BasisFunctionExpansion{1}
    activation::Function
    μ::Vector{Float64}
    σ::Float64
end

"""
    UniformRBFE(μ::Vector, σ::Float, activation)

Supply all parameters. OBS! `σ` can not be an integer, must be some kind of AbstractFloat
"""
function UniformRBFE(μ::AbstractVector, σ::AbstractFloat, activation)
    UniformRBFE(v->activation(v,μ,σ2γ(σ)),μ,σ)
end

"""
    UniformRBFE(v::Vector, Nv::Int; normalize=false, coulomb=false)

Supply scheduling signal and number of basis functions For automatic selection of centers and widths

The keyword `normalize` determines weather or not basis function activations are normalized to sum to one for each datapoint, normalized networks tend to extrapolate better ["The normalized radial basis function neural network" DOI: 10.1109/ICSMC.1998.728118](http://ieeexplore.ieee.org/document/728118/)
"""
function UniformRBFE(v::AbstractVector, Nv::Int; normalize=false, coulomb=false)
    activation, μ, γ = basis_activation_func_automatic(v,Nv,normalize,coulomb)
    UniformRBFE(activation,μ,γ2σ(γ))
end


### RBFE =======================================================================
"""
A `MultiUniformRBFE` has the same diagonal covariance matrix for all basis functions
See also `MultiDiagonalRBFE`, which has different covariance matrices for all basis functions
"""
struct MultiUniformRBFE{N} <: BasisFunctionExpansion{N}
    activation::Function
    μ::Matrix{Float64}
    Σ::Vector{Float64}
end

"""
    MultiUniformRBFE(μ::Matrix, Σ::Vector, activation)

Supply all parameters. Σ is the diagonal of the covariance matrix
"""
function MultiUniformRBFE(μ::AbstractMatrix, Σ::AbstractVector, activation)
    MultiUniformRBFE{size(μ,1)}(v->activation(v,μ,σ2γ(Σ)),μ,Σ)
end

"""
    MultiUniformRBFE(v::AbstractVector, Nv::Vector{Int}; normalize=false, coulomb=false)

Supply scheduling signal and number of basis functions For automatic selection of centers and widths

The keyword `normalize` determines weather or not basis function activations are normalized to sum to one for each datapoint, normalized networks tend to extrapolate better ["The normalized radial basis function neural network" DOI: 10.1109/ICSMC.1998.728118](http://ieeexplore.ieee.org/document/728118/)
"""
function MultiUniformRBFE(v::AbstractMatrix, Nv::AbstractVector{Int}; normalize=false, coulomb=false)
    @assert !coulomb "Coulomb not yet supported for multi-dimensional BFEs"
    @assert length(Nv) == size(v,2)
    activation, μ, γ = basis_activation_func_automatic(v,Nv,normalize,coulomb)
    MultiUniformRBFE{length(Nv)}(activation,μ,γ2σ.(γ))
end



## MultiDiagonalRBFE =======================================================================
"""
A `MultiDiagonalRBFE` has different diagonal covariance matrices for all basis functions
See also `MultiUniformRBFE`, which has the same covariance matrix for all basis functions
"""
struct MultiDiagonalRBFE{N} <: BasisFunctionExpansion{N}
    activation::Function
    μ::Matrix{Float64}
    Σ::Vector{Vector{Float64}}
end

"""
    MultiDiagonalRBFE(μ::Matrix, Σ::Vector{Vector{Float64}}, activation)

Supply all parameters. Σ is the diagonals of the covariance matrices
"""
function MultiDiagonalRBFE(μ::AbstractMatrix, Σ::AbstractVector{T}, activation) where T <: AbstractVector
    MultiDiagonalRBFE{size(μ,1)}(v->activation(v,μ,σ2γ(Σ)),μ,Σ)
end

"""
    MultiDiagonalRBFE(v::AbstractVector, nc; normalize=false, coulomb=false)

Supply scheduling signal `v` and numer of centers `nc` For automatic selection of covariance matrices and centers using K-means.

The keyword `normalize` determines weather or not basis function activations are normalized to sum to one for each datapoint, normalized networks tend to extrapolate better ["The normalized radial basis function neural network" DOI: 10.1109/ICSMC.1998.728118](http://ieeexplore.ieee.org/document/728118/)
"""
function MultiDiagonalRBFE(v::AbstractMatrix, nc; normalize=false, coulomb=false)
    @assert !coulomb "Coulomb not yet supported for multi-dimensional BFEs"
    μ, Σ = get_centers_Kmeans(v, nc)
    μ = hcat(μ...)
    Σ = [2diag(Σi) for Σi in Σ] # Heuristically inflate covariance by 2 "On the Kernel Widths in Radial-Basis Function Networks" NABIL BENOUDJIT and MICHEL VERLEYSEN
    activation = basis_activation_func(μ,σ2γ(Σ),normalize,coulomb)
    MultiDiagonalRBFE{size(v,2)}(activation,μ,Σ)
end

## MultiRBFE =======================================================================
"""
A `MultiRBFE` has different diagonal covariance matrices for all basis functions
See also `MultiUniformRBFE`, which has the same covariance matrix for all basis functions
"""
struct MultiRBFE{N} <: BasisFunctionExpansion{N}
    activation::Function
    μ::Matrix{Float64}
    Σ::Vector{Matrix{Float64}}
end

"""
    MultiRBFE(μ::Matrix, Σ::Vector{Vector{Float64}}, activation)

Supply all parameters. Σ is the diagonals of the covariance matrices
"""
function MultiRBFE(μ::AbstractMatrix, Σ::AbstractVector{T}, activation) where T <: AbstractVector
    MultiRBFE{size(μ,1)}(v->activation(v,μ,σ2γ(Σ)),μ,Σ)
end

"""
    MultiRBFE(v::AbstractVector, nc; normalize=false, coulomb=false)

Supply scheduling signal `v` and numer of centers `nc` For automatic selection of covariance matrices and centers using K-means.

The keyword `normalize` determines weather or not basis function activations are normalized to sum to one for each datapoint, normalized networks tend to extrapolate better ["The normalized radial basis function neural network" DOI: 10.1109/ICSMC.1998.728118](http://ieeexplore.ieee.org/document/728118/)
"""
function MultiRBFE(v::AbstractMatrix, nc; normalize=false, coulomb=false)
    @assert !coulomb "Coulomb not yet supported for multi-dimensional BFEs"
    μ, Σ = get_centers_Kmeans(v, nc)
    μ = hcat(μ...)
    Σ .*= 3 # Heuristically inflate covariance by 3 "On the Kernel Widths in Radial-Basis Function Networks" NABIL BENOUDJIT and MICHEL VERLEYSEN
    activation = basis_activation_func(μ,σ2γ(Σ),normalize,coulomb)
    MultiRBFE{size(v,2)}(activation,μ,Σ)
end

## Squared exponential functions

squared_exponential(v::Real,vc,gamma) = exp.(-gamma*(v.-vc).^2)
squared_exponential(v::AbstractVector,vc,gamma::Number) = exp.(-gamma*(v.-vc').^2)
squared_exponential(v::AbstractVector,vc,gamma::AbstractVector) = exp.(-((vc'.-v').^2)*gamma)
function squared_exponential{T}(v::AbstractMatrix{T},vc,gamma::AbstractVector)
    a = Matrix{T}(size(v,1),size(vc,2))
    for i = 1:size(v,1)
        a[i,:] = exp.(-sum(gamma.*(v[i,:].-vc).^2,1))
    end
    a
end

function squared_exponential(v::AbstractMatrix,vc,gamma::AbstractVector{T}) where T <: AbstractVector
    a = zeros(size(v,1),size(vc,2))
    for j = 1:size(vc,2)
        for i = 1:size(v,1)
            a[i,j] += exp(-gamma[j]⋅(v[i,:].-vc[:,j]).^2)
        end
    end
    a
end

function squared_exponential(v::AbstractMatrix,vc,gamma::AbstractVector{T}) where T <: AbstractMatrix
    a = zeros(size(v,1),size(vc,2))
    for j = 1:size(vc,2)
        for i = 1:size(v,1)
            d = v[i,:].-vc[:,j]
            a[i,j] += exp(-quadform(d,gamma[j]))
        end
    end
    a
end

function normalized_squared_exponential(v,vc,gamma::Number)
    r = squared_exponential(v,vc,gamma)
    r ./= (sum(r,2) + 1e-8)
end

function normalized_squared_exponential(v::AbstractVector,vc,gamma::AbstractVector)
    r = squared_exponential(v,vc,gamma)
    r ./= (sum(r) + 1e-8)
end

function normalized_squared_exponential(v::AbstractMatrix,vc,gamma::AbstractVector)
    r = squared_exponential(v,vc,gamma)
    r ./= (sum(r,2) + 1e-8)
end

squared_exponential_coulomb(v,vc,gamma) = squared_exponential(v,vc,gamma).*(sign(v) .== sign(vc))

function normalized_squared_exponential_coulomb(v,vc,gamma)
    r = squared_exponential_coulomb(v,vc,gamma)
    r ./= (sum(r,2) + 1e-8)
end

function squared_exponential(v::AbstractMatrix,vc, sigma, velocity::Int=0)
    error("This function has not yet been revised")
    @assert size(v,2) == length(sigma)
    N_basis = size(vc,2)
    y       = zeros(N_basis)
    iSIGMA  = sigma.^-2

    if velocity > 0
        y = [sign(vc[velocity,i]) == sign(v[velocity]) ? mnorm_pdf(v,vc[:,i],iSIGMA) : 0 for i = 1:N_basis]
    else
        y = [mnorm_pdf(v,vc[:,i],iSIGMA) for i = 1:N_basis]
    end
end

"""
    basis_activation_func_automatic(v,Nv,normalize,coulomb)

Returns a func v->ϕ(v) ∈ ℜ(Nv) that calculates the activation of `Nv` basis functions spread out to cover v nicely. If coulomb is true, then we get twice the number of basis functions, `2Nv`, with a hard split at `v=0` (useful to model Coulomb friction). coulomb is not yet fully supported for all expansion types.

The keyword `normalize` determines weather or not basis function activations are normalized to sum to one for each datapoint, normalized networks tend to extrapolate better ["The normalized radial basis function neural network" DOI: 10.1109/ICSMC.1998.728118](http://ieeexplore.ieee.org/document/728118/)
"""
function basis_activation_func_automatic(v,Nv,normalize,coulomb=false)
    vc,gamma = get_centers_automatic(v,Nv,coulomb)
    K = basis_activation_func(vc,gamma,normalize,coulomb)
    K,vc,gamma
end

function basis_activation_func(vc,gamma,normalize,coulomb=false)
    if coulomb
        K = normalize ? v -> normalized_squared_exponential_coulomb(v,vc,gamma) : v -> squared_exponential_coulomb(v,vc,gamma) # Use coulomb basis function instead
    else
        K = normalize ? v -> normalized_squared_exponential(v,vc,gamma) : v -> squared_exponential(v,vc,gamma)
    end
end

"""
    vc,γ = get_centers_automatic(v::AbstractVector,Nv::Int,coulomb = false)
"""
function get_centers_automatic(v::AbstractVector,Nv::Int,coulomb = false)
    if coulomb # If Coulomb setting is activated, double the number of basis functions and clip the activation at zero velocity (useful for data that exhibits a discontinuity at v=0, like coulomb friction)
        vc    = linspace(0,maximum(abs.(v)),Nv+2)
        vc    = vc[2:end-1]
        vc    = [-vc[end:-1:1]; vc]
        Nv    = 2Nv
        gamma = (Nv/(abs(vc[1]-vc[end])))^2
    else
        vc    = linspace(minimum(v),maximum(v),Nv)
        gamma = (Nv/(abs(vc[1]-vc[end])))^2
    end
    vc,gamma
end

"""
    vc,γ = get_centers_automatic(v::AbstractMatrix, Nv::AbstractVector{Int}, coulomb=false, coulombdims=0)
"""
function get_centers_automatic(v::AbstractMatrix, Nv::AbstractVector{Int}, coulomb=false, coulombdims=0)
    @assert !coulomb "Coulomb not yet supported for multi-dimensional BFEs"
    @assert size(v,2) == length(Nv) "size(v,2) != length(Nv)"
    dims   = size(v,2)
    minq   = minimum(v,1)[:]
    maxq   = maximum(v,1)[:]
    bounds = [minq maxq]
    get_centers(bounds, Nv)
end

"""
    vc,γ = get_centers(bounds, Nv, coulomb=false, coulombdims=0)
"""
function get_centers(bounds, Nv, coulomb=false, coulombdims=0)
    # TODO: split centers on velocity dim
    @assert !coulomb "Coulomb not yet supported for multi-dimensional BFEs"
    dims     = length(Nv)
    interval = [(bounds[n,2]-bounds[n,1])/Nv[n] for n = 1:dims]
    C = [linspace(bounds[n,1]+interval[n]/2,bounds[n,2]-interval[n]/2,(Nv)[n]) for n = 1:dims]

    Nbasis  = prod(Nv)
    centers = zeros(dims, Nbasis)
    v       = Nbasis
    h       = 1
    for i = 1:dims
        v = v ÷ Nv[i]
        centers[i,:] = vec(repmat(C[i]',v,h))'
        h *= Nv[i]
    end
    centers, (1./interval).^2
end

function get_centers_Kmeans(v, nc::Int; verbose=false)
    iters = 21
    n_state = size(v,2)
    errorvec = zeros(iters)
    params = Array{Float64}(nc*2*n_state,iters)
    methods = [:rand;:kmpp]
    Σ = [zeros(n_state,n_state) for i = 1:nc, j = 1:iters]
    μ = [zeros(n_state) for i = 1:nc, j = 1:iters]

    for iter = 1:iters
        clusterresult = Clustering.kmeans(v', nc; maxiter=200, display=:none, init=iter<iters ? methods[iter%2+1] : :kmcen)
        for i = 1:nc
            si = 1+(i-1)n_state*2
            μ[i,iter] .= clusterresult.centers[:,i]
            Σ[i,iter] .= cov(v[clusterresult.assignments .== i,:])
        end
        errorvec[iter] = clusterresult.totalcost
    end
    verbose && println("Std in errors among initial centers: ", round(std(errorvec),6))
    ind = indmin(errorvec)
    return μ[:,ind], Σ[:,ind]
end


## Utility functions

function quadform(x::AbstractVector,A::AbstractMatrix)
    x⋅(A*x)
end

function quadform(x::AbstractVector,A::AbstractVector)
    s = zero(eltype(x))
    for i in 1:length(x)
        s += A[i]*x[i]^2
    end
    s
end

γ2σ(γ) = √(1./(2γ))
σ2γ(σ) = 1./(2σ.^2)

function γ2σ(γ::Vector{T}) where T <: AbstractVector
    [1./(2γ) for γi in γ]
end

function σ2γ(σ::Vector{T}) where T <: AbstractVector
    [1./(2σi) for σi in σ]
end

function γ2σ(γ::Vector{T}) where T <: AbstractMatrix
    [inv(2γ) for γi in γ]
end

function σ2γ(σ::Vector{T}) where T <: AbstractMatrix
    [inv(2σi) for σi in σ]
end

function meshgrid(a::AbstractVector,b::AbstractVector)
    grid_a = [i for i in a, j in b]
    grid_b = [j for i in a, j in b]
    grid_a, grid_b
end

function meshgrid(a::AbstractVector,b::AbstractVector,c::AbstractVector)
    grid_a = [i for i in a, j in b, k in c]
    grid_b = [j for i in a, j in b, k in c]
    grid_c = [k for i in a, j in b, k in c]
    grid_a, grid_b, grid_c
end
# This code must be run after all type definitions!

# for T in subtypes(BasisFunctionExpansion)
#     (b::T)(x) = b.activation(x)
# end

(b::UniformRBFE)(x)       = b.activation(x)
(b::MultiUniformRBFE)(x)  = b.activation(x)
(b::MultiDiagonalRBFE)(x) = b.activation(x)
(b::MultiRBFE)(x)         = b.activation(x)



## Plot tools ==================================================================
include("plotting.jl")
include("dynamics.jl")
# plot

"""
    x,xm,u,n,m = testdata(T,r=1)

Generate `T` time steps of state-space data where the A-matrix changes from
`A = [0.95 0.1; 0 0.95]` to `A = [0.5 0.05; 0 0.5]` at time `t=T÷2`
`x,xm,u,n,m` = (state,noisy state, input, statesize, inputsize)
`r` is the seed to the random number generator.
"""
function testdata(T_,r=1)
    srand(r)
    n,m      = 2,1
    At_      = [0.95 0.1; 0 0.95]
    Bt_      = reshape([0.2; 1],2,1)
    u        = randn(1,T_)
    x        = zeros(n,T_)
    for t = 1:T_-1
        if t == T_÷2
            At_ = [0.5 0.05; 0 0.5]
        end
        x[:,t+1] = At_*x[:,t] + Bt_*u[:,t] #+ 0.2randn(n)
    end
    xm = x + 0.2randn(size(x));
    x',xm',u',n,m
end

end # module
