module BasisFunctionExpansions
export BasisFunctionExpansion, UniformRBFE, MultiUniformRBFE, BasisFunctionApproximation, get_centers, get_centers_multi, get_centers_automatic, quadform, γ2σ, σ2γ


## Types
abstract type BasisFunctionExpansion{N} end

function Base.show(b::BasisFunctionExpansion)
    s = string(typeof(b),"\n")
    for fn in fieldnames(b)
        s *= string(fn) * ": " * string(getfield(b,fn)) * "\n"
    end
    println(s)
end

Base.display(b::BasisFunctionExpansion) = show(b)

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
"""
function UniformRBFE(v::AbstractVector, Nv::Int; normalize=false, coulomb=false)
    activation, μ, γ = basis_activation_func_automatic(v,Nv,normalize,coulomb)
    UniformRBFE(activation,μ,γ2σ(γ))
end


### RBFE =======================================================================
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
    MultiUniformRBFE{size(μ,2)}(v->activation(v,μ,σ2γ(Σ)),μ,Σ)
end

"""
    MultiUniformRBFE(v::AbstractVector, Nv::Vector{Int}; normalize=false, coulomb=false)

Supply scheduling signal and number of basis functions For automatic selection of centers and widths
"""
function MultiUniformRBFE(v::AbstractMatrix, Nv::AbstractVector{Int}; normalize=false, coulomb=false)
    @assert !coulomb "Coulomb not yet supported for multi-dimensional BFEs"
    @assert length(Nv) == size(v,2)
    activation, μ, γ = basis_activation_func_automatic(v,Nv,normalize,coulomb)
    MultiUniformRBFE{length(Nv)}(activation,μ,γ2σ.(γ))
end






## Squared exponential functions

squared_exponential(v::Real,vc,gamma) = exp.(-gamma*(v.-vc).^2)
squared_exponential(v::AbstractVector,vc,gamma::Number) = exp.(-gamma*(v.-vc').^2)
squared_exponential(v::AbstractVector,vc,gamma::AbstractVector) = exp.(-((vc'.-v').^2)*gamma)
function squared_exponential(v::AbstractMatrix,vc,gamma::AbstractVector)
    a = Matrix{Float64}(size(v,1),size(vc,2))
    for i = 1:size(v,1)
        a[i,:] = exp.(-sum(gamma.*(v[i,:].-vc).^2,1))
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

Returns a func v->ϕ(v) ∈ ℜ(Nv) that calculates the activation of `Nv` basis functions spread out to cover v nicely. If coulomb is true, then we get twice the number of basis functions, 2Nv
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

(b::UniformRBFE)(x) = b.activation(x)
(b::MultiUniformRBFE)(x) = b.activation(x)



## Plot tools ==================================================================
include("plotting.jl")
# plot

end # module
