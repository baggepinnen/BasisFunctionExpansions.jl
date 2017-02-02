module BasisFunctionExpansions

# From LPVSpectral =============================================================

"""basis_activation_func(V,Nv,normalize,coulomb)

Returns a func v->ϕ(v) ∈ ℜ(Nv) that calculates the activation of `Nv` basis functions spread out to cover V nicely. If coulomb is true, then we get twice the number of basis functions, 2Nv
"""
function basis_activation_func(V,Nv,normalize,coulomb)
    if coulomb # If Coulomb setting is activated, double the number of basis functions and clip the activation at zero velocity (useful for data that exhibits a discontinuity at v=0, like coulomb friction)
        vc      = linspace(0,maximum(abs(V)),Nv+2)
        vc      = vc[2:end-1]
        vc      = [-vc[end:-1:1]; vc]
        Nv      = 2Nv
        gamma   = Nv/(abs(vc[1]-vc[end]))
        K       = normalize ? V -> _Kcoulomb_norm(V,vc,gamma) : V -> _Kcoulomb(V,vc,gamma) # Use coulomb basis function instead
    else
        vc      = linspace(minimum(V),maximum(V),Nv)
        gamma   = Nv/(abs(vc[1]-vc[end]))
        K       = normalize ? V -> _K_norm(V,vc,gamma) : V -> _K(V,vc,gamma)
    end
end



@inline _K(V,vc,gamma) = exp(-gamma*(V.-vc).^2)

@inline function _K_norm(V,vc,gamma)
    r = _K(V,vc,gamma)
    r ./=sum(r)
end

@inline _Kcoulomb(V,vc,gamma) = _K(V,vc,gamma).*(sign(V) .== sign(vc))

@inline function _Kcoulomb_norm(V,vc,gamma)
    r = _Kcoulomb(V,vc,gamma)
    r ./=sum(r)
end




# From Robotlib  ===============================================================


function mnorm_pdf(p,c,S)
    d  = p[:]-c
    exp(-vecdot(d.^2,S))
end

function basisParametersNd(p,centers, sigma, velocity::Int=0, normalize=true)
    @assert size(p,2) == length(sigma)
    N       = size(p,2)
    N_basis = size(centers,2)
    y       = zeros(N_basis)
    iSIGMA  = sigma.^-2

    if velocity > 0
        y = [sign(centers[velocity,i]) == sign(p[velocity]) ? mnorm_pdf(p,centers[:,i],iSIGMA) : 0 for i = 1:N_basis]
    else
        y = [mnorm_pdf(p,centers[:,i],iSIGMA) for i = 1:N_basis]
    end

    if normalize
        y ./= (sum(y)+0.000001)
    end
    return y
end


function getCenters(n_basis, bounds)
    # TODO: split centers on velocity dim
    warn("Not yet split in velocity dimension!")
    N = length(n_basis);
    interval = [(bounds[n,2]-bounds[n,1])/(n_basis)[n] for n = 1:N];
    C = [linspace(bounds[n,1]+interval[n]/2,bounds[n,2]-interval[n]/2,(n_basis)[n]) for n = 1:N];

    Nbasis = prod(n_basis)
    centers = zeros(N, Nbasis)
    v = Nbasis
    h = 1
    for i = 1:N
        v = convert(Int64,v / n_basis[i])
        centers[i,:] = vec(repmat(C[i]',v,h))'
        h *= n_basis[i]
    end
    centers
end


function getCenters(n_basis::Vector{Int64}, q::Matrix{Float64}, q̇::Matrix{Float64})
    n_joints = size(q,2)
    minq = minimum([q q̇],1)
    maxq = maximum([q q̇],1)
    centers = Array{Matrix{Float64}}(n_joints)
    for i = 1:n_joints
        bounds = [minq[[i i+n_joints]]' maxq[[i i+n_joints]]']
        centers[i] = getCenters(n_basis, bounds)
    end
    return centers
end



# From gym =====================================================================


cp = linspace(-1,0,4)
cv = linspace(-0.3,0.3,4)

grid1 = meshgrid(cp,cv)
const c1 = [grid1[1][:] grid1[2][:]]
const P = size(c1,1)
const gamma = 0.5P./(c1[end,:] - c1[1,:])'

function ϕ(s)
    a = exp(-sum((gamma.*(s'.-c1)).^2,2))[:]
    a ./= (sum(a)+1e-6)
    a
end



# Curated ======================================================================

## Types
abstract BasisFunctionExpansion{dim}
(b::BasisFunctionExpansion)(x) = b.activation(x)


immutable UniformRBFE{1} <: BasisFunctionExpansion{1}
    activation::Function
    μ::Array{Float64,1}
    σ::Float64

    """RBFE{1}(μ::AbstractVector, σ::Real, activation=squared_exponential)
    Supply all parameters
    """
    function RBFE{1}(μ::AbstractVector, σ::Real, activation=squared_exponential)
        new(squared_exponential,μ,σ)
    end

    """RBFE{1}(v::AbstractVector, Nv::Int; normalize=false, coulomb=false)
    Supply scheduling signal and number of basis functions for automatic selection of centers and widths
    """
    function RBFE{1}(v::AbstractVector, Nv::Int; normalize=false, coulomb=false)
        activation, μ, γ = basis_activation_func(v,Nv,normalize,coulomb)
        new(squared_exponential,μ,γ2σ(γ))
    end
end


## Squared exponential functions

squared_exponential(v,vc,gamma) = exp(-gamma*(v.-vc).^2)

function normalized_squared_exponential(v,vc,gamma)
    r = squared_exponential(v,vc,gamma)
    r ./= (sum(r) + 1e-8)
end

squared_exponential_coulomb(v,vc,gamma) = squared_exponential(v,vc,gamma).*(sign(v) .== sign(vc))

function normalized_squared_exponential_coulomb(v,vc,gamma)
    r = squared_exponential_coulomb(v,vc,gamma)
    r ./= (sum(r) + 1e-8)
end

"""basis_activation_func(v,Nv,normalize,coulomb)

Returns a func v->ϕ(v) ∈ ℜ(Nv) that calculates the activation of `Nv` basis functions spread out to cover v nicely. If coulomb is true, then we get twice the number of basis functions, 2Nv
"""
function basis_activation_func_automatic(v,Nv,normalize,coulomb=false)
    vc,gamma = get_centers_automatic(v,Nv,coulomb)
    K,vc,gamma = basis_activation_func(vc,gamma,normalize,coulomb)
end

function basis_activation_func(vc,gamma,normalize,coulomb=false)
    if coulomb
        K = normalize ? v -> normalized_squared_exponential_coulomb(v,vc,gamma) : v -> squared_exponential_coulomb(v,vc,gamma) # Use coulomb basis function instead
    else
        K = normalize ? v -> normalized_squared_exponential(v,vc,gamma) : v -> squared_exponential(v,vc,gamma)
    end
    K,vc,gamma
end


function get_centers_automatic(v::AbstractVector,Nv::Int,coulomb = false)
    if coulomb # If Coulomb setting is activated, double the number of basis functions and clip the activation at zero velocity (useful for data that exhibits a discontinuity at v=0, like coulomb friction)
        vc    = linspace(0,maximum(abs(v)),Nv+2)
        vc    = vc[2:end-1]
        vc    = [-vc[end:-1:1]; vc]
        Nv    = 2Nv
        gamma = Nv/(abs(vc[1]-vc[end]))
    else
        vc    = linspace(minimum(v),maximum(v),Nv)
        gamma = Nv/(abs(vc[1]-vc[end]))
    end
    vc,gamma
end

function get_centers_automatic(v::AbstractMatrix, Nv::AbstractVector{Int64}, coulomb=false, coulombdims=0)
    @assert !coulomb "Coulomb not yet supported for multi-dimensional BFEs"
    dims    = size(v,2)
    minq    = minimum(v,1)
    maxq    = maximum(v,1)
    N       = length(Nv)
    Nbasis  = prod(Nv)
    centers = Array{Float64,3}(N,Nbasis,dims)
    for i = 1:dims
        bounds = [minq[[i i+dims]]' maxq[[i i+dims]]']
        centers[:,:,i] = get_centers(bounds, Nv)
    end
    return centers
end

function get_centers(bounds, Nv, coulomb=false, coulombdims=0)
    # TODO: split centers on velocity dim
    @assert !coulomb "Coulomb not yet supported for multi-dimensional BFEs"
    N = length(Nv);
    interval = [(bounds[n,2]-bounds[n,1])/(Nv)[n] for n = 1:N];
    C = [linspace(bounds[n,1]+interval[n]/2,bounds[n,2]-interval[n]/2,(Nv)[n]) for n = 1:N];

    Nbasis = prod(Nv)
    centers = zeros(N, Nbasis)
    v = Nbasis
    h = 1
    for i = 1:N
        v = v ÷ Nv[i]
        centers[i,:] = vec(repmat(C[i]',v,h))'
        h *= Nv[i]
    end
    centers
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

end # module
