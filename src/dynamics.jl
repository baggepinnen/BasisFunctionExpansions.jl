"""
    toeplitz{T}(c::AbstractArray{T},r::AbstractArray{T})

Returns a Toeplitz matrix where `c` is the first column and `r` is the first row.
"""
function toeplitz(c::AbstractArray{T}, r::AbstractArray{T}) where T
    nc = length(c)
    nr = length(r)
    A = zeros(T, nc, nr)
    A[:,1] = c
    A[1,:] = r
    @views for i in 2:nr
        A[2:end,i] = A[1:end - 1,i - 1]
    end
    A
end

"""
    y,A = getARregressor(y::AbstractVector,na::Integer)

Returns a shortened output signal `y` and a regressor matrix `A` such that the least-squares
AR model estimate of order `na` is `y\\A`
"""
function getARregressor(y::AbstractVector, na)
    A = toeplitz(y[na + 1:end], y[na + 1:-1:1])
    y = copy(A[:,1])
    A = A[:,2:end]
    return y, A
end

"""
    getARXregressor(y::AbstractVector,u::AbstractVecOrMat, na, nb)

Returns a shortened output signal `y` and a regressor matrix `A` such that the least-squares
ARX model estimate of order `na,nb` is `y\\A`

Return a regressor matrix used to fit an ARX model on, e.g., the form
`A(z)y = B(z)f(u)`
with output `y` and input `u` where the order of autoregression is `na` and
the order of input moving average is `nb`

# Example
Here we test the model with the Function `f(u) = √(|u|)`
```julia
A     = [1,2*0.7*1,1] # A(z) coeffs
B     = [10,5] # B(z) coeffs
u     = randn(100) # Simulate 100 time steps with Gaussian input
y     = filt(B,A,sqrt.(abs.(u)))
yr,A  = getARXregressor(y,u,2,2) # We assume that we know the system order 2,2
bfe   = MultiUniformRBFE(A,[1,1,4,4,4], normalize=true)
bfa   = BasisFunctionApproximation(yr,A,bfe, 1e-3)
e_bfe = √(mean((yr - bfa(A)).^2))
plot([yr bfa(A)], lab=["Signal" "Prediction"])
```
See README (`?BasisFunctionExpansions`) for more details
"""
function getARXregressor(y::AbstractVector, u::AbstractVecOrMat, na, nb)
    length(nb) == size(u, 2) || throw(ArgumentError("Length of nb must equal number of input signals"))
    m    = max(na, maximum(nb)) + 1 # Start of yr
    @assert m >= 1
    n    = length(y) - m + 1 # Final length of yr
    @assert n <= length(y)
    A    = toeplitz(y[m:m + n - 1], y[m:-1:m - na])
    @assert size(A, 2) == na + 1
    y    = A[:,1] # extract yr
    A    = A[:,2:end]
    for i = 1:length(nb)
        nb[i] <= 0 && continue
        s = m - 1
        A = [A toeplitz(u[s:s + n - 1,i], u[s:-1:s - nb[i] + 1,i])]
    end
    return y, A
end

"size(y) = (T-1, n)"
function matricesn(x, u)
    A = [x[1:end - 1,:] u[1:end - 1,:]]
    y = x[2:end,:]
    y, A
end

"Convenience tyoe for estimation of LPV state-space models"
struct LPVSS
    bfe::BasisFunctionExpansion
    params
    cov
    σ
end

Base.show(io::IO, model::LPVSS) = print(io, "LPVSS model with $(typeof(model.bfe)) and $(length(model.params) * length(model.params[1])) parameters")

"""
    LPVSS(x, u, nc; normalize=true, λ = 1e-3)

Linear Parameter-Varying State-space model. Estimate a state-space model with
varying coefficient matrices `x(t+1) = A(v)x(t) + B(v)u(t)`. Internally a `MultiRBFE` spanning
the space of `X × U` is used. `x` and `u` should have time in first dimension. Centers are
found automatically using k-means, see `MultiRBFE`.

# Examples
```jldoctest
using Plots, BasisFunctionExpansions
x,xm,u,n,m = BasisFunctionExpansions.testdata(1000)
nc         = 10 # Number of centers
model      = LPVSS(x, u, nc; normalize=true, λ = 1e-3) # Estimate a model
xh         = model(x,u) # Form prediction

eRMS       = √(mean((xh[1:end-1,:]-x[2:end,:]).^2))

plot(xh[1:end-1,:], lab="Prediction", c=:red, layout=2)
plot!(x[2:end,:], lab="True", c=:blue); gui()
eRMS <= 0.37

# output

true
```
"""
function LPVSS(x, u, nc; normalize=true, λ=1e-3)
    y, A         = matricesn(x, u)
    bfe         = MultiRBFE(A, nc; normalize=normalize) # A is sched/regressor matrix
    params, cov, σ = fit_ss(x, u, A, bfe, λ)
    LPVSS(bfe, params, cov, σ)
end

"""
    LPVSS(x, u, v, nc; normalize=true, λ = 1e-3)

Linear Parameter-Varying State-space model. Estimate a state-space model with
varying coefficient matrices `x(t+1) = A(v)x(t) + B(v)u(t)`. Internally a `MultiRBFE` or
`UniformRBFE` spanning the space of `v` is used, depending on the dimensionality of
`v`. `x`, `u` and `v` should have time in first dimension. Centers are found automatically
using k-means, see `MultiRBFE`.

# Examples
```jldoctest
using Plots, BasisFunctionExpansions
T          = 1000
x,xm,u,n,m = BasisFunctionExpansions.testdata(T)
nc         = 4
v          = 1:T
model      = LPVSS(x, u, v, nc; normalize=true, λ = 1e-3)
xh         = model(x,u,v)

eRMS       = √(mean((xh[1:end-1,:]-x[2:end,:]).^2))

plot(xh[1:end-1,:], lab="Prediction", c=:red, layout=(2,1))
plot!(x[2:end,:], lab="True", c=:blue); gui()
eRMS <= 0.26

# output

true
```
"""
function LPVSS(x::AbstractArray, u::AbstractArray, v::AbstractVecOrMat, nc; normalize=true, λ=1e-3)
    if isa(v, AbstractMatrix)
        bfe  = MultiRBFE(v, nc; normalize=normalize)
    else
        bfe  = UniformRBFE(v, nc; normalize=normalize)
    end
    params, cov, σ = fit_ss(x, u, v, bfe, λ)
    LPVSS(bfe, params, cov, σ)
end

function mega_regressor(bfe, v, A)
    nc = length(bfe.μ) ÷ (ndims(v) > 1 ? size(v, 2) : 1)
    ϕ = bfe(v)
    if isa(ϕ, Vector)
        ϕ = ϕ'
    end
    ϕ = repeat(ϕ, 1, size(A, 2)) # Extend activations from nc to nc×(n+m)
    ϕ = ϕ .* repeat(A, 1, nc)  # Extend regressor from n+m to nc×(n+m)
end

shorten_v(v::AbstractVector) = v[1:end - 1]
shorten_v(v::AbstractMatrix) = v[1:end - 1,:]

function fit_ss(x, u, v, bfe, λ)
    y, A  = matricesn(x, u) # A is sched matrix [x[1:end-1,:] u[1:end-1,:]]
    if size(v, 1) > size(A, 1)
        v = shorten_v(v)
    end
    ϕ = mega_regressor(bfe, v, A)
    p = size(ϕ, 2)
    B = qr(λ == 0 ? ϕ : [ϕ; λ * I])
    params = mapslices(y, dims=1) do y
        if λ == 0
            x = B \ y
        else
            x = B \ [y;zeros(p)]
        end
    end # We now have n vectors of nc(n+m) parameters = nc(n+m)×n
    σ = [std(y[:,i] - ϕ * params[:,i]) for i = 1:size(params, 2)]
    ATA = Cholesky(B.R, :U, 0) # TODO: Isn't R already a cholesky factor of R'R??
    cov = λ == 0 ? inv(ATA) : ATA \ (ϕ'ϕ) / Matrix(ATA) # / not defined for factorizations https://github.com/JuliaLang/julia/issues/12436

    # icov = cholfact(Hermitian(q))
    # icov = cholfact(Symmetric(q[:R]'q[:R]))
    return params, cov, σ
end

"""
    predict(model::LPVSS, x::AbstractMatrix, u, v=[x u])

If no `v` provided, return a prediction of the output `x'` given the state `x` and input `u`
This function is called when a `model::LPVSS` object is called like `model(x,u)`

Provided `v`, return a prediction of the output `x'` given the state `x`, input `u` and
scheduling parameter `v`
"""
function predict(model::LPVSS, x::AbstractMatrix, u, v=[x u])
    A = [x u]
    ϕ = mega_regressor(model.bfe, v, A)
    y = similar(x)
    for i = 1:size(y, 2)
        y[:,i] = ϕ * model.params[:,i]
    end
    y
end

function predict(model::LPVSS, x::AbstractVector, u, v)
    A = [x' u']
    ϕ = mega_regressor(model.bfe, v, A)
    y = map(1:length(x)) do i
        vecdot(ϕ, model.params[:,i])
    end
end


(model::LPVSS)(x,u)                 = predict(model, x, u)
(model::LPVSS)(x,u,v)               = predict(model, x, u, v)
(model::LPVSS)(x::AbstractMatrix,u) = predict(model, x, u, [x u])
(model::LPVSS)(x::AbstractVector,u) = predict(model, x, u, [x' u'])


function AB_covariance(model::LPVSS)
    ATA = model.cov
    σ²  = model.σ^2
    # ATA has size (n+m)×(n+m)
    # Full covariance matrix should have size n(n+m)×n(n+m), hence ATA should be composed in
    # a block form and multiplied with the corresponding σ²

end

"""
    output_variance(model::LPVSS, x::AbstractVector, u::AbstractVector, v=[x u])

Return a vector of prediction variances. Note, no covariance between dimensions in output is
provided
"""
function output_variance(model::LPVSS, x::AbstractVector, u::AbstractVector, v=[x u])
    A = [x' u']
    ϕ = mega_regressor(model.bfe, v, A)[:]
    mean_var = [σ^2 * ϕ'model.cov * ϕ for σ in model.σ]
    pred_var = [model.σ[i] + sqrt(mean_var[i]) for i = 1:length(model.σ)]
end
