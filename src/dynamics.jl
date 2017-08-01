"""
    toeplitz{T}(c::AbstractArray{T},r::AbstractArray{T})

Returns a Toeplitz matrix where `c` is the first column and `r` is the first row.
"""
function toeplitz{T}(c::AbstractArray{T},r::AbstractArray{T})
    nc = length(c)
    nr = length(r)
    A = zeros(T, nc, nr)
    A[:,1] = c
    A[1,:] = r
    @views for i in 2:nr
        A[2:end,i] = A[1:end-1,i-1]
    end
    A
end

"""
    y,A = getARregressor(y::AbstractVector,na::Integer)

Returns a shortened output signal `y` and a regressor matrix `A` such that the least-squares AR model estimate of order `na` is `y\A`
"""
function getARregressor(y::AbstractVector,na)
    A = toeplitz(y[na+1:end],y[na+1:-1:1])
    y = copy(A[:,1])
    A = A[:,2:end]
    return y,A
end

"""
    getARXregressor(y::AbstractVector,u::AbstractVecOrMat, na, nb)

Returns a shortened output signal `y` and a regressor matrix `A` such that the least-squares ARX model estimate of order `na,nb` is `y\A` 

Return a regressor matrix used to fit an ARX model on, e.g., the form
`A(z)y = B(z)f(u)`
with output `y` and input `u` where the order of autoregression is `na` and
input moving average is `nb`

# Example
Here we test the model with the Function `f(u) = √(|u|)`
```julia
A     = [1,2*0.7*1,1] # A(z) coeffs
B     = [10,5] # B(z) coeffs
u     = randn(100) # Simulate 100 time steps with Gaussian input
y     = filt(B,A,sqrt.(abs.(u)))
yr,A  = getARXregressor(y,u,3,2) # We assume that we know the system order 3,2
bfe   = MultiUniformRBFE(A,[2,2,4,4,4], normalize=true)
bfa   = BasisFunctionApproximation(yr,A,bfe, 1e-3)
e_bfe = √(mean((yr - bfa(A)).^2)) # (0.005174261451622258)
plot([yr bfa(A)], lab=["Signal" "Prediction"])
```
See README for more details
"""
function getARXregressor(y::AbstractVector,u::AbstractVecOrMat, na, nb)
    assert(length(nb) == size(u,2))
    m    = max(na+1,maximum(nb))
    n    = length(y) - m+1
    offs = m-na-1
    A    = toeplitz(y[offs+na+1:n+na+offs],y[offs+na+1:-1:1])
    y    = copy(A[:,1])
    A    = A[:,2:end]
    for i = 1:length(nb)
        offs = m-nb[i]
        A = [A toeplitz(u[nb[i]+offs:n+nb[i]+offs-1,i],u[nb[i]+offs:-1:1+offs,i])]
    end
    return y,A
end

"size(y) = (T-1, n)"
function matricesn(x,u)
    A = [x[1:end-1,:] u[1:end-1,:]]
    y = x[2:end,:]
    y,A
end

"""
    LPVSS(x, u, nc; normalize=true, λ = 1e-3)

Linear Parameter-Varying State-space model. Estimate a state-space model with
varying coefficient matrices `x(t+1) = A(v)x(t) + B(v)u(t)`. Internally a `MultiRBFE` is used. `x` and `u` should have time in first dimension. Centers are found
automatically using k-means, see `MultiRBFE`.

# Example usage
```julia
using Plots
function testdata(T_)
    srand(1)

    n,m      = 2,1
    At_      = [0.95 0.1; 0 0.95]
    Bt_      = reshape([0.2; 1],2,1)
    u        = randn(1,T_)
    x        = zeros(n,T_)
    for t = 1:T_-1
        if t == T_÷2
            At_ = [0.5 0.05; 0 0.5]
        end
        x[:,t+1] = At_*x[:,t] + Bt_*u[:,t] + 0.2randn(n)
    end
    xm = x + 0.2randn(size(x));
    x',xm',u',n,m
end

x,xm,u,n,m = testdata(1000)
nc         = 10 # Number of centers
model      = LPVSS(x, u, nc; normalize=true, λ = 1e-3) # Estimate a model
xh         = model(x,u) # Form prediction

println("RMS error: ", √(mean((xh[1:end-1,:]-x[2:end,:]).^2)))

plot(xh[1:end-1,:], lab="Prediction", c=:red, layout=2)
plot!(x[2:end,:], lab="True", c=:blue); gui()
```
"""
struct LPVSS
    bfe::BasisFunctionExpansion
    bfas::Vector{BasisFunctionApproximation}
end

function LPVSS(x, u, nc; normalize=true, λ = 1e-3)
    y,A  = matricesn(x,u)
    bfe  = MultiRBFE(A, nc; normalize=normalize) # A is sched/regressor matrix
    bfas = mapslices(y->BasisFunctionApproximation(y,A,bfe, λ), y, 1)[:]
    LPVSS(bfe, bfas)
end

function predict(model::LPVSS, x::AbstractMatrix, u)
    A = [x u]
    y = similar(x)
    for i = 1:size(y,2)
        y[:,i] = model.bfas[i](A)
    end
    y
end

function predict(model::LPVSS, x::AbstractVector, u)
    A = [x' u']
    y = map(1:length(x)) do i
        model.bfas[i](A)
    end
end

(model::LPVSS)(x,u) = predict(model, x, u)
