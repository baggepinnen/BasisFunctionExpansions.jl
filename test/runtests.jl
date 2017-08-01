using BasisFunctionExpansions
using Base.Test

# write your own tests here
@test BasisFunctionExpansions.get_centers_automatic(1:10,5,false)[1] |> length == 5
@test BasisFunctionExpansions.get_centers_automatic(1:10,5,true)[1] |> length == 10

@test BasisFunctionExpansions.get_centers_automatic([1:10 2:11],[5,6])[1] |> size == (2,5*6)


rbf = UniformRBFE(1:3, 5)
@test size(rbf(randn(10))) == (10,5)
@test size(rbf(randn())) == (5,)

x = linspace(0,4,50)
a = rbf(x)
@test a[1] == a[end] # Should be completely symmetric in this case
@test isa(rbf, BasisFunctionExpansion{1})
@test !isa(rbf, BasisFunctionExpansion{2})



v   = [1:3 2:4]
Nv  = [2,3]
rbf = MultiUniformRBFE(v,Nv)
@test rbf(randn(2)) |> size == (prod(Nv),)
@test rbf(randn(10,2)) |> size == (10,prod(Nv))



# Single dim
N    = 1000
v    = linspace(0,10,N)
y    = randn(N)
y    = filt(ones(500)/500,[1],y)
Nv   = 10
rbf  = UniformRBFE(v,Nv, normalize=true)
bfa  = BasisFunctionApproximation(y,v,rbf,1)
yhat = bfa(v)
e = y-yhat
@test √(mean(e.^2)) < 0.012


# Multidim
N    = 1000
x    = linspace(0,2pi-0.2,N)
v    = [cos.(x) sin.(x)].*x
y    = randn(N)
y    = filt(ones(500)/500,[1],y)
Nv   = [10,10]
rbf  = MultiUniformRBFE(v,Nv, normalize=true)
bfa  = BasisFunctionApproximation(y,v,rbf,0.0001)
yhat = bfa(v)
e = y-yhat
@test isapprox.(sum(rbf(randn(10,2)), 2), 1, atol=1e-7) |> all
@test √(mean(e.^2)) < 0.08
# plot(rbf)



# Multidim Diagonal
N    = 1000
x    = linspace(0,2pi-0.2,N)
v    = [cos.(x) sin.(x)].*x
y    = randn(N)
y    = filt(ones(500)/500,[1],y)
Nc   = 8
rbf  = BasisFunctionExpansions.MultiDiagonalRBFE(v,Nc, normalize=true)
bfa  = BasisFunctionApproximation(y,v,rbf,0.0001)
yhat = bfa(v)
e = y-yhat
√(mean(e.^2))

# @test isapprox.(sum(rbf(randn(10,2)), 2), 1, atol=1e-7) |> all
@test √(mean(e.^2)) < 0.02



# Multidim
# using BasisFunctionExpansions, Plots
N    = 1000
x    = linspace(0,2pi-0.2,N)
v    = [cos.(x) sin.(x)].*x
y    = randn(N)
y    = filt(ones(500)/500,[1],y)
Nc   = 8
rbf  = MultiRBFE(v,Nc, normalize=true)
bfa  = BasisFunctionApproximation(y,v,rbf,0.0001)
yhat = bfa(v)
e = y-yhat
√(mean(e.^2))

# @test isapprox.(sum(rbf(randn(10,2)), 2), 1, atol=1e-7) |> all
@test √(mean(e.^2)) < 0.02
# scatter3d(v[:,1],v[:,2],y, lab="Signal")
# scatter3d!(v[:,1],v[:,2],yhat, lab="Reconstruction")



# Dynamics
# using BasisFunctionExpansions
# using Plots; plotly()
T = 100
A = [1,2*0.7*1,1]
B = [10,5]
u = 1randn(T)
y = filt(B,A,sqrt.(abs.(u)))

yr,A = getARXregressor(y,u,3,2)

rbf = MultiUniformRBFE(A,[2,2,4,4,4], normalize=true)
bfa = BasisFunctionApproximation(yr,A,rbf, 1e-4)
e = √(mean((yr - bfa(A)).^2))
x = A\yr
e2 = √(mean((yr - A*x).^2))

# plot([yr bfa(A) A*x]); gui()

@test e < e2
@test e < 0.04


include("test_LPVSS.jl")
