using BasisFunctionExpansions
using Base.Test

# write your own tests here
@test BasisFunctionExpansions.get_centers_automatic(1:10,5,false)[1] |> length == 5
@test BasisFunctionExpansions.get_centers_automatic(1:10,5,true)[1] |> length == 10

@test BasisFunctionExpansions.get_centers_automatic([1:10 2:11],[5,6])[1] |> size == (5*6,2)


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
