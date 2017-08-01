using Plots, Base.Test
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

nc = 20

model = BasisFunctionExpansions.LPVSS(x, u, nc; normalize=true, λ = 1e-3)
xh = model(x,u)

eRMS = √(mean((xh[1:end-1,:]-x[2:end,:]).^2))
println("RMS error: ", eRMS)
@test eRMS <= 1.1*0.6292304108390538
# plotlyjs(show=true)
# plot(xh[1:end-1,:], lab="Prediction", c=:red, layout=2)
# plot!(x[2:end,:], lab="True", c=:blue)
