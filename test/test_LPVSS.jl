# using Base.Test
function testdata(T_)
    # srand(1)

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

T = 1000
x,xm,u,n,m = testdata(T)

nc = 2

model = LPVSS(x, u, nc; normalize=true, λ = 1e-3)
xh = model(x,u)

eRMS = √(mean((xh[1:end-1,:]-x[2:end,:]).^2))
println("RMS error: ", eRMS)
@test eRMS <= 0.37

# using Plots
# plotlyjs(show=true)
# plot(xh[1:end-1,:], lab="Prediction", c=:red, layout=(2,1))
# plot!(x[2:end,:], lab="True", c=:blue)

nc = 4
v = 1:T
model = LPVSS(x, u, v, nc; normalize=true, λ = 1e-3)
xh = model(x,u,v)

eRMS = √(mean((xh[1:end-1,:]-x[2:end,:]).^2))
println("RMS error: ", eRMS)
@test eRMS <= 0.26

# using Plots
# plotlyjs(show=true)
# plot(xh[1:end-1,:], lab="Prediction", c=:red, layout=(2,1))
# plot!(x[2:end,:], lab="True", c=:blue)
