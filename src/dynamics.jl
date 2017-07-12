function toeplitz{T}(c::Array{T},r::Array{T})
    nc     = length(c)
    nr     = length(r)
    A      = zeros(T, nc, nr)
    A[:,1] = c
    A[1,:] = r
    for i in 2:nr
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
    y,A = getARXregressor(y::AbstractVector,u::AbstractVecOrMat, na::Integer, nb::Integer)

Returns a shortened output signal `y` and a regressor matrix `A` such that the least-squares ARX model estimate of order `na,nb` is `y\A` 
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
