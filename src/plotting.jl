
using RecipesBase

@recipe function f(b::UniformRBFE)
    Nv = size(b.μ,1)
    dist = b.μ[end]-b.μ[1]
    v = linspace(b.μ[1]-0.1dist, b.μ[end]+0.1dist,200)
    seriestype := :line
    color_palette --> [HSV(i,1,0.7) for i in linspace(0,255,Nv)]
    xguide --> "Scheduling signal \$v\$"
    title --> "Basis function expansion"
    lab --> ""
    v, b(v)
end

@recipe function f(rbf::MultiUniformRBFE, style=:default)
    # color_palette --> [HSV(i,1,0.7) for i in linspace(0,255,Nv)]
    xguide --> "Scheduling signal \$v_1\$"
    yguide --> "Scheduling signal \$v_2\$"
    title --> "Basis function expansion"
    lab --> ""
    seriestype --> :surface

    c       = rbf.μ
    minb    = minimum(c,2)
    maxb    = maximum(c,2)
    dist    = maxb-minb
    Npoints = 50
    v = [linspace(mi, ma, Npoints) for (mi,ma) in zip(minb,maxb)]
    vg = meshgrid(v...)
    v  = d[:seriestype] == :surface ? vg : v
    if style == :default
        bg = map(vg...) do v1,v2
            maximum(rbf([v1,v2]))
        end
        (v..., bg)
    elseif style == :full
        colorbar --> false
        seriesalpha --> 0.5
        bg = map(vg...) do v1,v2
            rbf([v1,v2])
        end
        bg = cat(3,bg...)
        bg = reshape(bg, size(bg,1), Npoints, Npoints)
        for i = 1:size(bg,1)
            @series (v..., bg[i,:,:])
        end
    end

end
