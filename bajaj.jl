# Experiments with fitting quadratic surfaces

module Bajaj

using LinearAlgebra


# Rational Bezier

struct RationalCurve
    cp::Vector{Vector{Float64}}
    w::Vector{Float64}
end

function bernstein(n, u)
    coeff = [1.0]
    for j in 1:n
        saved = 0.0
        for k in 1:j
            tmp = coeff[k]
            coeff[k] = saved + tmp * (1.0 - u)
            saved = tmp * u
        end
        push!(coeff, saved)
    end
    coeff
end

function evalRational(curve, u)
    n = length(curve.cp) - 1
    coeff = bernstein(n, u)
    p = [0, 0, 0]
    w = 0
    for k in 1:n+1
        p += curve.cp[k] * curve.w[k] * coeff[k]
        w += curve.w[k] * coeff[k]
    end
    p / w
end

function evalRationalDerivative(curve, u)
    n = length(curve.cp) - 1
    dcp = []
    dwi = []
    for i in 1:n
        push!(dcp, (curve.cp[i+1] * curve.w[i+1] - curve.cp[i] * curve.w[i]) * n)
        push!(dwi, (curve.w[i+1] - curve.w[i]) * n)
    end

    coeff = bernstein(n, u)
    p = [0, 0, 0]
    w = 0
    for k in 1:n+1
        p += curve.cp[k] * curve.w[k] * coeff[k]
        w += curve.w[k] * coeff[k]
    end

    coeff = bernstein(n - 1, u)
    dp = [0, 0, 0]
    dw = 0
    for k in 1:n
        dp += dcp[k] * coeff[k]
        dw += dwi[k] * coeff[k]
    end

    (dp - p / w * dw) / w
end


# Constraints

function pointConstraint(p)
    (x, y, z) = p
    [x^2, y^2, z^2, y*z, x*z, x*y, x, y, z, 1]
end

function normalConstraint(p, n)
    (x, y, z) = p
    derivatives = ([2x, 0, 0, 0, z, y, 1, 0, 0, 0],
                   [0, 2y, 0, z, 0, x, 0, 1, 0, 0],
                   [0, 0, 2z, y, x, 0, 0, 0, 1, 0])
    i = findmax(map(abs, n))[2] # index of max. absolute value in n
    j = mod1(i + 1, 3)
    k = mod1(j + 1, 3)
    (n[i] * derivatives[j] - n[j] * derivatives[i],
     n[i] * derivatives[k] - n[k] * derivatives[i])
end

function curveConstraint(curve)
    map([0, 0.25, 0.5, 0.75, 1]) do u
        pointConstraint(evalRational(curve, u))
    end
end

"""
    curveNormalConstraint(curve, n0, n1)

Fixes the normal sweep as a linear function from `n0` (at `u=0`) to `n1` (at `u=1`).
Returns 4 equations.
"""
function curveNormalConstraint(curve, n0, n1)
    map([0, 1/3, 2/3, 1]) do u
        n = n0 * (1 - u) + n1 * u
        t = evalRationalDerivative(curve, u)
        (x, y, z) = evalRational(curve, u)
        derivatives = ([2x, 0, 0, 0, z, y, 1, 0, 0, 0],
                       [0, 2y, 0, z, 0, x, 0, 1, 0, 0],
                       [0, 0, 2z, y, x, 0, 0, 0, 1, 0])
        i = findmax(map(abs, t))[2] # index of max. absolute value in t
        j = mod1(i + 1, 3)
        k = mod1(j + 1, 3)
        derivatives[j] * n[k] - n[j] * derivatives[k]
    end
end

function fitQuadratic(curve, n0, n1)
    rows = 10
    A = zeros(rows, 10)
    cc = curveConstraint(curve)
    for i in 1:5
        A[i,:] = cc[i]
    end
    cnc = curveNormalConstraint(curve, n0, n1)
    for i in 1:4
        A[5+i,:] = cnc[i]
    end
    A[rows,7:9] = ones(3)
    b = zeros(rows)
    b[rows] = 1
    x = qr(A, Val(true)) \ b
    println("S: $(svd(A).S)")
    println("x: $x\nError: $(maximum(map(abs,A*x-b)))")
    x
end

evalQuadratic(qf, p) = sum(pointConstraint(p) .* qf)

function test()
    resolution = 50
    curve = RationalCurve([[0., 0, 0], [0, 1, 0], [1, 1, 0]], [1, sqrt(2)/2, 1])
    n0 = [0.1, 0, 1]
    n1 = [0, 0.1, 1]
    res = 30
    f = fitQuadratic(curve, normalize(n0), normalize(n1))
    dc = Main.DualContouring.isosurface(0, ([-1,-1,-1], [2,2,2]), (res, res, res)) do p
        evalQuadratic(f, p)
    end
    open("/tmp/curve.obj", "w") do f
        for i in 0:resolution
            u = i / resolution
            p = evalRational(curve, u)
            println(f, "v $(p[1]) $(p[2]) $(p[3])")
        end
        p0 = curve.cp[1] + n0
        p1 = curve.cp[3] + n1
        println(f, "v $(p0[1]) $(p0[2]) $(p0[3])")
        println(f, "v $(p1[1]) $(p1[2]) $(p1[3])")
        for i in 1:resolution
            println(f, "l $i $(i+1)")
        end
        println(f, "l 1 $(resolution+2)")
        println(f, "l $(resolution+1) $(resolution+3)")
    end
    Main.DualContouring.writeOBJ(dc..., "/tmp/surface.obj")
end

end # module
