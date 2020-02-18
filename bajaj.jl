# Experiments with fitting quadratic & cubic surfaces

module Bajaj

using DelimitedFiles
using LinearAlgebra
using Printf

normal_scaling = 50


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

function writeCurve(curve, n0, n1, resolution, filename)
    open(filename, "w") do f
        for i in 0:resolution
            u = i / resolution
            p = evalRational(curve, u)
            println(f, "v $(p[1]) $(p[2]) $(p[3])")
        end
        p0 = curve.cp[1] + n0 * normal_scaling
        p1 = curve.cp[3] + n1 * normal_scaling
        println(f, "v $(p0[1]) $(p0[2]) $(p0[3])")
        println(f, "v $(p1[1]) $(p1[2]) $(p1[3])")
        for i in 1:resolution
            println(f, "l $i $(i+1)")
        end
        println(f, "l 1 $(resolution+2)")
        println(f, "l $(resolution+1) $(resolution+3)")
    end
end


# Quadratic constraints

function pointConstraint(p, ::Val{2})
    (x, y, z) = p
    [x^2, y^2, z^2, y*z, x*z, x*y, x, y, z, 1]
end

function normalConstraintExplicit(p, ::Val{2})
    (x, y, z) = p
    ([2x, 0, 0, 0, z, y, 1, 0, 0, 0],
     [0, 2y, 0, z, 0, x, 0, 1, 0, 0],
     [0, 0, 2z, y, x, 0, 0, 0, 1, 0])
end

function normalConstraint(p, n, ::Val{2})
    derivatives = normalConstraintExplicit(p, Val(2))
    i = findmax(map(abs, n))[2] # index of max. absolute value in n
    j = mod1(i + 1, 3)
    k = mod1(j + 1, 3)
    (n[i] * derivatives[j] - n[j] * derivatives[i],
     n[i] * derivatives[k] - n[k] * derivatives[i])
end

function curveConstraint(curve, ::Val{2})
    map(range(0, stop=1, length=5)) do u
        pointConstraint(evalRational(curve, u), Val(2))
    end
end

"""
    curveNormalConstraint(curve, n0, n1, Val(2))

Fixes the normal sweep as a linear function from `n0` (at `u=0`) to `n1` (at `u=1`).
Returns 4 equations.
"""
function curveNormalConstraint(curve, n0, n1, ::Val{2})
    map(range(0, stop=1, length=4)) do u
        n = n0 * (1 - u) + n1 * u
        t = evalRationalDerivative(curve, u)
        derivatives = normalConstraintExplicit(evalRational(curve, u), Val(2))
        i = findmax(map(abs, t))[2] # index of max. absolute value in t
        j = mod1(i + 1, 3)
        k = mod1(j + 1, 3)
        derivatives[j] * n[k] - n[j] * derivatives[k]
    end
end

function lineConstraint(p, d, ::Val{2})
    [pointConstraint(p + d, Val(2)),
     pointConstraint(p - d, Val(2))]
end

function prettyprint(x, ::Val{2})
    names = ["x**2", "y**2", "z**2", "y*z", "x*z", "x*y", "x", "y", "z", "1"]
    for i in 1:length(x)
        xi = x[i]
        if abs(xi) > 1e-5
            @printf "%.02f * %s + " xi names[i]
        end
    end
    println("0")
end

function fitQuadratic(curve, n0, n1)
    rows = curveConstraint(curve, Val(2))
    # append!(rows, curveNormalConstraint(curve, n0, n1, Val(2)))
    append!(rows, normalConstraint(evalRational(curve, 0), n0, Val(2)))
    append!(rows, normalConstraint(evalRational(curve, 1), n1, Val(2)))
    # append!(rows, normalConstraintExplicit(evalRational(curve, 0), Val(2)))
    # append!(rows, normalConstraintExplicit(evalRational(curve, 1), Val(2)))

    A1 = mapreduce(transpose, vcat, rows)
    n = size(A1, 1)
    b1 = zeros(n)
    # b1[6:8] = n0
    # b1[9:11] = n1


    m = 8                       # minimization constraints
    rows = []
    points = []
    scaling = norm(evalRational(curve, 0) - evalRational(curve, 1))
    for i in 1:m
        s = (i-1)/(m-1)
        ps = evalRational(curve, s)
        ts = evalRationalDerivative(curve, s)
        ns = n0 * (1 - s) + n1 * s
        bs = normalize(cross(ts, ns)) * scaling
        append!(rows, lineConstraint(ps, bs, Val(2)))
        append!(points, [ps + bs, ps - bs])
    end
    open("/tmp/points.obj", "w") do f
        for p in points
            println(f, "v $(p[1]) $(p[2]) $(p[3])")
        end
        for i in 1:length(points)
            println(f, "p $i")
        end
        for i in 1:m
            println(f, "l $(2*i-1) $(2*i)")
        end
    end

    # append!(rows, normalConstraintExplicit(evalRational(curve, 0), Val(2)))
    # append!(rows, normalConstraintExplicit(evalRational(curve, 1), Val(2)))
    A2 = mapreduce(transpose, vcat, rows)

    # Least-square minimization of A2 x = b2, while constraining A1 x = b1
    # A = [(A2' * A2)     A1'
    #          A1      zeros(n,n)]
    # b = vcat(A2' * b2, b1)
    # open("/tmp/mat2.txt", "w") do f
    #     println(f, A)
    # end

    # No minimization
    # A = A1
    # b = b1

    M = nullspace(A1)
    A = M' * A2' * A2 * M

    F = svd(A)
    println("S: $(F.S)")
    # display(F.V)
    # i = findfirst(s -> abs(s) < 1e-5, F.S)
    x = M * F.V[:,end]
    # x = qr(A, Val(true)) \ b

    x = x[1:10] / norm(x[1:10])

    println(x)
    print("x: ")
    prettyprint(x, Val(2))
    println("Error: $(maximum(map(abs,A1*x-b1)))")
    x
end

evalQuadratic(qf, p) = sum(pointConstraint(p, Val(2)) .* qf)


# Cubic constraints

function pointConstraint(p, ::Val{3})
    (x, y, z) = p
    [x^3, y^3, z^3, x^2*y, x^2*z, y^2*x, y^2*z, z^2*x, z^2*y, x*y*z,
     x^2, y^2, z^2, y*z, x*z, x*y, x, y, z, 1]
end

function normalConstraintExplicit(p, ::Val{3})
    (x, y, z) = p
    ([3x^2, 0, 0, 2x*y, 2x*z, y^2, 0, z^2, 0, y*z, 2x, 0, 0, 0, z, y, 1, 0, 0, 0],
     [0, 3y^2, 0, x^2, 0, 2y*x, 2y*z, 0, z^2, x*z, 0, 2y, 0, z, 0, x, 0, 1, 0, 0],
     [0, 0, 3z^2, 0, x^2, 0, y^2, 2z*x, 2z*y, x*y, 0, 0, 2z, y, x, 0, 0, 0, 1, 0])
end

function normalConstraint(p, n, ::Val{3})
    derivatives = normalConstraintExplicit(p, Val(3))
    i = findmax(map(abs, n))[2] # index of max. absolute value in n
    j = mod1(i + 1, 3)
    k = mod1(j + 1, 3)
    (n[i] * derivatives[j] - n[j] * derivatives[i],
     n[i] * derivatives[k] - n[k] * derivatives[i])
end

function curveConstraint(curve, ::Val{3})
    map(range(0, stop=1, length=7)) do u
        pointConstraint(evalRational(curve, u), Val(3))
    end
end

"""
    curveNormalConstraint(curve, n0, n1, Val(3))

Fixes the normal sweep from `n0` (at `u=0`) to `n1` (at `u=1`).
Returns 8 equations when the blend is cubic; 6 equations when the blend is linear.
"""
function curveNormalConstraint(curve, n0, n1, ::Val{3})
    linear = true
    map(range(0, stop=1, length=(linear ? 6 : 8))) do u
        coeff = bernstein(3, u)
        n = linear ? n0 * (1 - u) + n1 * u : n0 * (coeff[1] + coeff[2]) + n1 * (coeff[3] + coeff[4])
        t = evalRationalDerivative(curve, u)
        derivatives = normalConstraintExplicit(evalRational(curve, u), Val(3))
        i = findmax(map(abs, t))[2] # index of max. absolute value in t
        j = mod1(i + 1, 3)
        k = mod1(j + 1, 3)
        derivatives[j] * n[k] - n[j] * derivatives[k]
    end
end

function lineConstraint(p, d, ::Val{3})
    map(1:3) do i          # assumes that one other point (p) is already constrained
        pointConstraint(p + d * i, Val(3))
    end
end

function prettyprint(x, ::Val{3})
    names = ["x**3", "y**3", "z**3", "x**2*y", "x**2*z", "y**2*x", "y**2*z", "z**2*x", "z**2*y",
             "x*y*z", "x**2", "y**2", "z**2", "y*z", "x*z", "x*y", "x", "y", "z", "1"]
    for i in 1:length(x)
        xi = x[i]
        if abs(xi) > 1e-5
            @printf "%.02f * %s + " xi names[i]
        end
    end
    println("0")
end

function fitCubic(curve, n0, n1)
    rows = curveConstraint(curve, Val(3))
    # append!(rows, curveNormalConstraint(curve, n0, n1, Val(3)))
    append!(rows, normalConstraint(evalRational(curve, 0), n0, Val(3)))
    append!(rows, normalConstraint(evalRational(curve, 1), n1, Val(3)))
    # append!(rows, normalConstraintExplicit(evalRational(curve, 0), Val(3)))
    # append!(rows, normalConstraintExplicit(evalRational(curve, 1), Val(3)))

    A1 = mapreduce(transpose, vcat, rows)
    n = size(A1, 1)
    # b1 = zeros(n)
    # b1[8:10] = n0
    # b1[11:13] = n1

    m = 8                       # minimization constraints
    rows = []
    points = []
    scaling = norm(evalRational(curve, 0) - evalRational(curve, 1))
    for i in 1:m
        s = (i-1)/(m-1)
        ps = evalRational(curve, s)
        ts = evalRationalDerivative(curve, s)
        ns = n0 * (1 - s) + n1 * s
        bs = normalize(cross(ts, ns)) * scaling
        push!(rows, pointConstraint(ps + bs, Val(3)))
        push!(rows, pointConstraint(ps - bs, Val(3)))
        append!(points, [ps + bs, ps - bs])
    end
    open("/tmp/points.obj", "w") do f
        for p in points
            println(f, "v $(p[1]) $(p[2]) $(p[3])")
        end
        for i in 1:length(points)
            println(f, "p $i")
        end
        for i in 1:m
            println(f, "l $(2*i-1) $(2*i)")
        end
    end

    A2 = mapreduce(transpose, vcat, rows)

    # Least-square minimization of A2 x = b2, while constraining A1 x = b1
    # A = [(A2' * A2) A1'; A1 zeros(n,n)]
    # b = vcat(A2' * b2, b1)

    # No minimization
    # A = A1
    # b = b1

    M = nullspace(A1)
    A = M' * A2' * A2 * M
    F = svd(A)
    x = M * F.V[:,end]
    x = normalize(x[1:20])

    print("x: ")
    prettyprint(x, Val(3))
    # println("Error: $(maximum(map(abs,A*x-b)))")
    x
end

evalCubic(cf, p) = sum(pointConstraint(p, Val(3)) .* cf)

function readVector(f)
    x = read(f, Float64)
    y = read(f, Float64)
    z = read(f, Float64)
    [x, y, z]
end

function readExample(filename)
    open(filename) do f
        p0 = readVector(f)
        p1 = readVector(f)
        p2 = readVector(f)
        n0 = readVector(f)
        n1 = readVector(f)
        w = read(f, Float64)
        (p0, p1, p2, w, n0, n1)
    end
end

function test(filename; degree = 2)
    (p0, p1, p2, w, n0, n1) = readExample(filename)
    curve = RationalCurve([p0, p1, p2], [1, w, 1])


    res = 100
    fitter = degree == 2 ? fitQuadratic : fitCubic
    evaluator = degree == 2 ? evalQuadratic : evalCubic
    f = fitter(curve, normalize(n0), normalize(n1))
    dc = Main.DualContouring.isosurface(0, ([-200,-200,100], [100,100,400]), (res, res, res)) do p
        evaluator(f, p)
    end
    Main.DualContouring.writeOBJ(dc..., "/tmp/surface.obj")
    writeCurve(curve, n0, n1, 50, "/tmp/curve.obj")
end

end # module
