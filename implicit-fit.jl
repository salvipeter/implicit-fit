module ImpFit

using DelimitedFiles
using LinearAlgebra


# Constraints

function pointConstraint(p, degree)
    [p[1]^i * p[2]^j * p[3]^k for i in 0:degree for j in 0:degree-i for k in 0:degree-i-j]
end

function gradientConstraint(p, degree)
    pow(x,i) = i == 0 ? 0 : i * x^(i-1)
    ([pow(p[1],i) * p[2]^j * p[3]^k for i in 0:degree for j in 0:degree-i for k in 0:degree-i-j],
     [p[1]^i * pow(p[2],j) * p[3]^k for i in 0:degree for j in 0:degree-i for k in 0:degree-i-j],
     [p[1]^i * p[2]^j * pow(p[3],k) for i in 0:degree for j in 0:degree-i for k in 0:degree-i-j])
end

evalSurface(coeffs, degree, p) = dot(pointConstraint(p, degree), coeffs)

function normalConstraint(p, n, degree)
    derivatives = gradientConstraint(p, degree)
    i = findmax(map(abs, n))[2] # index of max. absolute value in n
    j = mod1(i + 1, 3)
    k = mod1(j + 1, 3)
    (n[i] * derivatives[j] - n[j] * derivatives[i],
     n[i] * derivatives[k] - n[k] * derivatives[i])
end

function curveConstraint(curve, degree)
    map(range(0, stop=1, length=2*degree+1)) do u
        pointConstraint(evalRational(curve, u), degree)
    end
end

function curveNormalConstraint(curve, n0, n1, degree)
    map(range(0, stop=1, length=2*degree)) do u
        n = n0 * (1 - u) + n1 * u
        t = evalRationalDerivative(curve, u)
        derivatives = gradientConstraint(evalRational(curve, u), degree)
        i = findmax(map(abs, t))[2] # index of max. absolute value in t
        j = mod1(i + 1, 3)
        k = mod1(j + 1, 3)
        derivatives[j] * n[k] - n[j] * derivatives[k]
    end
end

function fitSurface(curves, degree)
    n = length(curves)
    rows = []
    for i in 1:n
        curve = curves[i]
        prev, next = curves[mod1(i - 1, n)], curves[mod1(i + 1, n)]
        n0 = cross(evalRationalDerivative(curve, 0), evalRationalDerivative(prev, 1))
        n1 = cross(evalRationalDerivative(curve, 1), evalRationalDerivative(next, 0))
        append!(rows, curveConstraint(curve, degree))
        append!(rows, normalConstraint(evalRational(curve, 0), n0, degree))
        append!(rows, normalConstraint(evalRational(curve, 1), n1, degree))
    end
    A = mapreduce(transpose, vcat, rows)
    println("Matrix size: $(size(A))")

    # Solve the system
    F = svd(A)
    x = F.V[:,end]
    println("Nullspace size: $(size(nullspace(A),2))")

    println("Error: $(maximum(map(abs,A*x)))")
    x
end


# Rational Bezier curve

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

# Utilities

function writeCurves(curves, resolution, filename)
    n = length(curves)
    open(filename, "w") do f
        for j in 1:n
            curve = curves[j]
            for i in 1:resolution
                u = i / resolution
                p = evalRational(curve, u)
                println(f, "v $(p[1]) $(p[2]) $(p[3])")
            end
        end
        for i in 1:n*resolution-1
            println(f, "l $i $(i+1)")
        end
        println(f, "l $(n*resolution) 1")
    end
end

function readCurves(filename)
    data = readdlm(filename)
    n = size(data, 1)
    [RationalCurve([data[i,1:3], data[i,4:6], data[i,7:9]], [1, data[10], 1]) for i in 1:n]
end

function boundingBox(curves)
    a = curves[1].cp[1]
    b = a
    for curve in curves
        for p in curve.cp
            a = min.(a, p)
            b = max.(b, p)
        end
    end
    m = (a + b) / 2
    d = b - m
    scale = 1.3
    (m - d * scale, m + d * scale)
end


# Main program

function test(filename, degree)
    curves = readCurves(filename)
    bbox = boundingBox(curves)
    res = 100

    coeffs = fitSurface(curves, degree)
    dc = Main.DualContouring.isosurface(0, bbox, (res, res, res)) do p
        evalSurface(coeffs, degree, p)
    end
    Main.DualContouring.writeOBJ(dc..., "/tmp/surface.obj")
    writeCurves(curves, 50, "/tmp/curves.obj")
end

end
