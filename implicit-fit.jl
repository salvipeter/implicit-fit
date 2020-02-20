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
    d = length(curve.cp) - 1    # curve degree
    map(range(0, stop=1, length=degree*d+1)) do u
        pointConstraint(evalRational(curve, u), degree)
    end
end

function curveNormalConstraint(curve, n0, n1, degree, m)
    d = length(curve.cp) - 1    # curve degree
    map(range(0, stop=1, length=(degree-1)*d+m+1)) do u
        coeffs = bernstein(m, u)
        n = n0 * sum(coeffs[1:m÷2+1]) + n1 * sum(coeffs[m÷2+2:m+1])
        t = evalRationalDerivative(curve, u)
        derivatives = gradientConstraint(evalRational(curve, u), degree)
        i = findmax(map(abs, t))[2] # index of max. absolute value in t
        j = mod1(i + 1, 3)
        k = mod1(j + 1, 3)
        derivatives[j] * n[k] - n[j] * derivatives[k]
    end
end

"""
    fitSurface(curves, points, point_normals, degree, fence_degree)

Fits an implicit surface of degree `degree` on the given entities:
- `curves` is an array of `RationalCurve`s
- `points` is an array of 3D points
- `point_normals` is an array of (point, normal) pairs

When `fence_degree` is not zero, curves also get normal constraints.
It can take on any positive odd number, which will generate a Hermite blend
of the corresponding degree between the two normal vectors at the curve
endpoints (computed by the cross product of the tangent vectors).
"""
function fitSurface(curves, points, point_normals, degree, fence_degree)
    n = length(curves)
    rows = []
    for i in 1:n
        curve = curves[i]
        prev, next = curves[mod1(i - 1, n)], curves[mod1(i + 1, n)]
        n0 = cross(evalRationalDerivative(curve, 0), evalRationalDerivative(prev, 1))
        n1 = cross(evalRationalDerivative(curve, 1), evalRationalDerivative(next, 0))
        append!(rows, curveConstraint(curve, degree))
        if fence_degree > 0
            append!(rows, curveNormalConstraint(curve, n0, n1, degree, fence_degree))
        end
    end
    for p in points
        push!(rows, pointConstraint(p, degree))
    end
    for (p, n) in point_normals
        append!(rows, normalConstraint(p, n, degree))
    end
    A = mapreduce(transpose, vcat, rows)
    println("Matrix size: $(size(A))")

    # Solve the system
    F = svd(A)
    x = F.V[:,end]
    println("Matrix condition: $(cond(A))")
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

function readCurvesFromCrv(filename)
    data = readdlm(filename)
    n = size(data, 1)
    ([RationalCurve([data[i,1:3], data[i,4:6], data[i,7:9]], [1, data[i,10], 1]) for i in 1:n],
     nothing)
end

function readCurvesFromGbp(filename)
    read_numbers(f, numtype) = map(s -> parse(numtype, s), split(readline(f)))
    result = []
    local central_cp
    open(filename) do f
        n, d = read_numbers(f, Int)
        central_cp = read_numbers(f, Float64)
        cpts = []
        for i in 1:n*d
            p = read_numbers(f, Float64)
            push!(cpts, p)
            if i != 1 && (i - 1) % d == 0
                push!(result, RationalCurve(cpts, ones(d + 1)))
                cpts = [cpts[end]]
            end
        end
        push!(cpts, result[1].cp[1])
        push!(result, RationalCurve(cpts, ones(d + 1)))
    end
    (result, central_cp)
end

function readCurvesFromSbv(filename)
    data = readdlm(filename)
    n = size(data, 1) - 1
    (generateSetback(data[1,1:3], [data[i+1,1:3] for i in 1:n], [data[i+1,4] for i in 1:n]),
     nothing)
end

function readCurves(filename)
    match(r"\.crv$", filename) != nothing && return readCurvesFromCrv(filename)
    match(r"\.gbp$", filename) != nothing && return readCurvesFromGbp(filename)
    match(r"\.sbv$", filename) != nothing && return readCurvesFromSbv(filename)
    @error "Unknown file extension"
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

function intersectLines(ap, ad, bp, bd)
    ϵ = 1.0e-8
    a = dot(ad, ad)
    b = dot(ad, bd)
    c = dot(bd, bd)
    d = dot(ad, ap - bp)
    e = dot(bd, ap - bp)
    a * c - b * b < ϵ && return ap
    s = (b * e - c * d) / (a * c - b * b)
    ap + s * ad;
end

"""
    generateSetback(center, setbacks, ranges)

Given a center point and an array of setback points,
generate a list of curves defining a setback vertex blend.
The radii of the blends are defined by the given ranges.

The profile curves are circular arcs,
while the spring curves are parabolic arcs.
"""
function generateSetback(center, setbacks, ranges)
    n = length(setbacks)

    # Generate spring curves
    springs = map(1:n) do i
        ip = mod1(i + 1, n)
        p1, p2 = setbacks[i], setbacks[ip]
        r1, r2 = ranges[i], ranges[ip]

        # Compute directions
        normal = normalize(cross(p1 - center, p2 - center))
        d1 = normalize(cross(normal, center - p1))
        d2 = normalize(cross(normal, center - p2))

        # Fix orientation
        if dot(p2 - center, d1) < 0
            d1 *= -1
        end
        if dot(p1 - center, d2) < 0
            d2 *= -1
        end

        q1, q2 = p1 + d1 * r1, p2 + d2 * r2
        q12 = intersectLines(q1, center - p1, q2, center - p2)
        RationalCurve([q1, q12, q2], [1, 1, 1])
    end

    result = []
    for i in 1:n
        im = mod1(i - 1, n)
        push!(result, RationalCurve([springs[im].cp[3], setbacks[i], springs[i].cp[1]],
                                    [1, sqrt(2) / 2, 1])) # profile curve
        push!(result, springs[i])
    end

    result
end


# Main program

function test(filename, degree; fence_degree = 1, use_center = false)
    (curves, center) = readCurves(filename)
    bbox = boundingBox(curves)
    res = 100

    if center === nothing
        use_center = false
    end
    coeffs = fitSurface(curves, use_center ? [center] : [], [], degree, fence_degree)
    dc = Main.DualContouring.isosurface(0, bbox, (res, res, res)) do p
        evalSurface(coeffs, degree, p)
    end
    Main.DualContouring.writeOBJ(dc..., "/tmp/surface.obj")
    writeCurves(curves, 50, "/tmp/curves.obj")
end

end
