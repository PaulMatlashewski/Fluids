import Base: size
using FileIO, ImageMagick
using LinearAlgebra, SparseArrays, SuiteSparse
using Formatting, TimerOutputs
using ProgressMeter: Progress, next!

abstract type Integrator end
struct Euler <: Integrator end
struct RK3   <: Integrator end

struct FluidSolver{T1<:Real, T2<:Interpolation, T3<:Integrator}
    # Fluid data
    d::FluidValue{T1,T2}

    # Fluid velocity
    u::FluidValue{T1,T2}
    v::FluidValue{T1,T2}

    # Fluid forces
    fi::FluidValue{T1,T2}
    fj::FluidValue{T1,T2}

    # Grid size
    h::Int
    w::Int

    # Fluid density
    ρ::T1

    # Cholesky factorization of pressure Laplacian matrix
    A::SuiteSparse.CHOLMOD.Factor{T1}
    p::Array{T1, 1} # Pressure solution
    r::Array{T1, 1} # Discrete divergence right hand side

    # Solver options
    dt::T1
    tmax::T1
    bc::T1
    alg::T3
end

function FluidSolver(height, width, rho::T, dt, tmax, bc;
                     itp=Cubic(), alg=RK3()) where {T}
    # Fluid marker
    d = FluidValue(zeros(T, height, width), itp, 0.5, 0.5)

    # Fluid velocity
    u = FluidValue(zeros(T, height + 1, width + 1), itp, 0.0, 0.5)
    v = FluidValue(zeros(T, height + 1, width + 1), itp, 0.5, 0.0)

    # Fluid forces
    fi = FluidValue(zeros(T, height, width), itp, 0.5, 0.5)
    fj = FluidValue(zeros(T, height, width), itp, 0.5, 0.5)

    # Fluid pressure
    r = zeros(T, height * width)
    p = zeros(T, height * width)

    # Laplacian matrix for pressure
    L = ∇²(height, width)
    # Prefactor using Cholesky decomposition
    println("Performing Cholesky decomposition")
    A = cholesky(L)

    return FluidSolver(d, u, v, fi, fj, height, width, rho, A, p, r, dt, tmax, bc, alg)
end

function spdiagm_nonsquare(m, n, args...)
    I, J, V = SparseArrays.spdiagm_internal(args...)
    return sparse(I, J, V, m, n)
end

# returns -∇² (discrete Laplacian, real-symmetric positive-definite)
# on n₁×n₂ grid
function ∇²(n₁, n₂)
    o₁ = ones(n₁)
    ∂₁ = spdiagm_nonsquare(n₁+1,n₁,-1=>-o₁,0=>o₁)
    o₂ = ones(n₂)
    ∂₂ = spdiagm_nonsquare(n₂+1,n₂,-1=>-o₂,0=>o₂)
    return kron(sparse(I,n₂,n₂), ∂₁'*∂₁) + kron(∂₂'*∂₂, sparse(I,n₁,n₁))
end

Base.size(prob::FluidSolver) = (prob.h, prob.w)
@inline gridsize(prob::FluidSolver) = 1.0 / min(prob.w, prob.h)

# Euler integration
function euler!(a, grid_i, grid_j, prob)
    u = prob.u
    v = prob.v
    ox, oy = offset(a)
    hx = gridsize(a)
    dt = prob.dt

    i = grid_i + oy
    j = grid_j + ox

    u1 = linear_interp(u, i, j) / hx
    v1 = linear_interp(v, i, j) / hx

    i1 = i - v1 * dt
    j1 = j - u1 * dt

    a.dst[grid_i, grid_j] = a(i1, j1)
end

# RungeKutta3 integration
function rk3!(a, grid_i, grid_j, prob)
    u = prob.u
    v = prob.v
    ox, oy = offset(a)
    hx = gridsize(a)
    dt = prob.dt

    i = grid_i + oy
    j = grid_j + ox

    u1 = linear_interp(u, i, j) / hx
    v1 = linear_interp(v, i, j) / hx

    i1 = i - 0.5 * v1 * dt
    j1 = j - 0.5 * u1 * dt

    u2 = linear_interp(u, i1, j1) / hx
    v2 = linear_interp(v, i1, j1) / hx

    i2 = i - 0.75 * v2 * dt
    j2 = j - 0.75 * u2 * dt

    u3 = linear_interp(u, i2, j2) / hx
    v3 = linear_interp(v, i2, j2) / hx

    i3 = i - (2v1 + 3v2 + 4v3) / 9.0 * dt
    j3 = j - (2u1 + 3u2 + 4u3) / 9.0 * dt

    a.dst[grid_i, grid_j] = a(i3, j3)
end

function integrate!(a, i, j, prob::FluidSolver{T1, T2, Euler}) where {T1, T2}
    euler!(a, i, j, prob)
end

function integrate!(a, i, j, prob::FluidSolver{T1, T2, RK3}) where {T1, T2}
    rk3!(a, i, j, prob)
end

function advect!(a::FluidValue, prob::FluidSolver)
    h, w = size(a)
    Threads.@threads for j in 1:w
        for i in 1:h
            integrate!(a, i, j, prob)
        end
    end
end

function build_rhs!(prob::FluidSolver)
    # Velocity components
    u = prob.u
    v = prob.v
    
    # Incorporate Laplacian weights in the right hand side
    hx = gridsize(prob)
    dt = prob.dt
    ρ = prob.ρ
    scale = ρ * hx / dt
    h, w = size(prob)

    r = prob.r
    k = 1
    for j in 1:w
        for i in 1:h
            # Negative divergence of the velocity
            r[k] = -scale * (u[i, j+1] - u[i, j] + v[i+1, j] - v[i, j])
            k += 1
        end
    end
end

# Apply the computed pressure to the velocity field
function apply_pressure!(prob::FluidSolver)
    u, v = prob.u, prob.v
    rho = prob.ρ
    p = prob.p
    dt = prob.dt

    hx = gridsize(prob)
    scale = dt / (rho * hx)
    h, w = size(prob)

    k = 1
    for j in 1:w
        for i in 1:h
            u[i, j] -= scale * p[k]
            v[i, j] -= scale * p[k]
            u[i, j + 1] += scale * p[k]
            v[i + 1, j] += scale * p[k]
            k += 1
        end
    end
end

function vorticity(prob, i, j)
    u, v = prob.u, prob.v
    hx = gridsize(prob)
    return (v[i, j+1] - v[i, j] - u[i+1, j] + u[i, j]) / 2hx
end

# Calculate forces at cell centers for vorticity confinement
function vorticity_confinement!(prob, ϵ)
    u, v = prob.u, prob.v
    fi, fj = prob.fi, prob.fj
    h, w = size(prob)
    dt = prob.dt
    hx = gridsize(prob)
    Threads.@threads for j in 1:w-1
        for i in 1:h-1
            ω = vorticity(prob, i, j)
            ωi = vorticity(prob, i+1, j)
            ωj = vorticity(prob, i, j+1)

            # Vector pointing to local vortex: N = ∇ω / |∇ω|
            ∂ωi = (abs(ωi) - abs(ω)) / 2hx
            ∂ωj = (abs(ωj) - abs(ω)) / 2hx
            L = sqrt(∂ωi^2 + ∂ωj^2) + (1e-20/hx/dt) # Prevent divide by 0

            # Force to increase vorticity: f = (ω × N)dt
            fi[i, j] -= ϵ * dt * hx * (ω * ∂ωj/L)
            fj[i, j] += ϵ * dt * hx * (ω * ∂ωi/L)
        end
    end
end

# Apply body forces to velocity
function apply_bodyforces!(prob)
    u, v = prob.u, prob.v
    fi, fj = prob.fi, prob.fj
    h, w = size(prob)
    for j in 1:w
        for i in 1:h
            u[i, j] += 0.5 * fj[i, j]
            v[i, j] += 0.5 * fi[i, j]
            u[i, j + 1] += 0.5 * fj[i, j]
            v[i + 1, j] += 0.5 * fj[i, j]
        end
    end
end

function apply_bc!(prob)
    u, v = prob.u, prob.v
    bc = prob.bc
    h, w = size(prob)
    for i in 1:h+1
        u[i, 1] = bc
        u[i, w] = bc
        u[i, w + 1] = bc
    end
    for j in 1:w+1
        v[1, j] = bc
        v[h, j] = bc
        v[h + 1, j] = bc
    end
end

function update!(prob, to)
    # Pressure step
    @timeit to "Assemble Velocity Divergence" build_rhs!(prob)
    @timeit to "Solve Pressure" copyto!(prob.p, prob.A \ prob.r)
    @timeit to "Apply Pressure" apply_pressure!(prob)
    @timeit to "Apply BCs" apply_bc!(prob)

    # Advection step
    @timeit to "Advection" begin
        advect!(prob.d, prob)
        advect!(prob.u, prob)
        advect!(prob.v, prob)
        flip!(prob.d)
        flip!(prob.u)
        flip!(prob.v)
    end

    # Body force step
    @timeit to "Vorticity Confinement" begin
        vorticity_confinement!(prob, 10.0)
        apply_bodyforces!(prob)
        fill!(prob.fi, 0.0)
        fill!(prob.fj, 0.0)
    end
end

function solve(prob, filename, fps)
    d, u, v = prob.d, prob.u, prob.v
    dt = prob.dt
    tmax = prob.tmax
    t = 0.0
    n = floor(Int, tmax / dt)
    h, w = size(prob)
    data = zeros(eltype(d), h, w, n)
    dt = prob.dt
    tmax = prob.tmax
    t = 0.0

    to = TimerOutput()
    p = Progress(n, dt, "Solving: ", 50, :white)
    
    for k in 1:n
        # Add inflow conditions
        add_smooth_inflow!(d, prob, [0.45, 0.55], [0.1, 0.11], 1.0)
        add_smooth_inflow!(v, prob, [0.45, 0.55], [0.1, 0.11], 3.0)
        add_smooth_inflow!(d, prob, [0.45, 0.55], [0.89, 0.9], 1.0)
        add_smooth_inflow!(v, prob, [0.45, 0.55], [0.89, 0.9], -3.0)

        update!(prob, to)
        data[:, :, k] .= max.(min.(prob.d.src, 1.0), 0.0)
        t += dt
        next!(p)
    end
    println(to)
    println("Saving gif")
    save(filename, data; fps=fps)
end
