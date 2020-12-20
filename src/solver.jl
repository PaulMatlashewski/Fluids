import Base: size
using FileIO, ImageMagick
using LinearAlgebra, SparseArrays, SuiteSparse
using Formatting, TimerOutputs

abstract type Integrator end
struct Euler <: Integrator end
struct RK3   <: Integrator end

struct FluidSolver{T<:Real, T2<:Integrator}
    # Fluid velocities
    d::FluidValue{T}
    u::FluidValue{T}
    v::FluidValue{T}

    # Grid size
    h::Int
    w::Int

    # Fluid density
    ρ::T

    # Cholesky factorization of pressure Laplacian matrix
    A::SuiteSparse.CHOLMOD.Factor{T}
    p::Array{T, 1} # Pressure solution
    r::Array{T, 1} # Discrete divergence right hand side

    # Solver options
    dt::T
    maxiter::Int
    tol::T
    tmax::T
    bc::T
    alg::T2
end

function FluidSolver(height, width, rho::T, dt, maxiter, tol, tmax, bc;
                     itp=Cubic(), alg=RK3()) where {T}
    # Fluid data values
    d = FluidValue(zeros(T, height, width), itp, 0.5, 0.5)

    # Fluid velocity
    u = FluidValue(zeros(T, height, width + 1), itp, 0.0, 0.5)
    v = FluidValue(zeros(T, height + 1, width), itp, 0.5, 0.0)

    # Fluid pressure
    r = zeros(T, height * width)
    p = zeros(T, height * width)

    # Laplacian matrix for pressure
    L = ∇²(height, width)
    # Prefactor using Cholesky decomposition
    println("Performing Cholesky decomposition")
    A = cholesky(L)

    return FluidSolver(d, u, v, height, width, rho, A, p, r,
                       dt, maxiter, tol, tmax, bc, alg)
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

# Forward Euler integration
function euler!(dx, i, j, dt, hx, u::FluidValue, v::FluidValue)
    dx[1] = i - v(i, j) / hx * dt
    dx[2] = j - u(i, j) / hx * dt
end

# RungeKutta3 integration
function rk3!(dx, i, j, dt, hx, u::FluidValue, v::FluidValue)
    u1 = linear_interp(u, i, j) / hx
    v1 = linear_interp(v, i, j) / hx

    i1 = i - 0.5 * dt * v1
    j1 = j - 0.5 * dt * u1

    u2 = linear_interp(u, i1, j1) / hx
    v2 = linear_interp(v, i1, j1) / hx

    i2 = i - 0.75 * dt * v2
    j2 = j - 0.75 * dt * u2

    u3 = linear_interp(u, i2, j2)
    v3 = linear_interp(v, i2, j2)

    dx[1] = i - (2v1 + 3v2 + 4v3)*dt/9.0
    dx[2] = j - (2u1 + 3u2 + 4u3)*dt/9.0
end

function integrate!(dx, i, j, hx, prob::FluidSolver{T, Euler}) where {T}
    euler!(dx, i, j, prob.dt, hx, prob.u, prob.v)
end

function integrate!(dx, i, j, hx, prob::FluidSolver{T, RK3}) where {T}
    rk3!(dx, i, j, prob.dt, hx, prob.u, prob.v)
end

function advect!(a::FluidValue, prob::FluidSolver)
    h, w = size(a)
    ox, oy = offset(a)
    hx = gridsize(a)
    dx = [0.0, 0.0]
    for j in 1:w
        for i in 1:h
            integrate!(dx, i + oy, j + ox, hx, prob)
            a.dst[i, j] = a(dx[1], dx[2])
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

function apply_bc!(prob::FluidSolver)
    u, v = prob.u, prob.v
    bc = prob.bc
    h, w = size(prob)
    for i in 1:h
        u[i, 1] = bc
        u[i, w] = bc
        u[i, w + 1] = bc
    end
    for j in 1:w
        v[1, j] = bc
        v[h, j] = bc
        v[h + 1, j] = bc
    end
end

# CFL condition for maximum timestep
function max_dt(prob::FluidSolver)
    w, h = size(prob)
    hx = gridsize(prob)
    # Maximum velocity at center of grid
    max_velocity = 0.0
    for j in 1:w
        for i in 1:h
            u = prob.u(i + 0.5, j + 0.5)
            v = prob.v(i + 0.5, j + 0.5)
            velocity = sqrt(u^2 + v^2)
            max_velocity = max(max_velocity, velocity)
        end
    end
    dt = 2.0 * hx / max_velocity
    return min(dt, 1.0) # Clamp to a resonable value for small velocities
end

function update!(prob::FluidSolver, to)
    # Pressure step
    @timeit to "Assemble Velocity Divergence" build_rhs!(prob)
    @timeit to "Solve Pressure" copyto!(prob.p, prob.A \ prob.r)
    @timeit to "Apply Pressure" apply_pressure!(prob)
    @timeit to "Apply BCs" apply_bc!(prob)

    # Advection step
    @timeit to "Advection" begin
        advect!(prob.d, prob::FluidSolver)
        advect!(prob.u, prob::FluidSolver)
        advect!(prob.v, prob::FluidSolver)
        flip!(prob.d)
        flip!(prob.u)
        flip!(prob.v)
    end
end

function solve(prob, filename, fps)
    to = TimerOutput()
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
    for k in 1:n    
        # Add inflow conditions
        add_smooth_inflow!(d, [0.4, 0.6], [0.1, 0.13], 1.0)
        add_smooth_inflow!(u, [0.4, 0.6], [0.1, 0.13], 0.0)
        add_smooth_inflow!(v, [0.4, 0.6], [0.1, 0.13], 3.0)

        printfmtln("Solving time t: {:.3f}", t)
        update!(prob, to)

        # Save result
        data[:, :, k] .= prob.d.src

        t += dt
    end
    save(filename, data; fps=fps)
    show(to)
end
