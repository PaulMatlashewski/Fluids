import Base: size
using FileIO, ImageMagick

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
    rho::T

    # Pressure variables
    r::Array{T, 2} # Right hand side of pressure solve
    p::Array{T, 2} # Pressure solution

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
    r = zeros(T, height, width)
    p = zeros(T, height, width)

    return FluidSolver(d, u, v, height, width, rho, r,
                       p, dt, maxiter, tol, tmax, bc, alg)
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
    
    scale = 1.0 / gridsize(prob)
    h, w = size(prob)

    # Right hand side of pressure solve
    r = prob.r
    
    for j in 1:w
        for i in 1:h
            # Negative divergence of the velocity
            r[i, j] = -scale * (u[i, j+1] - u[i, j] + v[i+1, j] - v[i, j])
        end
    end
end

# Perform the pressure solve using Gauss-Seidel
function project!(prob::FluidSolver)
    p, r = prob.p, prob.r
    rho = prob.rho

    hx = gridsize(prob)
    dt = prob.dt
    maxiter = prob.maxiter
    scale = dt / (rho * hx^2)
    h, w = size(prob)

    max_delta = 0.0
    for k in 1:maxiter
        max_delta = 0.0
        for j in 1:w
            for i in 1:h
                diag = 0.0
                off_diag = 0.0
                
                # Build the matrix implicitly as the five point stencil
                # Grid borders are assumed to be solid (no fluid outside
                # the simulation domain)
                if i > 1
                    diag += scale
                    off_diag -= scale * p[i - 1, j]
                end
                if j > 1
                    diag += scale
                    off_diag -= scale * p[i, j - 1]
                end
                if i < h
                    diag += scale
                    off_diag -= scale * p[i + 1, j]
                end
                if j < w
                    diag += scale
                    off_diag -= scale * p[i, j + 1]
                end
                new_p = (r[i, j] - off_diag) / diag
                max_delta = max(max_delta, abs(p[i, j] - new_p))
                p[i, j] = new_p
            end
        end
        if max_delta < prob.tol
            println("    Exiting solver after $(k) iterations, res = $(max_delta)")
            return
        end
    end
    println("    Exceeded maxiter $(maxiter), res = $(max_delta)")
end

# Apply the computed pressure to the velocity field
function apply_pressure!(prob::FluidSolver)
    u, v = prob.u, prob.v
    rho = prob.rho
    p = prob.p
    dt = prob.dt

    hx = gridsize(prob)
    scale = dt / (rho * hx)
    h, w = size(prob)

    for j in 1:w
        for i in 1:h
            u[i, j] -= scale * p[i, j]
            v[i, j] -= scale * p[i, j]
            u[i, j + 1] += scale * p[i, j]
            v[i + 1, j] += scale * p[i, j]
        end
    end
end

function apply_bc!(prob::FluidSolver)
    u, v = prob.u, prob.v
    bc = prob.bc
    w, h = size(prob)
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

function update!(prob::FluidSolver)
    # Pressure step
    build_rhs!(prob)
    project!(prob)
    apply_pressure!(prob)
    apply_bc!(prob)

    # Advection step
    advect!(prob.d, prob::FluidSolver)
    advect!(prob.u, prob::FluidSolver)
    advect!(prob.v, prob::FluidSolver)
    flip!(prob.d)
    flip!(prob.u)
    flip!(prob.v)
end

function solve(prob, filename, fps)
    d, u, v = prob.d, prob.u, prob.v
    ρ = prob.rho

    dt = prob.dt
    tmax = prob.tmax
    t = 0.0
    n = floor(Int, tmax / dt)
    h, w = size(prob)
    data = zeros(eltype(d), h, w, n)

    # Initialize animation
    data[:, :, 1] .= prob.d.src

    dt = prob.dt
    tmax = prob.tmax
    t = 0.0
    for k in 2:n
        t += dt
        println("Solving time t: $(t)")
        update!(prob)
        
        # Add inflow conditions
        add_inflow!(d, [0.45, 0.55], [0.1, 0.11], 1.0)
        add_inflow!(u, [0.45, 0.55], [0.1, 0.11], 0.0)
        add_inflow!(v, [0.45, 0.55], [0.1, 0.11], 3.0)

        # Save result
        data[:, :, k] .= prob.d.src
    end
    save(filename, data; fps=fps)
end
