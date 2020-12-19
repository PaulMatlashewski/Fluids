import Base: size
using FileIO, ImageMagick

struct FluidSolver{T<:Real}
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
end

function FluidSolver(width, height, rho::T, dt, maxiter, tol, tmax, bc) where {T}
    # Grid spacing
    hx = 1.0 / min(width, height)

    # Fluid data values
    d = FluidValue(zeros(T, height, width), 0.5, 0.5, hx)

    # Fluid velocity
    u = FluidValue(zeros(T, height, width + 1), 0.0, 0.5, hx)
    v = FluidValue(zeros(T, height + 1, width), 0.5, 0.0, hx)

    # Fluid pressure
    r = zeros(T, height, width)
    p = zeros(T, height, width)

    return FluidSolver(d, u, v, width, height, rho, r,
                       p, dt, maxiter, tol, tmax, bc)
end

Base.size(prob::FluidSolver) = (prob.h, prob.w)
@inline gridsize(prob::FluidSolver) = 1.0 / min(prob.w, prob.h)

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
    build_rhs!(prob)
    project!(prob)
    apply_pressure!(prob)
    apply_bc!(prob)
    # Update interplation object after modifying velocity
    update_interp!(prob.u)
    update_interp!(prob.v)

    advect!(prob.d, prob.u, prob.v, prob.dt)
    advect!(prob.u, prob.u, prob.v, prob.dt)
    advect!(prob.v, prob.u, prob.v, prob.dt)

    flip!(prob.d)
    flip!(prob.u)
    flip!(prob.v)
end

function solve(prob::FluidSolver{T}, filename, fps) where {T}
    d, u, v = prob.d, prob.u, prob.v
    ρ = prob.rho

    dt = prob.dt
    tmax = prob.tmax
    t = 0.0
    n = floor(Int, tmax / dt)
    h, w = size(prob)
    data = zeros(T, h, w, n)

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
        add_inflow!(d, [0.45, 0.65], [0.1, 0.11], 1.0)
        add_inflow!(u, [0.45, 0.65], [0.1, 0.11], 0.0)
        add_inflow!(v, [0.45, 0.65], [0.1, 0.11], 3.0)

        # Save result
        data[:, :, k] .= prob.d.src
    end
    save(filename, data; fps=fps)
end
