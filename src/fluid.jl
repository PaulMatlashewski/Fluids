using Interpolations
using Interpolations: Extrapolation
using Base: @propagate_inbounds
import Base: size, getindex, setindex!

struct FluidValue{T<:Real, T_itp<:Extrapolation} <: AbstractArray{T,2}
    src::Array{T,2}
    dst::Array{T,2}

    # X and Y offsets from top left of grid cell
    ox::T
    oy::T

    # Cell grid size
    hx::T

    # Interpolation object
    itp::T_itp
end

function FluidValue(val::Array{T,2}, ox::T, oy::T, hx::T) where {T<:Real}
    height, width = size(val)
    dst = zeros(T, height, width)
    itp = extrapolate(interpolate(val, BSpline(Linear())), Flat())
    return FluidValue(val, dst, ox, oy, hx, itp)
end

# Array interface methods
Base.size(a::FluidValue) = size(a.src)
@propagate_inbounds Base.getindex(a::FluidValue, I...) = getindex(a.src, I...)
@propagate_inbounds Base.setindex!(a::FluidValue, v, I...) = setindex!(a.src, v, I...)

gridsize(a::FluidValue) = a.hx
offset(a::FluidValue) = (a.ox, a.oy)

# Interpolate fluid values
(a::FluidValue)(i, j) = a.itp(i - a.oy, j - a.ox)

# Set value at inside the given rectangular region to value v
function add_inflow!(a::FluidValue{T}, xlim, ylim, v::T) where {T}
    x1 = round(Int, xlim[1]/a.hx - a.ox)
    x2 = round(Int, xlim[2]/a.hx - a.ox)
    y1 = round(Int, ylim[1]/a.hx - a.oy + 1)
    y2 = round(Int, ylim[2]/a.hx - a.oy + 1)
    for j in max(x1, 1):min(x2, size(a)[1])
        for i in max(y1, 1):min(y2, size(a)[2])
            if abs(a[i, j]) < abs(v)
                a[i, j] = v
                # Update interpolation
                a.itp.itp.coefs[i, j] = v
            end
        end
    end
end

function euler!(dx, i, j, dt, hx, u::FluidValue, v::FluidValue)
    dx[1] = i - v(i, j) / hx * dt
    dx[2] = j - u(i, j) / hx * dt
end

function advect!(a::FluidValue, u::FluidValue, v::FluidValue, dt)
    h, w = size(a)
    ox, oy = offset(a)
    hx = gridsize(a)
    dx = [0.0, 0.0]
    for j in 1:w
        for i in 1:h
            # Advect to new position
            euler!(dx, i + oy, j + ox, dt, hx, u, v)
            # Interpolate value
            a.dst[i, j] = a(dx[1], dx[2])
        end
    end
end

function flip!(a::FluidValue)
    a.src .= a.dst
    update_interp!(a)
end

function update_interp!(a::FluidValue)
    a.itp.itp.coefs .= a.src
end
