using Base: @propagate_inbounds
import Base: size, getindex, setindex!

struct FluidValue{T<:Real} <: AbstractArray{T,2}
    src::Array{T,2}
    dst::Array{T,2}

    # X and Y offsets from top left of grid cell
    ox::T
    oy::T

    # Cell grid size
    hx::T
end

function FluidValue(val::Array{T,2}, ox::T, oy::T, hx::T) where {T<:Real}
    height, width = size(val)
    dst = zeros(T, height, width)
    return FluidValue(val, dst, ox, oy, hx)
end

# Array interface methods
Base.size(a::FluidValue) = size(a.src)
@propagate_inbounds Base.getindex(a::FluidValue, I...) = getindex(a.src, I...)
@propagate_inbounds Base.setindex!(a::FluidValue, v, I...) = setindex!(a.src, v, I...)

gridsize(a::FluidValue) = a.hx
offset(a::FluidValue) = (a.ox, a.oy)

# Linear interpolation between a and b for x ∈ [0, 1]
function linear_interp(x, a, b)
    return a * (1.0 - x) + b * x
end

# Bilinear interpolation at real valued grid coordinate (i, j)
function linear_interp(a::FluidValue, i, j)
    dj, di = offset(a)
    h, w = size(a)

    # Clamp values to lie in index domain
    i = min(max(i - di, 1.0), h - 0.001)
    j = min(max(j - dj, 1.0), w - 0.001)

    # Upper left grid point of cell containing point
    grid_i = floor(Int, i)
    grid_j = floor(Int, j)

    # Fractional part ∈ [0, 1]
    i -= grid_i
    j -= grid_j

    # Fluid value at 4 grid points of the cell
    a1 = a[grid_i,     grid_j    ]
    a2 = a[grid_i + 1, grid_j    ]
    a3 = a[grid_i,     grid_j + 1]
    a4 = a[grid_i + 1, grid_j + 1]

    # Bilinear interpolation
    i_interp_1 = linear_interp(i, a1, a2)
    i_interp_2 = linear_interp(i, a3, a4)
    return linear_interp(j, i_interp_1, i_interp_2)
end

# Interpolate fluid values
(a::FluidValue)(i, j) = linear_interp(a, i, j)

# Set value inside the given rectangular region to value v
function add_inflow!(a::FluidValue{T}, xlim, ylim, v::T) where {T}
    x1 = round(Int, xlim[1]/a.hx - a.ox) + 1
    x2 = round(Int, xlim[2]/a.hx - a.ox) + 1
    y1 = round(Int, ylim[1]/a.hx - a.oy) + 1
    y2 = round(Int, ylim[2]/a.hx - a.oy) + 1
    for j in max(x1, 1):min(x2, size(a)[1])
        for i in max(y1, 1):min(y2, size(a)[2])
            if abs(a[i, j]) < abs(v)
                a[i, j] = v
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
end
