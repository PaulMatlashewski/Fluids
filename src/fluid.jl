using Base: @propagate_inbounds
import Base: size, getindex, setindex!

# Traits
abstract type Interpolation end
struct Linear <: Interpolation end
struct Cubic <: Interpolation end

struct FluidValue{T<:Real,T2<:Interpolation} <: AbstractArray{T,2}
    src::Array{T,2}
    dst::Array{T,2}

    # X and Y offsets from top left of grid cell
    ox::T
    oy::T

    # Cell grid size
    hx::T

    # Interpolation type
    itp::T2
end

function FluidValue(val::Array{T,2}, itp, ox, oy) where {T<:Real}
    # Grid spacing
    height, width = size(val)
    hx = 1.0 / min(width, height)
    dst = zeros(T, height, width)
    return FluidValue(val, dst, ox, oy, hx, itp)
end

# Array interface methods
Base.size(a::FluidValue) = size(a.src)
@propagate_inbounds Base.getindex(a::FluidValue, I...) = getindex(a.src, I...)
@propagate_inbounds Base.setindex!(a::FluidValue, v, I...) = setindex!(a.src, v, I...)

gridsize(a::FluidValue) = a.hx
offset(a::FluidValue) = (a.ox, a.oy)

function locate(a, i, j)
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

    return i, j, grid_i, grid_j
end

# Linear interpolation between a and b for x ∈ [0, 1]
function linear_interp(x, a, b)
    return a * (1.0 - x) + b * x
end

# Cubic interpolation for values a, b, c, d for x ∈ [0, 1]
function cubic_interp(x, a, b, c, d)
    xs = x^2
    xc = x^3

    # Clamp values to prevent blow up
    min_val = min(a, min(b, min(c, d)))
    max_val = max(a, max(b, max(c, d)))

    s = (
        a*(0.0 - 0.5*x + 1.0*xs - 0.5*xc) +
        b*(1.0 + 0.0*x - 2.5*xs + 1.5*xc) +
        c*(0.0 + 0.5*x + 2.0*xs - 1.5*xc) +
        d*(0.0 + 0.0*x - 0.5*xs + 0.5*xc)
    )
    return min(max(s, min_val), max_val)
end

# Bilinear interpolation at real valued grid coordinate (i, j)
function interp(a::FluidValue{T, Linear}, i, j) where {T}
    i, j, grid_i, grid_j = locate(a, i, j)

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

# Cubic interpolation of real valued grid coordinate (i, j)
function interp(a::FluidValue{T, Cubic}, i, j) where {T}
    h, w = size(a)
    i, j, grid_i, grid_j = locate(a, i, j)

    # Grid points to interpolate over
    i1, i2, i3, i4 = max(grid_i - 1, 1), grid_i, grid_i + 1, min(grid_i + 2, h)
    j1, j2, j3, j4 = max(grid_j - 1, 1), grid_j, grid_j + 1, min(grid_j + 2, w)

    # Fluid value at interpolation points
    a1 = cubic_interp(i, a[i1, j1], a[i2, j1], a[i3, j1], a[i4, j1])
    a2 = cubic_interp(i, a[i1, j2], a[i2, j2], a[i3, j2], a[i4, j2])
    a3 = cubic_interp(i, a[i1, j3], a[i2, j3], a[i3, j3], a[i4, j3])
    a4 = cubic_interp(i, a[i1, j4], a[i2, j4], a[i3, j4], a[i4, j4])

    return cubic_interp(j, a1, a2, a3, a4)
end

# Interpolate fluid values
(a::FluidValue)(i, j) = interp(a, i, j)

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
