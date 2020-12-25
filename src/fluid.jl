using Base: @propagate_inbounds
import Base: size, getindex, setindex!

abstract type Interpolation end
struct Linear <: Interpolation end
struct Cubic  <: Interpolation end

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

# Find the grid cell corresponding to point (i, j),
# and return the fractional distance into the cell.
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

    # Clamp values to limit over and undershoot
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
function linear_interp(a::FluidValue, i, j)
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
function cubic_interp(a::FluidValue, i, j)
    h, w = size(a)
    i, j, grid_i, grid_j = locate(a, i, j)

    # Grid points to interpolate over
    i1, i2, i3, i4 = max(grid_i - 1, 1), grid_i, grid_i + 1, min(grid_i + 2, h)
    j1, j2, j3, j4 = max(grid_j - 1, 1), grid_j, grid_j + 1, min(grid_j + 2, w)

    # Cubic interpolation weights in i direction
    wi1 = -(1/3)i + 0.5i^2 - (1/6)i^3
    wi2 = 1.0 - 0.5i - i^2 + 0.5i^3
    wi3 = i + 0.5i^2 - 0.5i^3
    wi4 = -(1/6)i + (1/6)i^3

    # Cubic interpolation weights in j direction
    wj1 = -(1/3)j + 0.5j^2 - (1/6)j^3
    wj2 = 1.0 - 0.5j - j^2 + 0.5j^3
    wj3 = j + 0.5j^2 - 0.5j^3
    wj4 = -(1/6)j + (1/6)j^3

    # Fluid value at interpolation points
    a1 = wi1*a[i1, j1] + wi2*a[i2, j1] + wi3*a[i3, j1] + wi4*a[i4, j1]
    a2 = wi1*a[i1, j2] + wi2*a[i2, j2] + wi3*a[i3, j2] + wi4*a[i4, j2]
    a3 = wi1*a[i1, j3] + wi2*a[i2, j3] + wi3*a[i3, j3] + wi4*a[i4, j3]
    a4 = wi1*a[i1, j4] + wi2*a[i2, j4] + wi3*a[i3, j4] + wi4*a[i4, j4]

    return wj1*a1 + wj2*a2 + wj3*a3 + wj4*a4
end

# Interpolate fluid values
(a::FluidValue{T, Linear})(i, j) where {T} = linear_interp(a, i, j)
(a::FluidValue{T, Cubic})(i, j) where {T} = cubic_interp(a, i, j)

function flip!(a::FluidValue)
    a.src .= a.dst
end

function add_inflow!(a, prob, xlim, y, v)
    h, w = size(prob)
    hh = 1.0 / h
    hw = 1.0 / w
    j1 = round(Int, xlim[1]/hw - a.ox) + 1
    j2 = round(Int, xlim[2]/hw - a.ox) + 1
    i = round(Int, y/hh - a.oy) + 1
    for j in max(j1, 1):min(j2, w)
        if abs(a[i, j]) < abs(v)
            a[i, j] = v
        end
    end
end

function add_smooth_inflow!(a, prob, xlim, y, v)
    h, w = size(prob)
    hh = 1.0 / h
    hw = 1.0 / w
    j1 = round(Int, xlim[1]/hw - a.ox) + 1
    j2 = round(Int, xlim[2]/hw - a.ox) + 1
    i = round(Int, y/hh - a.oy) + 1
    jc = (j1 + j2) / 2
    L = j2 - j1
    for j in max(j1, 1):min(j2, w)
        x = 2 * (j - jc) / L
        vi = v * exp(-1/(1 - x^2))
        # if abs(a[i, j]) < abs(vi)
        a[i, j] = vi
        # end
    end
end
