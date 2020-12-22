module Fluids

include("fluid.jl")
export FluidValue, add_inflow!, flip!, Linear, Cubic, offset, linear_interp

include("solver.jl")
export FluidSolver, solve, update!, build_rhs!, Euler, RK3, ∇²,
       project!, apply_pressure!, apply_bc!, gridsize, advect!, integrate!,
       vorticity_confinement!, update_vorticity!

end # module
