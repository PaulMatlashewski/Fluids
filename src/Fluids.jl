module Fluids

include("fluid.jl")
export FluidValue, add_inflow!, flip!, Linear, Cubic

include("solver.jl")
export
    FluidSolver, solve, update!, build_rhs!, Euler, RK3,
    project!, apply_pressure!, apply_bc!, gridsize, advect!, integrate!

end # module
