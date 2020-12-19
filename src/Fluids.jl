module Fluids

include("fluid.jl")
export FluidValue, advect!, add_inflow!, flip!, euler!, Linear, Cubic

include("solver.jl")
export
    FluidSolver, solve, update!, build_rhs!,
    project!, apply_pressure!, apply_bc!, gridsize

end # module
