module Fluids

include("fluid.jl")
export FluidValue, advect!, add_inflow!, flip!, euler!, update_interp!

include("solver.jl")
export
    FluidSolver, solve, update!, build_rhs!,
    project!, apply_pressure!, apply_bc!, gridsize

end # module

# TODO: Interpolations is resulting in 0s
