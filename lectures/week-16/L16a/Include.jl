# setup paths -
const _ROOT = @__DIR__
const _PATH_TO_SRC = joinpath(_ROOT, "src");

# activate the local environment so the dependencies in Project.toml are used -
using Pkg
if (isfile(joinpath(_ROOT, "Manifest.toml")) == false)
    Pkg.activate(_ROOT); Pkg.resolve(); Pkg.instantiate(); Pkg.update();
else
    Pkg.activate(_ROOT);
end

# load external packages -
using LinearAlgebra
using Plots
using Colors
using Random

# load local source files -
include(joinpath(_PATH_TO_SRC, "Types.jl"));
include(joinpath(_PATH_TO_SRC, "Factory.jl"));
include(joinpath(_PATH_TO_SRC, "Compute.jl"));

# seed the random number generator for reproducibility -
Random.seed!(1234);
