# setup paths -
const _ROOT = @__DIR__
const _PATH_TO_DATA = joinpath(_ROOT, "data");
const _PATH_TO_SRC = joinpath(_ROOT, "src");
const _PATH_TO_FIGS = joinpath(_ROOT, "figs");

# check: do we have a data directory?
!isdir(_PATH_TO_DATA) && mkpath(_PATH_TO_DATA);
!isdir(_PATH_TO_FIGS) && mkpath(_PATH_TO_FIGS);

# check do we have a Manifest.toml file?
using Pkg;
Pkg.activate(_ROOT);
if (isfile(joinpath(_ROOT, "Manifest.toml")) == false)
    Pkg.activate("."); Pkg.resolve(); Pkg.instantiate(); Pkg.update();
end

# load external packages -
using Statistics
using LinearAlgebra
using Random
using Downloads
using DataFrames
using CSV
using PrettyTables
using Flux
using NNlib
using OneHotArrays
using Plots
using JLD2
using Colors
using Luxor

# load local source files -
include(joinpath(_PATH_TO_SRC, "GloVeUtils.jl"));
include(joinpath(_PATH_TO_SRC, "Reviews.jl"));
isfile(joinpath(_PATH_TO_SRC, "Embeddings.jl")) && include(joinpath(_PATH_TO_SRC, "Embeddings.jl"));
isfile(joinpath(_PATH_TO_SRC, "Attention.jl")) && include(joinpath(_PATH_TO_SRC, "Attention.jl"));
isfile(joinpath(_PATH_TO_SRC, "Figures.jl")) && include(joinpath(_PATH_TO_SRC, "Figures.jl"));
