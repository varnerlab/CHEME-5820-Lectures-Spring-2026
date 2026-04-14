# setup paths -
const _ROOT = @__DIR__
const _PATH_TO_DATA = joinpath(_ROOT, "data");
const _PATH_TO_SRC = joinpath(_ROOT, "src");
const _PATH_TO_FIGS = joinpath(_ROOT, "figs");

!isdir(_PATH_TO_DATA) && mkpath(_PATH_TO_DATA);
!isdir(_PATH_TO_FIGS) && mkpath(_PATH_TO_FIGS);

using Pkg;
Pkg.activate(_ROOT);
if (isfile(joinpath(_ROOT, "Manifest.toml")) == false)
    Pkg.resolve(); Pkg.instantiate(); Pkg.update();
end

# load external packages -
using Statistics
using LinearAlgebra
using Random
using Downloads
using DataFrames
using PrettyTables
using Flux
using NNlib
using OneHotArrays
using Plots
using JLD2
using StatsBase

# set the random seed for reproducibility -
Random.seed!(42);

# load local source files -
isfile(joinpath(_PATH_TO_SRC, "Shakespeare.jl"))     && include(joinpath(_PATH_TO_SRC, "Shakespeare.jl"));
isfile(joinpath(_PATH_TO_SRC, "CausalAttention.jl")) && include(joinpath(_PATH_TO_SRC, "CausalAttention.jl"));
isfile(joinpath(_PATH_TO_SRC, "DecoderBlock.jl"))    && include(joinpath(_PATH_TO_SRC, "DecoderBlock.jl"));
isfile(joinpath(_PATH_TO_SRC, "NanoGPT.jl"))         && include(joinpath(_PATH_TO_SRC, "NanoGPT.jl"));
isfile(joinpath(_PATH_TO_SRC, "Sample.jl"))          && include(joinpath(_PATH_TO_SRC, "Sample.jl"));
