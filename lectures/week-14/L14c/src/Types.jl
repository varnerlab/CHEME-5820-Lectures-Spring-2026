"""
    MyMimoLegSHippoModel

Discretized multi-input multi-output (MIMO) structured state space model with
block-diagonal HiPPO-LegS initialization.

The continuous-time system is

    dx/dt = A x + B u
        y = C x + D u

with vector input `u ∈ ℝ^{d_in}` and vector output `y ∈ ℝ^{d_out}`. The block-
diagonal construction runs one independent LegS filter per input channel: the
total hidden-state dimension is `H = h * d_in`, `A ∈ ℝ^{H×H}` is block-diagonal
with `d_in` copies of the `h×h` LegS state matrix, and `B ∈ ℝ^{H×d_in}` is
block-rectangular so that input channel `j` drives only its own `h`-dimensional
block of the hidden state. The readout `C ∈ ℝ^{d_out×H}` is dense, so every
output channel can mix information from every input channel's history.

# Fields
- `h::Int`: per-channel hidden-state dimension (LegS basis order).
- `d_in::Int`: number of input channels.
- `d_out::Int`: number of output channels.
- `Δt::Float64`: time step used to discretize the continuous system.
- `A::Matrix{Float64}`: continuous-time state matrix, size `(h*d_in, h*d_in)`.
- `B::Matrix{Float64}`: continuous-time input matrix, size `(h*d_in, d_in)`.
- `Ā::Matrix{Float64}`: discrete-time state matrix, size `(h*d_in, h*d_in)`.
- `B̄::Matrix{Float64}`: discrete-time input matrix, size `(h*d_in, d_in)`.
- `C::Matrix{Float64}`: readout matrix, size `(d_out, h*d_in)`; updated by `fit_C!`.
- `D::Matrix{Float64}`: feedforward matrix, size `(d_out, d_in)`.
- `x₀::Vector{Float64}`: initial hidden state, length `h*d_in`; defaults to zeros.
"""
mutable struct MyMimoLegSHippoModel
    h::Int
    d_in::Int
    d_out::Int
    Δt::Float64
    A::Matrix{Float64}
    B::Matrix{Float64}
    Ā::Matrix{Float64}
    B̄::Matrix{Float64}
    C::Matrix{Float64}
    D::Matrix{Float64}
    x₀::Vector{Float64}
end
