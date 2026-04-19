"""
    MySisoLegSHippoModel

Discretized single-input single-output structured state space model with
HiPPO-LegS initialization.

The continuous-time system is

    dx/dt = A x + B u
        y = C x + D u

where `A ∈ ℝ^{h×h}` and `B ∈ ℝ^{h×1}` are the LegS HiPPO matrices, `C ∈ ℝ^{1×h}`
is a learned readout row vector, and `D ∈ ℝ` is a scalar feedforward term
(fixed to 0 for the lectures). The model stores both the continuous matrices
`(A, B)` and the bilinear-discretized matrices `(Ā, B̄)` used for training
and inference.

# Fields
- `h::Int`: hidden-state dimension (number of LegS basis coefficients).
- `Δt::Float64`: time step used to discretize the continuous system.
- `A::Matrix{Float64}`: continuous-time state matrix, size `(h, h)`.
- `B::Matrix{Float64}`: continuous-time input matrix, size `(h, 1)`.
- `Ā::Matrix{Float64}`: discrete-time state matrix, size `(h, h)`.
- `B̄::Matrix{Float64}`: discrete-time input matrix, size `(h, 1)`.
- `C::Matrix{Float64}`: readout row vector, size `(1, h)`; updated by `fit_C!`.
- `D::Float64`: feedforward scalar; set to 0 unless the task requires it.
- `x₀::Vector{Float64}`: initial hidden state, length `h`; defaults to zeros.
"""
mutable struct MySisoLegSHippoModel
    h::Int
    Δt::Float64
    A::Matrix{Float64}
    B::Matrix{Float64}
    Ā::Matrix{Float64}
    B̄::Matrix{Float64}
    C::Matrix{Float64}
    D::Float64
    x₀::Vector{Float64}
end
