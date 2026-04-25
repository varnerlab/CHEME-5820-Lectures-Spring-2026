abstract type AbstractHebbianMemoryModel end

"""
    MyHMemMemory

Rate-domain Hebbian Memory module from Limbacher and Legenstein (NeurIPS 2020).
Holds the dynamic association matrix `W` together with the constants of the
write rule (`γ_plus`, `γ_minus`, `w_max`).

### Fields
- `m::Int64`: dimension of the key-vector and value-vector (square memory).
- `W::Matrix{Float64}`: association matrix of shape `(m, m)`. Element `W[k, j]`
   stores the synaptic weight from pre-synaptic key-coordinate `j` to
   post-synaptic value-coordinate `k`.
- `γ_plus::Float64`:  positive constant scaling the Hebbian "fire together,
   wire together" term.
- `γ_minus::Float64`: positive constant scaling the Oja-style forgetting term.
- `w_max::Float64`: soft upper bound on entries of `W`.
"""
mutable struct MyHMemMemory <: AbstractHebbianMemoryModel
    m::Int64
    W::Matrix{Float64}
    γ_plus::Float64
    γ_minus::Float64
    w_max::Float64

    MyHMemMemory() = new()
end
