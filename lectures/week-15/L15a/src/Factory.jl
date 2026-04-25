"""
    build(modeltype::Type{MyHMemMemory};
        m::Int64, γ_plus::Float64 = 0.3, γ_minus::Float64 = 0.3,
        w_max::Float64 = 1.0) -> MyHMemMemory

Construct an empty `MyHMemMemory` with association matrix `W = 0` of shape
`(m, m)`. The defaults for `γ_plus`, `γ_minus`, and `w_max` follow Table 2 of
[Limbacher, Özdenizci, and Legenstein (2022)](https://arxiv.org/abs/2205.11276).

### Keyword arguments
- `m`: dimension of the key and value vectors.
- `γ_plus`: positive constant scaling the Hebbian "fire together, wire together"
   term in the write rule.
- `γ_minus`: positive constant scaling the Oja-style forgetting term.
- `w_max`: soft upper bound on entries of `W`.

### Returns
- a populated `MyHMemMemory` instance with `W` initialized to zeros.
"""
function build(modeltype::Type{MyHMemMemory};
    m::Int64,
    γ_plus::Float64 = 0.3,
    γ_minus::Float64 = 0.3,
    w_max::Float64 = 1.0)::MyHMemMemory

    model = modeltype()
    model.m       = m
    model.W       = zeros(Float64, m, m)
    model.γ_plus  = γ_plus
    model.γ_minus = γ_minus
    model.w_max   = w_max
    return model
end
