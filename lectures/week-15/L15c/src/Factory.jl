"""
    build(modeltype::Type{MyGCNLayer};
        d_in::Int64, d_out::Int64,
        activation::Function = relu,
        rng::AbstractRNG = Random.GLOBAL_RNG) -> MyGCNLayer

Construct a `MyGCNLayer` with weight matrix `W` of shape `(d_in, d_out)`
initialized to Glorot uniform: each entry is drawn from
`Uniform(-bound, bound)` with `bound = sqrt(6 / (d_in + d_out))`.

### Keyword arguments
- `d_in`: input feature dimension.
- `d_out`: output feature dimension.
- `activation`: elementwise activation, applied componentwise. Defaults to `relu`.
- `rng`: random number generator used to draw `W`.

### Returns
- a populated `MyGCNLayer` instance with `W` initialized.
"""
function build(modeltype::Type{MyGCNLayer};
    d_in::Int64,
    d_out::Int64,
    activation::Function = relu,
    rng::AbstractRNG = Random.GLOBAL_RNG)::MyGCNLayer

    bound = sqrt(6.0 / (d_in + d_out))
    W = bound .* (2.0 .* rand(rng, d_in, d_out) .- 1.0)

    layer = modeltype()
    layer.d_in = d_in
    layer.d_out = d_out
    layer.W = W
    layer.activation = activation
    return layer
end
