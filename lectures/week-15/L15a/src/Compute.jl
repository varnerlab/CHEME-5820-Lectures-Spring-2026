"""
    random_sparse_pair(m::Int; sparsity::Float64 = 0.1,
        rng::AbstractRNG = Random.GLOBAL_RNG) -> Tuple{Vector{Float64}, Vector{Float64}}

Draw a random sparse non-negative key-value pair `(k, v)` in `R^m`. Each entry
is drawn from `Uniform(0, 1)` with probability `sparsity` and is zero otherwise.
This mimics the regime that the ReLU encoders in H-Mem produce: sparse,
non-negative key and value activations.

### Arguments
- `m::Int`: dimension of the key and value vectors.

### Keyword arguments
- `sparsity::Float64`: probability that any individual entry of `k` or `v` is
   non-zero. Defaults to `0.1`.
- `rng::AbstractRNG`: random number generator.

### Returns
- `(k::Vector{Float64}, v::Vector{Float64})`, both of length `m`.
"""
function random_sparse_pair(m::Int; sparsity::Float64 = 0.1,
    rng::AbstractRNG = Random.GLOBAL_RNG)::Tuple{Vector{Float64}, Vector{Float64}}

    @assert 0.0 < sparsity <= 1.0 "sparsity must lie in (0, 1]"
    k = [rand(rng) < sparsity ? rand(rng) : 0.0 for _ in 1:m]
    v = [rand(rng) < sparsity ? rand(rng) : 0.0 for _ in 1:m]
    return k, v
end

"""
    hebbian_write!(model::MyHMemMemory, k::Vector{Float64}, v::Vector{Float64})
        -> MyHMemMemory

Apply one step of the Hebbian write rule from
[Limbacher and Legenstein (NeurIPS 2020)](https://proceedings.neurips.cc/paper/2020/file/f6876a9f998f6472cc26708e27444456-Paper.pdf),
Eq. 1, to the in-place association matrix `model.W`:

    ΔW[k_idx, j] = γ_plus * (w_max - W[k_idx, j]) * v[k_idx] * k[j]
                 - γ_minus * W[k_idx, j] * k[j]^2

The first term strengthens the synapse between co-active pre-synaptic
(key-side) coordinate `j` and post-synaptic (value-side) coordinate `k_idx`.
The second term, in the spirit of [Oja's rule](https://link.springer.com/article/10.1007/BF00275687),
weakens connections from currently active key-coordinates to all targets, so
that fresh associations overwrite stale ones.

### Arguments
- `model`: `MyHMemMemory` whose association matrix `W` is updated in place.
- `k::Vector{Float64}`: key vector of length `m` (pre-synaptic activation).
- `v::Vector{Float64}`: value vector of length `m` (post-synaptic activation).

### Returns
- `model`, with `W` updated in place.
"""
function hebbian_write!(model::MyHMemMemory,
    k::Vector{Float64}, v::Vector{Float64})::MyHMemMemory

    @assert length(k) == model.m "key length $(length(k)) does not match m = $(model.m)"
    @assert length(v) == model.m "value length $(length(v)) does not match m = $(model.m)"

    γ_plus  = model.γ_plus
    γ_minus = model.γ_minus
    w_max   = model.w_max
    W       = model.W

    for j in 1:model.m, k_idx in 1:model.m
        plus_term  = γ_plus  * (w_max - W[k_idx, j]) * v[k_idx] * k[j]
        minus_term = γ_minus * W[k_idx, j] * k[j]^2
        W[k_idx, j] += plus_term - minus_term
    end
    return model
end

"""
    hebbian_read(model::MyHMemMemory, k_query::Vector{Float64}) -> Vector{Float64}

Read out the value vector associated with `k_query` by applying the matrix-vector
product `v_recall = W * k_query`. This is the simple feedforward read rule from
[Limbacher and Legenstein (NeurIPS 2020)](https://proceedings.neurips.cc/paper/2020/file/f6876a9f998f6472cc26708e27444456-Paper.pdf),
Eq. 5.

### Arguments
- `model`: `MyHMemMemory` whose association matrix `W` is queried.
- `k_query::Vector{Float64}`: query key of length `m`.

### Returns
- `v_recall::Vector{Float64}` of length `m`.
"""
function hebbian_read(model::MyHMemMemory, k_query::Vector{Float64})::Vector{Float64}
    @assert length(k_query) == model.m "query length $(length(k_query)) does not match m = $(model.m)"
    return model.W * k_query
end

"""
    corrupt_key(k::Vector{Float64}, p::Float64;
        rng::AbstractRNG = Random.GLOBAL_RNG) -> Vector{Float64}

Return a corrupted copy of the key `k` in which a fraction `p` of the entries
are replaced by an independent `Uniform(0, 1)` sample. The remaining `(1 - p)`
fraction of entries are passed through unchanged. `p = 0` returns a perfect
copy, `p = 1` returns an entirely fresh random key.

### Arguments
- `k::Vector{Float64}`: clean key vector.
- `p::Float64`: corruption probability per entry, in `[0, 1]`.

### Keyword arguments
- `rng::AbstractRNG`: random number generator.

### Returns
- `k_noisy::Vector{Float64}` of the same length as `k`.
"""
function corrupt_key(k::Vector{Float64}, p::Float64;
    rng::AbstractRNG = Random.GLOBAL_RNG)::Vector{Float64}

    @assert 0.0 <= p <= 1.0 "corruption probability p must lie in [0, 1]"
    return [rand(rng) < p ? rand(rng) : k[i] for i in eachindex(k)]
end
