"""
    SelfAttention(d, d_k, d_v)

Single-head scaled dot-product self-attention as a Flux layer. Holds three
learnable projection matrices `W_Q`, `W_K`, `W_V` of shape `(d, d_k)`,
`(d, d_k)`, and `(d, d_v)` respectively, where `d` is the input embedding
dimension and `d_k`, `d_v` are the key and value dimensions.

When called as `sa(X, mask)`:

* `X` is a `(d, n, B)` tensor of token embeddings (n = sequence length, B = batch size)
* `mask` is a `(n, B)` `Bool` matrix where `true` marks valid (non-padding) positions

Returns `(output, weights)` where `output` is `(d_v, n, B)` and `weights` is
`(n, n, B)` with `weights[i, j, b]` giving the attention weight from query `i`
to key `j` in batch `b`.

Implements the equation
    softmax( Q Kᵀ / sqrt(d_k) ) V
from the L13a lecture, with key positions corresponding to padding masked out
to large negative scores so they receive zero attention weight.
"""
struct SelfAttention
    WQ::Matrix{Float32}
    WK::Matrix{Float32}
    WV::Matrix{Float32}
end

Flux.@layer SelfAttention

function SelfAttention(d::Int, d_k::Int, d_v::Int)
    s = 1.0f0 / sqrt(Float32(d))
    return SelfAttention(
        randn(Float32, d, d_k) .* s,
        randn(Float32, d, d_k) .* s,
        randn(Float32, d, d_v) .* s,
    )
end

function (sa::SelfAttention)(X::AbstractArray{<:Real, 3},
                              mask::AbstractMatrix{Bool})
    d, n, B = size(X)
    d_k = size(sa.WQ, 2)

    # project to Q, K, V using a flattened (d, n*B) matmul, then reshape back
    Xmat = reshape(X, d, n * B)                       # (d, n*B)
    Q = reshape(sa.WQ' * Xmat, d_k, n, B)             # (d_k, n, B)
    K = reshape(sa.WK' * Xmat, d_k, n, B)             # (d_k, n, B)
    V = reshape(sa.WV' * Xmat, :, n, B)               # (d_v, n, B)

    # scores[i, j, b] = (1/sqrt(d_k)) * <Q[:, i, b], K[:, j, b]>
    scores = NNlib.batched_mul(permutedims(Q, (2, 1, 3)), K) ./ sqrt(Float32(d_k))

    # mask out padding key positions: subtract a large constant where mask is false
    mask_f = reshape(Float32.(mask), 1, n, B)         # (1, n, B), broadcasts over query dim
    scores = scores .+ (mask_f .- 1.0f0) .* 1.0f9

    # row-wise softmax over the key dimension
    weights = softmax(scores; dims = 2)               # (n, n, B), each row sums to 1

    # output[k, i, b] = sum_j weights[i, j, b] * V[k, j, b]
    output = NNlib.batched_mul(V, permutedims(weights, (2, 1, 3)))   # (d_v, n, B)

    return output, weights
end

"""
    masked_mean_pool(out, mask) -> Matrix{Float32}

Mean-pool a `(d, n, B)` tensor along its sequence dimension, averaging only over
positions where `mask` is true. Returns a `(d, B)` matrix.
"""
function masked_mean_pool(out::AbstractArray{<:Real, 3},
                           mask::AbstractMatrix{Bool})
    _, n, B = size(out)
    mask_f = reshape(Float32.(mask), 1, n, B)
    counts = sum(mask_f; dims = 2)                    # (1, 1, B)
    pooled = sum(out .* mask_f; dims = 2) ./ max.(counts, 1.0f0)
    return dropdims(pooled; dims = 2)                 # (d, B)
end

"""
    AttentionClassifier(d_emb, d_k, d_v, d_hidden, n_classes)

A classifier built from a single self-attention layer followed by mean-pool over
valid sequence positions and a small two-layer MLP head. Takes padded token
sequences `(X, mask)` and returns class logits.
"""
struct AttentionClassifier
    attention::SelfAttention
    head::Chain
end

Flux.@layer AttentionClassifier

function AttentionClassifier(d_emb::Int, d_k::Int, d_v::Int,
                              d_hidden::Int, n_classes::Int)
    return AttentionClassifier(
        SelfAttention(d_emb, d_k, d_v),
        Chain(Dense(d_v => d_hidden, relu), Dense(d_hidden => n_classes)),
    )
end

function (m::AttentionClassifier)(X::AbstractArray{<:Real, 3},
                                   mask::AbstractMatrix{Bool})
    out, _ = m.attention(X, mask)
    pooled = masked_mean_pool(out, mask)
    return m.head(pooled)
end

"""
    attention_weights(m::AttentionClassifier, X, mask) -> Array{Float32, 3}

Forward `m` over `(X, mask)` and return only the attention weights `(n, n, B)`,
discarding the classification logits. Useful for attention-pattern visualization.
"""
function attention_weights(m::AttentionClassifier,
                            X::AbstractArray{<:Real, 3},
                            mask::AbstractMatrix{Bool})::Array{Float32, 3}
    _, w = m.attention(X, mask)
    return w
end
