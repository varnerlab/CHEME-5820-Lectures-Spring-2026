"""
    CausalAttention(d_model, n_heads, ctx_len)

Multi-head causal self-attention as a Flux layer. Holds four learnable weight
matrices `W_Q`, `W_K`, `W_V`, `W_O` of shape `(d_model, d_model)` (with all `H`
heads packed into the column dimension), plus a precomputed `(ctx_len, ctx_len)`
causal mask whose entries are `-1e9` strictly above the diagonal and `0` on or
below it. The mask is stored as a non-trainable buffer.

When called as `ca(X)`:

* `X` is a `(d_model, T, B)` tensor of token embeddings (T = sequence length, B = batch size)
* `T` must satisfy `T ≤ ctx_len`

Returns the attention output as a `(d_model, T, B)` tensor. Use
[`causal_attention_weights`](@ref) to recover the per-head attention probability
matrices for visualization.
"""
struct CausalAttention
    WQ::Matrix{Float32}
    WK::Matrix{Float32}
    WV::Matrix{Float32}
    WO::Matrix{Float32}
    mask::Matrix{Float32}
    n_heads::Int
    d_head::Int
end

Flux.@layer CausalAttention trainable=(WQ, WK, WV, WO)

function CausalAttention(d_model::Int, n_heads::Int, ctx_len::Int)
    @assert d_model % n_heads == 0 "d_model ($d_model) must be divisible by n_heads ($n_heads)"
    d_head = d_model ÷ n_heads
    s = 1.0f0 / sqrt(Float32(d_model))

    # build causal mask: 0 on/below the main diagonal, -1e9 strictly above
    mask = zeros(Float32, ctx_len, ctx_len)
    for i in 1:ctx_len, j in 1:ctx_len
        if j > i
            mask[i, j] = -1.0f9
        end
    end

    return CausalAttention(
        randn(Float32, d_model, d_model) .* s,
        randn(Float32, d_model, d_model) .* s,
        randn(Float32, d_model, d_model) .* s,
        randn(Float32, d_model, d_model) .* s,
        mask,
        n_heads,
        d_head,
    )
end

function _causal_attention_forward(ca::CausalAttention, X::AbstractArray{<:Real, 3})
    d_model, T, B = size(X)
    n_heads = ca.n_heads
    d_head = ca.d_head

    # project to Q, K, V via a flattened matmul, then reshape back to 3D
    Xmat = reshape(X, d_model, T * B)
    Q = reshape(ca.WQ' * Xmat, d_model, T, B)
    K = reshape(ca.WK' * Xmat, d_model, T, B)
    V = reshape(ca.WV' * Xmat, d_model, T, B)

    # split each (d_model, T, B) into heads: (d_head, n_heads, T, B), then
    # move heads into the batch dimension: (d_head, T, n_heads * B)
    Q4 = reshape(Q, d_head, n_heads, T, B)
    K4 = reshape(K, d_head, n_heads, T, B)
    V4 = reshape(V, d_head, n_heads, T, B)
    Qh = reshape(permutedims(Q4, (1, 3, 2, 4)), d_head, T, n_heads * B)
    Kh = reshape(permutedims(K4, (1, 3, 2, 4)), d_head, T, n_heads * B)
    Vh = reshape(permutedims(V4, (1, 3, 2, 4)), d_head, T, n_heads * B)

    # scaled dot-product scores: (T, T, n_heads * B)
    scores = NNlib.batched_mul(permutedims(Qh, (2, 1, 3)), Kh) ./ sqrt(Float32(d_head))

    # apply causal mask: slice to (T, T) and broadcast across the batch dim
    mask_TT = ca.mask[1:T, 1:T]
    scores = scores .+ reshape(mask_TT, T, T, 1)

    # row-wise softmax over the key dimension
    weights = softmax(scores; dims = 2)

    # weighted sum of values: (d_head, T, n_heads * B)
    out = NNlib.batched_mul(Vh, permutedims(weights, (2, 1, 3)))

    # undo the head packing: (d_head, T, n_heads*B) → (d_head, n_heads, T, B) → (d_model, T, B)
    out4 = reshape(out, d_head, T, n_heads, B)
    out_perm = permutedims(out4, (1, 3, 2, 4))
    out_full = reshape(out_perm, d_model, T, B)

    # output projection
    out_mat = reshape(out_full, d_model, T * B)
    out_proj = reshape(ca.WO' * out_mat, d_model, T, B)

    return out_proj, weights
end

function (ca::CausalAttention)(X::AbstractArray{<:Real, 3})
    out, _ = _causal_attention_forward(ca, X)
    return out
end

"""
    causal_attention_weights(ca::CausalAttention, X) -> Array{Float32, 4}

Run the causal attention layer on `X` and return the attention probability
matrices, reshaped to `(T, T, n_heads, B)`. Useful for visualization.
"""
function causal_attention_weights(ca::CausalAttention,
                                   X::AbstractArray{<:Real, 3})::Array{Float32, 4}
    _, weights = _causal_attention_forward(ca, X)
    T, _, _ = size(weights)
    n_heads = ca.n_heads
    B = size(X, 3)
    return reshape(weights, T, T, n_heads, B)
end
