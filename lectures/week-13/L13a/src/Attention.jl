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
    WQ::Matrix{Float32}  # query projection matrix (d × d_k)
    WK::Matrix{Float32}  # key projection matrix (d × d_k)
    WV::Matrix{Float32}  # value projection matrix (d × d_v)
end

# Tell Flux that SelfAttention is a trainable layer (so WQ, WK, WV are updated during training)
Flux.@layer SelfAttention

"""
    SelfAttention(d, d_k, d_v)

Constructor: initialize projection matrices with small random values scaled by 1/√d
to keep initial outputs in a reasonable range (Xavier-style initialization).
"""
function SelfAttention(d::Int, d_k::Int, d_v::Int)
    s = 1.0f0 / sqrt(Float32(d))  # scale factor for initialization
    return SelfAttention(
        randn(Float32, d, d_k) .* s,   # W_Q: projects input → queries
        randn(Float32, d, d_k) .* s,   # W_K: projects input → keys
        randn(Float32, d, d_v) .* s,   # W_V: projects input → values
    )
end

"""
    (sa::SelfAttention)(X, mask)

Forward pass: compute scaled dot-product self-attention.

Steps:
1. Project X into Q, K, V using the learned weight matrices.
2. Compute attention scores as Q Kᵀ / √d_k.
3. Mask out padding positions by adding a large negative value to their scores.
4. Apply row-wise softmax so each query's weights sum to 1.
5. Compute the output as the weighted sum of value vectors.

Returns `(output, weights)` where:
- `output` is (d_v, n, B): the context-aware representation of each token
- `weights` is (n, n, B): the attention weight matrix (useful for visualization)
"""
function (sa::SelfAttention)(X::AbstractArray{<:Real, 3},
                              mask::AbstractMatrix{Bool})
    d, n, B = size(X)
    d_k = size(sa.WQ, 2)

    # Step 1: Project input X to queries (Q), keys (K), and values (V).
    # We flatten (d, n, B) → (d, n*B) for a single matrix multiply, then reshape back.
    Xmat = reshape(X, d, n * B)                       # (d, n*B)
    Q = reshape(sa.WQ' * Xmat, d_k, n, B)             # (d_k, n, B) — each column is a query vector
    K = reshape(sa.WK' * Xmat, d_k, n, B)             # (d_k, n, B) — each column is a key vector
    V = reshape(sa.WV' * Xmat, :, n, B)               # (d_v, n, B) — each column is a value vector

    # Step 2: Compute raw attention scores.
    # scores[i, j, b] = <Q[:, i, b], K[:, j, b]> / √d_k
    # This measures how much query i should attend to key j.
    scores = NNlib.batched_mul(permutedims(Q, (2, 1, 3)), K) ./ sqrt(Float32(d_k))  # (n, n, B)

    # Step 3: Mask out padding positions.
    # Where mask is false (padding), subtract a large constant (1e9) so softmax assigns ~0 weight.
    mask_f = reshape(Float32.(mask), 1, n, B)          # (1, n, B) — broadcasts over the query dimension
    scores = scores .+ (mask_f .- 1.0f0) .* 1.0f9     # padding positions get score ≈ -1e9

    # Step 4: Apply softmax along the key dimension (dim=2).
    # Each row of the resulting matrix is a probability distribution over keys.
    weights = softmax(scores; dims = 2)                # (n, n, B) — each row sums to 1

    # Step 5: Compute weighted sum of value vectors.
    # output[:, i, b] = Σⱼ weights[i, j, b] * V[:, j, b]
    output = NNlib.batched_mul(V, permutedims(weights, (2, 1, 3)))   # (d_v, n, B)

    return output, weights
end

"""
    masked_mean_pool(out, mask) -> Matrix{Float32}

Mean-pool a `(d, n, B)` tensor along its sequence dimension (dim=2), averaging only
over positions where `mask` is true. Returns a `(d, B)` matrix with one pooled
vector per batch element.

This collapses the sequence into a single fixed-size vector per review,
which can then be passed to the MLP classification head.
"""
function masked_mean_pool(out::AbstractArray{<:Real, 3},
                           mask::AbstractMatrix{Bool})
    _, n, B = size(out)
    mask_f = reshape(Float32.(mask), 1, n, B)          # (1, n, B) — broadcast-compatible with out
    counts = sum(mask_f; dims = 2)                     # (1, 1, B) — number of valid tokens per review
    pooled = sum(out .* mask_f; dims = 2) ./ max.(counts, 1.0f0)  # avoid division by zero
    return dropdims(pooled; dims = 2)                  # (d, B) — one vector per review
end

"""
    AttentionClassifier(d_emb, d_k, d_v, d_hidden, n_classes)

A classifier built from three components:
1. A single self-attention layer that produces context-aware token representations.
2. Masked mean-pooling that averages those representations over valid positions.
3. A two-layer MLP head (Dense → relu → Dense) that maps the pooled vector to class logits.

Takes padded token sequences `(X, mask)` and returns class logits of shape `(n_classes, B)`.
"""
struct AttentionClassifier
    attention::SelfAttention   # self-attention layer (produces context-aware token vectors)
    head::Chain                # MLP classification head (pooled vector → class logits)
end

# Tell Flux that AttentionClassifier is a trainable layer
Flux.@layer AttentionClassifier

function AttentionClassifier(d_emb::Int, d_k::Int, d_v::Int,
                              d_hidden::Int, n_classes::Int)
    return AttentionClassifier(
        SelfAttention(d_emb, d_k, d_v),                                     # attention layer
        Chain(Dense(d_v => d_hidden, relu), Dense(d_hidden => n_classes)),   # MLP head
    )
end

"""
    (m::AttentionClassifier)(X, mask) -> Matrix{Float32}

Forward pass: run self-attention over the token sequence, pool over valid positions,
and pass through the MLP head to produce class logits.
"""
function (m::AttentionClassifier)(X::AbstractArray{<:Real, 3},
                                   mask::AbstractMatrix{Bool})
    out, _ = m.attention(X, mask)    # (d_v, n, B) — context-aware token representations
    pooled = masked_mean_pool(out, mask)  # (d_v, B) — one vector per review
    return m.head(pooled)            # (n_classes, B) — class logits
end

"""
    attention_weights(m::AttentionClassifier, X, mask) -> Array{Float32, 3}

Forward `m` over `(X, mask)` and return only the attention weights `(n, n, B)`,
discarding the classification logits. Each entry `weights[i, j, b]` gives the
attention weight from query token `i` to key token `j` in batch element `b`.

This is used for post-hoc visualization of what the model attends to.
"""
function attention_weights(m::AttentionClassifier,
                            X::AbstractArray{<:Real, 3},
                            mask::AbstractMatrix{Bool})::Array{Float32, 3}
    _, w = m.attention(X, mask)
    return w
end
