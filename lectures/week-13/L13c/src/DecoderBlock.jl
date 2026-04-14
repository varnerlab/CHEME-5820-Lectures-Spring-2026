"""
    DecoderBlock(d_model, n_heads, ctx_len; d_ff=4*d_model)

A single decoder-only transformer block: pre-LayerNorm, causal multi-head
self-attention, residual; then pre-LayerNorm, position-wise feedforward,
residual. The block accepts a `(d_model, T, B)` tensor and returns one of the
same shape.

Architecture
============

```
Y = X + CausalAttention(LayerNorm(X))
Z = Y + FFN(LayerNorm(Y))
```

The position-wise FFN is a two-layer MLP `Dense(d_model => d_ff, gelu)
→ Dense(d_ff => d_model)` applied independently to each token position. Both
LayerNorms operate on the `d_model` dimension.
"""
struct DecoderBlock
    ln1::LayerNorm
    attn::CausalAttention
    ln2::LayerNorm
    ffn::Chain
end

Flux.@layer DecoderBlock

function DecoderBlock(d_model::Int, n_heads::Int, ctx_len::Int; d_ff::Int = 4 * d_model)
    return DecoderBlock(
        LayerNorm(d_model),
        CausalAttention(d_model, n_heads, ctx_len),
        LayerNorm(d_model),
        Chain(Dense(d_model => d_ff, gelu), Dense(d_ff => d_model)),
    )
end

function (b::DecoderBlock)(X::AbstractArray{<:Real, 3})
    Y = X .+ b.attn(b.ln1(X))
    Z = Y .+ b.ffn(b.ln2(Y))
    return Z
end
