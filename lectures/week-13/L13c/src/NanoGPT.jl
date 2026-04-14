"""
    NanoGPT(vocab_size, d_model, n_heads, n_layers, ctx_len; d_ff = 4 * d_model)

A small decoder-only language model in the GPT-style architecture. Maps a
`(T, B)` matrix of token ids to a `(vocab_size, T, B)` tensor of unnormalized
next-token logits.

Architecture
============

```
ids (T, B) → token embedding (d_model, T, B)
            + learned positional embedding (d_model, T, B)
            → DecoderBlock × L
            → final LayerNorm
            → LM head (Dense d_model → vocab_size)
            → logits (vocab_size, T, B)
```

Both the token embedding and the positional embedding are stored as raw
`(d_model, vocab_size)` and `(d_model, ctx_len)` matrices and looked up by
column indexing. The LM head is a linear layer with no bias.
"""
struct NanoGPT
    tok_emb::Matrix{Float32}     # (d_model, vocab_size)
    pos_emb::Matrix{Float32}     # (d_model, ctx_len)
    blocks::Vector{DecoderBlock}
    ln_final::LayerNorm
    lm_head::Dense
    ctx_len::Int
end

Flux.@layer NanoGPT trainable=(tok_emb, pos_emb, blocks, ln_final, lm_head)

function NanoGPT(vocab_size::Int, d_model::Int, n_heads::Int, n_layers::Int, ctx_len::Int;
                  d_ff::Int = 4 * d_model)
    s = 1.0f0 / sqrt(Float32(d_model))
    return NanoGPT(
        randn(Float32, d_model, vocab_size) .* s,
        randn(Float32, d_model, ctx_len) .* s,
        [DecoderBlock(d_model, n_heads, ctx_len; d_ff = d_ff) for _ in 1:n_layers],
        LayerNorm(d_model),
        Dense(d_model => vocab_size; bias = false),
        ctx_len,
    )
end

function (m::NanoGPT)(X_ids::AbstractMatrix{<:Integer})
    T, B = size(X_ids)
    @assert T <= m.ctx_len "input sequence length T=$T exceeds model context length $(m.ctx_len)"
    d_model = size(m.tok_emb, 1)

    # token embedding lookup: (d_model, T*B), then reshape to (d_model, T, B)
    flat_ids = vec(X_ids)
    tok_e = m.tok_emb[:, flat_ids]
    H = reshape(tok_e, d_model, T, B)

    # add positional embedding (broadcasts over the batch dim)
    pos_e = m.pos_emb[:, 1:T]
    H = H .+ reshape(pos_e, d_model, T, 1)

    # apply decoder blocks
    for block in m.blocks
        H = block(H)
    end

    # final layer norm
    H = m.ln_final(H)

    # LM head: (d_model, T, B) → (vocab_size, T, B)
    return m.lm_head(H)
end

"""
    nanogpt_loss(model::NanoGPT, X_ids, Y_ids) -> Float32

Cross-entropy loss for next-token prediction. `X_ids` and `Y_ids` are both
`(T, B)` integer matrices of input tokens and shifted targets respectively.
"""
function nanogpt_loss(model::NanoGPT,
                       X_ids::AbstractMatrix{<:Integer},
                       Y_ids::AbstractMatrix{<:Integer})
    logits = model(X_ids)
    V, T, B = size(logits)
    return Flux.logitcrossentropy(reshape(logits, V, T * B),
                                   Flux.onehotbatch(vec(Y_ids), 1:V))
end

"""
    n_parameters(model::NanoGPT) -> Int

Return the total number of trainable parameters in the model.
"""
n_parameters(model::NanoGPT)::Int = sum(length, Flux.trainables(model))
