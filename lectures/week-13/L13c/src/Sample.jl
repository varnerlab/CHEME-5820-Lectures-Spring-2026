"""
    sample_next_token(logits::AbstractVector{Float32}; temperature=1.0, top_k=nothing,
                       rng=Random.default_rng()) -> Int

Sample one token id from a logit vector using temperature scaling and optional
top-k truncation.

* `temperature = 0` (or `nothing`) returns the argmax (greedy decoding).
* `temperature > 0` divides the logits by `temperature`, applies softmax, and
  samples from the resulting distribution. Smaller temperatures concentrate
  probability mass on the most likely tokens; larger temperatures flatten it.
* `top_k`, if not `nothing`, restricts sampling to the `k` highest-logit tokens
  (everything else is set to `-Inf` before softmax).
"""
function sample_next_token(logits::AbstractVector{Float32};
                            temperature::Real = 1.0,
                            top_k::Union{Nothing, Int} = nothing,
                            rng::AbstractRNG = Random.default_rng())::Int
    if temperature == 0
        return argmax(logits)
    end

    scaled = Float32.(logits ./ Float32(temperature))

    if top_k !== nothing && top_k > 0 && top_k < length(scaled)
        # zero out everything except the top-k entries by sending them to -Inf
        threshold = partialsort(scaled, length(scaled) - top_k + 1; rev = false)
        scaled = ifelse.(scaled .< threshold, -1.0f9, scaled)
    end

    probs = softmax(scaled)
    return StatsBase.sample(rng, 1:length(probs), Weights(probs))
end

"""
    generate(model::NanoGPT, prompt_ids::Vector{Int}, n_new_tokens::Int;
              temperature=1.0, top_k=nothing, rng=Random.default_rng()) -> Vector{Int}

Autoregressively extend a prompt by `n_new_tokens` tokens. At each step the
model is fed the most recent `min(end, ctx_len)` tokens of the running sequence,
the next-position logits are extracted, and one token is sampled and appended.

Returns the full sequence (prompt followed by all newly generated tokens).
"""
function generate(model::NanoGPT, prompt_ids::AbstractVector{<:Integer},
                   n_new_tokens::Int;
                   temperature::Real = 1.0,
                   top_k::Union{Nothing, Int} = nothing,
                   rng::AbstractRNG = Random.default_rng())::Vector{Int}

    ids = Vector{Int}(prompt_ids)
    ctx = model.ctx_len

    for _ in 1:n_new_tokens
        # window the context: take the last ctx_len tokens
        window_start = max(1, length(ids) - ctx + 1)
        window = ids[window_start:end]
        T = length(window)

        # forward pass: shape (vocab_size, T, 1)
        X = reshape(Int.(window), T, 1)
        logits = model(X)

        # take logits at the LAST position
        last_logits = vec(logits[:, end, 1])

        next_id = sample_next_token(last_logits; temperature = temperature,
                                     top_k = top_k, rng = rng)
        push!(ids, next_id)
    end

    return ids
end

"""
    sample_text(model::NanoGPT, vocab::CharVocabulary, prompt::String, n_new_tokens::Int;
                 temperature=1.0, top_k=nothing, rng=Random.default_rng()) -> String

Convenience wrapper around `generate` that takes a string prompt and a character
vocabulary, encodes the prompt, generates `n_new_tokens` characters, and decodes
the full sequence back to a string.
"""
function sample_text(model::NanoGPT, vocab::CharVocabulary, prompt::String,
                      n_new_tokens::Int;
                      temperature::Real = 1.0,
                      top_k::Union{Nothing, Int} = nothing,
                      rng::AbstractRNG = Random.default_rng())::String
    ids = encode(prompt, vocab)
    full_ids = generate(model, ids, n_new_tokens;
                         temperature = temperature, top_k = top_k, rng = rng)
    return decode(full_ids, vocab)
end
