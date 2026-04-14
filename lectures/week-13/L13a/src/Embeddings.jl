"""
    _tokenize(text) -> Vector{String}

Lowercase the input text, split on whitespace, strip punctuation from each token,
and discard any empty tokens. This produces a clean list of lowercase words
suitable for GloVe lookup.
"""
function _tokenize(text::String)::Vector{String}
    raw = split(lowercase(text))                           # split on whitespace, lowercase
    tokens = [replace(w, r"[^a-z0-9]" => "") for w in raw] # remove all non-alphanumeric characters
    return filter(!isempty, tokens)                         # drop any tokens that became empty
end

"""
    embed_review(review, glove; dim=100) -> Vector{Float32}

Represent a text review as the element-wise mean of its word GloVe vectors
(bag-of-embeddings). Words not in the GloVe vocabulary are skipped.
Returns a zero vector if no words match.

This is the baseline representation used in the L10b lab. It collapses the
entire review into a single vector, losing all word order information.
"""
function embed_review(review::String, glove::Dict{String, Vector{Float32}};
                       dim::Int=100)::Vector{Float32}

    words = _tokenize(review)                               # tokenize the review text
    vecs = [glove[w] for w in words if haskey(glove, w)]    # look up GloVe vectors for in-vocabulary words
    isempty(vecs) && return zeros(Float32, dim)              # return zero vector if no words found
    return mean(vecs)                                        # element-wise mean of all word vectors
end

"""
    embed_reviews(reviews, glove; dim=100) -> Matrix{Float32}

Embed a vector of review strings into a `(dim, n)` matrix where each column
is the mean GloVe vector for one review (bag-of-embeddings baseline).
"""
function embed_reviews(reviews::Vector{String}, glove::Dict{String, Vector{Float32}};
                        dim::Int=100)::Matrix{Float32}

    n = length(reviews)
    X = zeros(Float32, dim, n)          # pre-allocate output matrix (dim × n_reviews)
    for i in 1:n
        X[:, i] = embed_review(reviews[i], glove; dim=dim)  # fill column i with mean embedding
    end
    return X
end

"""
    embed_review_sequence(review, glove; dim=100, n_max=32)
        -> Tuple{Matrix{Float32}, Vector{Bool}, Vector{String}}

Represent a text review as a padded sequence of word GloVe vectors (instead of a
single mean-pooled vector). Returns `(X, mask, tokens)` where:

* `X` is `(dim, n_max)`: column `i` holds the GloVe vector for the `i`-th in-vocabulary
  token, or zeros if `i` is past the end of the review (padding).
* `mask` is a length-`n_max` `Vector{Bool}`: true at valid token positions, false at padding.
* `tokens` is a length-`n_max` `Vector{String}`: the token string at each valid position
  (used for attention heatmap labels). Padding positions hold the empty string.

Reviews longer than `n_max` in-vocabulary tokens are truncated.
"""
function embed_review_sequence(review::String,
                                glove::Dict{String, Vector{Float32}};
                                dim::Int=100,
                                n_max::Int=32)::Tuple{Matrix{Float32}, Vector{Bool}, Vector{String}}

    words = _tokenize(review)                               # tokenize the review
    kept = [w for w in words if haskey(glove, w)]           # keep only words with GloVe vectors
    if length(kept) > n_max
        kept = kept[1:n_max]                                # truncate to max sequence length
    end

    # initialize outputs: zero embeddings, all-false mask, empty token strings
    X = zeros(Float32, dim, n_max)
    mask = falses(n_max)
    tokens = fill("", n_max)

    # fill in valid positions with GloVe vectors and token strings
    for (i, w) in enumerate(kept)
        X[:, i] = glove[w]      # embedding vector for token i
        mask[i] = true           # mark position i as valid (not padding)
        tokens[i] = w            # store the token string for visualization
    end
    return X, mask, tokens
end

"""
    embed_reviews_sequence(reviews, glove; dim=100, n_max=32)
        -> Tuple{Array{Float32,3}, Matrix{Bool}, Matrix{String}}

Batch version of `embed_review_sequence`. Embeds a vector of reviews as a 3D tensor
of padded token-vector sequences. Returns `(X, mask, tokens_str)` where:

* `X` is `(dim, n_max, N)` with `N = length(reviews)`.
* `mask` is `(n_max, N)`: true at valid positions for each review.
* `tokens_str` is `(n_max, N)`: token strings at valid positions, empty strings at padding.
"""
function embed_reviews_sequence(reviews::Vector{String},
                                 glove::Dict{String, Vector{Float32}};
                                 dim::Int=100,
                                 n_max::Int=32)::Tuple{Array{Float32, 3}, Matrix{Bool}, Matrix{String}}

    N = length(reviews)
    X = zeros(Float32, dim, n_max, N)       # 3D tensor: (embedding_dim, seq_len, n_reviews)
    mask = falses(n_max, N)                  # 2D mask: (seq_len, n_reviews)
    tokens_str = fill("", n_max, N)          # 2D token strings: (seq_len, n_reviews)

    # embed each review individually and stack into the batch tensors
    for i in 1:N
        Xi, mi, ti = embed_review_sequence(reviews[i], glove; dim=dim, n_max=n_max)
        X[:, :, i] = Xi             # embedding matrix for review i
        mask[:, i] = mi             # mask vector for review i
        tokens_str[:, i] = ti       # token strings for review i
    end
    return X, mask, tokens_str
end
