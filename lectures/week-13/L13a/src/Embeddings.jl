"""
    _tokenize(text) -> Vector{String}

Lowercase the input text, split on whitespace, strip punctuation from each token,
and discard any empty tokens.
"""
function _tokenize(text::String)::Vector{String}
    raw = split(lowercase(text))
    tokens = [replace(w, r"[^a-z0-9]" => "") for w in raw]
    return filter(!isempty, tokens)
end

"""
    embed_review(review, glove; dim=100) -> Vector{Float32}

Represent a text review as the element-wise mean of its word GloVe vectors.
Words not in the GloVe vocabulary are skipped. Returns a zero vector if no words
match. This is the bag-of-embeddings baseline used in the L10b lab.
"""
function embed_review(review::String, glove::Dict{String, Vector{Float32}};
                       dim::Int=100)::Vector{Float32}

    words = _tokenize(review)
    vecs = [glove[w] for w in words if haskey(glove, w)]
    isempty(vecs) && return zeros(Float32, dim)
    return mean(vecs)
end

"""
    embed_reviews(reviews, glove; dim=100) -> Matrix{Float32}

Embed a vector of review strings into a (dim x n) matrix where each column
is the mean GloVe vector for one review.
"""
function embed_reviews(reviews::Vector{String}, glove::Dict{String, Vector{Float32}};
                        dim::Int=100)::Matrix{Float32}

    n = length(reviews)
    X = zeros(Float32, dim, n)
    for i in 1:n
        X[:, i] = embed_review(reviews[i], glove; dim=dim)
    end
    return X
end

"""
    embed_review_sequence(review, glove; dim=100, n_max=32)
        -> Tuple{Matrix{Float32}, Vector{Bool}, Vector{String}}

Represent a text review as a padded sequence of word GloVe vectors instead of a
single mean-pooled vector. Returns `(X, mask, tokens)` where

* `X` is `(dim, n_max)`, with column `i` equal to the GloVe vector for the
  `i`-th in-vocabulary token of the review, or zero if `i` is past the end.
* `mask` is a length-`n_max` `Vector{Bool}`, true for valid positions and false
  for padding positions.
* `tokens` is a length-`n_max` `Vector{String}` holding the token strings at
  valid positions (used for attention visualization). Padded positions hold the
  empty string.

Reviews longer than `n_max` in-vocabulary tokens are truncated.
"""
function embed_review_sequence(review::String,
                                glove::Dict{String, Vector{Float32}};
                                dim::Int=100,
                                n_max::Int=32)::Tuple{Matrix{Float32}, Vector{Bool}, Vector{String}}

    words = _tokenize(review)
    kept = [w for w in words if haskey(glove, w)]
    if length(kept) > n_max
        kept = kept[1:n_max]
    end

    X = zeros(Float32, dim, n_max)
    mask = falses(n_max)
    tokens = fill("", n_max)
    for (i, w) in enumerate(kept)
        X[:, i] = glove[w]
        mask[i] = true
        tokens[i] = w
    end
    return X, mask, tokens
end

"""
    embed_reviews_sequence(reviews, glove; dim=100, n_max=32)
        -> Tuple{Array{Float32,3}, Matrix{Bool}, Matrix{String}}

Embed a vector of reviews as a batched tensor of padded token-vector sequences.
Returns `(X, mask, tokens_str)` where

* `X` is `(dim, n_max, N)` with `N = length(reviews)`.
* `mask` is `(n_max, N)`, true at valid positions.
* `tokens_str` is `(n_max, N)`, holding the token strings at valid positions and
  the empty string at padded positions.
"""
function embed_reviews_sequence(reviews::Vector{String},
                                 glove::Dict{String, Vector{Float32}};
                                 dim::Int=100,
                                 n_max::Int=32)::Tuple{Array{Float32, 3}, Matrix{Bool}, Matrix{String}}

    N = length(reviews)
    X = zeros(Float32, dim, n_max, N)
    mask = falses(n_max, N)
    tokens_str = fill("", n_max, N)
    for i in 1:N
        Xi, mi, ti = embed_review_sequence(reviews[i], glove; dim=dim, n_max=n_max)
        X[:, :, i] = Xi
        mask[:, i] = mi
        tokens_str[:, i] = ti
    end
    return X, mask, tokens_str
end
