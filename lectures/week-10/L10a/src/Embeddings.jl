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
Words not in the GloVe vocabulary are skipped. Returns a zero vector if no words match.
"""
function embed_review(review::String, glove::Dict{String, Vector{Float32}}; dim::Int=100)::Vector{Float32}

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
function embed_reviews(reviews::Vector{String}, glove::Dict{String, Vector{Float32}}; dim::Int=100)::Matrix{Float32}

    n = length(reviews)
    X = zeros(Float32, dim, n)
    for i in 1:n
        X[:, i] = embed_review(reviews[i], glove; dim=dim)
    end
    return X
end
