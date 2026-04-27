"""
    relu(x) = max(zero(x), x)

Elementwise rectified linear unit. Used as the default activation in
`MyGCNLayer`.
"""
relu(x) = max(zero(x), x)

"""
    karate_club_adjacency() -> Matrix{Float64}

Return the dense `(34, 34)` adjacency matrix of Zachary's Karate Club graph
(34 nodes, 78 undirected edges). The graph is loaded from
`Graphs.smallgraph(:karate)` and converted to a symmetric `Float64` matrix
with zeros on the diagonal.

### Returns
- `A::Matrix{Float64}` of shape `(34, 34)`, symmetric, with `A[i, j] = 1` if
   nodes `i` and `j` are connected and `0` otherwise.
"""
function karate_club_adjacency()::Matrix{Float64}
    g = smallgraph(:karate)
    n = nv(g)
    A = zeros(Float64, n, n)
    for e in edges(g)
        i, j = src(e), dst(e)
        A[i, j] = 1.0
        A[j, i] = 1.0
    end
    return A
end

"""
    karate_club_communities() -> Vector{Int64}

Return the original two-community labels from
[Zachary (1977)](https://www.jstor.org/stable/3629752): `1` for Mr. Hi's
faction (the instructor's group, 16 nodes), `2` for Officer John A.'s faction
(the administrator's group, 18 nodes). The labels are indexed `1:34`,
matching the node numbering of `Graphs.smallgraph(:karate)`.

### Returns
- `labels::Vector{Int64}` of length `34`, with each entry in `{1, 2}`.
"""
function karate_club_communities()::Vector{Int64}
    # Mr. Hi (label 1):  nodes 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 17, 18, 20, 22
    # Officer (label 2): nodes 9, 10, 15, 16, 19, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34
    mr_hi   = Set([1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 17, 18, 20, 22])
    labels  = ones(Int64, 34)
    for i in 1:34
        labels[i] = i in mr_hi ? 1 : 2
    end
    return labels
end

"""
    normalized_adjacency(A::AbstractMatrix) -> Matrix{Float64}

Return the symmetric-normalized propagation matrix
`P = D̂^(-1/2) (A + I) D̂^(-1/2)` of [Kipf and Welling (2017)](https://arxiv.org/abs/1609.02907),
where `Â = A + I` adds a self-loop at every node and `D̂` is the diagonal
degree matrix of `Â` (`D̂[i, i] = 1 + sum_j A[i, j]`).

### Arguments
- `A::AbstractMatrix`: adjacency matrix of shape `(N, N)`. Need not be binary.

### Returns
- `P::Matrix{Float64}` of shape `(N, N)`, symmetric.
"""
function normalized_adjacency(A::AbstractMatrix)::Matrix{Float64}
    @assert size(A, 1) == size(A, 2) "adjacency matrix must be square"
    n = size(A, 1)
    A_hat = A + I(n)
    d_hat = vec(sum(A_hat; dims = 2))
    D_inv_sqrt = Diagonal(1.0 ./ sqrt.(d_hat))
    return D_inv_sqrt * A_hat * D_inv_sqrt
end

"""
    gcn_forward(layer::MyGCNLayer, P::AbstractMatrix, H::AbstractMatrix) -> Matrix{Float64}

Apply one symmetric-normalized graph convolution layer to feature matrix `H`:

    H_out = layer.activation.(P * H * layer.W)

### Arguments
- `layer`: `MyGCNLayer` whose `W` and `activation` define the layer map.
- `P::AbstractMatrix`: propagation matrix of shape `(N, N)`, e.g. from
   `normalized_adjacency`.
- `H::AbstractMatrix`: input feature matrix of shape `(N, layer.d_in)`.

### Returns
- `H_out::Matrix{Float64}` of shape `(N, layer.d_out)`.
"""
function gcn_forward(layer::MyGCNLayer, P::AbstractMatrix, H::AbstractMatrix)::Matrix{Float64}
    @assert size(H, 2) == layer.d_in "H has $(size(H, 2)) input features but layer expects $(layer.d_in)"
    @assert size(P, 1) == size(P, 2) == size(H, 1) "P and H disagree on N"
    Z = P * H * layer.W
    return layer.activation.(Z)
end

"""
    gcn_stack_forward(layers::Vector{MyGCNLayer}, P::AbstractMatrix,
        X::AbstractMatrix) -> Matrix{Float64}

Apply a stack of `MyGCNLayer`s sequentially, sharing the same propagation
matrix `P` across all layers:

    H_0   = X
    H_l+1 = layers[l+1].activation.(P * H_l * layers[l+1].W)
    H_out = H_L

### Arguments
- `layers`: ordered vector of `MyGCNLayer` instances. The output dim of layer
   `l` must match the input dim of layer `l + 1`.
- `P::AbstractMatrix`: propagation matrix of shape `(N, N)`.
- `X::AbstractMatrix`: input feature matrix of shape `(N, layers[1].d_in)`.

### Returns
- `H_out::Matrix{Float64}` of shape `(N, layers[end].d_out)`.
"""
function gcn_stack_forward(layers::Vector{MyGCNLayer}, P::AbstractMatrix,
    X::AbstractMatrix)::Matrix{Float64}

    H = Matrix{Float64}(X)
    for layer in layers
        H = gcn_forward(layer, P, H)
    end
    return H
end

"""
    pca_2d(H::AbstractMatrix) -> Matrix{Float64}

Project the rows of `H` onto the top-2 principal components and return the
resulting `(N, 2)` matrix of 2D coordinates. Each column of `H` is
mean-centered before the SVD; the projection is onto the right-singular
vectors with the two largest singular values.

### Arguments
- `H::AbstractMatrix`: feature matrix of shape `(N, d)` with `d >= 2`.

### Returns
- `Y::Matrix{Float64}` of shape `(N, 2)`. Column 1 is PC1, column 2 is PC2.
"""
function pca_2d(H::AbstractMatrix)::Matrix{Float64}
    @assert size(H, 2) >= 2 "PCA needs at least 2 features"
    H_centered = H .- mean(H; dims = 1)
    F = svd(H_centered)
    return H_centered * F.V[:, 1:2]
end

"""
    pairwise_cosine_similarity(H::AbstractMatrix) -> Matrix{Float64}

Return the `(N, N)` matrix of pairwise cosine similarities between the rows
of `H`. Used to quantify how oversmoothing collapses node embeddings as the
GCN depth grows: at large depth every entry tends to one.

### Arguments
- `H::AbstractMatrix`: feature matrix of shape `(N, d)`.

### Returns
- `S::Matrix{Float64}` of shape `(N, N)`, with `S[i, j]` the cosine similarity
   between row `i` and row `j` of `H`. Rows of `H` with zero norm are mapped
   to a similarity of zero.
"""
function pairwise_cosine_similarity(H::AbstractMatrix)::Matrix{Float64}
    n = size(H, 1)
    norms = [norm(H[i, :]) for i in 1:n]
    S = zeros(Float64, n, n)
    for i in 1:n, j in 1:n
        if norms[i] > 0 && norms[j] > 0
            S[i, j] = dot(H[i, :], H[j, :]) / (norms[i] * norms[j])
        end
    end
    return S
end
