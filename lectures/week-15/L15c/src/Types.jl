abstract type AbstractGNNLayer end

"""
    MyGCNLayer

A single symmetric-normalized graph convolution layer in the convention of
[Kipf and Welling (2017)](https://arxiv.org/abs/1609.02907). Stores the
learnable weight matrix `W`, the input and output feature dimensions, and the
elementwise activation function.

The layer's forward map on a feature matrix `H` of shape `(N, d_in)` and a
fixed propagation matrix `P` of shape `(N, N)` is

    H_out = activation.(P * H * W)

where `W` has shape `(d_in, d_out)` and `H_out` has shape `(N, d_out)`.

### Fields
- `d_in::Int64`: input feature dimension.
- `d_out::Int64`: output feature dimension.
- `W::Matrix{Float64}`: learnable weight matrix of shape `(d_in, d_out)`.
- `activation::Function`: elementwise activation, applied componentwise.
"""
mutable struct MyGCNLayer <: AbstractGNNLayer
    d_in::Int64
    d_out::Int64
    W::Matrix{Float64}
    activation::Function

    MyGCNLayer() = new()
end
