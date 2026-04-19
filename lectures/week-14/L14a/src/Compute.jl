"""
    build_legS_matrices(h::Int) -> (A, B)

Return the continuous-time HiPPO-LegS state and input matrices for hidden
dimension `h`, using the sign convention from Gu et al. (2020) in which the
LTI system is the stable form `dx/dt = A x + B u`. The entries are

    A[i,k] = -√((2i+1)(2k+1))  if i > k
           = -(i + 1)           if i == k
           =  0                 if i < k

    B[i,1] =  √(2i+1)

with 1-based indexing. The minus signs on `A` make the continuous-time
eigenvalues negative (stable), matching the convention used in the S4
codebase.
"""
function build_legS_matrices(h::Int)
    A = zeros(Float64, h, h)
    B = zeros(Float64, h, 1)
    for i in 1:h
        for k in 1:h
            if i > k
                A[i, k] = -sqrt((2i + 1) * (2k + 1))
            elseif i == k
                A[i, k] = -(i + 1)
            end
        end
        B[i, 1] = sqrt(2i + 1)
    end
    return (A, B)
end

"""
    discretize(A, B; Δt, method=:bilinear) -> (Ā, B̄)

Discretize the continuous-time pair `(A, B)` with step size `Δt`. Only the
bilinear (Tustin) method is implemented, which maps the continuous dynamics
to

    Ā = (I - (Δt/2) A)⁻¹ (I + (Δt/2) A)
    B̄ = (I - (Δt/2) A)⁻¹ (Δt B)

preserving stability of the continuous system.
"""
function discretize(A::AbstractMatrix, B::AbstractMatrix; Δt::Float64, method::Symbol=:bilinear)
    method == :bilinear || error("only :bilinear is supported, got $(method)")
    h = size(A, 1)
    I_h = Matrix{Float64}(I, h, h)
    M = I_h - (Δt / 2) .* A
    Ā = M \ (I_h + (Δt / 2) .* A)
    B̄ = M \ (Δt .* B)
    return (Ā, B̄)
end

"""
    build(::Type{MySisoLegSHippoModel}; h, Δt, C=nothing, D=0.0, x₀=nothing) -> MySisoLegSHippoModel

Build a LegS HiPPO model instance. Constructs the continuous `(A, B)` and
the bilinear-discretized `(Ā, B̄)`. If `C` is not supplied, a zero row
vector of length `h` is used; `fit_C!` overwrites it.
"""
function build(::Type{MySisoLegSHippoModel};
    h::Int,
    Δt::Float64,
    C::Union{Nothing, AbstractMatrix, AbstractVector} = nothing,
    D::Float64 = 0.0,
    x₀::Union{Nothing, AbstractVector} = nothing,
)
    (A, B) = build_legS_matrices(h)
    (Ā, B̄) = discretize(A, B; Δt = Δt)
    Cmat = if C === nothing
        zeros(Float64, 1, h)
    elseif C isa AbstractVector
        reshape(collect(Float64, C), 1, h)
    else
        Matrix{Float64}(C)
    end
    x0 = x₀ === nothing ? zeros(Float64, h) : collect(Float64, x₀)
    return MySisoLegSHippoModel(h, Δt, A, B, Ā, B̄, Cmat, D, x0)
end

"""
    rollout(model::MySisoLegSHippoModel, u::AbstractVector) -> X

Run the discrete-time recursion `x_{t+1} = Ā x_t + B̄ u_t` with `x_0 = model.x₀`
and return the hidden-state matrix `X ∈ ℝ^{T×h}` whose `t`-th row is `x_t`
for `t = 1, …, T`. No readout is applied.
"""
function rollout(model::MySisoLegSHippoModel, u::AbstractVector)
    T = length(u)
    h = model.h
    X = zeros(Float64, T, h)
    x = copy(model.x₀)
    @inbounds for t in 1:T
        x = model.Ā * x + model.B̄ * u[t]
        X[t, :] = x
    end
    return X
end

"""
    solve(model::MySisoLegSHippoModel, u::AbstractVector) -> (X, y)

Roll the model forward on input `u` and apply the current readout `C` (and
scalar `D`) to produce output `y ∈ ℝ^T`. Returns the hidden-state matrix
`X ∈ ℝ^{T×h}` and `y`.
"""
function solve(model::MySisoLegSHippoModel, u::AbstractVector)
    X = rollout(model, u)
    y = (X * model.C')[:, 1] .+ model.D .* u
    return (X, y)
end

"""
    fit_C!(model::MySisoLegSHippoModel, u::AbstractVector, y::AbstractVector; λ=1e-4) -> model

Fit the readout `C` by ridge regression on the hidden-state rollout of `u`
against targets `y`. Solves

    Ĉᵀ = (XᵀX + λ I)⁻¹ Xᵀ y

and updates `model.C` in place. Returns the model.
"""
function fit_C!(model::MySisoLegSHippoModel, u::AbstractVector, y::AbstractVector; λ::Float64 = 1.0e-4)
    length(u) == length(y) || error("length(u) = $(length(u)) must equal length(y) = $(length(y))")
    X = rollout(model, u)
    h = model.h
    G = X' * X + λ .* Matrix{Float64}(I, h, h)
    rhs = X' * collect(Float64, y)
    c = G \ rhs
    model.C = reshape(c, 1, h)
    return model
end

"""
    predict(model::MySisoLegSHippoModel, u::AbstractVector; warmup::Int=0) -> (X, y)

Roll the model forward on an out-of-sample input `u` using the currently
trained `C`. If `warmup > 0`, the first `warmup` samples of `X` and `y` are
dropped to let the hidden state forget `x₀ = 0`.
"""
function predict(model::MySisoLegSHippoModel, u::AbstractVector; warmup::Int = 0)
    warmup >= 0 || error("warmup must be nonnegative")
    (X, y) = solve(model, u)
    if warmup > 0
        warmup < length(u) || error("warmup = $(warmup) must be smaller than length(u) = $(length(u))")
        return (X[(warmup + 1):end, :], y[(warmup + 1):end])
    end
    return (X, y)
end

"""
    log_growth_rate(prices::AbstractVector; Δt::Float64) -> g

Annualized log growth rate sequence `gₜ = (1/Δt) · log(Pₜ / Pₜ₋₁)` for a
price series, returned as a vector of length `length(prices) - 1`.
"""
function log_growth_rate(prices::AbstractVector; Δt::Float64)
    length(prices) >= 2 || error("need at least two prices")
    return (1.0 / Δt) .* diff(log.(collect(Float64, prices)))
end
