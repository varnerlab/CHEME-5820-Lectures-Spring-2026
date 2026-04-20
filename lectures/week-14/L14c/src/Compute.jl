"""
    _build_legS_single(h::Int) -> (A, b)

Return the single-channel LegS state matrix `A ∈ ℝ^{h×h}` and input vector
`b ∈ ℝ^{h}` in the 1-based sign convention from Gu et al. 2020 with the stable
form `dx/dt = A x + b u`. In Julia, `b` is returned as a length-`h` vector
rather than an explicit `h×1` matrix.
"""
function _build_legS_single(h::Int)
    h >= 1 || error("h must be positive")
    A = zeros(Float64, h, h)
    b = zeros(Float64, h)
    # Fill the lower-triangular LegS operator entry by entry so the code mirrors
    # the analytical definition used in the lecture notes.
    for i in 1:h
        for k in 1:h
            if i > k
                A[i, k] = -sqrt((2i + 1) * (2k + 1))
            elseif i == k
                A[i, k] = -(i + 1)
            end
        end
        b[i] = sqrt(2i + 1)
    end
    return (A, b)
end

"""
    build_legS_matrices_mimo(h::Int, d_in::Int) -> (A, B)

Build the block-diagonal MIMO LegS state matrix `A ∈ ℝ^{(h·d_in)×(h·d_in)}` and
input matrix `B ∈ ℝ^{(h·d_in)×d_in}`. Each input channel drives one independent
`h`-dim LegS filter: `A` is block-diagonal with `d_in` copies of the single-
channel LegS block on its diagonal, and column `j` of `B` is the standard LegS
input vector placed in block `j` (zero elsewhere).
"""
function build_legS_matrices_mimo(h::Int, d_in::Int)
    h >= 1   || error("h must be positive")
    d_in >= 1 || error("d_in must be positive")
    (A_single, b_single) = _build_legS_single(h)
    H = h * d_in
    A = zeros(Float64, H, H)
    B = zeros(Float64, H, d_in)
    # Hidden-state block `j` stores the LegS state driven by input channel `j`.
    for j in 1:d_in
        rows = ((j - 1) * h + 1):(j * h)
        A[rows, rows] = A_single
        B[rows, j]    = b_single
    end
    return (A, B)
end

"""
    discretize(A, B; Δt, method=:bilinear) -> (Ā, B̄)

Bilinear (Tustin) discretization of the continuous pair `(A, B)` at step `Δt`.
Valid for MIMO of arbitrary shape; `A` must be square. Returns discrete
matrices of the same shapes as the inputs:

    Ā = (I - Δt/2 A)⁻¹ (I + Δt/2 A)
    B̄ = (I - Δt/2 A)⁻¹ (Δt B)
"""
function discretize(A::AbstractMatrix, B::AbstractMatrix; Δt::Float64, method::Symbol=:bilinear)
    method == :bilinear || error("only :bilinear is supported, got $(method)")
    Δt > 0 || error("Δt must be positive")
    H, W = size(A)
    H == W || error("A must be square, got size $(size(A))")
    size(B, 1) == H || error("B has $(size(B, 1)) rows, expected $(H)")
    I_H = Matrix{Float64}(I, H, H)
    # The bilinear transform reuses the same solve for both discrete operators.
    M   = I_H - (Δt / 2) .* A
    Ā   = M \ (I_H + (Δt / 2) .* A)
    B̄   = M \ (Δt .* B)
    return (Ā, B̄)
end

"""
    build(::Type{MyMimoLegSHippoModel}; h, d_in, d_out, Δt, C=nothing, D=nothing, x₀=nothing) -> MyMimoLegSHippoModel

Build a MIMO LegS HiPPO model. Constructs the block-diagonal continuous pair
`(A, B)` and its bilinear-discretized counterpart `(Ā, B̄)`. `C` defaults to a
zero matrix of shape `(d_out, h·d_in)` (which `fit_C!` overwrites), `D`
defaults to a zero matrix of shape `(d_out, d_in)`, and `x₀` defaults to
zeros. The returned model stores both the continuous operators `(A, B)` and the
discrete operators `(Ā, B̄)` so either representation can be inspected.
"""
function build(::Type{MyMimoLegSHippoModel};
    h::Int, d_in::Int, d_out::Int, Δt::Float64,
    C::Union{Nothing, AbstractMatrix} = nothing,
    D::Union{Nothing, AbstractMatrix} = nothing,
    x₀::Union{Nothing, AbstractVector} = nothing,
)
    d_out >= 1 || error("d_out must be positive")
    Δt > 0 || error("Δt must be positive")
    (A, B) = build_legS_matrices_mimo(h, d_in)
    (Ā, B̄) = discretize(A, B; Δt = Δt)
    H = h * d_in
    # Convert optional user inputs to dense Float64 arrays with the expected shapes.
    Cmat = C === nothing ? zeros(Float64, d_out, H) : Matrix{Float64}(C)
    Dmat = D === nothing ? zeros(Float64, d_out, d_in) : Matrix{Float64}(D)
    x0   = x₀ === nothing ? zeros(Float64, H) : collect(Float64, x₀)
    size(Cmat) == (d_out, H)     || error("C has shape $(size(Cmat)), expected ($(d_out), $(H))")
    size(Dmat) == (d_out, d_in)  || error("D has shape $(size(Dmat)), expected ($(d_out), $(d_in))")
    length(x0) == H              || error("x₀ has length $(length(x0)), expected $(H)")
    return MyMimoLegSHippoModel(h, d_in, d_out, Δt, A, B, Ā, B̄, Cmat, Dmat, x0)
end

"""
    rollout(model::MyMimoLegSHippoModel, U::AbstractMatrix) -> X

Run the discrete-time recursion `xₜ = Ā xₜ₋₁ + B̄ uₜ` with `x₀ = model.x₀` on
the input matrix `U ∈ ℝ^{T×d_in}` (row `t` is the input vector at time `t`)
and return the hidden-state matrix `X ∈ ℝ^{T×(h·d_in)}` whose `t`-th row is
`xₜᵀ`. The readout is not applied, so `rollout` isolates the latent dynamics.
"""
function rollout(model::MyMimoLegSHippoModel, U::AbstractMatrix)
    T, d_in = size(U)
    d_in == model.d_in || error("U has $(d_in) columns, expected $(model.d_in)")
    H = model.h * model.d_in
    X = zeros(Float64, T, H)
    x = copy(model.x₀)
    # Store one hidden-state snapshot per row so `X` can be used directly as a
    # regression design matrix later.
    @inbounds for t in 1:T
        u = @view U[t, :]
        x = model.Ā * x + model.B̄ * u
        X[t, :] = x
    end
    return X
end

"""
    solve(model::MyMimoLegSHippoModel, U::AbstractMatrix) -> (X, Y)

Roll the model forward on the input matrix `U ∈ ℝ^{T×d_in}` and apply the
current readout to produce `Y ∈ ℝ^{T×d_out}` with `Yₜ = C xₜ + D uₜ`. Returns
the hidden-state matrix `X` and the output matrix `Y`. Row `t` of `Y`
corresponds to the output vector at time step `t`.
"""
function solve(model::MyMimoLegSHippoModel, U::AbstractMatrix)
    X = rollout(model, U)
    # Rows of `X` and `U` are time samples, so the batched readout uses the
    # transposed parameter matrices on the right.
    Y = X * model.C' .+ U * model.D'
    return (X, Y)
end

"""
    fit_C!(model::MyMimoLegSHippoModel, U::AbstractMatrix, Y::AbstractMatrix; λ=1e-4) -> model

Fit the readout `C` by closed-form ridge regression on the hidden-state
rollout of `U` against targets `Y`. With `D` fixed the loss decouples across
output channels, so the normal equations reduce to a single multi-output
solve

    Cᵀ = (XᵀX + λ I)⁻¹ Xᵀ (Y - U Dᵀ)

of size `(h·d_in)`. `U ∈ ℝ^{T×d_in}` and `Y ∈ ℝ^{T×d_out}` must have the same
number of rows. The function mutates `model.C` and returns `model` for
convenience.
"""
function fit_C!(model::MyMimoLegSHippoModel, U::AbstractMatrix, Y::AbstractMatrix; λ::Float64 = 1.0e-4)
    λ >= 0 || error("λ must be nonnegative")
    T_u, d_in = size(U)
    T_y, d_out = size(Y)
    T_u == T_y         || error("U has $(T_u) rows, Y has $(T_y) rows; must match")
    d_in == model.d_in || error("U has $(d_in) columns, expected $(model.d_in)")
    d_out == model.d_out || error("Y has $(d_out) columns, expected $(model.d_out)")
    X = rollout(model, U)
    # Remove the direct feedthrough term so the regression only learns the
    # hidden-state readout.
    Y_res = Y .- U * model.D'
    H = model.h * model.d_in
    G = X' * X + λ .* Matrix{Float64}(I, H, H)
    rhs = X' * collect(Float64, Y_res)
    # Solve all output channels in one linear system, then transpose back to the
    # stored `C :: (d_out, H)` layout.
    CT = G \ rhs
    model.C = Matrix{Float64}(CT')
    return model
end

"""
    predict(model::MyMimoLegSHippoModel, U::AbstractMatrix; warmup::Int=0) -> (X, Y)

Roll the model on a new input matrix `U` using the currently trained `C` and
`D`. If `warmup > 0`, the first `warmup` rows of `X` and `Y` are discarded so
the hidden state has time to forget `x₀`. This is useful when the initial
condition is artificial but the long-time forecast is what matters.
"""
function predict(model::MyMimoLegSHippoModel, U::AbstractMatrix; warmup::Int = 0)
    warmup >= 0 || error("warmup must be nonnegative")
    (X, Y) = solve(model, U)
    if warmup > 0
        warmup < size(U, 1) || error("warmup = $(warmup) must be smaller than length $(size(U, 1))")
        return (X[(warmup + 1):end, :], Y[(warmup + 1):end, :])
    end
    return (X, Y)
end

"""
    make_forecast_pair(U::AbstractMatrix, k::Int) -> (U_input, U_target)

Given a full input matrix `U ∈ ℝ^{T×d}` and a forecast horizon `k ≥ 1`, build
the aligned pair
  * `U_input  = U[1:T-k, :]`          (samples 1 through T-k)
  * `U_target = U[1+k:T, :]`          (samples 1+k through T)
so that a model fit on `(U_input, U_target)` learns to predict the input `k`
steps ahead, `uₜ ↦ uₜ₊ₖ`. Both returned matrices have `T - k` rows.
"""
function make_forecast_pair(U::AbstractMatrix, k::Int)
    k >= 1 || error("forecast horizon k must be at least 1, got $(k)")
    T = size(U, 1)
    k < T || error("forecast horizon k = $(k) must be smaller than length $(T)")
    # Pair each time sample with the observation `k` steps in the future.
    U_input  = U[1:(T - k), :]
    U_target = U[(1 + k):T, :]
    return (U_input, U_target)
end
