# L13a Advanced: Two Derivations for Self-Attention
This advanced companion notebook works out two short derivations that the [L13a lecture](CHEME-5820-L13a-Lecture-Spring-2026.ipynb) referenced but did not prove. Both are short and self-contained, and together they give a rigorous answer to two questions students often ask when they first see scaled dot-product attention:

__Underlying Questions:__
1. _Why divide by $\sqrt{d_{k}}$ inside the softmax?_ We show that the scaling factor follows from a one-line variance computation, and that without it the softmax saturates as the key dimension grows.
2. _Why does self-attention need positional encodings?_ We prove that self-attention is permutation equivariant: shuffling the input rows produces the same shuffled output, with no change to any row's content. So the layer cannot distinguish between sentences with different word order, and we need to break that symmetry by adding position information to the input.

These results are fundamental to understanding how self-attention works. We will also verify both results empirically with small Julia experiments.

> __Learning Objectives:__
>
> By the end of this notebook, you should be able to:
>
> * __Derive the $1/\sqrt{d_{k}}$ scaling factor from a variance argument:__ Show that for independent zero-mean unit-variance entries, the dot product $\langle\mathbf{q},\mathbf{k}\rangle$ has variance $d_{k}$, and explain why this leads to softmax saturation that the scaling factor fixes.
> * __Prove that self-attention is permutation equivariant:__ For any permutation matrix $\mathbf{P}$, show that $\operatorname{Attention}(\mathbf{P}\mathbf{X}) = \mathbf{P}\,\operatorname{Attention}(\mathbf{X})$, and identify the step in the proof that uses the row-wise structure of softmax.
> * __Verify both results empirically in code:__ Write a small Julia experiment that measures the variance of unscaled and scaled dot products as a function of $d_{k}$, and a separate experiment that confirms self-attention output rows shuffle exactly when the input rows are permuted.

Let's get started!
___

## Setup
We will use the same Julia environment as the rest of L13a, plus a few standard linear-algebra and plotting tools.


```julia
include("Include.jl");
Random.seed!(42);
```

    [32m[1m  Activating[22m[39m project at `~/Desktop/julia_work/CHEME-5820-instances/Spring-2026/CHEME-5820-Lectures-Spring-2026/lectures/week-13/L13a`


## Derivation 1: The $1/\sqrt{d_{k}}$ Scaling Factor
The scaled dot-product attention equation is given by:
$$
\operatorname{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \operatorname{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_{k}}}\right)\mathbf{V}.
$$
The factor of $1/\sqrt{d_{k}}$ inside the softmax is not arbitrary. It comes from a short statistical argument about the variance of a dot product of two random vectors.

### The Variance Computation

> __Variance of a dot product__
>
> Let $\mathbf{q}, \mathbf{k}\in\mathbb{R}^{d_{k}}$ be two random vectors whose entries are independent random variables satisfying the following conditions:
> * $\mathbb{E}[q_{i}] = \mathbb{E}[k_{i}] = 0$ for all $i = 1, 2, \ldots, d_{k}$ (zero mean)
> * $\operatorname{Var}(q_{i}) = \operatorname{Var}(k_{i}) = 1$ for all $i = 1, 2, \ldots, d_{k}$ (unit variance)
> * $q_{i}$ and $k_{j}$ are independent for all $i$ and $j$
>
> Then the dot product $\langle\mathbf{q},\mathbf{k}\rangle = \sum_{i=1}^{d_{k}} q_{i} k_{i}$ has mean zero and variance $d_{k}$.

We prove this in two steps.

> __Proof.__
>
> _Mean:_ By linearity of expectation, the mean of the dot product is given by:
> $$
\mathbb{E}[\langle\mathbf{q},\mathbf{k}\rangle] = \mathbb{E}\!\left[\sum_{i=1}^{d_{k}} q_{i} k_{i}\right] = \sum_{i=1}^{d_{k}} \mathbb{E}[q_{i} k_{i}].
> $$
> In general, $\mathbb{E}[q_{i} k_{i}] \neq \mathbb{E}[q_{i}]\,\mathbb{E}[k_{i}]$ for arbitrary random variables. However, because we assumed $q_{i}$ and $k_{i}$ are independent, the expectation of the product factors as $\mathbb{E}[q_{i} k_{i}] = \mathbb{E}[q_{i}]\,\mathbb{E}[k_{i}] = 0 \cdot 0 = 0$ for each $i$. Therefore:
> $$
\mathbb{E}[\langle\mathbf{q},\mathbf{k}\rangle] = \sum_{i=1}^{d_{k}} 0 = 0.
> $$
>
> _Variance:_ Because the products $q_{i} k_{i}$ for $i = 1, 2, \ldots, d_{k}$ are mutually independent (each pair $(q_{i}, k_{i})$ involves different underlying random variables), the variance of the sum equals the sum of the variances:
> $$
\operatorname{Var}(\langle\mathbf{q},\mathbf{k}\rangle) = \sum_{i=1}^{d_{k}} \operatorname{Var}(q_{i} k_{i}).
> $$
> We compute each term using the definition $\operatorname{Var}(Z) = \mathbb{E}[Z^{2}] - (\mathbb{E}[Z])^{2}$. From the mean calculation above, we already know $\mathbb{E}[q_{i} k_{i}] = 0$, so the second term vanishes and we get:
> $$
\operatorname{Var}(q_{i} k_{i}) = \mathbb{E}[(q_{i} k_{i})^{2}] - 0^{2} = \mathbb{E}[q_{i}^{2}\, k_{i}^{2}].
> $$
> Again, we use the independence of $q_{i}$ and $k_{i}$ to factor the expectation of the product:
> $$
\mathbb{E}[q_{i}^{2}\, k_{i}^{2}] = \mathbb{E}[q_{i}^{2}]\,\mathbb{E}[k_{i}^{2}].
> $$
> Finally, since $\mathbb{E}[q_{i}] = 0$, we can identify $\mathbb{E}[q_{i}^{2}] = \operatorname{Var}(q_{i}) + (\mathbb{E}[q_{i}])^{2} = \operatorname{Var}(q_{i}) = 1$ (and similarly $\mathbb{E}[k_{i}^{2}] = \operatorname{Var}(k_{i}) = 1$). Substituting back gives:
> $$
\operatorname{Var}(\langle\mathbf{q},\mathbf{k}\rangle) = \sum_{i=1}^{d_{k}} 1 \cdot 1 = d_{k}\quad\blacksquare
> $$

### Why This Variance Hurts the Softmax

The dot product has standard deviation $\sqrt{d_{k}}$, so for a typical query-key pair the scores take values on the order of $\pm\sqrt{d_{k}}$. As $d_{k}$ grows, the entries fed into the softmax grow as well.

The softmax function is _scale-sensitive_: doubling its input vector sharpens the output distribution, and quadrupling it sharpens it further. In the limit where one input is much larger than the others, the softmax assigns weight one to that entry and zero to all others. At that point the gradient of softmax through every entry is nearly zero, which makes the layer untrainable.

> __The fix.__ Dividing by $\sqrt{d_{k}}$ rescales the dot product to have unit variance, independent of $d_{k}$:
> $$
\operatorname{Var}\!\left(\frac{\langle\mathbf{q},\mathbf{k}\rangle}{\sqrt{d_{k}}}\right) = \frac{1}{d_{k}}\,\operatorname{Var}(\langle\mathbf{q},\mathbf{k}\rangle) = \frac{1}{d_{k}}\cdot d_{k} = 1.
> $$
> The softmax inputs now have a fixed scale regardless of how wide the model gets, so the softmax is neither saturated nor flat, and gradients flow through it normally.

### Empirical Verification: Variance Grows with $d_{k}$
Let's confirm the variance computation numerically. For each value of $d_{k}$ in a sweep, we sample many independent pairs $(\mathbf{q}, \mathbf{k})$ from the standard normal distribution, compute the unscaled dot product $\langle\mathbf{q},\mathbf{k}\rangle$ and the scaled dot product $\langle\mathbf{q},\mathbf{k}\rangle/\sqrt{d_{k}}$, and compare their sample variances against the theoretical predictions of $d_{k}$ and $1$.


```julia
Random.seed!(42);
d_ks = [4, 16, 64, 256, 1024];
n_samples = 50_000;
results = DataFrame(d_k = Int[], var_raw = Float64[], var_scaled = Float64[]);
for d_k in d_ks
    raw = [dot(randn(d_k), randn(d_k)) for _ in 1:n_samples];
    scaled = raw ./ sqrt(d_k);
    push!(results, (d_k, var(raw), var(scaled)));
end
pretty_table(results)
```

    ┌───────┬─────────┬────────────┐
    │[1m   d_k [0m│[1m var_raw [0m│[1m var_scaled [0m│
    │[90m Int64 [0m│[90m Float64 [0m│[90m    Float64 [0m│
    ├───────┼─────────┼────────────┤
    │     4 │ 3.99362 │   0.998404 │
    │    16 │ 15.9669 │   0.997933 │
    │    64 │  64.447 │    1.00698 │
    │   256 │ 258.015 │    1.00787 │
    │  1024 │ 1029.67 │    1.00554 │
    └───────┴─────────┴────────────┘


The unscaled variance tracks $d_{k}$ closely, and the scaled variance stays near $1$ regardless of how large $d_{k}$ becomes. This confirms the variance computation above empirically.

### Empirical Verification: Softmax Saturation
Now let's see what happens to the softmax when we feed it unscaled vs. scaled dot products. We pick one query $\mathbf{q}$ and a small set of $n_{\text{keys}} = 8$ keys, compute the scores both ways, and report the maximum softmax weight. A value close to $1$ means the softmax has saturated and assigns almost all of its mass to a single key.


```julia
Random.seed!(7);
d_ks = [4, 16, 64, 256, 1024];
n_keys = 8;
sat = DataFrame(d_k = Int[], max_weight_raw = Float64[], max_weight_scaled = Float64[]);
for d_k in d_ks
    q = randn(Float32, d_k);
    K = randn(Float32, d_k, n_keys);
    raw_scores = vec(K' * q);
    scaled_scores = raw_scores ./ sqrt(Float32(d_k));
    raw_w = softmax(raw_scores);
    scaled_w = softmax(scaled_scores);
    push!(sat, (d_k, maximum(raw_w), maximum(scaled_w)));
end
pretty_table(sat)
```

    ┌───────┬────────────────┬───────────────────┐
    │[1m   d_k [0m│[1m max_weight_raw [0m│[1m max_weight_scaled [0m│
    │[90m Int64 [0m│[90m        Float64 [0m│[90m           Float64 [0m│
    ├───────┼────────────────┼───────────────────┤
    │     4 │       0.470821 │          0.326617 │
    │    16 │       0.562191 │          0.250198 │
    │    64 │       0.999996 │          0.633766 │
    │   256 │       0.809442 │          0.199625 │
    │  1024 │            1.0 │          0.365179 │
    └───────┴────────────────┴───────────────────┘


For small $d_{k}$, the unscaled and scaled softmax are close. As $d_{k}$ grows, the unscaled softmax pushes the maximum weight close to $1$, indicating the distribution has collapsed onto a single key. The scaled softmax stays diffuse: the maximum weight does not collapse, and gradients still flow through every entry. This is why the $1/\sqrt{d_{k}}$ factor is non-optional in deep transformers, where $d_{k}$ is typically $64$ or larger.

___

## Derivation 2: Permutation Equivariance of Self-Attention
The lecture claimed that self-attention is _permutation equivariant_: if you shuffle the rows of the input, the rows of the output are shuffled in the same way, but no row's contents change. We now prove this rigorously.

### Setup and Statement
Let $\mathbf{X}\in\mathbb{R}^{n\times d}$ be a matrix of token embeddings (one row per token), and let the self-attention output be given by:
$$
\operatorname{Attention}(\mathbf{X}) = \operatorname{softmax}\!\left(\frac{\mathbf{X}\mathbf{W}_{Q}(\mathbf{X}\mathbf{W}_{K})^{\top}}{\sqrt{d_{k}}}\right)\mathbf{X}\mathbf{W}_{V}
$$
where $\mathbf{W}_{Q}, \mathbf{W}_{K}\in\mathbb{R}^{d\times d_{k}}$ and $\mathbf{W}_{V}\in\mathbb{R}^{d\times d_{v}}$ are the learned projection matrices, and the softmax is applied row-wise.

> __Theorem (Permutation Equivariance of Self-Attention).__
>
> Let $\mathbf{P}\in\mathbb{R}^{n\times n}$ be a permutation matrix satisfying $\mathbf{P}\mathbf{P}^{\top} = \mathbf{P}^{\top}\mathbf{P} = \mathbf{I}_{n}$, with exactly one entry of each row and column equal to $1$. Then:
> $$
\boxed{
\operatorname{Attention}(\mathbf{P}\mathbf{X}) = \mathbf{P}\,\operatorname{Attention}(\mathbf{X}).
}
> $$

The proof tracks how each piece of the attention computation transforms when the input is permuted on the left.

> __Proof.__
>
> _Step 1: Q, K, V transform as left-multiplication by $\mathbf{P}$._
> When we replace $\mathbf{X}$ with $\mathbf{P}\mathbf{X}$, the permuted queries are given by:
> $$
\mathbf{Q}_{\mathbf{P}} = (\mathbf{P}\mathbf{X})\mathbf{W}_{Q} = \mathbf{P}(\mathbf{X}\mathbf{W}_{Q}) = \mathbf{P}\mathbf{Q},
> $$
> where the second equality uses associativity of matrix multiplication. By the same argument, $\mathbf{K}_{\mathbf{P}} = \mathbf{P}\mathbf{K}$ and $\mathbf{V}_{\mathbf{P}} = \mathbf{P}\mathbf{V}$.
>
> _Step 2: The scaled score matrix transforms as $\mathbf{P} \mathbf{S}\mathbf{P}^{\top}$._
> Let $\mathbf{S} = \mathbf{Q}\mathbf{K}^{\top}/\sqrt{d_{k}}$ be the scaled scores from the original input. Then the permuted scores are given by:
> $$
\begin{align*}
\mathbf{S}_{\mathbf{P}} &= \frac{\mathbf{Q}_{\mathbf{P}}\mathbf{K}_{\mathbf{P}}^{\top}}{\sqrt{d_{k}}} \\
&= \frac{(\mathbf{P}\mathbf{Q})(\mathbf{P}\mathbf{K})^{\top}}{\sqrt{d_{k}}} \\
&= \frac{\mathbf{P}\mathbf{Q}\mathbf{K}^{\top}\mathbf{P}^{\top}}{\sqrt{d_{k}}} \\
&= \mathbf{P}\,\mathbf{S}\,\mathbf{P}^{\top},
\end{align*}
> $$
> where the third equality uses $(\mathbf{P}\mathbf{K})^{\top} = \mathbf{K}^{\top}\mathbf{P}^{\top}$.
>
> _Step 3: Row-wise softmax commutes with the conjugation $\mathbf{S}\mapsto\mathbf{P}\mathbf{S}\mathbf{P}^{\top}$._
> Let $\sigma$ denote the permutation of $\{1, \ldots, n\}$ corresponding to $\mathbf{P}$, so that $(\mathbf{P}\mathbf{A})_{i j} = \mathbf{A}_{\sigma^{-1}(i), j}$ and $(\mathbf{A}\mathbf{P}^{\top})_{i j} = \mathbf{A}_{i, \sigma^{-1}(j)}$ for any compatible matrix $\mathbf{A}$. Combining these, entry $(i,j)$ of the conjugated score matrix is:
> $$
(\mathbf{P}\mathbf{S}\mathbf{P}^{\top})_{i j} = \mathbf{S}_{\sigma^{-1}(i),\,\sigma^{-1}(j)}.
> $$
> Now apply softmax row-wise. Row $i$ of $\mathbf{P}\mathbf{S}\mathbf{P}^{\top}$ is row $\sigma^{-1}(i)$ of $\mathbf{S}$ with its entries reordered according to $\sigma^{-1}$. Because softmax is _equivariant under permutations of its input_ (the softmax of a permuted vector is the same permutation of the softmax of the original vector), we have:
> $$
\operatorname{softmax}(\mathbf{u})_{\pi(j)} = \operatorname{softmax}(\pi\cdot\mathbf{u})_{j}\quad\text{for any permutation }\pi.
> $$
> Applying this within each row gives:
> $$
\operatorname{softmax}(\mathbf{P}\mathbf{S}\mathbf{P}^{\top})_{i j} = \operatorname{softmax}(\mathbf{S})_{\sigma^{-1}(i),\,\sigma^{-1}(j)} = (\mathbf{P}\,\operatorname{softmax}(\mathbf{S})\,\mathbf{P}^{\top})_{i j}.
> $$
> So $\operatorname{softmax}(\mathbf{P}\mathbf{S}\mathbf{P}^{\top}) = \mathbf{P}\,\operatorname{softmax}(\mathbf{S})\,\mathbf{P}^{\top}$. This is the only step in the proof that uses anything specific about softmax; the rest is pure linear algebra.
>
> _Step 4: Combine to recover $\mathbf{P}\,\operatorname{Attention}(\mathbf{X})$._
> Let $\mathbf{A} = \operatorname{softmax}(\mathbf{S})$ denote the original attention weights. Combining the results from Steps 1-3, the full permuted attention output is:
> $$
\begin{align*}
\operatorname{Attention}(\mathbf{P}\mathbf{X}) &= \operatorname{softmax}(\mathbf{S}_{\mathbf{P}})\,\mathbf{V}_{\mathbf{P}} \\
&= (\mathbf{P}\mathbf{A}\mathbf{P}^{\top})(\mathbf{P}\mathbf{V}) \\
&= \mathbf{P}\mathbf{A}(\mathbf{P}^{\top}\mathbf{P})\mathbf{V} \\
&= \mathbf{P}\mathbf{A}\mathbf{V} \\
&= \mathbf{P}\,\operatorname{Attention}(\mathbf{X}),
\end{align*}
> $$
> where the third line uses $\mathbf{P}^{\top}\mathbf{P} = \mathbf{I}_{n}$. This is the desired identity. $\quad\blacksquare$

> __Why this matters.__ The theorem says that self-attention has _no internal mechanism_ for distinguishing the order of its input rows. The sentence "the cat sat on the mat" and any of its $6! = 720$ permutations produce the same set of output rows in different orders. Self-attention treats its input as a set, not a sequence. To recover sequence semantics, we have to break this symmetry by adding _positional encodings_ to the input embeddings before they enter the attention layer. After position information is added, the rows of $\mathbf{X}$ are no longer interchangeable, the proof above no longer applies, and the layer can finally tell word order from word identity.

### Empirical Verification: Self-Attention Output Permutes with Input
Let's verify the theorem in code. We instantiate a small self-attention layer, run it on a random input, then permute the input rows and confirm that the output rows are shuffled in exactly the same way, with no change to any row's contents.


```julia
# Build a small single-head self-attention layer (no padding mask needed)
Random.seed!(99);
d_in, n_seq, d_k_test, d_v_test = 16, 7, 8, 8;

W_Q = Float32.(randn(d_in, d_k_test) ./ sqrt(d_in));
W_K = Float32.(randn(d_in, d_k_test) ./ sqrt(d_in));
W_V = Float32.(randn(d_in, d_v_test) ./ sqrt(d_in));

# X shape: (d, n) — each column is one token embedding
X = Float32.(randn(d_in, n_seq));

function attn_layer(X::Matrix{Float32}, WQ, WK, WV)
    Q = WQ' * X            # (d_k, n)
    K = WK' * X            # (d_k, n)
    V = WV' * X            # (d_v, n)
    d_k = size(WQ, 2)
    scores = (Q' * K) ./ sqrt(Float32(d_k))   # (n, n)
    weights = softmax(scores; dims = 2)       # (n, n)
    return V * weights'                       # (d_v, n)
end

original_out = attn_layer(X, W_Q, W_K, W_V);
println("original output shape: ", size(original_out))
```

    original output shape: (8, 7)



```julia
# Permute the columns of X (i.e. the rows of the n × d input)
sigma = [3, 1, 6, 4, 7, 2, 5];
X_perm = X[:, sigma];

permuted_out = attn_layer(X_perm, W_Q, W_K, W_V);

# Theorem: permuted_out[:, j] == original_out[:, sigma[j]] for every j
println("Checking permutation equivariance row-by-row:");
all_ok = true;
for j in 1:n_seq
    ok = isapprox(permuted_out[:, j], original_out[:, sigma[j]]; atol = 1.0f-5);
    println("  output column j=", j, "  matches original column σ(j)=", sigma[j], "  : ", ok);
    all_ok &= ok;
end
println();
println("All ", n_seq, " columns match: ", all_ok)
```

    Checking permutation equivariance row-by-row:
      output column j=1  matches original column σ(j)=3  : true
      output column j=2  matches original column σ(j)=1  : true
      output column j=3  matches original column σ(j)=6  : true
      output column j=4  matches original column σ(j)=4  : true
      output column j=5  matches original column σ(j)=7  : true
      output column j=6  matches original column σ(j)=2  : true
      output column j=7  matches original column σ(j)=5  : true
    
    All 7 columns match: true


Every output row of the permuted run matches the corresponding original-input output row exactly, modulo a `1e-5` floating-point tolerance, which is the empirical confirmation of the theorem above. Self-attention is permutation equivariant, and that is why we have to add positional encoding to make it usable on sequences.

___

## Summary
Both derivations are short, but they answer questions that show up the moment you write down scaled dot-product attention.

> __Key Takeaways:__
>
> * **The $1/\sqrt{d_{k}}$ scaling has a one-line variance derivation.** Independent zero-mean unit-variance entries make $\langle\mathbf{q},\mathbf{k}\rangle$ a sum of $d_{k}$ independent unit-variance random variables, so it has variance $d_{k}$. Dividing by $\sqrt{d_{k}}$ rescales the variance to $1$ and prevents the softmax from saturating as the model dimension grows.
> * **Self-attention is permutation equivariant by direct computation.** The proof tracks how $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$, and the softmax matrix transform when the input is left-multiplied by a permutation matrix $\mathbf{P}$, and uses $\mathbf{P}^{\top}\mathbf{P} = \mathbf{I}$ to recover $\mathbf{P}\,\operatorname{Attention}(\mathbf{X})$. Only the row-wise softmax step uses anything specific about the activation function.
> * **Positional encoding is the symmetry-breaking step.** Without it, self-attention treats its input rows as a set and cannot distinguish word order. Adding a position-dependent vector to each row breaks the row-interchangeability assumption used in the proof, which is why every transformer for sequence modeling includes positional information of some kind.

For the main lecture see [L13a Lecture: Transformers and Self-Attention](CHEME-5820-L13a-Lecture-Spring-2026.ipynb), and for an applied example contrasting self-attention with the L10b mean-pool baseline see the [L13a example notebook](CHEME-5820-L13a-Example-Attention-Sentiment-Spring-2026.ipynb).
___
