# NotebookLM Audio Overview Prompt: L15a, Spiking Neural Networks and the H-Mem Memory Module

Paste the block below into NotebookLM's "Customize" box when generating the audio overview for this lecture.

```
Audience: advanced undergraduate and early graduate students in chemical engineering who know linear algebra and ODEs, but who are new to spiking neural networks, neuromorphic computing, and Hebbian associative memory.

Open with the intuition: why this exists. Conventional artificial neural networks propagate dense floating-point activations through every layer on every forward pass, which is energetically expensive at deployment and a step removed from how biological neurons compute. Spiking neural networks (SNNs) instead use binary spike events on an explicit clock; between spikes, no work is done. Each neuron is a leaky integrator that charges up to a firing threshold, emits a single spike, and resets, much like a capacitor charging across a trip voltage. This event-driven design pays off as orders-of-magnitude lower power on neuromorphic hardware (Intel Loihi 2, IBM NorthPole), and a natural representation of time-structured signals like audio or event-based vision.

Hit these math highlights without dwelling on derivations:
- the leaky integrate-and-fire (LIF) neuron is a one-dimensional linear ODE in the membrane potential plus a Heaviside threshold-and-reset rule, so all the nonlinearity lives at the spike event
- in discrete time the LIF dynamics become a per-step decay-and-add recursion governed by one decay factor that controls how long the neuron remembers recent input
- a network of LIF neurons is one matrix-vector product per time step plus a Heaviside threshold and a refractory mask, so per-step compute matches a dense ANN layer of the same width
- real-valued inputs are turned into spike trains by rate coding (Poisson, noise-robust but slow) or temporal coding (one spike per dimension, fast but fragile)
- the Hebbian Memory module (H-Mem) puts a key-value association matrix on top of the LIF network; it is written at inference time by a local outer-product rule and read back by one matrix-vector product, giving constant-time hetero-associative memory

Tradeoffs worth naming out loud:
Upside: sparse spike traffic on neuromorphic chips like Loihi 2 and NorthPole costs orders of magnitude less power than dense activations on a GPU; each LIF neuron is a continuous-time linear filter, so time-structured data sits naturally on the architecture; H-Mem stores new key-value pairs in a single forward pass with no retraining (one-shot learning at inference time); the read is a linear matvec so recall degrades smoothly under noisy or partial queries rather than collapsing all at once.
Downside: the Heaviside spike rule has zero gradient almost everywhere, so backpropagation does not apply directly, and all three training routes (surrogate gradients, spike-timing-dependent plasticity, ANN-to-SNN conversion) are workarounds with their own costs. Real-valued inputs must be re-encoded as spike trains, and rate coding needs long simulation windows to recover small differences. H-Mem's association matrix has fixed size ℓ², so capacity is bounded; recall correlation stays high while the load factor K/ℓ is small but falls past a knee near one, and the standard deviation across random instantiations grows once the memory is overloaded. Whether the energy and latency savings outweigh the training complexity is task-dependent and still an open research question rather than a settled engineering tradeoff.

Keep the tone conversational, not lecture-y. Prefer analogies (the "capacitor charging up to a trip voltage and discharging" image, or "one fixed-size association matrix that one-shot learns key-value pairs at inference time") over formulas read aloud.
```

## What was emphasized

- Opening intuition frames SNNs against conventional ANNs via the energy and event-driven contrast (drawing from the lecture's "Why Spiking Neural Networks?" section), with the capacitor charge-and-trip analogy seeded for the hosts to reuse.
- Math highlights mirror the lecture's blockquote sections: the continuous-time LIF ODE, the discrete-time recursion with its decay factor, the vector LIF network as one matvec plus a threshold, the rate-vs-temporal encoding choice, and the H-Mem architecture/write/recall on top of the LIF network.
- Downsides combine the lecture's own caveats (Heaviside non-differentiability, the three training workarounds, the rate-coding window cost) with the example notebook's empirical sweeps (load-factor knee near K/ℓ ≈ 1, replicate-variance growth past overload, smooth degradation under query corruption); the capacity-knee phrasing is grounded in the example notebook's K_SWEEP rather than asserted in the lecture.
