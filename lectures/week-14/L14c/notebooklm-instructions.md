# NotebookLM Audio Overview Prompt: L14c, MIMO HiPPO-LegS State Space Models and Forecasting

Paste the block below into NotebookLM's "Customize" box when generating the audio overview for this lecture.

```
Audience: advanced undergraduate and early graduate students in chemical engineering who know linear algebra and ODEs, who have seen the single-input single-output HiPPO-LegS state space model from L14a, but who are new to multi-channel time series and to forecasting with SSMs.

Open with the intuition: why this lecture exists. You have a vector time series and want to predict future values of some channels from the past of all of them. Classical tools force a choice between remembering only the last p lags (VAR) or fitting the full state-space dynamics yourself (Kalman). A MIMO HiPPO-LegS SSM freezes the memory mechanism as a structured polynomial summary of the entire past per channel and only learns a small linear readout, so you keep long-range memory and still train with one closed-form ridge solve. A good analogy: one compressed history sketch per input channel, plus a linear decoder that mixes those sketches into every output channel.

Hit these math highlights without dwelling on derivations:
- the continuous-time MIMO SSM with the same four matrices A, B, C, D as the SISO case, where only the shapes change to accommodate vector inputs and outputs
- a block-diagonal A built by stacking one LegS filter per input channel (total hidden dimension H equals h times d_in)
- bilinear discretization preserves the block structure, so the stability argument from L14a carries over unchanged
- a dense readout C is where all the cross-channel mixing lives, and it is the only thing the model learns
- forecasting alignment: shifting the target k samples ahead turns the same machinery from memorize into predict with no change to the training math

Tradeoffs worth naming out loud:
Upside: one closed-form ridge solve fits all output channels simultaneously through a shared Gram matrix, the parameter count scales as d_out times H (smaller than VAR or Kalman), long-range memory comes for free from the HiPPO-LegS basis, and on genuinely structured signals the forecast correlation can approach one both in sample and out of sample.
Downside: the block-diagonal A is less expressive than an unrestricted MIMO SSM, which is why S5 exists (diagonal A shared across channels, dense B). Placing cross-channel mixing in the single-layer readout is pedagogically simpler than canonical S4, which instead keeps each SSM layer SISO and mixes channels in an MLP between stacked layers. On white-noise-like signals no h is large enough to rescue the forecast. And the LegS basis has slow modes whose transients last more than a hundred samples, so naively resetting the hidden state on a test split inflates error and must be avoided.

Keep the tone conversational, not lecture-y. Prefer analogies (the "one history sketch per channel plus a linear decoder" image above) over formulas read aloud.
```

## What was emphasized

- Opening intuition frames the MIMO SSM against VAR and Kalman (the three-way comparison table in the lecture), positioning "frozen memory, learned readout" as the structural story.
- Math highlights mirror the lecture's section blockquotes: the four-matrix MIMO definition, Kronecker block-diagonal construction, bilinear discretization, dense readout, and the memorize-to-forecast alignment (make_forecast_pair).
- Downsides combine the lecture's own caveats (the S4 architectural deviation blockquote, the S5 pointer blockquote, the "why forecasting is harder than memorizing" blockquote) with the example notebook's warm-state OOS rationale; nothing required inference beyond the notebooks.
