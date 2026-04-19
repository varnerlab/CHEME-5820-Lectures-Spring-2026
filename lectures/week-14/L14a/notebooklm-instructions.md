# NotebookLM Audio Overview Instructions

Template for the "Customize" box when generating NotebookLM audio overviews
of CHEME-5820 lectures. Paste the filled-in version into NotebookLM, not this
file itself.

## Reusable pattern

Fill in each block, then concatenate into the Customize box. Aim for
roughly 500 characters; NotebookLM tends to ignore much longer inputs.

1. **Audience line.** One sentence naming student background and what they
   don't know yet. NotebookLM calibrates depth off this.
2. **Opening intuition.** "Why does this exist / what problem does it solve."
   Framing, not derivation.
3. **Math highlights as a bulleted checklist.** Three to five items, each
   naming the object *and why it matters*. Describe what formulas mean in
   words; NotebookLM reads raw formulas awkwardly.
4. **Tradeoffs section with explicit upside / downside bullets.** Without
   this block the hosts default to cheerleading. This is the single biggest
   lever for a balanced episode.
5. **Tone instruction.** One sentence. "Conversational, analogies over
   formulas" steers away from a textbook-reading voice.

## Worked example: L14a (HiPPO-LegS SSMs)

```
Audience: advanced undergrad / early graduate students in chemical
engineering who know linear algebra and ODEs but are new to sequence
models.

Open with the intuition: why RNNs and Transformers struggle on very
long sequences, and how a structured state space model sidesteps both
problems by replacing a nonlinear hidden state with a linear dynamical
system whose hidden state encodes a polynomial approximation of the
input history.

Hit these math highlights without dwelling on derivations:
- the continuous-time LTI form dx/dt = A x + B u, y = C x + D u
- the HiPPO-LegS choice of A and B (lower triangular, eigenvalues all
  strictly negative, so the system is stable)
- bilinear discretization as the step that preserves stability
- why training reduces to a single ridge regression on the rolled-out
  hidden states, not an SGD loop

Close with a candid tradeoffs discussion:
- upside: linear-in-sequence-length cost, provable long-range memory,
  closed-form training, strong Long Range Arena results
- downside: frozen A and B limit expressivity compared to learned
  attention, SISO memorize task is a toy target, ridge fit only works
  because C is the single learned piece, and the polynomial basis
  assumes the signal is well-approximated by smooth functions

Keep the tone conversational, not lecture-y. Prefer analogies (e.g.
"the hidden state is a running polynomial fit of everything you've
heard so far") over formulas read aloud.
```

## Notes on what works

- **Name the tradeoffs explicitly.** Generic prompts like "discuss pros
  and cons" produce vague both-sides filler. Specific downsides
  ("frozen A and B limit expressivity", "polynomial basis assumes
  smooth signals") give the hosts something concrete to push back on.
- **Say "without dwelling on derivations."** Otherwise the hosts try to
  walk through proofs out loud, which rarely lands in audio.
- **Give an analogy in the prompt.** The hosts tend to reuse analogies
  you seed them with. A good one is worth more than three bullet
  points of formulas.
- **Tone line goes last.** NotebookLM weights the tail of the prompt
  more heavily, so put the voice guidance where it will stick.
