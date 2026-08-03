# Style Playbook — Narrative Tutorial Voice

This file documents the distilled writing devices and the three source articles they come from. Use it
when you need a concrete pattern to copy for a specific chapter.

## Source articles

- **Ref A** — *Probability: Beginner to Advanced for Data Science, Part 1 (Foundations → Bayes Theorem)*,
  Medium (priyanshuthetechintel). Style: quote hook → "real problem → definition → numeric example →
  code" closed loop; dozens of numbered everyday examples; counter-intuitive numbers to build Bayes
  intuition; closing "embrace uncertainty, beyond point estimates".
- **Ref B** — *Variational Inference: The Basics*, Medium (data-science). Style: when/why/how backbone;
  conversational encouragement ("Congrats!", "Lo and behold, a perfect match!"); one concrete toy example
  whose approximate posterior *exactly matches* the true one to build trust; ELBO framed transparently
  with the engineering pain (can't differentiate through an expectation → reparameterisation); contrast
  "traditional ML = a single point Θ" vs "Bayesian VI = a fuzzy posterior distribution p(Θ|X)".
- **Ref C** — *Bayesian Statistics: A Beginner's Guide*, QuantStart. Style: teaching-driven, first/second
  person ("we will…", "our beliefs"); concept + math but derivations in a skippable box; everyday
  analogies (moon hits Earth, unfair coin, election); goals listed up front; frequentist-vs-Bayesian
  comparison table; probability reframed as *confidence/belief*; dynamic "prior washed out by data"
  narrative; the four Bayes-rule roles personified with plain glosses; progressive coin-flip
  visualization (0→2→10→20→50→500 flips) showing the posterior density shift and narrow; heavily
  commented Python (SciPy/Matplotlib) at the end.

## Device → source map

| Device | From | How to apply |
|---|---|---|
| Quote or real-problem hook | A, C | Open the chapter with a memorable line or a puzzle, never a definition. |
| Upfront goals list | C | Bullet 3–5 what the reader will be able to do afterward. |
| Skippable derivation box | C | Put proofs in a marked, optional block; keep the main thread intuitive. |
| Causal chain connectors | A, B, C | Use "because / so / now that we understand X" between sections. |
| Relatable analogy | C | Coin, election, disease test, moon—make the abstract tangible. |
| Guess-then-reveal number | A | Ask the reader for a number, then show the surprising true value. |
| Belief-update narrative | C | Show prior → posterior evolving as data count grows (e.g. 0→n). |
| Personified formula roles | C | Prior/Posterior/Likelihood/Evidence each get a one-line plain gloss. |
| Point-vs-distribution | B | Contrast MLE's single point with Bayes' full distribution to motivate methods. |
| when / why / how | B | Structure any method section on this backbone. |
| Toy example "perfect match" | B | One small case where the approximation provably matches truth, to build trust. |
| Runnable commented code | A, C | numpy/scipy in this repo; place at end, optional to run. |
| Zoom-out closing | A | Reframe the whole chapter as one big idea ("embrace uncertainty"). |

## Checklist before publishing a chapter

- [ ] Opens with a hook, not a definition.
- [ ] States goals/roadmap near the top.
- [ ] Every formula is preceded by a plain-language "why".
- [ ] At least one "guess the number" counter-intuitive beat.
- [ ] Bayesian content includes a belief-update demonstration (prior → posterior as data grows).
- [ ] Methods use when/why/how; point-vs-distribution contrast appears where relevant.
- [ ] Runs a closed loop: concept → example → runnable code.
- [ ] Conversational voice; jargon defined in parentheses on first use.
- [ ] Ends with a zoom-out "why it matters" remark.
- [ ] Makes sense if every equation is deleted.
