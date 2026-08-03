---
name: narrative-tutorial-style
description: This skill should be used when writing or rewriting explanatory content—study guides, book chapters, tutorials, or documentation—in a human, narrative, intuition-first teaching style. It is distilled from three reference articles (a probability-foundations-to-Bayes tutorial, a variational-inference basics tutorial, and a QuantStart Bayesian beginners guide) and codifies a "friendly teacher" voice that builds causal logic and intuition before formalism, rather than dumping formulas. Trigger it whenever the user asks to author guide/book/chapters, "write like a tutorial", "make it read like a human guide", or "inherit the book's writing style".
---

# Narrative Tutorial Style

Use this skill to make any explanatory text read like a patient teacher talking to one student:
intuition first, formulas last, a clear causal chain from "why" to "so what". It is the house style for
the book in this repository and should be applied consistently across all chapters.

## When to use

- Writing or rewriting a chapter, study guide, or tutorial in this repo.
- The user asks for content that "reads like a human guide", "isn't a formula dump", or should match
  "the book's style".
- Any explanatory prose where the goal is *understanding*, not reference lookup.

Do NOT use this style for: API docs, strict reference tables, or spec sheets where skimming and
exact lookup matter more than narrative. (Even there, keep the voice friendly; just skip the
story arc.)

## The core principle

**Formulas are punctuation for intuition, never the skeleton.** A reader should be able to follow the
causal story with every equation removed. State *why a concept exists* in plain words, then make the
concept precise with one formula, then show it with a concrete number or example.

## The writing playbook (apply as many as fit)

1. **Hook, don't define.** Open with a quote, a real problem, or a puzzling observation—not a definition.
   (Ref A opens with a Deming quote; Ref C opens by reframing probability as *confidence/belief*.)
2. **State the roadmap.** List 3–5 goals up front so the reader knows what they'll get.
3. **Intuition before formalism.** Explain the idea in plain language first. Put heavy derivations in an
   optional, clearly skippable "box" (Ref C: "feel free to skip this derivation").
4. **Build a causal chain.** Every section answers "why does this step exist, and what does it enable
   next?" Connect with *because / so / now that we understand X*. No section stands alone.
5. **Use relatable analogies.** Coins, elections, disease tests, "the moon hitting Earth" (Ref C) — concrete
   scenarios that make abstract probability tangible.
6. **Engineer a counter-intuitive "aha".** Make the reader *guess a number first*, then reveal the
   surprising true value (Ref A: a 95%-sensitive test yields only an 8.76% posterior because the base rate
   is low). The gap is what they remember.
7. **Tell a belief-update story.** Frame probability as *belief that evolves with evidence*. Show a prior
   being "washed out" as data accumulates (Ref C: posterior density at 0 → 2 → 10 → 50 → 500 coin flips
   shifts and narrows). Concretely demonstrate "prior mean pulled toward data".
8. **Personify the formula's roles.** Introduce Prior / Posterior / Likelihood / Evidence each with a
   one-line plain-language gloss in parentheses on first use (Ref C).
9. **Point-vs-distribution contrast.** To motivate Bayesian or uncertainty methods, contrast "frequentist /
   MLE hands you a single point" vs "Bayes hands you a whole distribution" (Ref B).
10. **when / why / how for methods.** Structure method sections as: *when is this useful → why it's hard →
    how it works* (Ref B's backbone).
11. **Close the loop with runnable code.** End with heavily-commented, runnable code (numpy/scipy in this
    repo) that reproduces the phenomenon the chapter taught. Keep it at the end, optional to run (Ref A, C).
12. **Conversational, encouraging voice.** Use "you / we", light asides, define jargon in parentheses on
    first use. Never stack unexplained terms.
13. **Zoom out at the end.** A short closing "why this matters" that reframes the whole chapter as one idea
    (Ref A: "embrace uncertainty, beyond point estimates").
14. **One narrative recap, not a formula grid.** If a summary is needed, write it as causal bullet points
    ("why → so"), not a table of isolated equations.

## Workflow

1. Identify the target concept(s) and the single causal through-line connecting them.
2. Draft the hook + roadmap.
3. For each concept: write the plain-language "why", then the one anchor formula, then a concrete
   number/example (prefer one from the source material so the book stays self-consistent).
4. Insert at least one counter-intuitive guess-then-reveal and one belief-update demonstration where
   Bayesian content is involved.
5. Add the runnable code block last.
6. End with the zoom-out remark.
7. Read the result aloud; if any paragraph makes sense *only* with its formula, rewrite the prose.

## Bundled resources

- `references/style-playbook.md` — detailed technique breakdown and a per-reference-article analysis
  (what each of the three source articles does and the exact devices to borrow).
- `assets/chapter-template.md` — a copy-paste skeleton enforcing the hook → roadmap → intuition →
  example → code → zoom-out structure for a single chapter.

## Quality check

Before finishing, confirm: (a) the piece survives with every equation deleted; (b) at least one
"guess the number" intuition beat exists; (c) sections connect with causal language; (d) code is
present and runnable; (e) a zoom-out closing exists. If any fail, revise.
