# Internal audit & planning docs

Living documents that drove the post-MVP hardening pass of `TokenEfficiencyEnv`.
They are kept in the repo so reviewers can see the *reasoning trail*, not just the
final code.

| File | What it is |
|---|---|
| [`vulnerability_audit.md`](./vulnerability_audit.md) | First-pass audit of the v0.2 environment. Lists the concrete reward-hacking, prompt-bank, and judge failure modes we observed, with line-level references. The "what's broken and why" document. |
| [`VULNERABILITY_FIX_PLAN.md`](./VULNERABILITY_FIX_PLAN.md) | The execution plan that fell out of the audit. Twelve numbered sections (§1–§12) covering reward redesign, judge swap, prompt-bank scale-up, curriculum learning, Colab training, HF Spaces deploy, etc. Every merged commit maps back to a section here. |
| [`comparative_analysis.md`](./comparative_analysis.md) | Short comparison of our design against the hackathon's reference environments and against the common reward-hacking anti-patterns called out in the organiser FAQ. Written before §1 was merged, to sanity-check the direction. |
| [`HACKATHON_ALIGNMENT_REPORT.md`](./HACKATHON_ALIGNMENT_REPORT.md) | Final self-audit (commit `3efe2d7`). Scores the project against 20 dimensions derived from the hackathon guidance, calls out the one remaining open issue (hidden chain-of-thought loophole), and lists prioritised non-blocking recommendations. |

## Why these are in the repo

Hackathon judges explicitly asked for reward-hacking evidence and mitigation
narrative, not just working code. These four docs are that narrative. Commit
history alone would force a judge to reconstruct the story from 60+ diffs —
these docs compress it to four short reads.

## What is *not* in here (and why)

The organiser's own materials — the participant FAQ PDF, the self-serve guide
`.docx`, and the theme artwork — live outside the repo. They're Meta's
copyrighted documents and re-publishing them here would be inappropriate.
