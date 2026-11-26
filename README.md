# training — the model factory

This system teaches small AI models new skills and personalities, on rented
cloud GPUs, without you babysitting anything. You write the teaching material;
everything else — the machine rental, the lesson, the receipts, the delivery —
is automated, and every claim it makes can be proven with a receipt.

## How you use it

- **Teach a model something.** Pick a "domain" (a subject, or a personality),
  point at a recipe, and run one command (`qtrain submit`). The system rents
  a GPU by the hour, trains a small patch (a "LoRA") on your material, packs
  everything it learned into a labeled artifact, and sends the machine home.
  A smoke test (`--mini`) checks the whole pipeline for a few cents before
  the real lesson runs.
- **Ask for proof.** `qtrain verify <run>` answers, in one screen: *did the
  machine actually learn from the material we think it did?* Every lesson is
  fingerprinted and checked three ways — our records, what we shipped to the
  machine, and what the machine reports it consumed. No vibes.
- **Ask what a model can do.** `qtrain runs`, `qtrain events`, `qtrain evals`
  — history of every lesson, every test score, per model, forever. The test
  questions come from a reserved slice of the material that no model ever
  trained on, so a score means "this generalizes", not "this memorized".
- **Test a personality.** `qtrain bench <domain>` interrogates a live model
  with scripted questions (who are you? how do you answer under pressure?)
  and grades the answers. Results go into the same permanent history.
- **Turn a lesson into a product.** One command (`qtrain stage`) converts
  what was learned into the formats models actually serve from — with hard
  count-checks so a package that lost any piece of the learning fails
  loudly instead of shipping silently — and `qtrain register` delivers it,
  fingerprint-verified, to the serving machine.

## What it promises you

- **It won't overspend.** Every job carries a budget cap set from history,
  not hope; jobs die at the cap with a clear reason. The balance is checked
  before anything launches, and one bad estimate can't bankrupt the week.
- **It can't run away.** If the controller dies mid-lesson, the rented GPU
  notices the silence (a "heartbeat file" the controller refreshes every
  few seconds), saves its work, files an honest receipt marked
  "cut short by silence", and shuts itself off — within about ten minutes.
  When the controller comes back, it adopts the open job and closes it
  properly. This was proven live today: killed mid-run, the pod delivered
  its stamped receipt 89 seconds after the watchdog's grace window.
- **It remembers.** Crashes, reboots, network blips: open jobs get adopted,
  finished ones get collected, nothing is lost, nothing is paid for twice.
- **It refuses stupid questions.** If a conversion has been proven
  impossible for a model family (we tried, we have the receipts), asking
  again is rejected instantly at zero cost, and re-asking requires new
  evidence.
- **It shows its work.** Every artifact is fingerprinted; every claim in
  every document here cites a run id you can verify with one command.

## Where everything lives

- **This repo** (`quadra/training`): the machinery only — no lessons, no
  personal data, no secrets. Safe to read anywhere.
- **`quadra/corpora`**: the teaching material itself (the personality's
  memories, the test questions, the recipes). The domain truth.
- **`quadra/infra`**: where machines live and how they're wired.
- Images (the software that runs the lessons) come from our CI, pinned by
  fingerprint, from ghcr.io.
- Everything a run produces lives in the lane store until deliberately
  deleted, with fingerprints so it can be audited later. The lane's quota
  is shared account-wide, so housekeeping deletes only *proven* junk and
  verifies the deletion.

## The Kandi story so far (plainly)

Kandi is a personality taught into a small Qwen model, across three rounds.

- **Round 1** appeared to half-work, with a crutch (a forced greeting script)
  that also leaked onto the base model. The crutch was removed by ruling:
  the identity must live in the weights, not in a serving-side trick.
- **Round 2** failed the questions — "I'm Qwen". The post-mortem first
  blamed thin teaching material; the deeper truth came later: **for months,
  the packaging step had been silently attaching nothing** — every
  "finished" model was the vanilla base wearing a name tag. The round-2
  test had never actually tested the teaching at all.
- **The packaging bug is now fixed and proven** — a package that moves zero
  (or not exactly all) of the learned pieces fails loudly. And round 3
  taught the identity properly: varied material, honestly measured.
- **Round 3's verdict: it works.** On questions the model had never seen,
  the round-3 package scores **4.5× better than the base model**. The
  taught personality finally survives packaging.

## Current state

- Round-3 lesson: done, proven, packaged, and **delivered to the serving
  machine** (fingerprint-checked, clean directory).
- The runaway-cost safety switch: proven live, as described above.
- Serving: **the personality is live on the engine** — the round-3
  alias answers, and its voice is measurably its own: at zero creativity
  settings it talks terse and lowercase ("Qwen is what you're getting
  to.") where the base recites corporate boilerplate. The honest catch
  from the first recorded test: when asked who it *is*, it still names
  the base model. Round 3 taught the voice into the weights; the *name
  claim* is what round 4's teaching material is now aimed at.
- Queue: the next training round targets the name claim in the material
  itself; the permanent score history now holds the first honest
  round-3 row next to its base control; and the optional
  compact-version pilot waits behind it.

The technical companion is [`docs/SPEC.md`](docs/SPEC.md) — terse, dated,
receipt-backed; start at the feature index (§0).
