# Implementation notes

This file records where the code follows the white paper exactly, where it fills in details the paper leaves open, and what is still limited. Every default parameter matches the paper's Section 11.2.

## Followed exactly

- Strength steps of +0.1 and −0.05, cap 2.0, weaken floor 0.1, pruning at 0.2.
- Quality score is the mean of four binary signals; the gate is `score >= 0.65`, so 3 of 4 passes and 2 of 4 does not.
- A reinforced turn strengthens every active rule and extracts nothing. A failed turn weakens every active rule and may extract new ones.
- Extraction keywords: should, must, need to, avoid, ensure.
- Mirror loop fires when `message_count % 3 == 0`, nudges traits by 0.05, then pulls each trait 15% toward the group mean.

## Gaps filled in

The paper's Section 10 code is a sketch. These are the places where this implementation had to decide something the sketch leaves open, or where the sketch has a bug.

1. **Extraction reads the self-evaluation, not the reply.** The sketch passes the raw response to the extractor, but Section 5.2 says the self-evaluation text is what gets mined. On a failed turn the agent is asked to critique its own reply, and rules are extracted from that critique. If the model can't produce one, rules are derived from the problems the evaluator detected.
2. **Deduplication checks stored rules too.** The sketch only deduplicates within one batch, so the same rule piles up after repeated bad turns. Here a new rule is rejected if it matches, or overlaps 80% or more with, any rule already stored.
3. **The mirror loop includes the trait nudge.** The sketch's `_mirror_loop` only runs drift correction. Section 4.2 describes a self-evaluation and nudge step first; this implementation runs both, nudging `curious` when most recent replies asked questions, `verbose` by reply length, and `formal` by use of contractions. Nudges only touch traits the persona has.
4. **No silent trait creation.** The sketch's `adjust()` creates a new trait at 0.5 if the name is misspelled. Here it raises an error instead.
5. **Pruning tolerates float rounding.** 1.0 minus sixteen steps of 0.05 lands a hair above 0.2 in floating point, which would keep a rule alive one turn longer than the paper intends. A tiny tolerance makes the 16th failure prune it.
6. **Prompt assembly.** The strongest 12 rules go into the system prompt, labeled Always (1.5+), Usually (0.8+), or When possible, so stronger rules carry more weight. The last 20 messages of history are sent with each request.

## Optional controls (off by default)

Long-run simulations in an earlier round of testing found two ways the default design isn't fully self-limiting. Both controls are off by default so the paper's behavior is reproduced exactly.

- `max_constraints`: the rule set can grow without limit if failures keep producing new, distinct rules faster than old ones are pruned. Setting a cap drops the weakest rules first.
- `baseline_pull`: mean reversion can't bring traits down if they are all pushed to the ceiling together, because the mean is then at the ceiling too. A small pull toward each persona's starting values gives the profile a way home.

## Known limitations

- The heuristic evaluator is simple. In particular, `followed_constraints` only detects broken "avoid / never / don't" rules; it can't tell whether a positive rule like "ask a clarifying question" was followed. `--evaluator llm` lets the model grade itself, at the cost of an extra model call per turn, and falls back to the heuristics if the model's answer isn't valid JSON.
- All active rules are reinforced or weakened together, as the paper specifies, even rules that had nothing to do with that reply. Over many good turns, every rule drifts toward the 2.0 cap.
- Keyword extraction is brittle, as the paper notes: it misses rules phrased without a trigger word and can pick up incidental ones.
- The parameters are the paper's starting points, not tuned values.
