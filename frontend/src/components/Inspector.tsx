import type { AgentState } from "../types";
import RuleList from "./RuleList";
import ScoreMeter from "./ScoreMeter";
import TraitBars from "./TraitBars";

const SIGNAL_LABELS: [keyof NonNullable<AgentState["last_turn"]>["signals"], string][] = [
  ["asked_questions", "Asked for clarity when needed"],
  ["appropriate_length", "Length fit the request"],
  ["followed_constraints", "Followed active rules"],
  ["shows_self_awareness", "Admitted uncertainty when needed"],
];

export default function Inspector({ state }: { state: AgentState }) {
  const turn = state.last_turn;
  const cfg = state.config;
  const mirrorText =
    state.next_mirror_in === cfg.mirror_loop_interval && state.message_count > 0
      ? "Mirror loop just ran"
      : `Mirror loop in ${state.next_mirror_in} ${state.next_mirror_in === 1 ? "message" : "messages"}`;

  return (
    <aside className="inspector" aria-label="Agent state">
      <section className="panel">
        <h2>Last reply</h2>
        {turn ? (
          <>
            <div className="score-row">
              <span className="score-big">{turn.signals.quality_score.toFixed(2)}</span>
              <span className={`verdict ${turn.reinforced ? "is-reinforce" : "is-learn"}`}>
                {turn.reinforced ? "Reinforced active rules" : "Below the gate, learning"}
              </span>
            </div>
            <ScoreMeter score={turn.signals.quality_score} threshold={cfg.quality_threshold} reinforced={turn.reinforced} />
            <ul className="signals">
              {SIGNAL_LABELS.map(([key, label]) => (
                <li key={key} className={turn.signals[key] ? "pass" : "fail"}>
                  <span aria-hidden="true">{turn.signals[key] ? "✓" : "✗"}</span> {label}
                </li>
              ))}
            </ul>
            {turn.signals.notes.length > 0 && (
              <ul className="notes">
                {turn.signals.notes.map((n) => (
                  <li key={n}>{n}</li>
                ))}
              </ul>
            )}
            {turn.self_evaluation && (
              <details className="selfeval">
                <summary>Self-evaluation</summary>
                <p>{turn.self_evaluation}</p>
              </details>
            )}
          </>
        ) : (
          <p className="empty">Send a message to see how the reply is scored.</p>
        )}
      </section>

      <section className="panel">
        <div className="panel__head">
          <h2>Personality</h2>
          <span className="hint">{mirrorText}</span>
        </div>
        <TraitBars
          traits={state.traits.traits}
          baseline={state.traits.baseline}
          changes={turn?.mirror_loop_ran ? turn.trait_changes : undefined}
        />
        {state.last_reflection && <blockquote className="reflection">{state.last_reflection}</blockquote>}
      </section>

      <section className="panel">
        <div className="panel__head">
          <h2>Learned rules</h2>
          <span className="hint">{state.constraints.length} active</span>
        </div>
        <RuleList
          rules={state.constraints}
          maxStrength={cfg.max_strength}
          floor={cfg.prune_floor}
          fresh={turn?.rules_added ?? []}
        />
        {turn && turn.rules_pruned.length > 0 && (
          <p className="pruned">Pruned: {turn.rules_pruned.join("; ")}</p>
        )}
      </section>
    </aside>
  );
}
