import type { Rule } from "../types";

interface Props {
  rules: Rule[];
  maxStrength: number;
  floor: number;
  fresh: string[];
}

export default function RuleList({ rules, maxStrength, floor, fresh }: Props) {
  if (rules.length === 0) {
    return (
      <p className="empty">
        No rules yet. Rules are learned only when a reply scores below the gate, so a run of
        good replies keeps this list empty.
      </p>
    );
  }
  return (
    <ol className="rules">
      {rules.map((r) => (
        <li key={r.id} className={`rule ${fresh.includes(r.text) ? "rule--fresh" : ""}`}>
          <p className="rule__text">{r.text}</p>
          <div className="rule__meta">
            <div className="rule__track" title={`Pruned at ${floor}, capped at ${maxStrength}`}>
              <div className="rule__fill" style={{ width: `${(r.strength / maxStrength) * 100}%` }} />
              <div className="rule__floor" style={{ left: `${(floor / maxStrength) * 100}%` }} />
            </div>
            <span className="rule__num">{r.strength.toFixed(2)}</span>
            <span className="rule__count" title="Times reinforced / weakened">
              {r.successes} up, {r.failures} down
            </span>
          </div>
        </li>
      ))}
    </ol>
  );
}
