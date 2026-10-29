interface Props {
  score: number;
  threshold: number;
  reinforced: boolean;
  compact?: boolean;
}

/** Quality score against the Method 3 gate. Amber = reinforced, blue = learned. */
export default function ScoreMeter({ score, threshold, reinforced, compact }: Props) {
  return (
    <div
      className={`meter ${reinforced ? "is-reinforce" : "is-learn"} ${compact ? "meter--compact" : ""}`}
      role="meter"
      aria-valuemin={0}
      aria-valuemax={1}
      aria-valuenow={score}
      aria-label={`Quality score ${score.toFixed(2)}, gate at ${threshold}`}
    >
      <div className="meter__track">
        <div className="meter__fill" style={{ width: `${score * 100}%` }} />
        <div className="meter__gate" style={{ left: `${threshold * 100}%` }} title={`Gate at ${threshold}`} />
      </div>
    </div>
  );
}
