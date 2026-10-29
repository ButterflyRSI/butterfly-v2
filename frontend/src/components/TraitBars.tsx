interface Props {
  traits: Record<string, number>;
  baseline: Record<string, number>;
  changes?: Record<string, number>;
}

export default function TraitBars({ traits, baseline, changes }: Props) {
  return (
    <ul className="traits">
      {Object.entries(traits).map(([name, value]) => {
        const delta = changes?.[name] ?? 0;
        return (
          <li key={name} className="trait">
            <div className="trait__head">
              <span className="trait__name">{name}</span>
              <span className="trait__value">
                {value.toFixed(2)}
                {delta !== 0 && (
                  <span className={delta > 0 ? "delta up" : "delta down"}>
                    {delta > 0 ? "+" : ""}
                    {delta.toFixed(3)}
                  </span>
                )}
              </span>
            </div>
            <div className="trait__track">
              <div className="trait__fill" style={{ width: `${value * 100}%` }} />
              {name in baseline && (
                <div
                  className="trait__baseline"
                  style={{ left: `${baseline[name] * 100}%` }}
                  title={`Starting value ${baseline[name].toFixed(2)}`}
                />
              )}
            </div>
          </li>
        );
      })}
    </ul>
  );
}
