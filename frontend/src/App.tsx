import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import Inspector from "./components/Inspector";
import ScoreMeter from "./components/ScoreMeter";
import { Speaker, sortVoices, voiceLabel } from "./speech";
import type { AgentState, ChatMessage, Health, Persona, TTSStatus } from "./types";

const speechSupported = typeof window !== "undefined" && "speechSynthesis" in window;

function readStored(key: string, fallback: string): string {
  try {
    return localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}

function store(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* storage unavailable; setting just won't persist */
  }
}

export default function App() {
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [persona, setPersona] = useState(() => readStored("butterfly.persona", "nova"));
  const [state, setState] = useState<AgentState | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [readAloud, setReadAloud] = useState(() => readStored("butterfly.readAloud", "off") === "on");
  const [showInspector, setShowInspector] = useState(false);
  const [tts, setTts] = useState<TTSStatus | null>(null);
  const [voiceChoice, setVoiceChoice] = useState<Record<string, string>>({});
  const speaker = useRef(new Speaker());
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    api.personas().then(setPersonas).catch((e) => setError(e.message));
    api.tts().then(setTts).catch(() => setTts(null));
    api.health().then(setHealth).catch(() =>
      setHealth({ ok: false, backend: "server", error: "Can't reach the Butterfly server. Is it running?" }),
    );
  }, []);

  const loadPersona = useCallback((key: string) => {
    setError(null);
    api
      .state(key)
      .then((s) => {
        setState(s);
        setMessages(s.history.map((m) => ({ role: m.role, content: m.content })));
      })
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    loadPersona(persona);
    store("butterfly.persona", persona);
    speaker.current.stop();
  }, [persona, loadPersona]);

  useEffect(() => {
    const saved = readStored(`butterfly.voice.${persona}`, "");
    setVoiceChoice((v) => (persona in v ? v : { ...v, [persona]: saved }));
  }, [persona]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, busy]);

  async function send() {
    const text = draft.trim();
    if (!text || busy) return;
    setDraft("");
    setError(null);
    setMessages((m) => [...m, { role: "user", content: text }]);
    setBusy(true);
    try {
      const { turn, state: next } = await api.chat(persona, text);
      setMessages((m) => [...m, { role: "assistant", content: turn.response, turn }]);
      setState(next);
      if (readAloud) {
        speaker.current.speak(turn.response, {
          persona,
          voice: voiceChoice[persona] || undefined,
          useKokoro: !!tts?.available,
        });
      }
    } catch (e) {
      setError((e as Error).message);
      setMessages((m) => m.slice(0, -1));
      setDraft(text);
    } finally {
      setBusy(false);
      inputRef.current?.focus();
    }
  }

  async function reset() {
    if (!window.confirm(`Clear ${state?.name ?? "this persona"}'s rules, personality changes, and chat history?`)) return;
    try {
      const s = await api.reset(persona);
      setState(s);
      setMessages([]);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  function toggleReadAloud() {
    const next = !readAloud;
    setReadAloud(next);
    store("butterfly.readAloud", next ? "on" : "off");
    if (!next) speaker.current.stop();
  }

  function chooseVoice(id: string) {
    setVoiceChoice((v) => ({ ...v, [persona]: id }));
    store(`butterfly.voice.${persona}`, id);
    speaker.current.speak(`Hi, I'm ${current?.name ?? "here"}. This is how I sound.`, {
      persona,
      voice: id || undefined,
      useKokoro: true,
    });
  }

  const current = personas.find((p) => p.key === persona);
  const canSpeak = speechSupported || !!tts?.available;
  const defaultVoice = current?.voice?.name;
  const threshold = state?.config.quality_threshold ?? 0.65;

  return (
    <div className="app">
      <header className="topbar">
        <h1 className="wordmark">Butterfly</h1>
        <nav className="personas" aria-label="Persona">
          {personas.map((p) => (
            <button
              key={p.key}
              className={p.key === persona ? "persona is-active" : "persona"}
              aria-pressed={p.key === persona}
              onClick={() => setPersona(p.key)}
              disabled={busy}
            >
              {p.name}
            </button>
          ))}
        </nav>
        <div className="topbar__tools">
          <span className={`status ${health?.ok ? "ok" : "bad"}`} title={health?.error ?? health?.backend ?? ""}>
            {health ? (health.ok ? health.backend : "Model offline") : "Checking model"}
          </span>
          {canSpeak && (
            <button className="tool" aria-pressed={readAloud} onClick={toggleReadAloud}>
              {readAloud ? "Reading aloud" : "Read aloud"}
            </button>
          )}
          {tts?.available && (
            <label className="voice">
              <span className="sr-only">Voice for {current?.name}</span>
              <select value={voiceChoice[persona] ?? ""} onChange={(e) => chooseVoice(e.target.value)}>
                <option value="">{defaultVoice ? `${voiceLabel(defaultVoice)} (default)` : "Default voice"}</option>
                {sortVoices(tts.voices).map((v) => (
                  <option key={v} value={v}>
                    {voiceLabel(v)}
                  </option>
                ))}
              </select>
            </label>
          )}
          <button className="tool tool--inspector" aria-pressed={showInspector} onClick={() => setShowInspector((v) => !v)}>
            {showInspector ? "Show chat" : "Show state"}
          </button>
          <button className="tool" onClick={reset} disabled={busy || !state}>
            Reset
          </button>
        </div>
      </header>

      {health && !health.ok && health.error && <div className="banner">{health.error}</div>}
      {error && (
        <div className="banner banner--error" role="alert">
          {error}
        </div>
      )}

      <main className={`layout ${showInspector ? "show-inspector" : ""}`}>
        <section className="chat" aria-label="Conversation">
          <div className="log" aria-live="polite">
            {messages.length === 0 && (
              <div className="intro">
                <p className="intro__name">{current?.name ?? "…"}</p>
                <p className="intro__desc">{current?.description}</p>
                <p className="intro__how">
                  Every reply is scored. Good ones strengthen the rules that were active; weak ones
                  teach a new rule. Every {state?.config.mirror_loop_interval ?? 3} messages the
                  personality checks itself for drift. Watch it happen on the right.
                </p>
              </div>
            )}
            {messages.map((m, i) => (
              <article key={i} className={`msg msg--${m.role}`}>
                <p className="msg__text">{m.content}</p>
                {m.turn && (
                  <div className="msg__loop">
                    <ScoreMeter compact score={m.turn.signals.quality_score} threshold={threshold} reinforced={m.turn.reinforced} />
                    <span>
                      {m.turn.signals.quality_score.toFixed(2)}{" "}
                      {m.turn.reinforced
                        ? "reinforced"
                        : m.turn.rules_added.length
                          ? `learned ${m.turn.rules_added.length} ${m.turn.rules_added.length === 1 ? "rule" : "rules"}`
                          : "weakened rules"}
                      {m.turn.mirror_loop_ran && ", mirror loop ran"}
                    </span>
                  </div>
                )}
              </article>
            ))}
            {busy && <p className="thinking">{current?.name ?? "Agent"} is thinking</p>}
            <div ref={endRef} />
          </div>

          <form
            className="composer"
            onSubmit={(e) => {
              e.preventDefault();
              send();
            }}
          >
            <label htmlFor="msg" className="sr-only">
              Message
            </label>
            <textarea
              id="msg"
              ref={inputRef}
              value={draft}
              rows={2}
              placeholder={`Message ${current?.name ?? ""}`}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
            />
            <button type="submit" className="send" disabled={busy || !draft.trim()}>
              Send
            </button>
          </form>
        </section>

        {state && <Inspector state={state} />}
      </main>
    </div>
  );
}
