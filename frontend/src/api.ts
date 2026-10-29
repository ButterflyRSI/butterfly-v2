import type { AgentState, Health, Persona, TTSStatus, Turn } from "./types";

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* keep default */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => call<Health>("/health"),
  personas: () => call<Persona[]>("/personas"),
  tts: () => call<TTSStatus>("/tts"),
  state: (persona: string) => call<AgentState>(`/state/${persona}`),
  reset: (persona: string) => call<AgentState>(`/reset/${persona}`, { method: "POST" }),
  chat: (persona: string, message: string) =>
    call<{ turn: Turn; state: AgentState }>("/chat", {
      method: "POST",
      body: JSON.stringify({ persona, message }),
    }),
};
