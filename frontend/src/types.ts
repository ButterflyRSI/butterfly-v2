export interface Signals {
  asked_questions: boolean;
  appropriate_length: boolean;
  followed_constraints: boolean;
  shows_self_awareness: boolean;
  quality_score: number;
  deserves_reinforcement: boolean;
  word_count: number;
  source: string;
  notes: string[];
}

export interface Turn {
  response: string;
  signals: Signals;
  reinforced: boolean;
  rules_added: string[];
  rules_pruned: string[];
  self_evaluation: string | null;
  mirror_loop_ran: boolean;
  mirror_reflection: string | null;
  trait_changes: Record<string, number>;
  message_count: number;
  elapsed_ms: number;
}

export interface Rule {
  id: string;
  text: string;
  strength: number;
  successes: number;
  failures: number;
  created_turn: number;
}

export interface AgentState {
  name: string;
  message_count: number;
  next_mirror_in: number;
  constraints: Rule[];
  traits: { traits: Record<string, number>; baseline: Record<string, number> };
  history: { role: "user" | "assistant"; content: string }[];
  last_turn: Turn | null;
  last_reflection: string | null;
  config: { quality_threshold: number; mirror_loop_interval: number; max_strength: number; prune_floor: number };
}

export interface Persona {
  key: string;
  name: string;
  description: string;
}

export interface Health {
  ok: boolean;
  backend: string;
  error: string | null;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  turn?: Turn;
}
