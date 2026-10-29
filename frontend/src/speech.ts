// Reads replies aloud. Uses the server's Kokoro voices when available,
// otherwise the browser's built-in voice.
//
// Long replies are split into chunks so the first sentence starts playing
// quickly; the next chunk is generated while the current one plays.

const LANGS: Record<string, string> = {
  a: "US", b: "UK", e: "Spanish", f: "French", h: "Hindi",
  i: "Italian", j: "Japanese", p: "Portuguese", z: "Chinese",
};

export function voiceLabel(id: string): string {
  const [code, raw = id] = id.split("_");
  const name = raw.charAt(0).toUpperCase() + raw.slice(1);
  const lang = LANGS[code?.[0] ?? ""] ?? "";
  const gender = code?.[1] === "f" ? "female" : code?.[1] === "m" ? "male" : "";
  return [name, [lang, gender].filter(Boolean).join(" ")].filter(Boolean).join(", ");
}

/** English voices first, then everything else, alphabetical within each group. */
export function sortVoices(ids: string[]): string[] {
  const english = (v: string) => (v[0] === "a" || v[0] === "b" ? 0 : 1);
  return [...ids].sort((x, y) => english(x) - english(y) || x.localeCompare(y));
}

export function chunkText(text: string, max = 260): string[] {
  const sentences = text.replace(/\s+/g, " ").match(/[^.!?]+[.!?]+["')\]]*\s*|[^.!?]+$/g) ?? [text];
  const chunks: string[] = [];
  let current = "";
  for (const s of sentences) {
    if (current && (current + s).length > max) {
      chunks.push(current.trim());
      current = "";
    }
    current += s;
  }
  if (current.trim()) chunks.push(current.trim());
  return chunks;
}

const browserVoice = typeof window !== "undefined" && "speechSynthesis" in window;

export class Speaker {
  private audio: HTMLAudioElement | null = null;
  private session = 0;
  private urls: string[] = [];

  stop() {
    this.session++;
    if (this.audio) {
      this.audio.pause();
      this.audio = null;
    }
    this.urls.forEach((u) => URL.revokeObjectURL(u));
    this.urls = [];
    if (browserVoice) window.speechSynthesis.cancel();
  }

  /** Speak with Kokoro; falls back to the browser voice if the server can't. */
  async speak(text: string, opts: { persona: string; voice?: string; useKokoro: boolean }) {
    this.stop();
    const id = this.session;
    if (!opts.useKokoro) return this.speakBrowser(text);

    const chunks = chunkText(text);
    const fetchChunk = async (chunk: string): Promise<string> => {
      const res = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ persona: opts.persona, text: chunk, voice: opts.voice || undefined }),
      });
      if (!res.ok) throw new Error(`voice request failed (${res.status})`);
      const url = URL.createObjectURL(await res.blob());
      this.urls.push(url);
      return url;
    };

    let i = 0;
    try {
      let next = fetchChunk(chunks[0]);
      for (; i < chunks.length; i++) {
        const url = await next;
        if (id !== this.session) return;
        if (i + 1 < chunks.length) next = fetchChunk(chunks[i + 1]);
        await this.play(url, id);
        if (id !== this.session) return;
      }
    } catch {
      if (id === this.session) this.speakBrowser(chunks.slice(i).join(" "));
    }
  }

  private play(url: string, id: number): Promise<void> {
    return new Promise((resolve) => {
      if (id !== this.session) return resolve();
      const audio = new Audio(url);
      this.audio = audio;
      audio.onended = () => resolve();
      audio.onerror = () => resolve();
      audio.play().catch(() => resolve());
    });
  }

  private speakBrowser(text: string) {
    if (!browserVoice) return;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
  }
}
