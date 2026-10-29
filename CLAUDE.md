# CLAUDE.md

Notes for Claude Code working in this repo.

## What this is

Butterfly is the reference implementation of RJ Thomas's white paper (`docs/Butterfly_White_Paper.rtf`): a self-correcting constraint layer that sits above any LLM. It's a homeostatic control layer, closer to a thermostat than a tuner. It keeps agent behavior stable; it does not optimize prompts or claim to improve model output quality.

## Commands

- Run tests: `pytest` (all 55 must pass before committing)
- Run app: `.\start.ps1` (Windows) or `python -m butterfly`; `--demo` needs no model
- Build UI: `cd frontend; npm run build`
- UI dev server: `cd frontend; npm run dev` (proxies /api to port 8000)

## Rules for changes

- The paper's default parameters in `butterfly/config.py` are the spec. Don't change a default without the owner's say-so; add an opt-in setting instead.
- Keep the three methods in their own modules: Method 1 in `constraints.py`, Method 2 in `traits.py` plus `agent._mirror_loop`, Method 3 in `extractor.py`. The shared quality score lives in `quality.py`.
- Any change to loop behavior needs a test, and the test should be shown to fail without the change.
- Record deviations from the paper in `docs/IMPLEMENTATION_NOTES.md`.
- Work on a branch. Never commit `data/` (runtime state) or secrets.
- Owner's workflow: talk the design through first, patch problems up front, then build. Explain what you changed and why.

## Stack

Python 3.10+, FastAPI, httpx, pytest. Frontend is React 18 + TypeScript + Vite. Default model backend is Ollama (`llama3.1`) on an RTX 3060 12GB; OpenAI-compatible servers and a mock backend are also supported.
