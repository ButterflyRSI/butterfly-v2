"""Start the Butterfly server.

    python -m butterfly                    # Ollama + llama3.1
    python -m butterfly --model qwen2.5:14b
    python -m butterfly --demo             # no model needed
"""

import argparse
import os
import threading
import webbrowser


def main() -> None:
    p = argparse.ArgumentParser(prog="butterfly", description="Run the Butterfly server and UI.")
    p.add_argument("--model", help="model name, e.g. llama3.1 or qwen2.5:14b")
    p.add_argument("--backend", choices=["ollama", "openai", "mock"], help="model backend")
    p.add_argument("--demo", action="store_true", help="use the built-in mock model (no Ollama needed)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--evaluator", choices=["heuristic", "llm"], default="heuristic",
                   help="how replies are graded (llm = the model grades itself)")
    p.add_argument("--no-browser", action="store_true", help="don't open the browser")
    args = p.parse_args()

    if args.model:
        os.environ["BUTTERFLY_MODEL"] = args.model
    if args.backend:
        os.environ["BUTTERFLY_BACKEND"] = args.backend
    if args.demo:
        os.environ["BUTTERFLY_BACKEND"] = "mock"

    import uvicorn

    from .config import ButterflyConfig
    from .server import DIST, create_app

    app = create_app(config=ButterflyConfig(evaluator_mode=args.evaluator))
    url = f"http://{args.host}:{args.port}"
    if DIST.exists():
        print(f"\n  Butterfly is running at {url}\n")
        if not args.no_browser:
            threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    else:
        print(f"\n  API running at {url}/docs")
        print("  UI not built yet: cd frontend, npm install, npm run build\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
