# Query fan-out tool

A web tool for **query fan-out analysis**: model the sub-queries an AI search engine would fan a
question into (optionally conditioned on a buyer persona), pool repeated runs, and get back a
deterministic **entity analysis** plus a writer's **brief** — with every angle and source tagged by
where it came from (live engine vs. model).

It wraps a small Python core behind a [Streamlit](https://streamlit.io) interface so you don't have to
run scripts by hand.

> **Status: under construction.** Being built in phases — see the build plan. The core logic is
> already written and tested; this repo is the importable-core + UI rebuild.

## Bring your own keys

This tool runs on **your** API keys (OpenAI / Google Gemini / Anthropic). You paste them into the app;
they are held in memory for the duration of your run only, passed straight to the providers, and
**never logged, stored, or written to disk**. This repo is public precisely so you can verify that.
(Because the app runs server-side, your keys do pass through the app process in memory while a run is
executing — that's unavoidable for a server app, and they are never persisted.)

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

## License

MIT — see [LICENSE](LICENSE).
