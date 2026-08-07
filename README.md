# AEROTHON 2026 — OMEGA-EMS Dashboard

This repository contains the Streamlit dashboard for the Aerothon 2026 hybrid-electric propulsion project.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run aerothon_2026/dashboard/app.py
```

## Deploy on Replit

1. Create a new Replit project and upload this repository.
2. Replit will use `.replit` to start the app.
3. The app command is:

```bash
cd aerothon_2026 && streamlit run dashboard/app.py --server.port $PORT --server.enableCORS false
```

## Deploy on other platforms

- Use `requirements.txt` to install dependencies.
- Use `Procfile` if the platform supports Heroku-style deployment.
- The app entrypoint is:

```bash
streamlit run aerothon_2026/dashboard/app.py --server.port $PORT --server.enableCORS false
```

## Notes

- The main Streamlit dashboard file is `aerothon_2026/dashboard/app.py`.
- Make sure the `models/` folder is present and `src/` imports resolve correctly.
- If using a platform with a different startup command, set the command to the same Streamlit launch line above.
