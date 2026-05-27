# photo-to-json

Cookiecutter-style data science project for extracting bibliography index entries from publication images into structured JSON using Gemini.

## Project layout

- `/data/raw`: source images (front matter + index pages)
- `/data/interim`: optional intermediate artifacts
- `/data/processed`: processed outputs
- `/notebooks`: exploratory notebooks
- `/references`: publication/reference materials
- `/reports`: generated reports
- `/src/photo_to_json`: extraction pipeline and data models
- `/tests`: focused unit tests

## Usage

```bash
python -m photo_to_json.pipeline /absolute/path/to/input/images --output /absolute/path/to/output.json
```

Set `GEMINI_API_KEY` in your environment (or pass `--api-key`).
