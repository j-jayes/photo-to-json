import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import google.genai as genai
from PIL import Image

from photo_to_json.models import DocumentMetadata, FinalDocument, IndexPage, ReportEntry

MODEL_NAME = "gemini-3.5-flash"
LOGGER = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def list_image_paths(input_dir: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ],
        key=lambda path: path.name.lower(),
    )


def split_front_matter_and_index(
    image_paths: Sequence[Path], front_matter_count: int = 4
) -> Tuple[List[Path], List[Path]]:
    return list(image_paths[:front_matter_count]), list(image_paths[front_matter_count:])


def _load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.copy()


def extract_document_metadata(client: genai.Client, front_matter_paths: Sequence[Path]) -> DocumentMetadata:
    images = [_load_image(path) for path in front_matter_paths]
    prompt = "Analyze these front matter pages of a book. Extract the main title, publisher, and publication year."
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[*images, prompt],
        config={
            "response_format": {
                "text": {
                    "mime_type": "application/json",
                    "schema": DocumentMetadata.model_json_schema(),
                }
            }
        },
    )
    metadata = json.loads(response.text)
    return DocumentMetadata.model_validate(metadata)


def extract_index_page_entries(
    client: genai.Client, first_page_path: Path, second_page_path: Path
) -> IndexPage:
    first_page = _load_image(first_page_path)
    second_page = _load_image(second_page_path)
    prompt = (
        "You are an expert data extraction assistant. I have provided two consecutive pages from a bibliography index. "
        "Your task is to extract all report entries that START on the FIRST page provided. "
        "If an entry starts at the bottom of the first page and continues onto the second page, "
        "you MUST stitch the information together into a single complete record and set 'is_stitched' to true. "
        "ONLY extract entries that begin on the FIRST page. Ignore entries that start on the second page. "
        "Ignore page headers, footers, and general explanatory text."
    )
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[first_page, second_page, prompt],
        config={
            "response_format": {
                "text": {
                    "mime_type": "application/json",
                    "schema": IndexPage.model_json_schema(),
                }
            },
            "temperature": 0.0,
        },
    )
    page_data = json.loads(response.text)
    return IndexPage.model_validate(page_data)


def extract_reports_with_sliding_window(
    client: genai.Client,
    index_page_paths: Sequence[Path],
    max_retries: int = 3,
    retry_delay_seconds: int = 5,
) -> List[ReportEntry]:
    all_reports: List[ReportEntry] = []
    if not index_page_paths:
        return all_reports

    for page_index, first_page_path in enumerate(index_page_paths):
        second_page_path = (
            index_page_paths[page_index + 1]
            if page_index + 1 < len(index_page_paths)
            else index_page_paths[page_index]
        )
        LOGGER.info(
            "Starting extraction for pair %s/%s: %s + %s",
            page_index + 1,
            len(index_page_paths),
            first_page_path.name,
            second_page_path.name,
        )

        for attempt in range(1, max_retries + 1):
            try:
                page = extract_index_page_entries(client, first_page_path, second_page_path)
                all_reports.extend(page.entries)
                LOGGER.info(
                    "Completed extraction for pair %s/%s on attempt %s",
                    page_index + 1,
                    len(index_page_paths),
                    attempt,
                )
                break
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as error:  # noqa: BLE001
                LOGGER.error(
                    "Failed extraction for pair %s/%s on attempt %s: %s",
                    page_index + 1,
                    len(index_page_paths),
                    attempt,
                    error,
                )
                if attempt < max_retries:
                    time.sleep(retry_delay_seconds)
                else:
                    LOGGER.error(
                        "Skipping pair %s/%s after %s failed attempts",
                        page_index + 1,
                        len(index_page_paths),
                        max_retries,
                    )

    return all_reports


def build_final_document(client: genai.Client, image_paths: Sequence[Path]) -> FinalDocument:
    front_matter_paths, index_page_paths = split_front_matter_and_index(image_paths)
    if len(front_matter_paths) < 4:
        raise ValueError("At least 4 images are required (used as front matter for metadata extraction).")

    metadata = extract_document_metadata(client, front_matter_paths[:4])
    all_reports = extract_reports_with_sliding_window(client, index_page_paths)
    return FinalDocument(metadata=metadata, all_reports=all_reports)


def run_pipeline(input_dir: Path, output_path: Path, api_key: str) -> FinalDocument:
    image_paths = list_image_paths(input_dir)
    client = genai.Client(api_key=api_key)
    final_document = build_final_document(client, image_paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(final_document.model_dump_json(indent=4), encoding="utf-8")
    return final_document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract bibliography index entries from images using Gemini."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing ordered document images")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output.json"),
        help="Destination JSON file",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Gemini API key (defaults to GEMINI_API_KEY env var)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    args = parse_args()
    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Gemini API key is required via --api-key or GEMINI_API_KEY")

    run_pipeline(args.input_dir, args.output, api_key)


if __name__ == "__main__":
    main()
