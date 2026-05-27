import tempfile
import unittest
from pathlib import Path
from unittest import mock

from photo_to_json.models import DocumentMetadata, ReportEntry
from photo_to_json.pipeline import (
    build_final_document,
    extract_reports_with_sliding_window,
    list_image_paths,
    split_front_matter_and_index,
)


class TestPipeline(unittest.TestCase):
    def test_list_image_paths_and_split_front_matter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            for filename in ["10.png", "2.png", "001.jpg", "a.txt", "03.jpeg"]:
                (tmp_dir / filename).touch()

            sorted_images = list_image_paths(tmp_dir)
            self.assertEqual([path.name for path in sorted_images], ["001.jpg", "03.jpeg", "10.png", "2.png"])

            front, index = split_front_matter_and_index(sorted_images)
            self.assertEqual(len(front), 4)
            self.assertEqual(len(index), 0)

    def test_extract_reports_retries_and_handles_last_page(self) -> None:
        paths = [Path("page1.png"), Path("page2.png")]
        calls = []

        def fake_extract(_client, first_page, second_page):
            calls.append((first_page.name, second_page.name))
            if len(calls) == 1:
                raise RuntimeError("rate limit")
            entry_id = len(calls)
            return mock.Mock(
                entries=[
                    ReportEntry(
                        entry_id=entry_id,
                        authors=[],
                        title=f"entry-{entry_id}",
                        translated_title=None,
                        publication_info="publisher info",
                        is_stitched=False,
                    )
                ]
            )

        with mock.patch("photo_to_json.pipeline.extract_index_page_entries", side_effect=fake_extract):
            reports = extract_reports_with_sliding_window(
                client=mock.Mock(),
                index_page_paths=paths,
                max_retries=3,
                retry_delay_seconds=0,
            )

        self.assertEqual(calls, [("page1.png", "page2.png"), ("page1.png", "page2.png"), ("page2.png", "page2.png")])
        self.assertEqual([report.entry_id for report in reports], [2, 3])

    def test_extract_reports_skips_pair_after_retry_exhaustion(self) -> None:
        paths = [Path("page1.png"), Path("page2.png")]

        with mock.patch(
            "photo_to_json.pipeline.extract_index_page_entries",
            side_effect=RuntimeError("always failing"),
        ) as extract_mock:
            reports = extract_reports_with_sliding_window(
                client=mock.Mock(),
                index_page_paths=paths,
                max_retries=2,
                retry_delay_seconds=0,
            )

        self.assertEqual(reports, [])
        self.assertEqual(extract_mock.call_count, 4)

    def test_build_final_document_uses_first_four_front_matter_images(self) -> None:
        image_paths = [Path(f"image_{index}.png") for index in range(8)]

        with (
            mock.patch(
                "photo_to_json.pipeline.extract_document_metadata",
                return_value=DocumentMetadata(
                    book_title="Economic and Social Implications of Automation",
                    publisher="Michigan State Labor & Industrial Relations Center",
                    publication_year=1966,
                ),
            ) as metadata_mock,
            mock.patch(
                "photo_to_json.pipeline.extract_reports_with_sliding_window",
                return_value=[
                    ReportEntry(
                        entry_id=1,
                        authors=["DOE J"],
                        title="Automation study",
                        translated_title=None,
                        publication_info="City: Publisher",
                        is_stitched=True,
                    )
                ],
            ) as reports_mock,
        ):
            final_document = build_final_document(client=mock.Mock(), image_paths=image_paths)

        metadata_mock.assert_called_once()
        self.assertEqual(len(metadata_mock.call_args.args[1]), 4)
        reports_mock.assert_called_once_with(mock.ANY, image_paths[4:])
        self.assertEqual(final_document.metadata.publication_year, 1966)
        self.assertEqual(len(final_document.all_reports), 1)


if __name__ == "__main__":
    unittest.main()
