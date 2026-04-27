import base64
import io

from PIL import Image

from palm_reader import (
    build_contour_prompt,
    encode_image_data_url,
    optimize_upload_image,
    parse_validation_response,
    split_report_sections,
)


def test_encode_image_data_url() -> None:
    data_url = encode_image_data_url(b"abc", "image/png")

    assert data_url == f"data:image/png;base64,{base64.b64encode(b'abc').decode('ascii')}"


def test_optimize_upload_image_returns_reasonable_jpeg() -> None:
    source = io.BytesIO()
    Image.new("RGB", (3200, 2000), color="white").save(source, format="PNG")

    optimized, mime_type = optimize_upload_image(source.getvalue(), max_side=800)
    result = Image.open(io.BytesIO(optimized))

    assert mime_type == "image/jpeg"
    assert max(result.size) <= 800


def test_build_contour_prompt_includes_labels_and_context() -> None:
    prompt = build_contour_prompt("## Major Lines\nHeart line is clear.")

    assert "Heart Line" in prompt
    assert "Life Line" in prompt
    assert "Heart line is clear." in prompt


def test_parse_validation_response_extracts_json_from_text() -> None:
    validation = parse_validation_response(
        'Result: {"is_valid": false, "score": 42, "issues": ["blurry"], "tips": ["use daylight"]}'
    )

    assert not validation.is_valid
    assert validation.score == 42
    assert validation.issues == ["blurry"]
    assert validation.tips == ["use daylight"]


def test_split_report_sections_uses_markdown_headings() -> None:
    sections = split_report_sections("Intro\n\n## At a Glance\nBright path.\n\n## Your Palm Lines\nHeart line.")

    assert sections["Overview"] == "Intro"
    assert sections["At a Glance"] == "Bright path."
    assert sections["Your Palm Lines"] == "Heart line."
