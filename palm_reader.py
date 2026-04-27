from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass
from typing import Any
from xml.sax.saxutils import escape

from PIL import Image


DEFAULT_VISION_MODEL = "gpt-5.5"
DEFAULT_IMAGE_MODEL = "gpt-image-2"


@dataclass(frozen=True)
class PalmReading:
    report_markdown: str
    palm_photo_jpeg: bytes | None = None
    contour_png: bytes | None = None


@dataclass(frozen=True)
class PalmPhotoValidation:
    is_valid: bool
    score: int
    issues: list[str]
    tips: list[str]


VALIDATION_PROMPT = """
Check whether this image is good enough for a palm-reading app.

Return only JSON with this shape:
{
  "is_valid": true,
  "score": 0,
  "issues": [],
  "tips": []
}

Accept only if:
- a human open palm is clearly visible
- the palm faces the camera
- the fingers are mostly visible
- the palm lines are reasonably sharp
- lighting is good enough, without heavy shadows or blur
- the palm is the main subject, not tiny or cropped badly

Reject if the image is a fist, back of hand, face/body photo, object, very blurry image, dark image, or badly cropped palm.
Keep issues and tips short and user-friendly.
""".strip()


REPORT_PROMPT = """
Just attach a photo of your open palm.

Based on my hand I want you to make a complete palm reading guide. Analyze the palm. The style of the guide should be clean and minimal, thin lines, rounded cards, overall very expensive looking.

Focus on the palm.

Create a premium entertainment-only palm reading guide in markdown. Preserve uncertainty: do not claim this is real fortune telling, biometric identity analysis, medical analysis, or professional advice.

    Include:
- ## At a Glance
- ## Your Palm Lines
- ## Your Palm Print
- ## The Major Lines
- ## Palm Features
- ## What This Means For You
- ## Your Path

For palm lines, cover heart line, head line, life line, fate line, and sun line if visible. For meanings, include strengths, challenges, love & relationships, career & life path, and guidance.

Keep the writing specific to visible palm features, polished, minimal, and premium.
""".strip()


CONTOUR_PROMPT = """
Can you also get my palm print extracted from this photo and present the specular highlights of it in black and white.

Create a clean premium black-and-white palm print page from the uploaded palm photo:
- focus on the palm
- extract the handprint/palm-print feeling from the source image
- show ridges, palm lines, and specular highlights in black and white
- use a clean minimal layout with thin lines and rounded-card editorial style
- include subtle labels for Heart Line, Head Line, Life Line, Fate Line, and Sun Line where visible
- no color, no mystical symbols, no clutter
""".strip()


def encode_image_data_url(image_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def optimize_upload_image(image_bytes: bytes, max_side: int = 1600) -> tuple[bytes, str]:
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    image.thumbnail((max_side, max_side))

    output = io.BytesIO()
    image.save(output, format="JPEG", quality=88, optimize=True)
    return output.getvalue(), "image/jpeg"


def build_contour_prompt(report_markdown: str) -> str:
    return f"{CONTOUR_PROMPT.strip()}\n\nPalm-reading context:\n{report_markdown[:1600]}"


def validate_palm_photo(
    client: Any,
    image_bytes: bytes,
    mime_type: str,
    model: str = DEFAULT_VISION_MODEL,
) -> PalmPhotoValidation:
    data_url = encode_image_data_url(image_bytes, mime_type)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": VALIDATION_PROMPT},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
    )
    return parse_validation_response(_response_text(response))


def generate_report(client: Any, image_bytes: bytes, mime_type: str, model: str = DEFAULT_VISION_MODEL) -> str:
    data_url = encode_image_data_url(image_bytes, mime_type)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": REPORT_PROMPT.strip()},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
    )
    return _response_text(response).strip()


def generate_contour_image(
    client: Any,
    report_markdown: str,
    image_bytes: bytes | None = None,
    model: str = DEFAULT_IMAGE_MODEL,
) -> bytes:
    if image_bytes:
        image_file = io.BytesIO(image_bytes)
        image_file.name = "palm.jpg"
        response = client.images.edit(
            model=model,
            image=image_file,
            prompt=build_contour_prompt(report_markdown),
            size="1024x1536",
        )
        return base64.b64decode(response.data[0].b64_json)

    response = client.images.generate(
        model=model,
        prompt=build_contour_prompt(report_markdown),
        size="1024x1024",
    )
    b64_png = response.data[0].b64_json
    return base64.b64decode(b64_png)


def create_pdf_bytes(reading: PalmReading) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image as ReportImage
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    output = io.BytesIO()
    doc = SimpleDocTemplate(
        output,
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
    )
    styles = getSampleStyleSheet()
    story: list[Any] = [
        Paragraph("Palm Reading Guide", styles["Title"]),
        Spacer(1, 0.18 * inch),
        Paragraph("For entertainment only. We do not store your palm photo.", styles["Italic"]),
        Spacer(1, 0.18 * inch),
    ]

    if reading.contour_png:
        image_stream = io.BytesIO(reading.contour_png)
        story.extend([ReportImage(image_stream, width=3.8 * inch, height=3.8 * inch), Spacer(1, 0.18 * inch)])

    if reading.palm_photo_jpeg:
        image_stream = io.BytesIO(reading.palm_photo_jpeg)
        story.extend([ReportImage(image_stream, width=3.0 * inch, height=3.0 * inch), Spacer(1, 0.18 * inch)])

    for block in _markdown_blocks(reading.report_markdown):
        style = styles["Heading2"] if block.startswith("#") else styles["BodyText"]
        text = escape(block.lstrip("# ")).replace("\n", "<br/>")
        story.extend([Paragraph(text, style), Spacer(1, 0.09 * inch)])

    doc.build(story)
    return output.getvalue()


def _response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    if isinstance(response, dict):
        if response.get("output_text"):
            return str(response["output_text"])
        return json.dumps(response)

    return str(response)


def parse_validation_response(text: str) -> PalmPhotoValidation:
    payload = _extract_json_object(text)
    is_valid = bool(payload.get("is_valid", False))
    score = int(payload.get("score", 0) or 0)
    issues = _string_list(payload.get("issues"))
    tips = _string_list(payload.get("tips"))
    return PalmPhotoValidation(is_valid=is_valid, score=max(0, min(score, 100)), issues=issues, tips=tips)


def split_report_sections(markdown: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current_title = "Overview"
    current_lines: list[str] = []

    for line in markdown.splitlines():
        match = re.match(r"^\s*#{1,3}\s+(.+?)\s*$", line)
        if match:
            if current_lines:
                sections[current_title] = "\n".join(current_lines).strip()
            current_title = match.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections[current_title] = "\n".join(current_lines).strip()

    return {title: body for title, body in sections.items() if body}


def _extract_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            value = json.loads(text[start : end + 1])
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _markdown_blocks(markdown: str) -> list[str]:
    blocks = [block.strip() for block in markdown.split("\n\n")]
    return [block for block in blocks if block]
