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
Analyze the palm in this image and create a comprehensive palm reading report.

Create a professional palm reading guide using clean HTML formatting. Use proper section symbols and structure. 

**IMPORTANT FORMATTING RULES:**
- Use HTML tags, NOT markdown
- Use section symbols: ⭐ 📏 ✋ 💫 🎯 ✨ 🛤️
- No ** bold markdown - use <strong> tags
- Use <div class="section"> for each section
- Use <h2>, <h3> for headings
- Use <ul>, <li> for lists
- Use <p> for paragraphs

Generate sections with these exact symbols and structure:

<div class="section">
<h2>⭐ At a Glance</h2>
<p>[Brief overview of the palm characteristics]</p>
</div>

<div class="section">
<h2>📏 Your Palm Lines</h2>
<h3>♥️ Heart Line</h3>
<p>[Analysis of heart line]</p>
<h3>🧠 Head Line</h3>
<p>[Analysis of head line]</p>
<h3>🌱 Life Line</h3>
<p>[Analysis of life line]</p>
<h3>🎭 Fate Line</h3>
<p>[Analysis of fate line if visible]</p>
</div>

<div class="section">
<h2>✋ Palm Features</h2>
<p>[Analysis of mounts, palm shape, texture]</p>
</div>

<div class="section">
<h2>💫 What This Means For You</h2>
<h3>💪 Strengths</h3>
<ul><li>[List strengths]</li></ul>
<h3>⚖️ Challenges</h3>
<ul><li>[List challenges]</li></ul>
<h3>💝 Love & Relationships</h3>
<p>[Relationship insights]</p>
<h3>🎯 Career & Life Path</h3>
<p>[Career guidance]</p>
</div>

<div class="section">
<h2>✨ Guidance</h2>
<p>[Personal guidance and recommendations]</p>
</div>

<div class="section">
<h2>🛤️ Your Path Forward</h2>
<p>[Future focus and direction]</p>
</div>

**Requirements:**
- Use only HTML formatting with proper tags
- Include section symbols as shown
- Be specific to visible palm features
- Keep content premium and polished
- Include entertainment disclaimer
- Focus on visible palm characteristics only
""".strip()


CONTOUR_PROMPT = """
Transform this palm photo into a professional black and white palm analysis diagram.

REQUIREMENTS FOR PALM IMAGE TRANSFORMATION:

1. **Complete Background Removal**: 
   - Remove ALL background elements
   - Make background pure white (#FFFFFF)
   - Keep ONLY the palm and fingers

2. **Professional Palm Line Enhancement**:
   - Draw clear, bold black lines for major palm lines
   - Heart Line: Upper curved line across palm
   - Head Line: Middle horizontal line
   - Life Line: Curved line around thumb base
   - Fate Line: Vertical line through palm center (if visible)
   - Minor lines: Other visible creases and lines

3. **Clean Medical Diagram Style**:
   - Pure black lines (#000000) on pure white background
   - Remove all skin texture, color, and shading
   - High contrast, sharp line definition
   - Professional palmistry chart appearance
   - Vector-like clean drawing style

4. **Image Processing Standards**:
   - Remove fingerprints, skin pores, and texture
   - Focus only on major palm creases and lines
   - Make lines bold and clearly visible
   - Ensure high contrast for easy analysis
   - Create a clean, medical-grade hand diagram

5. **Final Output**:
   - Should look like a professional hand analysis chart
   - Suitable for medical or scientific documentation
   - Clean enough for report inclusion
   - Black lines only, no gray or colored elements

Transform the original palm photo into a clinical, professional palm line diagram that clearly shows all major lines for analysis.
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

    # Parse HTML/Markdown sections for PDF
    sections = split_report_sections(reading.report_markdown)
    
    for title, content in sections.items():
        # Add section title
        clean_title = re.sub(r'<[^>]+>', '', title)  # Remove HTML tags
        clean_title = re.sub(r'[^\w\s]', '', clean_title).strip()  # Remove emojis
        
        if clean_title:
            story.extend([
                Paragraph(clean_title, styles["Heading2"]), 
                Spacer(1, 0.12 * inch)
            ])
        
        # Process content
        if content.strip():
            # Convert HTML to plain text for PDF
            clean_content = re.sub(r'<[^>]+>', '', content)  # Remove HTML tags
            clean_content = clean_content.replace('&nbsp;', ' ').replace('&amp;', '&')
            clean_content = clean_content.replace('&lt;', '<').replace('&gt;', '>')
            
            # Split into paragraphs
            paragraphs = [p.strip() for p in clean_content.split('\n') if p.strip()]
            
            for paragraph in paragraphs:
                if paragraph:
                    story.extend([
                        Paragraph(escape(paragraph), styles["BodyText"]), 
                        Spacer(1, 0.06 * inch)
                    ])
            
            story.append(Spacer(1, 0.12 * inch))

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


def split_report_sections(report: str) -> dict[str, str]:
    """Split HTML/markdown report into sections based on headings."""
    sections: dict[str, str] = {}
    
    # First try to split by HTML sections with div class="section"
    html_sections = re.findall(r'<div class="section">(.*?)</div>', report, re.DOTALL)
    if html_sections:
        for section_html in html_sections:
            # Extract title from h2 tag and remove HTML
            title_match = re.search(r'<h2>(.*?)</h2>', section_html)
            if title_match:
                title = re.sub(r'<[^>]+>', '', title_match.group(1)).strip()  # Remove HTML tags and emojis
                title = re.sub(r'[^\w\s]', '', title).strip()  # Remove emojis and special chars
                content = section_html.replace(title_match.group(0), '').strip()
                sections[title] = content
    
    # Fallback: Split by markdown/HTML headers
    if not sections:
        current_title = "Overview"
        current_lines: list[str] = []

        for line in report.splitlines():
            # Match both markdown and HTML headers
            md_match = re.match(r"^\s*#{1,3}\s+(.+?)\s*$", line)
            html_match = re.match(r"^\s*<h[1-3]>(.*?)</h[1-3]>\s*$", line)
            
            if md_match or html_match:
                if current_lines:
                    sections[current_title] = "\n".join(current_lines).strip()
                
                if md_match:
                    current_title = md_match.group(1).strip()
                else:
                    current_title = re.sub(r'<[^>]+>', '', html_match.group(1)).strip()
                
                # Clean title of emojis and special characters for key
                clean_title = re.sub(r'[^\w\s]', '', current_title).strip()
                if clean_title:
                    current_title = clean_title
                    
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections[current_title] = "\n".join(current_lines).strip()
    
    return {title: body for title, body in sections.items() if body}


def format_report_content(content: str) -> str:
    """Clean and format report content for display."""
    # If it's already HTML, return as is
    if '<p>' in content or '<div>' in content or '<ul>' in content:
        return content
    
    # Remove excessive markdown formatting
    content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
    content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', content)
    
    # Convert markdown lists to HTML
    lines = content.split('\n')
    formatted_lines = []
    in_list = False
    
    for line in lines:
        line = line.strip()
        if line.startswith('- '):
            if not in_list:
                formatted_lines.append('<ul>')
                in_list = True
            formatted_lines.append(f'<li>{line[2:]}</li>')
        else:
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            if line:
                formatted_lines.append(f'<p>{line}</p>')
    
    if in_list:
        formatted_lines.append('</ul>')
    
    return '\n'.join(formatted_lines)


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
