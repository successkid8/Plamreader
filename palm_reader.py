from __future__ import annotations

import base64
import io
import json
import re
import numpy as np
from dataclasses import dataclass
from typing import Any
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps


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
Analyze this palm image and create a comprehensive professional palm reading report in the style of a premium paid consultation.

**PROFESSIONAL REPORT REQUIREMENTS:**
- NO emojis or casual symbols - professional business format only
- Use precise palmistry terminology and technical analysis
- Provide specific, actionable insights based on visible features
- Structure like a professional consultation document
- Use proper HTML formatting for clean presentation

**REQUIRED PROFESSIONAL STRUCTURE:**

<div class="section">
<h2>EXECUTIVE SUMMARY</h2>
<ul>
<li><strong>Palm Classification:</strong> [Square/Spatulate/Conic/Pointed palm type with technical details]</li>
<li><strong>Dominant Characteristics:</strong> [List 3 most significant palmistry features observed]</li>
<li><strong>Primary Insights:</strong> [Key personality and life path indicators identified]</li>
<li><strong>Consultation Date:</strong> [Current analysis timestamp]</li>
</ul>
</div>

<div class="section">
<h2>MAJOR LINE ANALYSIS</h2>
<h3>Heart Line Assessment</h3>
<ul>
<li><strong>Technical Position:</strong> [Precise location - high/medium/low placement]</li>
<li><strong>Line Quality:</strong> [Deep/moderate/light, continuous/broken, length measurement]</li>
<li><strong>Emotional Profile:</strong> [Specific relationship and emotional patterns indicated]</li>
<li><strong>Professional Interpretation:</strong> [What this reveals about emotional intelligence and relationships]</li>
</ul>

<h3>Head Line Evaluation</h3>
<ul>
<li><strong>Direction & Slope:</strong> [Straight/curved angle and significance]</li>
<li><strong>Line Characteristics:</strong> [Depth, clarity, any branches or markings]</li>
<li><strong>Cognitive Style:</strong> [Analytical vs intuitive thinking patterns]</li>
<li><strong>Decision-Making Profile:</strong> [How you process information and make choices]</li>
</ul>

<h3>Life Line Analysis</h3>
<ul>
<li><strong>Arc Pattern:</strong> [Wide/narrow curve around Venus mount]</li>
<li><strong>Line Strength:</strong> [Vitality and energy levels indicated]</li>
<li><strong>Energy Distribution:</strong> [Physical and mental stamina patterns]</li>
<li><strong>Health Indicators:</strong> [General vitality and resilience factors]</li>
</ul>

<h3>Fate Line Investigation</h3>
<ul>
<li><strong>Presence & Clarity:</strong> [Strong/weak/absent with implications]</li>
<li><strong>Origin Point:</strong> [Starting location and meaning]</li>
<li><strong>Career Trajectory:</strong> [Professional path and goal orientation]</li>
<li><strong>Life Purpose Alignment:</strong> [Direction and focus in life goals]</li>
</ul>
</div>

<div class="section">
<h2>MOUNT DEVELOPMENT ANALYSIS</h2>
<ul>
<li><strong>Venus Mount (Passion & Relationships):</strong> [Development level and personality implications]</li>
<li><strong>Jupiter Mount (Leadership & Ambition):</strong> [Prominence and leadership capabilities]</li>
<li><strong>Saturn Mount (Discipline & Structure):</strong> [Organization and responsibility traits]</li>
<li><strong>Apollo Mount (Creativity & Recognition):</strong> [Artistic abilities and public recognition potential]</li>
<li><strong>Mercury Mount (Communication & Business):</strong> [Communication skills and commercial aptitude]</li>
<li><strong>Mars Mounts (Action & Courage):</strong> [Passive and active Mars development]</li>
</ul>
</div>

<div class="section">
<h2>PERSONALITY PROFILE & STRENGTHS</h2>
<h3>Core Personality Traits</h3>
<ul>
<li><strong>Primary Temperament:</strong> [Based on palm shape and mount development]</li>
<li><strong>Communication Style:</strong> [How you interact with others professionally and personally]</li>
<li><strong>Leadership Qualities:</strong> [Natural authority and influence patterns]</li>
<li><strong>Creative Expression:</strong> [Artistic and innovative capabilities]</li>
</ul>

<h3>Professional Strengths</h3>
<ul>
<li><strong>Natural Talents:</strong> [Specific abilities indicated by palm features]</li>
<li><strong>Work Style Preferences:</strong> [Independent vs collaborative tendencies]</li>
<li><strong>Problem-Solving Approach:</strong> [Analytical vs intuitive methods]</li>
<li><strong>Success Indicators:</strong> [Factors that drive achievement and recognition]</li>
</ul>
</div>

<div class="section">
<h2>RELATIONSHIP & COMPATIBILITY ANALYSIS</h2>
<ul>
<li><strong>Emotional Expression Style:</strong> [How you show and receive affection]</li>
<li><strong>Relationship Priorities:</strong> [What you value most in partnerships]</li>
<li><strong>Compatibility Factors:</strong> [Personality types that complement your nature]</li>
<li><strong>Communication Patterns:</strong> [How you handle conflict and express needs]</li>
<li><strong>Long-term Relationship Potential:</strong> [Commitment style and partnership approach]</li>
</ul>
</div>

<div class="section">
<h2>CAREER & FINANCIAL PROSPECTS</h2>
<ul>
<li><strong>Ideal Career Fields:</strong> [Specific industries and roles aligned with your palmistry profile]</li>
<li><strong>Entrepreneurial Potential:</strong> [Business ownership and leadership capabilities]</li>
<li><strong>Financial Management Style:</strong> [Approach to money and resource management]</li>
<li><strong>Professional Growth Areas:</strong> [Skills to develop for career advancement]</li>
<li><strong>Success Timeline:</strong> [Periods of opportunity and achievement potential]</li>
</ul>
</div>

<div class="section">
<h2>STRATEGIC RECOMMENDATIONS</h2>
<h3>Immediate Focus Areas (Next 90 Days)</h3>
<ul>
<li><strong>Priority Development:</strong> [Specific skill or trait to enhance immediately]</li>
<li><strong>Relationship Focus:</strong> [How to improve personal connections]</li>
<li><strong>Professional Action:</strong> [Career move or skill development to pursue]</li>
</ul>

<h3>Medium-Term Strategy (6-18 Months)</h3>
<ul>
<li><strong>Personal Growth:</strong> [Character development and self-improvement focus]</li>
<li><strong>Career Advancement:</strong> [Professional goals and positioning strategy]</li>
<li><strong>Life Balance:</strong> [Areas requiring attention for optimal well-being]</li>
</ul>

<h3>Long-Term Vision (2-5 Years)</h3>
<ul>
<li><strong>Life Direction:</strong> [Major life path and purpose alignment]</li>
<li><strong>Legacy Building:</strong> [How to create lasting impact and meaning]</li>
<li><strong>Personal Fulfillment:</strong> [Keys to long-term satisfaction and happiness]</li>
</ul>
</div>

**ANALYSIS STANDARDS:**
- Base all interpretations on classical palmistry principles
- Provide specific, actionable guidance for real-world application  
- Use professional consultation language and terminology
- Include technical palmistry observations with practical meanings
- Maintain professional disclaimer about entertainment purpose
- Focus on empowerment and personal development
- Avoid generic statements - personalize to visible palm characteristics
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


def process_palm_to_blackwhite(image_bytes: bytes) -> bytes:
    """Process palm image to create actual black and white line drawing"""
    try:
        # Load the image
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert('RGB')
        
        # Convert to numpy array for processing
        img_array = np.array(image)
        
        # Convert to grayscale
        gray = Image.fromarray(img_array).convert('L')
        
        # Enhance contrast to make lines more visible
        enhancer = ImageEnhance.Contrast(gray)
        contrast_img = enhancer.enhance(2.0)
        
        # Apply edge detection using PIL filters
        # First apply a slight blur to reduce noise
        blurred = contrast_img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Apply edge enhancement
        edges = blurred.filter(ImageFilter.FIND_EDGES)
        
        # Invert the image (make lines dark, background light)
        inverted = ImageOps.invert(edges)
        
        # Apply threshold to create pure black and white
        threshold = 128
        bw_image = inverted.point(lambda x: 255 if x > threshold else 0, mode='1')
        
        # Convert back to RGB for further processing
        bw_rgb = bw_image.convert('RGB')
        
        # Create a new image with white background
        processed = Image.new('RGB', bw_rgb.size, 'white')
        
        # Convert black pixels to lines, ignore gray areas
        width, height = bw_rgb.size
        pixels = bw_rgb.load()
        new_pixels = processed.load()
        
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                # If pixel is dark enough, make it black (palm line)
                if r < 100 and g < 100 and b < 100:
                    new_pixels[x, y] = (0, 0, 0)  # Black line
                else:
                    new_pixels[x, y] = (255, 255, 255)  # White background
        
        # Apply morphological operations to clean up the lines
        # Convert to grayscale for morphological operations
        final_gray = processed.convert('L')
        
        # Apply closing to connect broken lines
        kernel_size = 2
        final_processed = final_gray.filter(ImageFilter.MinFilter(size=kernel_size))
        final_processed = final_processed.filter(ImageFilter.MaxFilter(size=kernel_size))
        
        # Convert back to RGB and ensure pure black/white
        final_rgb = final_processed.convert('RGB')
        final_pixels = final_rgb.load()
        width, height = final_rgb.size
        
        for y in range(height):
            for x in range(width):
                r, g, b = final_pixels[x, y]
                if r < 128:
                    final_pixels[x, y] = (0, 0, 0)  # Pure black
                else:
                    final_pixels[x, y] = (255, 255, 255)  # Pure white
        
        # Save to bytes
        output = io.BytesIO()
        final_rgb.save(output, format='PNG', optimize=True)
        return output.getvalue()
        
    except Exception as e:
        # Fallback: create a simple processed version
        try:
            image = Image.open(io.BytesIO(image_bytes))
            # Simple grayscale with high contrast
            gray = image.convert('L')
            enhancer = ImageEnhance.Contrast(gray)
            high_contrast = enhancer.enhance(3.0)
            
            # Convert to black and white
            bw = high_contrast.point(lambda x: 0 if x < 128 else 255, '1')
            bw_rgb = bw.convert('RGB')
            
            output = io.BytesIO()
            bw_rgb.save(output, format='PNG')
            return output.getvalue()
        except:
            # Ultimate fallback: return original
            return image_bytes


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
    """Generate black and white palm line image using actual image processing"""
    if image_bytes:
        # Use actual image processing instead of AI generation
        try:
            # First process the image to black and white lines
            processed_bytes = process_palm_to_blackwhite(image_bytes)
            return processed_bytes
        except Exception as e:
            # Fallback to AI generation if processing fails
            try:
                image_file = io.BytesIO(image_bytes)
                image_file.name = "palm.jpg"
                response = client.images.edit(
                    model=model,
                    image=image_file,
                    prompt=build_contour_prompt(report_markdown),
                    size="1024x1024",
                )
                return base64.b64decode(response.data[0].b64_json)
            except Exception as ai_error:
                # Ultimate fallback: process with simple method
                return process_palm_to_blackwhite(image_bytes)
    else:
        # Generate from scratch using AI
        try:
            response = client.images.generate(
                model=model,
                prompt=build_contour_prompt(report_markdown),
                size="1024x1024",
            )
            return base64.b64decode(response.data[0].b64_json)
        except:
            # Return a placeholder black and white image
            placeholder = Image.new('RGB', (512, 512), 'white')
            draw = ImageDraw.Draw(placeholder)
            draw.text((50, 50), "Palm processing unavailable", fill='black')
            output = io.BytesIO()
            placeholder.save(output, format='PNG')
            return output.getvalue()


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
    """Split professional HTML report into sections based on headings."""
    sections: dict[str, str] = {}
    
    # First try to split by HTML sections with div class="section"
    html_sections = re.findall(r'<div class="section">(.*?)</div>', report, re.DOTALL)
    if html_sections:
        for section_html in html_sections:
            # Extract title from h2 tag and remove HTML
            title_match = re.search(r'<h2>(.*?)</h2>', section_html)
            if title_match:
                title = re.sub(r'<[^>]+>', '', title_match.group(1)).strip()  # Remove HTML tags
                # Keep professional titles as-is, just clean HTML
                content = section_html.replace(title_match.group(0), '').strip()
                sections[title] = content
    
    # Fallback: Split by markdown/HTML headers
    if not sections:
        current_title = "Executive Summary"
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
