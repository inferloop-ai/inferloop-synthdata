#!/usr/bin/env python3
"""
Image renderer for generating document images with realistic appearance.

Provides capabilities for converting documents to images, adding visual
effects, and creating training data for OCR and document analysis systems.
"""

import io
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

from ...core import get_logger

logger = get_logger(__name__)


class ImageRenderer:
    """High-quality document image renderer with realistic effects"""
    
    def __init__(self, dpi: int = 300, background_color: str = 'white'):
        self.dpi = dpi
        self.background_color = background_color
        self.default_fonts = self._get_default_fonts()
        
    def _get_default_fonts(self) -> Dict[str, str]:
        """Get available system fonts"""
        import platform
        
        fonts = {
            'regular': None,
            'bold': None,
            'italic': None
        }
        
        # Try to find system fonts
        system = platform.system()
        
        if system == "Darwin":  # macOS
            font_paths = {
                'regular': '/System/Library/Fonts/Arial.ttf',
                'bold': '/System/Library/Fonts/Arial Bold.ttf',
                'italic': '/System/Library/Fonts/Arial Italic.ttf'
            }
        elif system == "Linux":
            font_paths = {
                'regular': '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
                'bold': '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
                'italic': '/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf'
            }
        elif system == "Windows":
            font_paths = {
                'regular': 'C:/Windows/Fonts/arial.ttf',
                'bold': 'C:/Windows/Fonts/arialbd.ttf',
                'italic': 'C:/Windows/Fonts/ariali.ttf'
            }
        else:
            font_paths = {}
        
        # Check which fonts exist
        for font_type, path in font_paths.items():
            if path and Path(path).exists():
                fonts[font_type] = path
        
        return fonts
    
    async def render_document_page(self, content: Dict[str, Any], 
                                 page_size: Tuple[int, int] = (2480, 3508),  # A4 at 300 DPI
                                 output_path: Optional[Union[str, Path]] = None) -> Union[Image.Image, bytes]:
        """
        Render document content to image.
        
        Args:
            content: Document content dictionary
            page_size: Image size in pixels (width, height)
            output_path: Output image file path (if None, returns PIL Image)
        
        Returns:
            PIL Image object or bytes if output_path specified
        """
        try:
            # Create blank page
            image = Image.new('RGB', page_size, self.background_color)
            draw = ImageDraw.Draw(image)
            
            # Set up fonts
            fonts = self._setup_fonts()
            
            # Current Y position for content placement
            y_pos = 100
            margin_left = 150
            margin_right = page_size[0] - 150
            line_height = 40
            
            # Render title
            if 'title' in content:
                y_pos = self._draw_title(draw, content['title'], fonts['title'], 
                                       margin_left, margin_right, y_pos)
                y_pos += 60
            
            # Render metadata
            if 'metadata' in content:
                y_pos = self._draw_metadata(draw, content['metadata'], fonts['small'], 
                                          margin_left, margin_right, y_pos)
                y_pos += 40
            
            # Render main content
            if 'content' in content:
                y_pos = self._draw_content(draw, content['content'], fonts, 
                                         margin_left, margin_right, y_pos, line_height)
            
            # Render tables
            if 'tables' in content:
                for table in content['tables']:
                    y_pos = self._draw_table(draw, table, fonts, 
                                           margin_left, margin_right, y_pos)
                    y_pos += 40
            
            # Save or return image
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(output_path, 'PNG', dpi=(self.dpi, self.dpi))
                logger.info(f"Rendered document image: {output_path}")
                return output_path
            else:
                return image
                
        except Exception as e:
            logger.error(f"Failed to render document image: {e}")
            raise
    
    def _setup_fonts(self) -> Dict[str, ImageFont.FreeTypeFont]:
        """Setup fonts for different text elements"""
        fonts = {}
        
        try:
            # Try to load system fonts
            if self.default_fonts['regular']:
                fonts['body'] = ImageFont.truetype(self.default_fonts['regular'], 32)
                fonts['title'] = ImageFont.truetype(self.default_fonts['bold'] or self.default_fonts['regular'], 48)
                fonts['heading'] = ImageFont.truetype(self.default_fonts['bold'] or self.default_fonts['regular'], 36)
                fonts['small'] = ImageFont.truetype(self.default_fonts['regular'], 24)
            else:
                # Fallback to default font
                fonts['body'] = ImageFont.load_default()
                fonts['title'] = ImageFont.load_default()
                fonts['heading'] = ImageFont.load_default()
                fonts['small'] = ImageFont.load_default()
                
        except Exception as e:
            logger.warning(f"Could not load custom fonts: {e}")
            # Use default font for all
            default_font = ImageFont.load_default()
            fonts = {
                'body': default_font,
                'title': default_font,
                'heading': default_font,
                'small': default_font
            }
        
        return fonts
    
    def _draw_title(self, draw: ImageDraw.Draw, title: str, font: ImageFont.FreeTypeFont,
                   left: int, right: int, y_pos: int) -> int:
        """Draw document title"""
        # Center the title
        bbox = draw.textbbox((0, 0), title, font=font)
        text_width = bbox[2] - bbox[0]
        x_pos = (right - left - text_width) // 2 + left
        
        draw.text((x_pos, y_pos), title, fill='black', font=font)
        
        # Draw underline
        line_y = y_pos + bbox[3] + 5
        draw.line([(left, line_y), (right, line_y)], fill='black', width=2)
        
        return line_y + 10
    
    def _draw_metadata(self, draw: ImageDraw.Draw, metadata: Dict[str, Any], 
                      font: ImageFont.FreeTypeFont, left: int, right: int, y_pos: int) -> int:
        """Draw document metadata"""
        for key, value in metadata.items():
            if key not in ['tags', 'annotations']:
                text = f"{key.replace('_', ' ').title()}: {value}"
                draw.text((left, y_pos), text, fill='gray', font=font)
                y_pos += 30
        
        return y_pos
    
    def _draw_content(self, draw: ImageDraw.Draw, content: Union[str, List, Dict], 
                     fonts: Dict[str, ImageFont.FreeTypeFont], left: int, right: int, 
                     y_pos: int, line_height: int) -> int:
        """Draw main document content"""
        if isinstance(content, str):
            # Simple text content
            paragraphs = content.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    y_pos = self._draw_paragraph(draw, para.strip(), fonts['body'], 
                                                left, right, y_pos, line_height)
                    y_pos += 20
        
        elif isinstance(content, list):
            # List of content items
            for item in content:
                y_pos = self._draw_content(draw, item, fonts, left, right, y_pos, line_height)
        
        elif isinstance(content, dict):
            # Structured content
            if content.get('type') == 'paragraph':
                y_pos = self._draw_paragraph(draw, content['text'], fonts['body'], 
                                           left, right, y_pos, line_height)
            elif content.get('type') == 'heading':
                level = content.get('level', 1)
                font = fonts['heading'] if level <= 2 else fonts['body']
                y_pos = self._draw_paragraph(draw, content['text'], font, 
                                           left, right, y_pos, line_height)
                y_pos += 20
        
        return y_pos
    
    def _draw_paragraph(self, draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont,
                       left: int, right: int, y_pos: int, line_height: int) -> int:
        """Draw a paragraph with word wrapping"""
        words = text.split()
        lines = []
        current_line = []
        
        # Word wrapping
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            text_width = bbox[2] - bbox[0]
            
            if text_width <= (right - left - 20):  # 20px margin
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Draw lines
        for line in lines:
            draw.text((left, y_pos), line, fill='black', font=font)
            y_pos += line_height
        
        return y_pos
    
    def _draw_table(self, draw: ImageDraw.Draw, table_data: Dict[str, Any], 
                   fonts: Dict[str, ImageFont.FreeTypeFont], left: int, right: int, 
                   y_pos: int) -> int:
        """Draw a table"""
        if 'title' in table_data:
            draw.text((left, y_pos), table_data['title'], fill='black', font=fonts['heading'])
            y_pos += 50
        
        if 'data' not in table_data:
            return y_pos
        
        data = table_data['data']
        headers = table_data.get('headers', [])
        
        # Calculate column widths
        num_cols = len(headers) if headers else len(data[0]) if data else 0
        if num_cols == 0:
            return y_pos
        
        col_width = (right - left) // num_cols
        row_height = 35
        
        # Draw headers
        if headers:
            for i, header in enumerate(headers):
                x = left + i * col_width
                # Header background
                draw.rectangle([(x, y_pos), (x + col_width, y_pos + row_height)], 
                             fill='lightgray', outline='black')
                # Header text
                draw.text((x + 5, y_pos + 5), str(header), fill='black', font=fonts['small'])
            y_pos += row_height
        
        # Draw data rows
        for row in data:
            for i, cell in enumerate(row):
                x = left + i * col_width
                # Cell background
                draw.rectangle([(x, y_pos), (x + col_width, y_pos + row_height)], 
                             fill='white', outline='black')
                # Cell text
                draw.text((x + 5, y_pos + 5), str(cell), fill='black', font=fonts['small'])
            y_pos += row_height
        
        return y_pos + 20
    
    async def add_realistic_effects(self, image: Image.Image, 
                                  effects: Optional[List[str]] = None) -> Image.Image:
        """
        Add realistic effects to make document look scanned/photographed.
        
        Args:
            image: Source PIL Image
            effects: List of effects to apply
        
        Returns:
            Modified PIL Image
        """
        if effects is None:
            effects = ['noise', 'blur', 'rotation', 'brightness']
        
        result_image = image.copy()
        
        try:
            # Apply rotation (slight angle)
            if 'rotation' in effects:
                angle = random.uniform(-2, 2)  # Small rotation
                result_image = result_image.rotate(angle, expand=True, fillcolor=self.background_color)
            
            # Add noise
            if 'noise' in effects:
                result_image = self._add_noise(result_image)
            
            # Add blur
            if 'blur' in effects:
                blur_radius = random.uniform(0.1, 0.5)
                result_image = result_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
            # Adjust brightness/contrast
            if 'brightness' in effects:
                enhancer = ImageEnhance.Brightness(result_image)
                brightness_factor = random.uniform(0.95, 1.05)
                result_image = enhancer.enhance(brightness_factor)
                
                enhancer = ImageEnhance.Contrast(result_image)
                contrast_factor = random.uniform(0.98, 1.02)
                result_image = enhancer.enhance(contrast_factor)
            
            # Add shadows or creases
            if 'shadows' in effects:
                result_image = self._add_shadows(result_image)
            
            # Simulate paper texture
            if 'texture' in effects:
                result_image = self._add_paper_texture(result_image)
            
            return result_image
            
        except Exception as e:
            logger.error(f"Failed to add realistic effects: {e}")
            return image
    
    def _add_noise(self, image: Image.Image) -> Image.Image:
        """Add random noise to image"""
        img_array = np.array(image)
        
        # Add Gaussian noise
        noise = np.random.normal(0, 5, img_array.shape).astype(np.uint8)
        noisy_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy_array)
    
    def _add_shadows(self, image: Image.Image) -> Image.Image:
        """Add subtle shadow effects"""
        # Create shadow layer
        shadow = Image.new('RGBA', image.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        
        # Add random shadow patches
        for _ in range(random.randint(1, 3)):
            x = random.randint(0, image.size[0] // 2)
            y = random.randint(0, image.size[1] // 2)
            width = random.randint(100, 300)
            height = random.randint(100, 300)
            
            shadow_draw.ellipse(
                [(x, y), (x + width, y + height)],
                fill=(0, 0, 0, 10)  # Very light shadow
            )
        
        # Blend shadow with original image
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        result = Image.alpha_composite(image, shadow)
        return result.convert('RGB')
    
    def _add_paper_texture(self, image: Image.Image) -> Image.Image:
        """Add paper-like texture"""
        # Create texture pattern
        texture = Image.new('L', image.size)
        texture_array = np.random.normal(128, 10, image.size[::-1]).astype(np.uint8)
        texture = Image.fromarray(texture_array, 'L')
        
        # Convert image to grayscale temporarily for blending
        if image.mode == 'RGB':
            # Blend texture with image
            img_array = np.array(image)
            texture_array = np.array(texture)
            
            # Apply texture to each channel
            for c in range(3):
                img_array[:, :, c] = np.clip(
                    img_array[:, :, c].astype(np.float32) * (texture_array / 255.0),
                    0, 255
                ).astype(np.uint8)
            
            return Image.fromarray(img_array)
        
        return image
    
    async def create_form_image(self, form_data: Dict[str, Any], 
                              output_path: Optional[Union[str, Path]] = None) -> Union[Image.Image, Path]:
        """
        Create form-like document image.
        
        Args:
            form_data: Form structure and data
            output_path: Output path (if None, returns PIL Image)
        
        Returns:
            PIL Image or output path
        """
        try:
            page_size = (2480, 3508)  # A4 at 300 DPI
            image = Image.new('RGB', page_size, 'white')
            draw = ImageDraw.Draw(image)
            fonts = self._setup_fonts()
            
            y_pos = 100
            margin_left = 150
            margin_right = page_size[0] - 150
            
            # Form title
            title = form_data.get('title', 'Form Document')
            y_pos = self._draw_title(draw, title, fonts['title'], 
                                   margin_left, margin_right, y_pos)
            y_pos += 60
            
            # Form fields
            fields = form_data.get('fields', [])
            for field in fields:
                y_pos = self._draw_form_field(draw, field, fonts, 
                                            margin_left, margin_right, y_pos)
                y_pos += 60
            
            # Add realistic effects
            image = await self.add_realistic_effects(image, ['noise', 'blur'])
            
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(output_path, 'PNG', dpi=(self.dpi, self.dpi))
                return output_path
            else:
                return image
                
        except Exception as e:
            logger.error(f"Failed to create form image: {e}")
            raise
    
    def _draw_form_field(self, draw: ImageDraw.Draw, field: Dict[str, Any], 
                        fonts: Dict[str, ImageFont.FreeTypeFont], 
                        left: int, right: int, y_pos: int) -> int:
        """Draw a form field"""
        field_type = field.get('type', 'text')
        label = field.get('label', 'Field')
        value = field.get('value', '')
        
        # Draw label
        draw.text((left, y_pos), f"{label}:", fill='black', font=fonts['body'])
        y_pos += 35
        
        if field_type == 'text':
            # Draw text input box
            box_width = min(400, right - left - 20)
            draw.rectangle([(left, y_pos), (left + box_width, y_pos + 40)], 
                         outline='black', width=2)
            
            # Draw value if present
            if value:
                draw.text((left + 5, y_pos + 8), str(value), fill='black', font=fonts['small'])
        
        elif field_type == 'checkbox':
            # Draw checkbox
            box_size = 25
            draw.rectangle([(left, y_pos), (left + box_size, y_pos + box_size)], 
                         outline='black', width=2)
            
            # Draw checkmark if checked
            if value:
                draw.line([(left + 5, y_pos + 12), (left + 10, y_pos + 18)], 
                         fill='black', width=3)
                draw.line([(left + 10, y_pos + 18), (left + 20, y_pos + 8)], 
                         fill='black', width=3)
        
        elif field_type == 'signature':
            # Draw signature line
            line_width = min(300, right - left - 20)
            draw.line([(left, y_pos + 20), (left + line_width, y_pos + 20)], 
                     fill='black', width=2)
            
            # Draw signature if present
            if value:
                # Simulate handwritten signature
                self._draw_signature(draw, str(value), left + 10, y_pos + 5, fonts['body'])
        
        return y_pos + 20
    
    def _draw_signature(self, draw: ImageDraw.Draw, text: str, x: int, y: int, 
                       font: ImageFont.FreeTypeFont):
        """Draw signature-like text"""
        # Make text look more handwritten by adding slight variations
        for i, char in enumerate(text):
            char_x = x + i * 15 + random.randint(-2, 2)
            char_y = y + random.randint(-3, 3)
            draw.text((char_x, char_y), char, fill='blue', font=font)
    
    async def render_to_bytes(self, content: Dict[str, Any], 
                            format: str = 'PNG') -> bytes:
        """
        Render document to bytes.
        
        Args:
            content: Document content
            format: Image format ('PNG', 'JPEG')
        
        Returns:
            Image bytes
        """
        try:
            image = await self.render_document_page(content)
            
            buffer = io.BytesIO()
            image.save(buffer, format=format, dpi=(self.dpi, self.dpi))
            buffer.seek(0)
            
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Failed to render image to bytes: {e}")
            raise


def create_image_renderer(dpi: int = 300, background_color: str = 'white') -> ImageRenderer:
    """Factory function to create image renderer"""
    return ImageRenderer(dpi=dpi, background_color=background_color)


__all__ = ['ImageRenderer', 'create_image_renderer']