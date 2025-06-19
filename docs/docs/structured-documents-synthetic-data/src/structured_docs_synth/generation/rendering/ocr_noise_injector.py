#!/usr/bin/env python3
"""
OCR noise injector for creating realistic document scanning artifacts.

Provides capabilities for adding various types of noise and distortions
to document images to simulate real-world scanning conditions for
OCR training and testing.
"""

import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
from scipy import ndimage

from ...core import get_logger

logger = get_logger(__name__)


class OCRNoiseInjector:
    """Injects realistic OCR noise and artifacts into document images"""
    
    def __init__(self, noise_level: str = 'medium'):
        self.noise_level = noise_level
        self.noise_params = self._get_noise_parameters(noise_level)
        
    def _get_noise_parameters(self, level: str) -> Dict[str, any]:
        """Get noise parameters based on level"""
        params = {
            'low': {
                'gaussian_noise_std': 2,
                'blur_radius_range': (0.1, 0.3),
                'rotation_range': (-0.5, 0.5),
                'brightness_range': (0.98, 1.02),
                'contrast_range': (0.99, 1.01),
                'jpeg_quality_range': (90, 95)
            },
            'medium': {
                'gaussian_noise_std': 5,
                'blur_radius_range': (0.2, 0.8),
                'rotation_range': (-1.0, 1.0),
                'brightness_range': (0.95, 1.05),
                'contrast_range': (0.95, 1.05),
                'jpeg_quality_range': (80, 90)
            },
            'high': {
                'gaussian_noise_std': 10,
                'blur_radius_range': (0.5, 1.5),
                'rotation_range': (-2.0, 2.0),
                'brightness_range': (0.9, 1.1),
                'contrast_range': (0.9, 1.1),
                'jpeg_quality_range': (60, 80)
            }
        }
        return params.get(level, params['medium'])
    
    async def inject_noise(self, image: Image.Image, 
                         noise_types: Optional[List[str]] = None) -> Image.Image:
        """
        Inject various types of noise into document image.
        
        Args:
            image: Source PIL Image
            noise_types: List of noise types to apply
        
        Returns:
            Image with noise applied
        """
        if noise_types is None:
            noise_types = ['gaussian', 'blur', 'rotation', 'brightness', 'jpeg']
        
        result_image = image.copy()
        
        try:
            # Apply each noise type
            for noise_type in noise_types:
                if noise_type == 'gaussian':
                    result_image = self._add_gaussian_noise(result_image)
                elif noise_type == 'blur':
                    result_image = self._add_blur(result_image)
                elif noise_type == 'rotation':
                    result_image = self._add_rotation(result_image)
                elif noise_type == 'brightness':
                    result_image = self._adjust_brightness_contrast(result_image)
                elif noise_type == 'jpeg':
                    result_image = self._add_jpeg_compression(result_image)
                elif noise_type == 'salt_pepper':
                    result_image = self._add_salt_pepper_noise(result_image)
                elif noise_type == 'scan_lines':
                    result_image = self._add_scan_lines(result_image)
                elif noise_type == 'fold_marks':
                    result_image = self._add_fold_marks(result_image)
                elif noise_type == 'stains':
                    result_image = self._add_stains(result_image)
            
            return result_image
            
        except Exception as e:
            logger.error(f"Failed to inject noise: {e}")
            return image
    
    def _add_gaussian_noise(self, image: Image.Image) -> Image.Image:
        """Add Gaussian noise to image"""
        img_array = np.array(image)
        noise = np.random.normal(0, self.noise_params['gaussian_noise_std'], img_array.shape)
        noisy_array = np.clip(img_array.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_array)
    
    def _add_blur(self, image: Image.Image) -> Image.Image:
        """Add blur to simulate camera shake or motion blur"""
        blur_radius = random.uniform(*self.noise_params['blur_radius_range'])
        return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    def _add_rotation(self, image: Image.Image) -> Image.Image:
        """Add slight rotation to simulate document not being perfectly aligned"""
        angle = random.uniform(*self.noise_params['rotation_range'])
        return image.rotate(angle, expand=True, fillcolor='white')
    
    def _adjust_brightness_contrast(self, image: Image.Image) -> Image.Image:
        """Adjust brightness and contrast to simulate lighting conditions"""
        # Brightness adjustment
        brightness_factor = random.uniform(*self.noise_params['brightness_range'])
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)
        
        # Contrast adjustment
        contrast_factor = random.uniform(*self.noise_params['contrast_range'])
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)
        
        return image
    
    def _add_jpeg_compression(self, image: Image.Image) -> Image.Image:
        """Add JPEG compression artifacts"""
        import io
        
        quality = random.randint(*self.noise_params['jpeg_quality_range'])
        
        # Save to bytes with JPEG compression
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        
        # Load back from compressed data
        return Image.open(buffer)
    
    def _add_salt_pepper_noise(self, image: Image.Image) -> Image.Image:
        """Add salt and pepper noise"""
        img_array = np.array(image)
        noise_density = 0.001  # 0.1% of pixels
        
        # Generate random coordinates for noise
        num_pixels = int(noise_density * img_array.size)
        coords = [np.random.randint(0, i - 1, num_pixels) for i in img_array.shape[:2]]
        
        # Add salt (white) noise
        salt_coords = (coords[0][:num_pixels//2], coords[1][:num_pixels//2])
        img_array[salt_coords] = 255
        
        # Add pepper (black) noise
        pepper_coords = (coords[0][num_pixels//2:], coords[1][num_pixels//2:])
        img_array[pepper_coords] = 0
        
        return Image.fromarray(img_array)
    
    def _add_scan_lines(self, image: Image.Image) -> Image.Image:
        """Add horizontal scan lines to simulate scanner artifacts"""
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        # Add random horizontal lines
        num_lines = random.randint(2, 8)
        for _ in range(num_lines):
            y = random.randint(0, height)
            opacity = random.randint(10, 30)
            color = (128, 128, 128, opacity)
            
            # Draw subtle line
            for x in range(0, width, 2):  # Dotted line effect
                if random.random() < 0.7:
                    draw.point((x, y), fill=color[:3])
        
        return image
    
    def _add_fold_marks(self, image: Image.Image) -> Image.Image:
        """Add fold marks to simulate creased paper"""
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Add vertical fold mark
        if random.random() < 0.3:
            fold_x = random.randint(width//4, 3*width//4)
            fold_width = random.randint(2, 8)
            
            # Create fold shadow
            for i in range(fold_width):
                x = fold_x + i
                if x < width:
                    intensity = 1.0 - (i / fold_width) * 0.1
                    img_array[:, x] = (img_array[:, x] * intensity).astype(np.uint8)
        
        # Add horizontal fold mark
        if random.random() < 0.2:
            fold_y = random.randint(height//4, 3*height//4)
            fold_height = random.randint(2, 6)
            
            for i in range(fold_height):
                y = fold_y + i
                if y < height:
                    intensity = 1.0 - (i / fold_height) * 0.1
                    img_array[y, :] = (img_array[y, :] * intensity).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def _add_stains(self, image: Image.Image) -> Image.Image:
        """Add coffee stains or other marks"""
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Add random stains
        num_stains = random.randint(0, 3)
        for _ in range(num_stains):
            # Random position and size
            x = random.randint(0, image.size[0])
            y = random.randint(0, image.size[1])
            radius = random.randint(20, 100)
            
            # Random stain color (brownish)
            color = (
                random.randint(150, 200),
                random.randint(120, 170),
                random.randint(80, 130),
                random.randint(10, 30)  # Low opacity
            )
            
            # Draw irregular stain
            for i in range(3):
                offset_x = random.randint(-radius//2, radius//2)
                offset_y = random.randint(-radius//2, radius//2)
                stain_radius = radius + random.randint(-radius//3, radius//3)
                
                draw.ellipse(
                    [(x + offset_x - stain_radius//2, y + offset_y - stain_radius//2),
                     (x + offset_x + stain_radius//2, y + offset_y + stain_radius//2)],
                    fill=color
                )
        
        # Composite with original image
        result = Image.alpha_composite(image, overlay)
        return result.convert('RGB')
    
    async def simulate_photocopying(self, image: Image.Image, 
                                  generations: int = 1) -> Image.Image:
        """
        Simulate multiple generations of photocopying.
        
        Args:
            image: Source image
            generations: Number of photocopy generations
        
        Returns:
            Image with photocopy artifacts
        """
        result_image = image.copy()
        
        for generation in range(generations):
            # Each generation adds more noise
            noise_factor = 1 + (generation * 0.5)
            
            # Add noise specific to photocopying
            result_image = self._add_gaussian_noise(result_image)
            result_image = self._add_contrast_loss(result_image, factor=0.02 * noise_factor)
            result_image = self._add_dust_spots(result_image)
            
            # Slight blur accumulation
            if generation > 0:
                result_image = result_image.filter(ImageFilter.GaussianBlur(radius=0.1 * generation))
        
        return result_image
    
    def _add_contrast_loss(self, image: Image.Image, factor: float = 0.02) -> Image.Image:
        """Reduce contrast to simulate photocopy quality loss"""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(1.0 - factor)
    
    def _add_dust_spots(self, image: Image.Image) -> Image.Image:
        """Add small dust spots"""
        draw = ImageDraw.Draw(image)
        
        num_spots = random.randint(5, 20)
        for _ in range(num_spots):
            x = random.randint(0, image.size[0])
            y = random.randint(0, image.size[1])
            size = random.randint(1, 3)
            
            # Random dark spot
            color = random.randint(0, 100)
            draw.ellipse(
                [(x - size, y - size), (x + size, y + size)],
                fill=(color, color, color)
            )
        
        return image
    
    async def simulate_fax_transmission(self, image: Image.Image) -> Image.Image:
        """
        Simulate fax transmission artifacts.
        
        Args:
            image: Source image
        
        Returns:
            Image with fax artifacts
        """
        # Convert to grayscale first
        if image.mode != 'L':
            image = image.convert('L')
        
        # Add horizontal lines (fax transmission errors)
        img_array = np.array(image)
        height, width = img_array.shape
        
        # Random horizontal line dropouts
        num_dropouts = random.randint(2, 10)
        for _ in range(num_dropouts):
            y = random.randint(0, height - 1)
            start_x = random.randint(0, width // 2)
            end_x = random.randint(start_x, width)
            
            # Make line white (dropout)
            img_array[y, start_x:end_x] = 255
        
        # Add noise
        noise = np.random.normal(0, 10, img_array.shape)
        img_array = np.clip(img_array.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # Reduce resolution (fax quality)
        reduced_size = (width // 2, height // 2)
        fax_image = Image.fromarray(img_array).resize(reduced_size, Image.LANCZOS)
        fax_image = fax_image.resize((width, height), Image.NEAREST)
        
        return fax_image.convert('RGB')
    
    async def add_perspective_distortion(self, image: Image.Image, 
                                       intensity: float = 0.1) -> Image.Image:
        """
        Add perspective distortion to simulate angled scanning.
        
        Args:
            image: Source image
            intensity: Distortion intensity (0.0 to 1.0)
        
        Returns:
            Image with perspective distortion
        """
        try:
            from PIL import ImageTransform
            
            width, height = image.size
            
            # Calculate perspective transform coefficients
            # Create slight perspective effect
            dx = width * intensity * random.uniform(-1, 1)
            dy = height * intensity * random.uniform(-1, 1)
            
            # Define source and destination points
            src_points = [(0, 0), (width, 0), (width, height), (0, height)]
            dst_points = [
                (max(0, dx), max(0, dy)),
                (width - max(0, -dx), max(0, dy)),
                (width - max(0, dx), height - max(0, -dy)),
                (max(0, -dx), height - max(0, dy))
            ]
            
            # Calculate transform coefficients
            coeffs = self._get_perspective_coefficients(src_points, dst_points)
            
            # Apply transform
            transformed = image.transform(
                (width, height),
                ImageTransform.PerspectiveTransform(coeffs),
                fillcolor='white'
            )
            
            return transformed
            
        except Exception as e:
            logger.warning(f"Could not apply perspective distortion: {e}")
            return image
    
    def _get_perspective_coefficients(self, src_points, dst_points):
        """Calculate perspective transform coefficients"""
        # Simplified perspective transform calculation
        # For a more robust implementation, use OpenCV or similar
        return [1, 0, 0, 0, 1, 0, 0, 0]  # Identity transform as fallback


def create_noise_injector(noise_level: str = 'medium') -> OCRNoiseInjector:
    """Factory function to create OCR noise injector"""
    return OCRNoiseInjector(noise_level=noise_level)


__all__ = ['OCRNoiseInjector', 'create_noise_injector']