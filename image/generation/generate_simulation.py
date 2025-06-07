import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import json
import colorsys

logger = logging.getLogger(__name__)

class SimulationGenerator:
    """Generate synthetic images using procedural simulation techniques."""
    
    def __init__(self):
        self.environments = {
            'urban_city': self._generate_urban_scene,
            'highway': self._generate_highway_scene,
            'suburban': self._generate_suburban_scene,
            'industrial': self._generate_industrial_scene,
            'parking_lot': self._generate_parking_scene
        }
        
        self.weather_effects = {
            'clear': self._apply_clear_weather,
            'rain': self._apply_rain_effect,
            'fog': self._apply_fog_effect,
            'night': self._apply_night_effect,
            'snow': self._apply_snow_effect
        }
        
        self.object_templates = self._load_object_templates()
    
    def generate_scenes(self,
                       environment: str = 'urban_city',
                       weather: str = 'clear',
                       time_of_day: str = 'day',
                       num_images: int = 10,
                       resolution: Tuple[int, int] = (512, 512),
                       object_density: float = 0.5) -> List[Image.Image]:
        """Generate synthetic scenes with specified parameters."""
        
        if environment not in self.environments:
            raise ValueError(f"Unknown environment: {environment}. Available: {list(self.environments.keys())}")
        
        if weather not in self.weather_effects:
            raise ValueError(f"Unknown weather: {weather}. Available: {list(self.weather_effects.keys())}")
        
        logger.info(f"Generating {num_images} {environment} scenes with {weather} weather")
        
        images = []
        
        for i in range(num_images):
            try:
                # Generate base scene
                scene = self.environments[environment](resolution, object_density, time_of_day)
                
                # Apply weather effects
                scene = self.weather_effects[weather](scene, time_of_day)
                
                # Add realistic noise and artifacts
                scene = self._add_realistic_effects(scene)
                
                images.append(scene)
                
                if (i + 1) % 5 == 0:
                    logger.info(f"Generated {i + 1}/{num_images} simulation images")
                    
            except Exception as e:
                logger.error(f"Failed to generate simulation image {i + 1}: {e}")
                continue
        
        return images
    
    def generate_from_profile(self,
                             profile_path: str,
                             num_images: int = 10,
                             resolution: Tuple[int, int] = (512, 512)) -> List[Image.Image]:
        """Generate images based on a distribution profile."""
        
        # Load profile
        with open(profile_path, 'r') as f:
            profile = json.load(f)
        
        # Extract simulation guidance
        guidance = profile.get('generation_guidance', {}).get('simulation', {})
        
        # Use profile parameters
        environment = guidance.get('environment_type', 'urban_city')
        weather = guidance.get('weather_conditions', 'clear')
        object_density = guidance.get('object_density', 0.5)
        
        # Determine time of day from brightness if available
        time_of_day = 'day'
        if 'brightness' in profile.get('distributions', {}):
            brightness_info = profile['distributions']['brightness']
            if 'basic_stats' in brightness_info:
                avg_brightness = brightness_info['basic_stats']['mean']
                if avg_brightness < 80:
                    time_of_day = 'night'
                elif avg_brightness < 120:
                    time_of_day = 'evening'
        
        logger.info(f"Generating simulation images from profile: {environment}, {weather}, {time_of_day}")
        
        return self.generate_scenes(
            environment=environment,
            weather=weather,
            time_of_day=time_of_day,
            num_images=num_images,
            resolution=resolution,
            object_density=object_density
        )
    
    def _generate_urban_scene(self, resolution: Tuple[int, int], density: float, time_of_day: str) -> Image.Image:
        """Generate an urban city scene."""
        
        width, height = resolution
        
        # Create base image with sky gradient
        img = Image.new('RGB', (width, height), color=(135, 206, 235))  # Sky blue
        draw = ImageDraw.Draw(img)
        
        # Sky gradient
        for y in range(height // 3):
            color_intensity = int(235 - (y / (height // 3)) * 100)
            color = (135, 206, color_intensity)
            draw.line([(0, y), (width, y)], fill=color)
        
        # Ground
        ground_color = (105, 105, 105) if time_of_day == 'day' else (50, 50, 50)
        draw.rectangle([(0, height * 2 // 3), (width, height)], fill=ground_color)
        
        # Buildings
        self._add_buildings(draw, width, height, density, time_of_day)
        
        # Roads
        self._add_roads(draw, width, height)
        
        # Vehicles
        self._add_vehicles(draw, width, height, density)
        
        # Pedestrians
        self._add_pedestrians(draw, width, height, density)
        
        # Traffic lights and signs
        self._add_traffic_elements(draw, width, height)
        
        return img
    
    def _generate_highway_scene(self, resolution: Tuple[int, int], density: float, time_of_day: str) -> Image.Image:
        """Generate a highway scene."""
        
        width, height = resolution
        
        # Create base image
        img = Image.new('RGB', (width, height), color=(135, 206, 235))
        draw = ImageDraw.Draw(img)
        
        # Sky and horizon
        horizon_y = height // 3
        ground_color = (90, 140, 90) if time_of_day == 'day' else (30, 50, 30)
        draw.rectangle([(0, horizon_y), (width, height)], fill=ground_color)
        
        # Highway with perspective
        road_color = (60, 60, 60)
        road_top_width = width // 3
        road_bottom_width = width
        
        # Highway surface
        highway_points = [
            (width // 2 - road_bottom_width // 2, height),
            (width // 2 + road_bottom_width // 2, height),
            (width // 2 + road_top_width // 2, horizon_y),
            (width // 2 - road_top_width // 2, horizon_y)
        ]
        draw.polygon(highway_points, fill=road_color)
        
        # Lane markings
        self._add_highway_markings(draw, highway_points, width, height)
        
        # Vehicles on highway
        self._add_highway_vehicles(draw, highway_points, density)
        
        # Roadside elements
        self._add_roadside_elements(draw, width, height, density)
        
        return img
    
    def _generate_suburban_scene(self, resolution: Tuple[int, int], density: float, time_of_day: str) -> Image.Image:
        """Generate a suburban scene."""
        
        width, height = resolution
        
        img = Image.new('RGB', (width, height), color=(135, 206, 235))
        draw = ImageDraw.Draw(img)
        
        # Ground with grass
        grass_color = (90, 140, 90) if time_of_day == 'day' else (30, 50, 30)
        draw.rectangle([(0, height * 2 // 3), (width, height)], fill=grass_color)
        
        # Houses
        self._add_houses(draw, width, height, density)
        
        # Trees
        self._add_trees(draw, width, height, density)
        
        # Suburban roads
        self._add_suburban_roads(draw, width, height)
        
        # Parked cars
        self._add_parked_cars(draw, width, height, density)
        
        return img
    
    def _generate_industrial_scene(self, resolution: Tuple[int, int], density: float, time_of_day: str) -> Image.Image:
        """Generate an industrial scene."""
        
        width, height = resolution
        
        img = Image.new('RGB', (width, height), color=(120, 120, 120))  # Overcast sky
        draw = ImageDraw.Draw(img)
        
        # Industrial ground
        ground_color = (80, 80, 80)
        draw.rectangle([(0, height * 2 // 3), (width, height)], fill=ground_color)
        
        # Industrial buildings
        self._add_industrial_buildings(draw, width, height, density)
        
        # Smokestacks
        self._add_smokestacks(draw, width, height)
        
        # Industrial vehicles
        self._add_industrial_vehicles(draw, width, height, density)
        
        return img
    
    def _generate_parking_scene(self, resolution: Tuple[int, int], density: float, time_of_day: str) -> Image.Image:
        """Generate a parking lot scene."""
        
        width, height = resolution
        
        img = Image.new('RGB', (width, height), color=(135, 206, 235))
        draw = ImageDraw.Draw(img)
        
        # Parking lot surface
        asphalt_color = (70, 70, 70)
        draw.rectangle([(0, height // 3), (width, height)], fill=asphalt_color)
        
        # Parking spaces
        self._add_parking_spaces(draw, width, height)
        
        # Parked cars in grid
        self._add_parking_cars(draw, width, height, density)
        
        # Parking lot elements
        self._add_parking_elements(draw, width, height)
        
        return img
    
    def _add_buildings(self, draw: ImageDraw.Draw, width: int, height: int, density: float, time_of_day: str):
        """Add buildings to urban scene."""
        
        num_buildings = int(5 + density * 10)
        building_width = width // num_buildings
        
        for i in range(num_buildings):
            x_start = i * building_width
            building_height = random.randint(height // 6, height // 2)
            y_start = height * 2 // 3 - building_height
            
            # Building colors
            if time_of_day == 'day':
                building_color = (random.randint(100, 180), random.randint(100, 180), random.randint(100, 180))
            else:
                building_color = (random.randint(30, 80), random.randint(30, 80), random.randint(30, 80))
            
            # Building body
            draw.rectangle([
                (x_start, y_start),
                (x_start + building_width - 2, height * 2 // 3)
            ], fill=building_color, outline=(0, 0, 0))
            
            # Windows
            self._add_windows(draw, x_start, y_start, building_width, building_height, time_of_day)
    
    def _add_windows(self, draw: ImageDraw.Draw, x: int, y: int, width: int, height: int, time_of_day: str):
        """Add windows to buildings."""
        
        window_size = 8
        window_spacing = 12
        
        for win_y in range(y + 10, y + height - 10, window_spacing):
            for win_x in range(x + 5, x + width - 10, window_spacing):
                if random.random() < 0.8:  # 80% chance of window
                    if time_of_day == 'night' and random.random() < 0.6:
                        # Lit windows at night
                        window_color = (255, 255, 200)
                    else:
                        window_color = (100, 150, 200)
                    
                    draw.rectangle([
                        (win_x, win_y),
                        (win_x + window_size, win_y + window_size)
                    ], fill=window_color)
    
    def _add_roads(self, draw: ImageDraw.Draw, width: int, height: int):
        """Add roads to the scene."""
        
        road_color = (60, 60, 60)
        
        # Main horizontal road
        road_y = height * 3 // 4
        draw.rectangle([(0, road_y), (width, road_y + 30)], fill=road_color)
        
        # Lane markings
        for x in range(0, width, 40):
            draw.rectangle([(x, road_y + 13), (x + 20, road_y + 17)], fill=(255, 255, 255))
    
    def _add_vehicles(self, draw: ImageDraw.Draw, width: int, height: int, density: float):
        """Add vehicles to the scene."""
        
        num_vehicles = int(2 + density * 8)
        road_y = height * 3 // 4
        
        for _ in range(num_vehicles):
            x = random.randint(10, width - 30)
            y = road_y + random.randint(2, 8)
            
            # Car colors
            car_color = random.choice([
                (255, 0, 0),    # Red
                (0, 0, 255),    # Blue
                (255, 255, 255), # White
                (0, 0, 0),      # Black
                (128, 128, 128) # Gray
            ])
            
            # Car body
            draw.rectangle([(x, y), (x + 25, y + 12)], fill=car_color, outline=(0, 0, 0))
            
            # Windows
            draw.rectangle([(x + 3, y + 2), (x + 22, y + 6)], fill=(150, 200, 255))
    
    def _add_pedestrians(self, draw: ImageDraw.Draw, width: int, height: int, density: float):
        """Add pedestrians to the scene."""
        
        num_pedestrians = int(density * 6)
        sidewalk_y = height * 2 // 3
        
        for _ in range(num_pedestrians):
            x = random.randint(5, width - 5)
            y = sidewalk_y + random.randint(5, 20)
            
            # Simple stick figure
            # Head
            draw.ellipse([(x, y), (x + 4, y + 4)], fill=(255, 220, 177))
            
            # Body
            draw.line([(x + 2, y + 4), (x + 2, y + 12)], fill=(0, 0, 0), width=2)
            
            # Arms
            draw.line([(x - 1, y + 7), (x + 5, y + 7)], fill=(0, 0, 0), width=1)
            
            # Legs
            draw.line([(x + 2, y + 12), (x, y + 18)], fill=(0, 0, 0), width=2)
            draw.line([(x + 2, y + 12), (x + 4, y + 18)], fill=(0, 0, 0), width=2)
    
    def _add_traffic_elements(self, draw: ImageDraw.Draw, width: int, height: int):
        """Add traffic lights and signs."""
        
        # Traffic light
        if random.random() < 0.7:
            x = width // 4
            y = height * 2 // 3 - 40
            
            # Post
            draw.rectangle([(x, y), (x + 3, height * 2 // 3)], fill=(100, 100, 100))
            
            # Light box
            draw.rectangle([(x - 5, y - 15), (x + 8, y)], fill=(50, 50, 50))
            
            # Lights
            light_color = random.choice([(255, 0, 0), (255, 255, 0), (0, 255, 0)])
            draw.ellipse([(x - 3, y - 12), (x + 6, y - 3)], fill=light_color)
    
    def _add_highway_markings(self, draw: ImageDraw.Draw, highway_points: List, width: int, height: int):
        """Add lane markings to highway."""
        
        # Center line markings
        for i in range(10):
            y = height - (i * (height // 15))
            line_width = 2 + i // 3
            x_center = width // 2
            
            draw.rectangle([
                (x_center - line_width, y),
                (x_center + line_width, y - 10)
            ], fill=(255, 255, 255))
    
    def _add_highway_vehicles(self, draw: ImageDraw.Draw, highway_points: List, density: float):
        """Add vehicles to highway with perspective."""
        
        num_vehicles = int(3 + density * 12)
        
        for _ in range(num_vehicles):
            # Random position on highway
            progress = random.random()
            y_pos = highway_points[3][1] + progress * (highway_points[0][1] - highway_points[3][1])
            
            # Calculate vehicle size based on distance (perspective)
            scale = 0.3 + progress * 0.7
            vehicle_width = int(20 * scale)
            vehicle_height = int(8 * scale)
            
            # Lane offset
            lane_offset = random.randint(-30, 30) * scale
            x_pos = highway_points[3][0] + (highway_points[0][0] - highway_points[3][0]) * progress + lane_offset
            
            # Vehicle color
            car_color = random.choice([
                (255, 0, 0), (0, 0, 255), (255, 255, 255),
                (0, 0, 0), (128, 128, 128), (255, 255, 0)
            ])
            
            # Draw vehicle
            draw.rectangle([
                (int(x_pos - vehicle_width // 2), int(y_pos - vehicle_height)),
                (int(x_pos + vehicle_width // 2), int(y_pos))
            ], fill=car_color, outline=(0, 0, 0))
    
    def _add_roadside_elements(self, draw: ImageDraw.Draw, width: int, height: int, density: float):
        """Add roadside elements like signs and barriers."""
        
        # Guard rails
        for x in range(0, width, 50):
            rail_height = 20
            y = height // 3 + rail_height
            draw.rectangle([(x, y), (x + 30, y + 5)], fill=(150, 150, 150))
    
    def _apply_clear_weather(self, img: Image.Image, time_of_day: str) -> Image.Image:
        """Apply clear weather effects."""
        
        if time_of_day == 'day':
            # Enhance brightness and contrast
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.1)
            
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.1)
        
        return img
    
    def _apply_rain_effect(self, img: Image.Image, time_of_day: str) -> Image.Image:
        """Apply rain effects."""
        
        # Darken the image
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.7)
        
        # Add rain drops
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        num_drops = random.randint(200, 500)
        for _ in range(num_drops):
            x = random.randint(0, width)
            y = random.randint(0, height)
            length = random.randint(3, 8)
            
            # Rain drop
            draw.line([(x, y), (x + 1, y + length)], fill=(200, 200, 255), width=1)
        
        # Add slight blur for atmosphere
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return img
    
    def _apply_fog_effect(self, img: Image.Image, time_of_day: str) -> Image.Image:
        """Apply fog effects."""
        
        # Create fog overlay
        fog_overlay = Image.new('RGBA', img.size, (200, 200, 200, 100))
        
        # Blend with original
        img = img.convert('RGBA')
        img = Image.alpha_composite(img, fog_overlay)
        img = img.convert('RGB')
        
        # Reduce contrast
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(0.6)
        
        return img
    
    def _apply_night_effect(self, img: Image.Image, time_of_day: str) -> Image.Image:
        """Apply night effects."""
        
        # Darken significantly
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.3)
        
        # Add blue tint
        blue_overlay = Image.new('RGBA', img.size, (0, 0, 50, 50))
        img = img.convert('RGBA')
        img = Image.alpha_composite(img, blue_overlay)
        img = img.convert('RGB')
        
        return img
    
    def _apply_snow_effect(self, img: Image.Image, time_of_day: str) -> Image.Image:
        """Apply snow effects."""
        
        # Brighten the image
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.2)
        
        # Add snowflakes
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        num_flakes = random.randint(100, 300)
        for _ in range(num_flakes):
            x = random.randint(0, width)
            y = random.randint(0, height)
            size = random.randint(1, 3)
            
            # Snowflake
            draw.ellipse([(x, y), (x + size, y + size)], fill=(255, 255, 255))
        
        return img
    
    def _add_realistic_effects(self, img: Image.Image) -> Image.Image:
        """Add realistic camera effects and noise."""
        
        # Add slight noise
        np_img = np.array(img)
        noise = np.random.normal(0, 2, np_img.shape)
        np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(np_img)
        
        # Slight blur for realism
        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.2))
        
        return img
    
    def _load_object_templates(self) -> Dict:
        """Load object templates for procedural generation."""
        
        return {
            'car_colors': [
                (255, 0, 0), (0, 0, 255), (255, 255, 255),
                (0, 0, 0), (128, 128, 128), (255, 255, 0),
                (0, 128, 0), (128, 0, 128)
            ],
            'building_colors': [
                (120, 120, 120), (150, 150, 150), (100, 100, 100),
                (140, 130, 120), (160, 140, 120)
            ]
        }
    
    def save_images(self, images: List[Image.Image], output_dir: str, prefix: str = "simulation") -> List[str]:
        """Save generated images to directory."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        for i, image in enumerate(images):
            filename = f"{prefix}_{i:04d}.png"
            filepath = output_path / filename
            
            try:
                image.save(filepath)
                saved_paths.append(str(filepath))
            except Exception as e:
                logger.error(f"Failed to save image {filename}: {e}")
        
        logger.info(f"Saved {len(saved_paths)} simulation images to {output_dir}")
        return saved_paths
    
    # Additional helper methods for other scene types
    def _add_houses(self, draw: ImageDraw.Draw, width: int, height: int, density: float):
        """Add houses to suburban scene."""
        num_houses = int(2 + density * 5)
        house_width = width // (num_houses + 1)
        
        for i in range(num_houses):
            x = (i + 1) * house_width - house_width // 2
            y = height * 2 // 3 - 40
            
            # House body
            draw.rectangle([(x, y), (x + house_width // 2, height * 2 // 3)], 
                          fill=(200, 180, 140), outline=(0, 0, 0))
            
            # Roof
            roof_points = [
                (x - 5, y),
                (x + house_width // 2 + 5, y),
                (x + house_width // 4, y - 20)
            ]
            draw.polygon(roof_points, fill=(150, 50, 50))
    
    def _add_trees(self, draw: ImageDraw.Draw, width: int, height: int, density: float):
        """Add trees to scene."""
        num_trees = int(density * 8)
        
        for _ in range(num_trees):
            x = random.randint(20, width - 20)
            y = random.randint(height * 2 // 3 - 10, height * 2 // 3 + 20)
            
            # Trunk
            draw.rectangle([(x, y), (x + 4, y + 20)], fill=(101, 67, 33))
            
            # Foliage
            draw.ellipse([(x - 8, y - 15), (x + 12, y + 5)], fill=(34, 139, 34))
    
    def _add_suburban_roads(self, draw: ImageDraw.Draw, width: int, height: int):
        """Add suburban roads."""
        road_color = (80, 80, 80)
        
        # Main road
        draw.rectangle([(0, height * 3 // 4), (width, height * 3 // 4 + 20)], fill=road_color)
        
        # Sidewalks
        draw.rectangle([(0, height * 2 // 3), (width, height * 2 // 3 + 8)], fill=(180, 180, 180))
    
    def _add_parked_cars(self, draw: ImageDraw.Draw, width: int, height: int, density: float):
        """Add parked cars to suburban scene."""
        num_cars = int(density * 4)
        
        for _ in range(num_cars):
            x = random.randint(10, width - 30)
            y = height * 2 // 3 + 10
            
            car_color = random.choice(self.object_templates['car_colors'])
            draw.rectangle([(x, y), (x + 25, y + 12)], fill=car_color, outline=(0, 0, 0))
    
    def _add_industrial_buildings(self, draw: ImageDraw.Draw, width: int, height: int, density: float):
        """Add industrial buildings."""
        num_buildings = int(2 + density * 4)
        building_width = width // num_buildings
        
        for i in range(num_buildings):
            x = i * building_width
            building_height = random.randint(height // 4, height // 2)
            y = height * 2 // 3 - building_height
            
            # Industrial building colors (more muted)
            building_color = random.choice([(100, 100, 100), (120, 120, 120), (80, 80, 80)])
            
            draw.rectangle([(x, y), (x + building_width - 5, height * 2 // 3)], 
                          fill=building_color, outline=(0, 0, 0))
    
    def _add_smokestacks(self, draw: ImageDraw.Draw, width: int, height: int):
        """Add industrial smokestacks."""
        num_stacks = random.randint(1, 3)
        
        for _ in range(num_stacks):
            x = random.randint(width // 4, 3 * width // 4)
            stack_height = random.randint(height // 3, height // 2)
            y = height * 2 // 3 - stack_height
            
            # Smokestack
            draw.rectangle([(x, y), (x + 12, height * 2 // 3)], fill=(60, 60, 60))
            
            # Smoke
            for i in range(5):
                smoke_y = y - i * 8
                smoke_size = 8 + i * 2
                draw.ellipse([(x - smoke_size // 2, smoke_y), 
                             (x + 12 + smoke_size // 2, smoke_y + smoke_size)], 
                            fill=(150, 150, 150, 100))
    
    def _add_industrial_vehicles(self, draw: ImageDraw.Draw, width: int, height: int, density: float):
        """Add industrial vehicles like trucks."""
        num_vehicles = int(density * 3)
        
        for _ in range(num_vehicles):
            x = random.randint(20, width - 50)
            y = height * 3 // 4 + 5
            
            # Truck (larger than regular car)
            truck_color = random.choice([(255, 255, 0), (255, 165, 0), (255, 255, 255)])
            draw.rectangle([(x, y), (x + 35, y + 15)], fill=truck_color, outline=(0, 0, 0))
    
    def _add_parking_spaces(self, draw: ImageDraw.Draw, width: int, height: int):
        """Add parking space markings."""
        space_width = 30
        space_height = 40
        
        rows = (height - height // 3) // (space_height + 10)
        cols = width // (space_width + 5)
        
        for row in range(rows):
            for col in range(cols):
                x = col * (space_width + 5) + 5
                y = height // 3 + row * (space_height + 10) + 10
                
                # Parking space outline
                draw.rectangle([(x, y), (x + space_width, y + space_height)], 
                              outline=(255, 255, 255), width=2)
    
    def _add_parking_cars(self, draw: ImageDraw.Draw, width: int, height: int, density: float):
        """Add cars to parking spaces."""
        space_width = 30
        space_height = 40
        
        rows = (height - height // 3) // (space_height + 10)
        cols = width // (space_width + 5)
        
        for row in range(rows):
            for col in range(cols):
                if random.random() < density:  # Occupancy rate
                    x = col * (space_width + 5) + 7
                    y = height // 3 + row * (space_height + 10) + 15
                    
                    car_color = random.choice(self.object_templates['car_colors'])
                    draw.rectangle([(x, y), (x + space_width - 4, y + space_height - 10)], 
                                  fill=car_color, outline=(0, 0, 0))
    
    def _add_parking_elements(self, draw: ImageDraw.Draw, width: int, height: int):
        """Add parking lot elements like light poles."""
        # Light poles
        for i in range(3):
            x = (i + 1) * width // 4
            y = height // 3 + 20
            
            # Pole
            draw.rectangle([(x, y), (x + 3, height)], fill=(100, 100, 100))
            
            # Light
            draw.ellipse([(x - 5, y), (x + 8, y + 8)], fill=(255, 255, 200))

if __name__ == "__main__":
    # Test the simulation generator
    try:
        generator = SimulationGenerator()
        
        # Generate urban scenes
        urban_images = generator.generate_scenes(
            environment='urban_city',
            weather='clear',
            time_of_day='day',
            num_images=3,
            resolution=(512, 512),
            object_density=0.7
        )
        
        print(f"Generated {len(urban_images)} urban images")
        
        # Save test images
        saved_paths = generator.save_images(urban_images, "./data/generated", "test_simulation")
        print(f"Saved to: {saved_paths}")
        
        # Test weather effects
        rainy_images = generator.generate_scenes(
            environment='highway',
            weather='rain',
            time_of_day='day',
            num_images=2,
            object_density=0.5
        )
        
        rainy_paths = generator.save_images(rainy_images, "./data/generated", "test_rain")
        print(f"Rainy images saved to: {rainy_paths}")
        
    except Exception as e:
        print(f"Test failed: {e}")
