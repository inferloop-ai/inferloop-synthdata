"""
Traffic Scenarios Generator for Autonomous Vehicle Synthetic Data
Part of the Inferloop SynthData Video Pipeline

This module provides configurable traffic scenarios for autonomous vehicle
synthetic data generation, including urban, highway, and complex intersection scenarios.
"""

import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WeatherCondition(Enum):
    """Weather conditions for scenario generation"""
    CLEAR = "clear"
    RAIN = "rain"
    SNOW = "snow"
    FOG = "fog"
    OVERCAST = "overcast"

class TimeOfDay(Enum):
    """Time of day for scenario generation"""
    DAWN = "dawn"
    MORNING = "morning"
    NOON = "noon"
    AFTERNOON = "afternoon"
    DUSK = "dusk"
    NIGHT = "night"

class RoadType(Enum):
    """Road types for scenario generation"""
    URBAN = "urban"
    SUBURBAN = "suburban"
    HIGHWAY = "highway"
    RURAL = "rural"
    INTERSECTION = "intersection"
    ROUNDABOUT = "roundabout"
    PARKING = "parking"

@dataclass
class VehicleConfig:
    """Configuration for a vehicle in the scenario"""
    vehicle_type: str  # car, truck, bus, motorcycle, bicycle
    behavior: str  # normal, aggressive, cautious
    speed_factor: float  # 0.5 = half speed limit, 1.0 = speed limit, 1.2 = 20% over
    trajectory: List[Dict[str, float]]  # list of waypoints with x, y, z coordinates
    color: str = "white"
    model: str = "sedan"

@dataclass
class PedestrianConfig:
    """Configuration for a pedestrian in the scenario"""
    behavior: str  # normal, distracted, rushing
    trajectory: List[Dict[str, float]]  # list of waypoints
    appearance: str = "casual"  # casual, business, child
    group_size: int = 1  # number of pedestrians in group

@dataclass
class TrafficScenarioConfig:
    """Configuration for a traffic scenario"""
    name: str
    road_type: RoadType
    weather: WeatherCondition
    time_of_day: TimeOfDay
    duration_seconds: int
    vehicles: List[VehicleConfig]
    pedestrians: List[PedestrianConfig]
    traffic_signals: bool = True
    road_markings: bool = True
    obstacles: List[Dict[str, Any]] = None

class TrafficScenarioGenerator:
    """
    Generator for realistic traffic scenarios for autonomous vehicle testing
    and synthetic data generation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the traffic scenario generator
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.scenario_templates = self._load_scenario_templates()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return {}
    
    def _load_scenario_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load scenario templates from configuration or use defaults"""
        templates_path = self.config.get('templates_path', os.path.join(os.path.dirname(__file__), 'templates'))
        templates = {}
        
        # Default templates if file loading fails
        default_templates = {
            "urban_intersection": {
                "road_type": RoadType.INTERSECTION,
                "vehicle_count_range": [5, 15],
                "pedestrian_count_range": [3, 10],
                "traffic_signals": True,
                "description": "Busy urban intersection with traffic lights and pedestrian crossings"
            },
            "highway": {
                "road_type": RoadType.HIGHWAY,
                "vehicle_count_range": [10, 30],
                "pedestrian_count_range": [0, 0],
                "traffic_signals": False,
                "description": "Multi-lane highway with varying traffic density"
            },
            "residential": {
                "road_type": RoadType.URBAN,
                "vehicle_count_range": [2, 8],
                "pedestrian_count_range": [1, 5],
                "traffic_signals": False,
                "description": "Residential area with parked cars and occasional pedestrians"
            },
            "roundabout": {
                "road_type": RoadType.ROUNDABOUT,
                "vehicle_count_range": [4, 12],
                "pedestrian_count_range": [0, 3],
                "traffic_signals": False,
                "description": "Multi-entry roundabout with merging traffic"
            },
            "parking_lot": {
                "road_type": RoadType.PARKING,
                "vehicle_count_range": [5, 20],
                "pedestrian_count_range": [2, 8],
                "traffic_signals": False,
                "description": "Parking lot with moving and stationary vehicles and pedestrians"
            }
        }
        
        # Try to load templates from files
        try:
            if os.path.exists(templates_path):
                for file in os.listdir(templates_path):
                    if file.endswith('.json'):
                        with open(os.path.join(templates_path, file), 'r') as f:
                            template_data = json.load(f)
                            templates[template_data.get('name', file.split('.')[0])] = template_data
            
            if not templates:
                templates = default_templates
                
        except Exception as e:
            logger.warning(f"Failed to load scenario templates: {e}")
            templates = default_templates
            
        return templates
    
    def generate_scenario(self, template_name: str = None, **kwargs) -> TrafficScenarioConfig:
        """
        Generate a traffic scenario based on template and parameters
        
        Args:
            template_name: Name of scenario template to use
            **kwargs: Override parameters for the scenario
            
        Returns:
            TrafficScenarioConfig object with the generated scenario
        """
        # Select random template if none specified
        if template_name is None:
            template_name = random.choice(list(self.scenario_templates.keys()))
        
        # Get template or use default urban intersection
        template = self.scenario_templates.get(template_name, self.scenario_templates.get("urban_intersection"))
        
        # Generate weather and time of day if not specified
        weather = kwargs.get('weather', random.choice(list(WeatherCondition)))
        time_of_day = kwargs.get('time_of_day', random.choice(list(TimeOfDay)))
        
        # Generate vehicles
        vehicle_count = kwargs.get('vehicle_count', 
                                  random.randint(template['vehicle_count_range'][0], 
                                                template['vehicle_count_range'][1]))
        
        vehicles = self._generate_vehicles(vehicle_count, template['road_type'])
        
        # Generate pedestrians
        pedestrian_count = kwargs.get('pedestrian_count', 
                                     random.randint(template['pedestrian_count_range'][0], 
                                                   template['pedestrian_count_range'][1]))
        
        pedestrians = self._generate_pedestrians(pedestrian_count, template['road_type'])
        
        # Create scenario config
        scenario_config = TrafficScenarioConfig(
            name=kwargs.get('name', f"{template_name}_{weather.value}_{time_of_day.value}"),
            road_type=template['road_type'] if isinstance(template['road_type'], RoadType) else RoadType(template['road_type']),
            weather=weather if isinstance(weather, WeatherCondition) else WeatherCondition(weather),
            time_of_day=time_of_day if isinstance(time_of_day, TimeOfDay) else TimeOfDay(time_of_day),
            duration_seconds=kwargs.get('duration_seconds', 60),
            vehicles=vehicles,
            pedestrians=pedestrians,
            traffic_signals=kwargs.get('traffic_signals', template.get('traffic_signals', True)),
            road_markings=kwargs.get('road_markings', True),
            obstacles=kwargs.get('obstacles', [])
        )
        
        return scenario_config
    
    def _generate_vehicles(self, count: int, road_type: RoadType) -> List[VehicleConfig]:
        """Generate vehicle configurations for the scenario"""
        vehicles = []
        
        vehicle_types = ["car", "car", "car", "truck", "motorcycle"]  # More cars than other types
        behaviors = ["normal", "normal", "normal", "aggressive", "cautious"]  # More normal behavior
        
        for i in range(count):
            vehicle_type = random.choice(vehicle_types)
            behavior = random.choice(behaviors)
            
            # Speed factor based on behavior
            if behavior == "aggressive":
                speed_factor = random.uniform(1.1, 1.3)
            elif behavior == "cautious":
                speed_factor = random.uniform(0.7, 0.9)
            else:
                speed_factor = random.uniform(0.9, 1.1)
            
            # Generate trajectory based on road type
            trajectory = self._generate_trajectory(road_type)
            
            # Car colors and models
            colors = ["white", "black", "silver", "red", "blue", "gray"]
            models = ["sedan", "suv", "hatchback", "pickup", "van"]
            
            vehicles.append(VehicleConfig(
                vehicle_type=vehicle_type,
                behavior=behavior,
                speed_factor=speed_factor,
                trajectory=trajectory,
                color=random.choice(colors),
                model=random.choice(models) if vehicle_type == "car" else vehicle_type
            ))
        
        return vehicles
    
    def _generate_pedestrians(self, count: int, road_type: RoadType) -> List[PedestrianConfig]:
        """Generate pedestrian configurations for the scenario"""
        pedestrians = []
        
        behaviors = ["normal", "normal", "distracted", "rushing"]
        appearances = ["casual", "casual", "business", "child"]
        
        for i in range(count):
            behavior = random.choice(behaviors)
            
            # Generate trajectory based on road type
            trajectory = self._generate_pedestrian_trajectory(road_type)
            
            # Determine if this is a group
            group_size = 1
            if random.random() < 0.3:  # 30% chance of being a group
                group_size = random.randint(2, 4)
            
            pedestrians.append(PedestrianConfig(
                behavior=behavior,
                trajectory=trajectory,
                appearance=random.choice(appearances),
                group_size=group_size
            ))
        
        return pedestrians
    
    def _generate_trajectory(self, road_type: RoadType) -> List[Dict[str, float]]:
        """Generate a vehicle trajectory based on road type"""
        # This is a simplified version - in a real implementation, 
        # this would use splines or predefined paths based on the road network
        
        trajectory = []
        
        # Number of waypoints depends on road type
        if road_type == RoadType.HIGHWAY:
            num_points = 10
            x_range = (-100, 100)
            y_variation = (-5, 5)  # Less variation on highway
            
        elif road_type == RoadType.INTERSECTION:
            num_points = 15
            x_range = (-50, 50)
            y_variation = (-50, 50)  # More variation at intersection
            
        else:
            num_points = 20
            x_range = (-75, 75)
            y_variation = (-20, 20)
        
        # Generate waypoints
        for i in range(num_points):
            # Linear progression along x-axis
            x = x_range[0] + (x_range[1] - x_range[0]) * (i / (num_points - 1))
            
            # Add some variation to y based on road type
            y = random.uniform(y_variation[0], y_variation[1])
            
            # Add some variation to z (elevation)
            z = 0
            if road_type != RoadType.PARKING:
                z = random.uniform(-1, 1)
            
            trajectory.append({"x": x, "y": y, "z": z})
        
        return trajectory
    
    def _generate_pedestrian_trajectory(self, road_type: RoadType) -> List[Dict[str, float]]:
        """Generate a pedestrian trajectory based on road type"""
        trajectory = []
        
        # Pedestrians have more varied paths than vehicles
        num_points = 15
        
        # Different trajectory patterns based on road type
        if road_type == RoadType.INTERSECTION:
            # Crossing pattern
            start_x = random.choice([-30, 30])
            end_x = -start_x
            
            for i in range(num_points):
                progress = i / (num_points - 1)
                x = start_x + (end_x - start_x) * progress
                
                # Add sidewalk waiting
                if 0.4 < progress < 0.6:
                    y = random.uniform(-2, 2)  # Waiting at crossing
                else:
                    y = random.uniform(-15, 15)  # On sidewalk
                
                trajectory.append({"x": x, "y": y, "z": 0})
                
        elif road_type == RoadType.PARKING:
            # Walking between cars
            for i in range(num_points):
                x = random.uniform(-40, 40)
                y = random.uniform(-40, 40)
                trajectory.append({"x": x, "y": y, "z": 0})
                
        else:
            # Walking along sidewalk
            for i in range(num_points):
                x = -50 + 100 * (i / (num_points - 1))
                y = random.choice([-15, 15])  # Either side of the road
                y += random.uniform(-2, 2)  # Small variation
                trajectory.append({"x": x, "y": y, "z": 0})
        
        return trajectory
    
    def save_scenario(self, scenario: TrafficScenarioConfig, output_path: str) -> None:
        """
        Save scenario configuration to file
        
        Args:
            scenario: TrafficScenarioConfig to save
            output_path: Path to save the scenario
        """
        # Convert dataclasses to dictionaries
        scenario_dict = {
            "name": scenario.name,
            "road_type": scenario.road_type.value,
            "weather": scenario.weather.value,
            "time_of_day": scenario.time_of_day.value,
            "duration_seconds": scenario.duration_seconds,
            "traffic_signals": scenario.traffic_signals,
            "road_markings": scenario.road_markings,
            "vehicles": [
                {
                    "vehicle_type": v.vehicle_type,
                    "behavior": v.behavior,
                    "speed_factor": v.speed_factor,
                    "trajectory": v.trajectory,
                    "color": v.color,
                    "model": v.model
                } for v in scenario.vehicles
            ],
            "pedestrians": [
                {
                    "behavior": p.behavior,
                    "trajectory": p.trajectory,
                    "appearance": p.appearance,
                    "group_size": p.group_size
                } for p in scenario.pedestrians
            ],
            "obstacles": scenario.obstacles or []
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(scenario_dict, f, indent=2)
            
        logger.info(f"Saved scenario '{scenario.name}' to {output_path}")

# Example usage
if __name__ == "__main__":
    generator = TrafficScenarioGenerator()
    
    # Generate urban intersection scenario
    urban_scenario = generator.generate_scenario("urban_intersection", 
                                               weather=WeatherCondition.RAIN,
                                               time_of_day=TimeOfDay.NIGHT)
    
    print(f"Generated {urban_scenario.name} scenario with {len(urban_scenario.vehicles)} vehicles "
          f"and {len(urban_scenario.pedestrians)} pedestrians")
    
    # Save scenario
    generator.save_scenario(urban_scenario, "output/urban_rain_night.json")
    
    # Generate highway scenario
    highway_scenario = generator.generate_scenario("highway", 
                                                 vehicle_count=20,
                                                 weather=WeatherCondition.CLEAR,
                                                 time_of_day=TimeOfDay.NOON)
    
    print(f"Generated {highway_scenario.name} scenario with {len(highway_scenario.vehicles)} vehicles")
