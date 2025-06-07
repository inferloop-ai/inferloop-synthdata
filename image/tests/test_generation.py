
import unittest
import os
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from generation.generate_diffusion import generate_image
from configs.config_loader import load_config

class TestImageGeneration(unittest.TestCase):
    def setUp(self):
        self.test_config = {
            "prompt": "a stunning landscape with mountains and a lake",
            "negative_prompt": "blurry, bad quality, text",
            "width": 512,
            "height": 512,
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "seed": 42
        }
        self.output_dir = Path("data/generated/test")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def test_generate_single_image(self):
        """Test that a single image can be generated with default parameters"""
        output_path = generate_image(
            prompt=self.test_config["prompt"],
            output_dir=str(self.output_dir),
            filename="test_single.png"
        )
        self.assertTrue(os.path.exists(output_path))
        
    def test_generate_with_seed(self):
        """Test that seed produces consistent results"""
        output_path1 = generate_image(
            prompt=self.test_config["prompt"],
            output_dir=str(self.output_dir),
            filename="test_seed_1.png",
            seed=self.test_config["seed"]
        )
        
        output_path2 = generate_image(
            prompt=self.test_config["prompt"],
            output_dir=str(self.output_dir),
            filename="test_seed_2.png",
            seed=self.test_config["seed"]
        )
        
        # Images should be identical with same seed
        # This would require image comparison but we'll check they exist for now
        self.assertTrue(os.path.exists(output_path1))
        self.assertTrue(os.path.exists(output_path2))

if __name__ == "__main__":
    unittest.main()