# audio_synth/sdk/pipeline.py
"""
Audio Generation and Processing Pipeline implementation

This module provides a pipeline interface for chaining together different
audio generation, processing, and validation components from the audio_synth
framework in a consistent and modular way.
"""

from typing import Dict, List, Any, Optional, Union, Callable
import os
import uuid
import logging
import torch
import torchaudio
import numpy as np
from pathlib import Path
import json
import tempfile

from ..core.generators.base import BaseGenerator
from ..core.validators.base import BaseValidator
from ..core.utils.config import load_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineStage:
    """Represents a single stage in the audio processing pipeline"""
    
    def __init__(self, name: str, processor: Callable, config: Optional[Dict[str, Any]] = None):
        """
        Initialize a pipeline stage
        
        Args:
            name: Name of the stage
            processor: Callable function or object with __call__ method
            config: Configuration for this stage
        """
        self.name = name
        self.processor = processor
        self.config = config or {}
        
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute this stage's processor on the inputs"""
        try:
            logger.debug(f"Executing pipeline stage: {self.name}")
            result = self.processor(inputs, **self.config)
            return result
        except Exception as e:
            logger.error(f"Error in pipeline stage '{self.name}': {e}")
            raise


class Pipeline:
    """Audio processing pipeline that chains multiple stages together"""
    
    def __init__(self, name: str = "default"):
        """
        Initialize a new pipeline
        
        Args:
            name: Name of the pipeline
        """
        self.name = name
        self.stages: List[PipelineStage] = []
        self.artifacts_dir = tempfile.mkdtemp(prefix=f"audio_pipeline_{name}_")
        
    def add_stage(self, name: str, processor: Callable, config: Optional[Dict[str, Any]] = None) -> 'Pipeline':
        """
        Add a stage to the pipeline
        
        Args:
            name: Name of the stage
            processor: Processing function or object
            config: Configuration for this stage
            
        Returns:
            Self for chaining
        """
        stage = PipelineStage(name, processor, config)
        self.stages.append(stage)
        return self
    
    def add_generator(self, generator: BaseGenerator, config: Optional[Dict[str, Any]] = None) -> 'Pipeline':
        """
        Add a generator stage to the pipeline
        
        Args:
            generator: Audio generator instance
            config: Configuration for the generator
            
        Returns:
            Self for chaining
        """
        return self.add_stage(f"generator_{generator.__class__.__name__}", generator.generate, config)
    
    def add_validator(self, validator: BaseValidator, config: Optional[Dict[str, Any]] = None) -> 'Pipeline':
        """
        Add a validator stage to the pipeline
        
        Args:
            validator: Audio validator instance
            config: Configuration for the validator
            
        Returns:
            Self for chaining
        """
        return self.add_stage(f"validator_{validator.__class__.__name__}", validator.validate, config)
    
    def add_processor(self, name: str, processor_fn: Callable, config: Optional[Dict[str, Any]] = None) -> 'Pipeline':
        """
        Add a custom processing stage to the pipeline
        
        Args:
            name: Name of the processor
            processor_fn: Processing function
            config: Configuration for the processor
            
        Returns:
            Self for chaining
        """
        return self.add_stage(f"processor_{name}", processor_fn, config)
    
    def run(self, initial_inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the complete pipeline
        
        Args:
            initial_inputs: Initial inputs to the pipeline
            
        Returns:
            Dict containing the pipeline results
        """
        if not self.stages:
            logger.warning("Running empty pipeline")
            return initial_inputs or {}
        
        # Initialize pipeline context
        context = initial_inputs or {}
        context["pipeline_id"] = str(uuid.uuid4())
        context["artifacts_dir"] = self.artifacts_dir
        context["stage_results"] = {}
        
        # Run each stage in sequence
        for i, stage in enumerate(self.stages):
            logger.info(f"Pipeline '{self.name}': Running stage {i+1}/{len(self.stages)} - {stage.name}")
            try:
                # Execute stage
                stage_result = stage(context)
                
                # Store results
                if isinstance(stage_result, dict):
                    # Update context with stage results
                    context.update(stage_result)
                    # Also store in stage_results
                    context["stage_results"][stage.name] = stage_result
                else:
                    # If not a dict, store as output of this stage
                    context["stage_results"][stage.name] = {"output": stage_result}
                    
            except Exception as e:
                logger.error(f"Pipeline '{self.name}': Error in stage {stage.name} - {e}")
                context["error"] = str(e)
                context["failed_stage"] = stage.name
                break
        
        return context
    
    def save(self, path: Union[str, Path]) -> str:
        """
        Save pipeline configuration to a file
        
        Args:
            path: Path to save the pipeline configuration
            
        Returns:
            Path where configuration was saved
        """
        path = Path(path)
        os.makedirs(path.parent, exist_ok=True)
        
        # Create serializable representation of pipeline
        config = {
            "name": self.name,
            "stages": []
        }
        
        # Currently this only saves the pipeline structure, not the actual processors
        # which would require more complex serialization
        for stage in self.stages:
            config["stages"].append({
                "name": stage.name,
                "config": stage.config
            })
        
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
            
        return str(path)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'Pipeline':
        """
        Load a pipeline configuration
        
        Note: This only loads the structure, processors need to be reattached
        
        Args:
            path: Path to the pipeline configuration file
            
        Returns:
            Pipeline instance with the configured structure
        """
        path = Path(path)
        with open(path, "r") as f:
            config = json.load(f)
            
        pipeline = cls(name=config.get("name", "loaded_pipeline"))
        # Note: This only loads the structure, processors need to be reattached
        
        return pipeline
    
    def __del__(self):
        """Clean up temporary artifacts directory"""
        import shutil
        try:
            if os.path.exists(self.artifacts_dir):
                shutil.rmtree(self.artifacts_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Failed to clean up pipeline artifacts: {e}")


# Common pipeline factories
def create_generation_pipeline(generator_name: str, config: Optional[Dict[str, Any]] = None) -> Pipeline:
    """
    Create a standard audio generation pipeline
    
    Args:
        generator_name: Name of the generator to use
        config: Configuration for the pipeline
        
    Returns:
        Configured pipeline instance
    """
    from ..core.generators import get_generator
    from ..core.validators.quality import QualityValidator
    
    config = config or {}
    generator_config = config.get("generator", {})
    validation_config = config.get("validation", {})
    
    # Create pipeline
    pipeline = Pipeline(name=f"{generator_name}_generation")
    
    # Add generator
    generator = get_generator(generator_name)
    pipeline.add_generator(generator, generator_config)
    
    # Add quality validation if enabled
    if config.get("validate_quality", True):
        quality_validator = QualityValidator()
        pipeline.add_validator(quality_validator, validation_config)
    
    return pipeline


def create_validation_pipeline(validators: List[str], config: Optional[Dict[str, Any]] = None) -> Pipeline:
    """
    Create a standard audio validation pipeline
    
    Args:
        validators: List of validator names to include
        config: Configuration for the pipeline
        
    Returns:
        Configured pipeline instance
    """
    from ..core.validators import get_validator
    
    config = config or {}
    
    # Create pipeline
    pipeline = Pipeline(name="validation_pipeline")
    
    # Add each requested validator
    for validator_name in validators:
        validator_config = config.get(validator_name, {})
        validator = get_validator(validator_name)
        pipeline.add_validator(validator, validator_config)
    
    return pipeline
