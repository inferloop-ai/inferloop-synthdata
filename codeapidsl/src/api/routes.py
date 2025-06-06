# src/api/routes.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
from ..generators.code_llama_generator import CodeLlamaGenerator
from ..generators.starcoder_generator import StarCoderGenerator
from ..validators.syntax_validator import SyntaxValidator
from ..validators.compilation_validator import CompilationValidator
from ..delivery.formatters import JSONLFormatter, CSVFormatter
from ..delivery.grpc_mocks import GRPCMockGenerator

app = FastAPI(title="Synthetic Code Generation API", version="1.0.0")

# Pydantic models
class CodeGenerationRequest(BaseModel):
    prompts: List[str]
    language: str = "python"
    framework: Optional[str] = None
    complexity: str = "medium"
    count: int = 100
    include_tests: bool = True
    include_validation: bool = True
    output_format: str = "jsonl"  # jsonl, csv, grpc

class GenerationResponse(BaseModel):
    generated_code: List[Dict[str, Any]]
    validation_results: List[Dict[str, Any]]
    metadata: Dict[str, Any]

# Dependencies
def get_generator(language: str, framework: Optional[str] = None):
    """Factory function to get appropriate generator"""
    config = GenerationConfig(
        language=language,
        framework=framework,
        complexity="medium",
        count=100
    )
    
    if language in ["python", "javascript", "typescript"]:
        return CodeLlamaGenerator(config)
    else:
        return StarCoderGenerator(config)

def get_validators():
    """Get validation pipeline"""
    return {
        "syntax": SyntaxValidator(),
        "compilation": CompilationValidator()
    }

@app.post("/generate/code", response_model=GenerationResponse)
async def generate_code(
    request: CodeGenerationRequest,
    generator = Depends(get_generator),
    validators = Depends(get_validators)
):
    """Generate synthetic code based on prompts"""
    try:
        # Configure generator
        generator.config.language = request.language
        generator.config.framework = request.framework
        generator.config.complexity = request.complexity
        generator.config.count = request.count
        
        # Generate code
        generated_code = generator.generate_batch(request.prompts)
        
        # Validate if requested
        validation_results = []
        if request.include_validation:
            for code_sample in generated_code:
                syntax_result = validators["syntax"].validate_code(
                    code_sample["code"], 
                    request.language
                )
                compilation_result = validators["compilation"].validate_compilation(
                    code_sample["code"], 
                    request.language
                )
                
                validation_results.append({
                    "id": code_sample["id"],
                    "syntax_validation": syntax_result,
                    "compilation_validation": compilation_result
                })
        
        # Format output
        if request.output_format == "jsonl":
            formatter = JSONLFormatter()
        elif request.output_format == "csv":
            formatter = CSVFormatter()
        elif request.output_format == "grpc":
            grpc_generator = GRPCMockGenerator()
            return grpc_generator.generate_mocks(generated_code)
        
        formatted_data = formatter.format(generated_code, validation_results)
        
        return GenerationResponse(
            generated_code=generated_code,
            validation_results=validation_results,
            metadata={
                "total_generated": len(generated_code),
                "language": request.language,
                "framework": request.framework,
                "validation_enabled": request.include_validation,
                "output_format": request.output_format
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generate/code/templates")
async def get_code_templates():
    """Get available code generation templates"""
    return {
        "languages": ["python", "javascript", "typescript", "java", "go", "rust"],
        "frameworks": {
            "python": ["fastapi", "django", "flask"],
            "javascript": ["express", "koa", "nestjs"],
            "typescript": ["nestjs", "express"],
            "java": ["spring", "springboot"],
            "go": ["gin", "echo", "gorilla"]
        },
        "complexity_levels": ["low", "medium", "high"],
        "output_formats": ["jsonl", "csv", "grpc"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "synthetic-code-generator"}
