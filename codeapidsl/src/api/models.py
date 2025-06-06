class GenerateCodeRequest(BaseModel):
    prompts: List[str]
    language: str = "python"
    framework: Optional[str] = None
    complexity: str = "medium"
    count: int = 10
    include_validation: bool = True
    output_format: str = "jsonl"
    
    class Config:
        schema_extra = {
            "example": {
                "prompts": ["fibonacci function", "binary search algorithm"],
                "language": "python",
                "framework": "fastapi",
                "count": 5,
                "include_validation": True
            }
        }

class GenerateCodeResponse(BaseModel):
    samples: List[Dict[str, Any]]
    metadata: Dict[str, Any]
