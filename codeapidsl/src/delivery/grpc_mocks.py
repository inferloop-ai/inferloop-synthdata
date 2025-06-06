# src/delivery/grpc_mocks.py
class GRPCMockGenerator:
    """Generate gRPC service definitions and mocks"""
    
    def generate_mocks(self, generated_code: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate gRPC service mocks"""
        proto_definition = self._generate_proto_definition()
        mock_responses = self._generate_mock_responses(generated_code)
        
        return {
            "proto_definition": proto_definition,
            "mock_responses": mock_responses,
            "service_implementation": self._generate_service_implementation()
        }
    
    def _generate_proto_definition(self) -> str:
        """Generate .proto file definition"""
        return '''
syntax = "proto3";

package synthcode;

service CodeGenerationService {
  rpc GenerateCode(GenerateCodeRequest) returns (GenerateCodeResponse);
  rpc ValidateCode(ValidateCodeRequest) returns (ValidateCodeResponse);
  rpc GetTemplates(GetTemplatesRequest) returns (GetTemplatesResponse);
}

message GenerateCodeRequest {
  repeated string prompts = 1;
  string language = 2;
  optional string framework = 3;
  string complexity = 4;
  int32 count = 5;
  bool include_tests = 6;
  bool include_validation = 7;
}

message GenerateCodeResponse {
  repeated GeneratedCodeSample samples = 1;
  GenerationMetadata metadata = 2;
}

message GeneratedCodeSample {
  string id = 1;
  string prompt = 2;
  string code = 3;
  string language = 4;
  CodeMetadata metadata = 5;
}

message CodeMetadata {
  int32 lines_of_code = 1;
  string estimated_complexity = 2;
  repeated string dependencies = 3;
}

message GenerationMetadata {
  int32 total_generated = 1;
  string language = 2;
  optional string framework = 3;
  bool validation_enabled = 4;
}

message ValidateCodeRequest {
  string code = 1;
  string language = 2;
}

message ValidateCodeResponse {
  bool valid = 1;
  repeated string errors = 2;
  repeated string warnings = 3;
}

message GetTemplatesRequest {}

message GetTemplatesResponse {
  repeated string languages = 1;
  map<string, FrameworkList> frameworks = 2;
  repeated string complexity_levels = 3;
}

message FrameworkList {
  repeated string frameworks = 1;
}
'''
    
    def _generate_mock_responses(self, generated_code: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate mock responses for gRPC calls"""
        return [
            {
                "method": "GenerateCode",
                "response": {
                    "samples": generated_code[:5],  # Limit for demo
                    "metadata": {
                        "total_generated": len(generated_code),
                        "language": "python",
                        "validation_enabled": True
                    }
                }
            },
            {
                "method": "ValidateCode",
                "response": {
                    "valid": True,
                    "errors": [],
                    "warnings": ["Line too long"]
                }
            }
        ]
    
    def _generate_service_implementation(self) -> str:
        """Generate Python gRPC service implementation"""
        return '''
import grpc
from concurrent import futures
import synthcode_pb2
import synthcode_pb2_grpc

class CodeGenerationServiceImpl(synthcode_pb2_grpc.CodeGenerationServiceServicer):
    
    def GenerateCode(self, request, context):
        # Mock implementation
        samples = []
        for i, prompt in enumerate(request.prompts[:request.count]):
            sample = synthcode_pb2.GeneratedCodeSample(
                id=f"sample_{i}",
                prompt=prompt,
                code=f"def generated_function():\\n    # {prompt}\\n    pass",
                language=request.language,
                metadata=synthcode_pb2.CodeMetadata(
                    lines_of_code=3,
                    estimated_complexity="low",
                    dependencies=[]
                )
            )
            samples.append(sample)
        
        metadata = synthcode_pb2.GenerationMetadata(
            total_generated=len(samples),
            language=request.language,
            framework=request.framework,
            validation_enabled=request.include_validation
        )
        
        return synthcode_pb2.GenerateCodeResponse(
            samples=samples,
            metadata=metadata
        )
    
    def ValidateCode(self, request, context):
        # Mock validation
        return synthcode_pb2.ValidateCodeResponse(
            valid=True,
            errors=[],
            warnings=[]
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    synthcode_pb2_grpc.add_CodeGenerationServiceServicer_to_server(
        CodeGenerationServiceImpl(), server
    )
    listen_addr = '[::]:50051'
    server.add_insecure_port(listen_addr)
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
'''

