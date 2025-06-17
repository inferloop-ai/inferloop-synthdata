#!/bin/bash
# TSIoT API Examples using cURL
# These examples demonstrate how to interact with the TSIoT API using cURL commands.

# Set base URL (change as needed)
BASE_URL="http://localhost:8080"
API_KEY="your-api-key-here" # Optional

# Helper function to make authenticated requests
curl_api() {
    local method="$1"
    local endpoint="$2"
    local data="$3"
    
    local headers=()
    headers+=("-H" "Content-Type: application/json")
    
    if [[ -n "$API_KEY" ]]; then
        headers+=("-H" "X-API-Key: $API_KEY")
    fi
    
    if [[ -n "$data" ]]; then
        curl -s -X "$method" "${BASE_URL}${endpoint}" "${headers[@]}" -d "$data"
    else
        curl -s -X "$method" "${BASE_URL}${endpoint}" "${headers[@]}"
    fi
}

echo "TSIoT API Examples"
echo "=================="
echo

# Health Check
echo "1. Health Check"
echo "----------------"
echo "Request: GET /health"
response=$(curl_api "GET" "/health")
echo "Response: $response"
echo

# Readiness Check
echo "2. Readiness Check"
echo "------------------"
echo "Request: GET /ready"
response=$(curl_api "GET" "/ready")
echo "Response: $response"
echo

# Basic Data Generation
echo "3. Basic Data Generation"
echo "------------------------"
echo "Request: POST /api/v1/generate"
request_data='{
  "generator": "statistical",
  "sensor_type": "temperature",
  "points": 100,
  "frequency": "1m"
}'
echo "Data: $request_data"
response=$(curl_api "POST" "/api/v1/generate" "$request_data")
echo "Response: $response"
echo

# Advanced Data Generation with Anomalies
echo "4. Advanced Data Generation"
echo "---------------------------"
echo "Request: POST /api/v1/generate"
request_data='{
  "generator": "timegan",
  "sensor_type": "energy",
  "sensor_id": "sensor_001",
  "points": 1000,
  "frequency": "5m",
  "anomalies": 10,
  "anomaly_magnitude": 2.5,
  "seasonality": true,
  "seasonal_period": "24h",
  "seed": 42,
  "metadata": {
    "location": "building_a",
    "floor": 2
  }
}'
echo "Data: $request_data"
response=$(curl_api "POST" "/api/v1/generate" "$request_data")
echo "Response: $response"
echo

# Time Range Generation
echo "5. Time Range Generation"
echo "------------------------"
echo "Request: POST /api/v1/generate"
start_time=$(date -u -d '7 days ago' +%Y-%m-%dT%H:%M:%SZ)
end_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)
request_data=$(cat <<EOF
{
  "generator": "arima",
  "sensor_type": "humidity",
  "start_time": "$start_time",
  "end_time": "$end_time",
  "frequency": "1h"
}
EOF
)
echo "Data: $request_data"
response=$(curl_api "POST" "/api/v1/generate" "$request_data")
echo "Response: $response"
echo

# Privacy-Preserving Generation
echo "6. Privacy-Preserving Generation"
echo "--------------------------------"
echo "Request: POST /api/v1/generate"
request_data='{
  "generator": "ydata",
  "sensor_type": "medical",
  "points": 5000,
  "privacy": true,
  "privacy_epsilon": 0.5,
  "noise_level": 0.1
}'
echo "Data: $request_data"
response=$(curl_api "POST" "/api/v1/generate" "$request_data")
echo "Response: $response"
echo

# Data Validation
echo "7. Data Validation"
echo "------------------"
echo "Request: POST /api/v1/validate"
request_data='{
  "data": {
    "points": [
      {
        "timestamp": "2024-01-15T10:00:00Z",
        "sensor_id": "test_sensor",
        "sensor_type": "temperature",
        "value": 25.5
      },
      {
        "timestamp": "2024-01-15T10:01:00Z",
        "sensor_id": "test_sensor",
        "sensor_type": "temperature",
        "value": 25.7
      }
    ]
  },
  "metrics": ["basic", "distribution"],
  "statistical_tests": ["ks"]
}'
echo "Data: $request_data"
response=$(curl_api "POST" "/api/v1/validate" "$request_data")
echo "Response: $response"
echo

# Data Analysis
echo "8. Data Analysis"
echo "----------------"
echo "Request: POST /api/v1/analyze"
request_data='{
  "data": {
    "points": [
      {
        "timestamp": "2024-01-15T10:00:00Z",
        "sensor_id": "test_sensor",
        "sensor_type": "temperature",
        "value": 25.5
      },
      {
        "timestamp": "2024-01-15T10:01:00Z",
        "sensor_id": "test_sensor",
        "sensor_type": "temperature",
        "value": 25.7
      }
    ]
  },
  "detect_anomalies": true,
  "decompose": true,
  "forecast": true,
  "forecast_periods": 10
}'
echo "Data: $request_data"
response=$(curl_api "POST" "/api/v1/analyze" "$request_data")
echo "Response: $response"
echo

# List Generators
echo "9. List Generators"
echo "------------------"
echo "Request: GET /api/v1/generators"
response=$(curl_api "GET" "/api/v1/generators")
echo "Response: $response"
echo

# List Sensor Types
echo "10. List Sensor Types"
echo "---------------------"
echo "Request: GET /api/v1/sensors"
response=$(curl_api "GET" "/api/v1/sensors")
echo "Response: $response"
echo

# List Jobs
echo "11. List Jobs"
echo "-------------"
echo "Request: GET /api/v1/jobs"
response=$(curl_api "GET" "/api/v1/jobs?limit=5")
echo "Response: $response"
echo

# Stream Data (start stream in background)
echo "12. Stream Data"
echo "---------------"
echo "Request: GET /api/v1/stream"
echo "Starting stream for 10 seconds..."
stream_url="${BASE_URL}/api/v1/stream?generator=statistical&sensor_type=temperature&frequency=1s"
if [[ -n "$API_KEY" ]]; then
    timeout 10s curl -s -H "X-API-Key: $API_KEY" "$stream_url" | head -10
else
    timeout 10s curl -s "$stream_url" | head -10
fi
echo
echo "Stream ended."
echo

# File Upload Example (if you have a data file)
echo "13. File Upload Validation"
echo "--------------------------"
echo "Note: This example requires a data file. Skipping for now."
echo "Example command:"
echo 'curl -X POST "${BASE_URL}/api/v1/validate" \'
echo '  -H "Content-Type: multipart/form-data" \'
echo '  -F "data=@your_data_file.csv" \'
echo '  -F "metrics=[\"basic\",\"temporal\"]"'
echo

# Batch Processing Example
echo "14. Batch Generation"
echo "--------------------"
echo "Generating data for multiple sensors..."
for i in {1..3}; do
    sensor_id="sensor_$(printf "%03d" $i)"
    echo "Generating data for $sensor_id"
    request_data=$(cat <<EOF
{
  "generator": "statistical",
  "sensor_type": "temperature",
  "sensor_id": "$sensor_id",
  "points": 100,
  "frequency": "1m"
}
EOF
    )
    response=$(curl_api "POST" "/api/v1/generate" "$request_data")
    job_id=$(echo "$response" | grep -o '"job_id":"[^"]*"' | cut -d'"' -f4)
    echo "Job ID: $job_id"
done
echo

# Error Handling Example
echo "15. Error Handling"
echo "------------------"
echo "Request with invalid parameters:"
request_data='{
  "generator": "invalid_generator",
  "sensor_type": "temperature",
  "points": -1
}'
echo "Data: $request_data"
response=$(curl_api "POST" "/api/v1/generate" "$request_data")
echo "Response: $response"
echo

# Custom Parameters Example
echo "16. Custom Parameters"
echo "---------------------"
echo "Request: POST /api/v1/generate"
request_data='{
  "generator": "statistical",
  "sensor_type": "custom",
  "points": 500,
  "custom_params": {
    "distribution": "exponential",
    "scale": 2.0,
    "offset": 10.0,
    "seasonality_components": [
      {"period": "24h", "amplitude": 5.0},
      {"period": "7d", "amplitude": 2.0}
    ]
  }
}'
echo "Data: $request_data"
response=$(curl_api "POST" "/api/v1/generate" "$request_data")
echo "Response: $response"
echo

echo "Examples completed!"
echo
echo "Useful cURL Tips:"
echo "- Add -v flag for verbose output"
echo "- Add -i flag to include response headers"
echo "- Add -o filename to save response to file"
echo "- Use jq to format JSON responses: curl ... | jq ."
echo "- Use -w '%{http_code}\n' to show HTTP status code"
echo
echo "Example with formatting:"
echo "curl -s '${BASE_URL}/health' | jq ."
echo
echo "Example saving to file:"
echo "curl -s '${BASE_URL}/api/v1/generate' -d '{}' -o generated_data.json"