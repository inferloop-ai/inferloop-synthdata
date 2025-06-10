#!/bin/bash

# Autonomous Driving Pipeline Example

echo "🚗 Running Autonomous Driving Video Synthesis Pipeline Example"

# Configuration
API_URL="http://localhost:8080"
PIPELINE_CONFIG='{
  "vertical": "autonomous_vehicles",
  "data_sources": [
    {
      "source_type": "web",
      "url": "https://example.com/traffic-datasets",
      "quality_filters": {
        "min_resolution": "1080p",
        "min_duration": 30
      }
    }
  ],
  "generation_config": {
    "engine": "unreal",
    "scenarios": ["highway_driving", "urban_navigation", "weather_conditions"],
    "duration_seconds": 120,
    "resolution": "1920x1080",
    "weather_conditions": ["clear", "rain", "fog"],
    "traffic_density": "medium",
    "vehicle_count": 15
  },
  "quality_requirements": {
    "min_label_accuracy": 0.95,
    "max_frame_lag_ms": 100,
    "min_object_detection_precision": 0.95,
    "safety_critical": true
  },
  "delivery_config": {
    "format": "mp4",
    "delivery_method": "streaming",
    "include_annotations": true
  }
}'

# Start pipeline
echo "🚀 Starting autonomous driving pipeline..."
RESPONSE=$(curl -s -X POST "$API_URL/api/v1/pipeline/start" \
  -H "Content-Type: application/json" \
  -d "$PIPELINE_CONFIG")

PIPELINE_ID=$(echo $RESPONSE | jq -r '.pipeline_id')
echo "📋 Pipeline ID: $PIPELINE_ID"

# Monitor progress
echo "📊 Monitoring pipeline progress..."
while true; do
  STATUS=$(curl -s "$API_URL/api/v1/pipeline/status/$PIPELINE_ID")
  CURRENT_STATUS=$(echo $STATUS | jq -r '.status')
  PROGRESS=$(echo $STATUS | jq -r '.progress')
  STAGE=$(echo $STATUS | jq -r '.current_stage')
  
  echo "Status: $CURRENT_STATUS | Stage: $STAGE | Progress: $PROGRESS%"
  
  if [ "$CURRENT_STATUS" = "completed" ] || [ "$CURRENT_STATUS" = "failed" ]; then
    break
  fi
  
  sleep 10
done

echo "✅ Pipeline $CURRENT_STATUS!"

# Show final results
if [ "$CURRENT_STATUS" = "completed" ]; then
  echo "🎉 Autonomous driving video synthesis completed successfully!"
  echo "📊 Final status:"
  curl -s "$API_URL/api/v1/pipeline/status/$PIPELINE_ID" | jq '.'
else
  echo "❌ Pipeline failed. Check logs for details."
fi
