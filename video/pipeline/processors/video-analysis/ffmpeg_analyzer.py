"""
FFmpeg-based Video Analysis Module
Part of the Inferloop SynthData Video Pipeline

This module provides advanced video analysis capabilities using FFmpeg,
focusing on codec analysis, stream processing, and format conversion.
"""

import subprocess
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import re
import tempfile
import shutil
from dataclasses import dataclass
from datetime import timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class StreamInfo:
    """Information about a video/audio stream"""
    index: int
    codec_name: str
    codec_type: str
    bit_rate: Optional[int]
    duration: Optional[float]
    nb_frames: Optional[int]
    width: Optional[int]
    height: Optional[int]
    fps: Optional[float]
    pix_fmt: Optional[str]
    sample_rate: Optional[int]
    channels: Optional[int]
    
class FFmpegAnalyzer:
    """
    Video analysis using FFmpeg for the synthetic data pipeline.
    Provides deep codec analysis, quality metrics, and format conversions.
    """
    
    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe"):
        """
        Initialize the FFmpeg analyzer.
        
        Args:
            ffmpeg_path: Path to ffmpeg executable
            ffprobe_path: Path to ffprobe executable
        """
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        
        # Verify FFmpeg installation
        if not self._verify_ffmpeg():
            raise RuntimeError("FFmpeg not found or not accessible")
    
    def _verify_ffmpeg(self) -> bool:
        """Verify FFmpeg and FFprobe are installed and accessible"""
        try:
            subprocess.run([self.ffmpeg_path, "-version"], capture_output=True, check=True)
            subprocess.run([self.ffprobe_path, "-version"], capture_output=True, check=True)
            logger.info("FFmpeg and FFprobe verified successfully")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("FFmpeg or FFprobe not found")
            return False
    
    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """
        Comprehensive video analysis using FFprobe
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with detailed video analysis
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Get basic video information
        probe_data = self._probe_video(video_path)
        
        # Extract streams information
        streams = self._parse_streams(probe_data)
        
        # Get quality metrics
        quality_metrics = self._analyze_quality_metrics(video_path)
        
        # Get scene detection
        scene_changes = self._detect_scene_changes(video_path)
        
        # Compile results
        results = {
            "file_path": video_path,
            "file_size_mb": Path(video_path).stat().st_size / (1024 * 1024),
            "format": probe_data.get("format", {}),
            "streams": streams,
            "quality_metrics": quality_metrics,
            "scene_changes": scene_changes,
            "metadata": self._extract_metadata(probe_data)
        }
        
        return results
    
    def _probe_video(self, video_path: str) -> Dict[str, Any]:
        """Use ffprobe to get video information"""
        cmd = [
            self.ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"FFprobe failed: {e}")
            raise
    
    def _parse_streams(self, probe_data: Dict[str, Any]) -> Dict[str, List[StreamInfo]]:
        """Parse stream information from probe data"""
        streams = {"video": [], "audio": [], "subtitle": [], "data": []}
        
        for stream in probe_data.get("streams", []):
            codec_type = stream.get("codec_type")
            
            if codec_type == "video":
                info = StreamInfo(
                    index=stream.get("index"),
                    codec_name=stream.get("codec_name"),
                    codec_type=codec_type,
                    bit_rate=int(stream.get("bit_rate", 0)) if stream.get("bit_rate") else None,
                    duration=float(stream.get("duration", 0)) if stream.get("duration") else None,
                    nb_frames=int(stream.get("nb_frames", 0)) if stream.get("nb_frames") else None,
                    width=stream.get("width"),
                    height=stream.get("height"),
                    fps=self._parse_fps(stream.get("r_frame_rate")),
                    pix_fmt=stream.get("pix_fmt"),
                    sample_rate=None,
                    channels=None
                )
                streams["video"].append(info)
                
            elif codec_type == "audio":
                info = StreamInfo(
                    index=stream.get("index"),
                    codec_name=stream.get("codec_name"),
                    codec_type=codec_type,
                    bit_rate=int(stream.get("bit_rate", 0)) if stream.get("bit_rate") else None,
                    duration=float(stream.get("duration", 0)) if stream.get("duration") else None,
                    nb_frames=None,
                    width=None,
                    height=None,
                    fps=None,
                    pix_fmt=None,
                    sample_rate=int(stream.get("sample_rate", 0)) if stream.get("sample_rate") else None,
                    channels=stream.get("channels")
                )
                streams["audio"].append(info)
                
            elif codec_type in ["subtitle", "data"]:
                info = StreamInfo(
                    index=stream.get("index"),
                    codec_name=stream.get("codec_name"),
                    codec_type=codec_type,
                    bit_rate=None,
                    duration=None,
                    nb_frames=None,
                    width=None,
                    height=None,
                    fps=None,
                    pix_fmt=None,
                    sample_rate=None,
                    channels=None
                )
                streams[codec_type].append(info)
        
        return streams
    
    def _parse_fps(self, fps_string: Optional[str]) -> Optional[float]:
        """Parse FPS from fraction string (e.g., '30/1' -> 30.0)"""
        if not fps_string:
            return None
            
        try:
            if '/' in fps_string:
                num, den = map(int, fps_string.split('/'))
                return num / den if den != 0 else None
            return float(fps_string)
        except:
            return None
    
    def _analyze_quality_metrics(self, video_path: str) -> Dict[str, Any]:
        """Analyze video quality metrics"""
        metrics = {}
        
        # Get bitrate variability
        metrics["bitrate_stats"] = self._analyze_bitrate(video_path)
        
        # Get keyframe intervals
        metrics["keyframe_intervals"] = self._analyze_keyframes(video_path)
        
        # Get black frames
        metrics["black_frames"] = self._detect_black_frames(video_path)
        
        # Get silence detection for audio
        metrics["silence_periods"] = self._detect_silence(video_path)
        
        return metrics
    
    def _analyze_bitrate(self, video_path: str) -> Dict[str, float]:
        """Analyze bitrate statistics"""
        cmd = [
            self.ffprobe_path,
            "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "packet=size,pts_time",
            "-of", "json",
            video_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            # Calculate bitrate over time windows
            packets = data.get("packets", [])
            if not packets:
                return {}
            
            # Group packets by second
            bitrates_per_second = {}
            for packet in packets:
                if "pts_time" in packet and "size" in packet:
                    second = int(float(packet["pts_time"]))
                    size = int(packet["size"])
                    
                    if second not in bitrates_per_second:
                        bitrates_per_second[second] = 0
                    bitrates_per_second[second] += size * 8  # Convert to bits
            
            if bitrates_per_second:
                bitrates = list(bitrates_per_second.values())
                return {
                    "average_bitrate": sum(bitrates) / len(bitrates),
                    "min_bitrate": min(bitrates),
                    "max_bitrate": max(bitrates),
                    "bitrate_variance": self._calculate_variance(bitrates)
                }
            
        except Exception as e:
            logger.warning(f"Failed to analyze bitrate: {e}")
        
        return {}
    
    def _analyze_keyframes(self, video_path: str) -> Dict[str, Any]:
        """Analyze keyframe intervals"""
        cmd = [
            self.ffprobe_path,
            "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "packet=pts_time,flags",
            "-of", "json",
            video_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            keyframe_times = []
            for packet in data.get("packets", []):
                if "K" in packet.get("flags", "") and "pts_time" in packet:
                    keyframe_times.append(float(packet["pts_time"]))
            
            if len(keyframe_times) > 1:
                intervals = [keyframe_times[i+1] - keyframe_times[i] 
                           for i in range(len(keyframe_times)-1)]
                
                return {
                    "total_keyframes": len(keyframe_times),
                    "average_interval": sum(intervals) / len(intervals),
                    "min_interval": min(intervals),
                    "max_interval": max(intervals)
                }
                
        except Exception as e:
            logger.warning(f"Failed to analyze keyframes: {e}")
        
        return {}
    
    def _detect_black_frames(self, video_path: str) -> List[Dict[str, float]]:
        """Detect black frames in video"""
        cmd = [
            self.ffmpeg_path,
            "-i", video_path,
            "-vf", "blackdetect=d=0.1:pix_th=0.10",
            "-an",
            "-f", "null",
            "-"
        ]
        
        black_frames = []
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse black frame detection from stderr
            for line in result.stderr.split('\n'):
                if "blackdetect" in line:
                    # Extract start and duration
                    start_match = re.search(r'black_start:(\d+\.?\d*)', line)
                    end_match = re.search(r'black_end:(\d+\.?\d*)', line)
                    
                    if start_match and end_match:
                        start = float(start_match.group(1))
                        end = float(end_match.group(1))
                        black_frames.append({
                            "start": start,
                            "end": end,
                            "duration": end - start
                        })
                        
        except Exception as e:
            logger.warning(f"Failed to detect black frames: {e}")
        
        return black_frames
    
    def _detect_silence(self, video_path: str) -> List[Dict[str, float]]:
        """Detect silence periods in audio"""
        cmd = [
            self.ffmpeg_path,
            "-i", video_path,
            "-af", "silencedetect=n=-30dB:d=0.5",
            "-vn",
            "-f", "null",
            "-"
        ]
        
        silence_periods = []
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse silence detection from stderr
            for line in result.stderr.split('\n'):
                if "silence_start" in line:
                    match = re.search(r'silence_start: (\d+\.?\d*)', line)
                    if match:
                        start = float(match.group(1))
                        silence_periods.append({"start": start})
                elif "silence_end" in line and silence_periods:
                    match = re.search(r'silence_end: (\d+\.?\d*)', line)
                    if match and "end" not in silence_periods[-1]:
                        end = float(match.group(1))
                        silence_periods[-1]["end"] = end
                        silence_periods[-1]["duration"] = end - silence_periods[-1]["start"]
                        
        except Exception as e:
            logger.warning(f"Failed to detect silence: {e}")
        
        return [s for s in silence_periods if "end" in s]
    
    def _detect_scene_changes(self, video_path: str) -> List[Dict[str, Any]]:
        """Detect scene changes in video"""
        cmd = [
            self.ffmpeg_path,
            "-i", video_path,
            "-vf", "select='gt(scene,0.4)',showinfo",
            "-vsync", "vfr",
            "-f", "null",
            "-"
        ]
        
        scene_changes = []
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse scene changes from stderr
            for line in result.stderr.split('\n'):
                if "showinfo" in line and "pts_time" in line:
                    pts_match = re.search(r'pts_time:(\d+\.?\d*)', line)
                    if pts_match:
                        scene_changes.append({
                            "timestamp": float(pts_match.group(1)),
                            "frame": len(scene_changes)
                        })
                        
        except Exception as e:
            logger.warning(f"Failed to detect scene changes: {e}")
        
        return scene_changes
    
    def _extract_metadata(self, probe_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from probe data"""
        format_data = probe_data.get("format", {})
        
        metadata = {
            "duration": float(format_data.get("duration", 0)),
            "size": int(format_data.get("size", 0)),
            "bit_rate": int(format_data.get("bit_rate", 0)),
            "format_name": format_data.get("format_name", ""),
            "format_long_name": format_data.get("format_long_name", ""),
            "nb_streams": format_data.get("nb_streams", 0),
            "nb_programs": format_data.get("nb_programs", 0),
            "tags": format_data.get("tags", {})
        }
        
        return metadata
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values"""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def extract_frames(self, video_path: str, output_dir: str, 
                      interval: float = 1.0, format: str = "jpg") -> List[str]:
        """
        Extract frames from video at specified intervals
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save extracted frames
            interval: Interval between frames in seconds
            format: Output image format
            
        Returns:
            List of extracted frame paths
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        cmd = [
            self.ffmpeg_path,
            "-i", video_path,
            "-vf", f"fps=1/{interval}",
            "-q:v", "2",
            f"{output_dir}/frame_%06d.{format}"
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            
            # Get list of extracted frames
            frame_paths = sorted(Path(output_dir).glob(f"*.{format}"))
            return [str(p) for p in frame_paths]
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract frames: {e}")
            raise
    
    def generate_thumbnail(self, video_path: str, output_path: str, 
                         timestamp: Optional[float] = None) -> str:
        """
        Generate thumbnail from video
        
        Args:
            video_path: Path to video file
            output_path: Path to save thumbnail
            timestamp: Specific timestamp for thumbnail (seconds)
            
        Returns:
            Path to generated thumbnail
        """
        if timestamp is None:
            # Use 10% of video duration
            probe_data = self._probe_video(video_path)
            duration = float(probe_data.get("format", {}).get("duration", 10))
            timestamp = duration * 0.1
        
        cmd = [
            self.ffmpeg_path,
            "-ss", str(timestamp),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to generate thumbnail: {e}")
            raise
    
    def convert_video(self, input_path: str, output_path: str, 
                     codec: str = "h264", quality: str = "medium",
                     resolution: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert video to different format/codec
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            codec: Video codec (h264, h265, vp9)
            quality: Quality preset (low, medium, high)
            resolution: Target resolution (e.g., "1920x1080")
            
        Returns:
            Conversion statistics
        """
        # Build FFmpeg command
        cmd = [self.ffmpeg_path, "-i", input_path]
        
        # Add codec settings
        codec_map = {
            "h264": "libx264",
            "h265": "libx265",
            "vp9": "libvpx-vp9"
        }
        cmd.extend(["-c:v", codec_map.get(codec, codec)])
        
        # Add quality settings
        quality_map = {
            "low": {"crf": "28", "preset": "faster"},
            "medium": {"crf": "23", "preset": "medium"},
            "high": {"crf": "18", "preset": "slow"}
        }
        
        if codec in ["h264", "h265"]:
            settings = quality_map.get(quality, quality_map["medium"])
            cmd.extend(["-crf", settings["crf"], "-preset", settings["preset"]])
        
        # Add resolution if specified
        if resolution:
            cmd.extend(["-s", resolution])
        
        # Copy audio
        cmd.extend(["-c:a", "copy"])
        
        # Output file
        cmd.append(output_path)
        
        # Run conversion
        start_time = Path(input_path).stat().st_mtime
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            
            # Calculate statistics
            input_size = Path(input_path).stat().st_size
            output_size = Path(output_path).stat().st_size
            
            return {
                "input_file": input_path,
                "output_file": output_path,
                "input_size_mb": input_size / (1024 * 1024),
                "output_size_mb": output_size / (1024 * 1024),
                "compression_ratio": input_size / output_size if output_size > 0 else 0,
                "codec": codec,
                "quality": quality,
                "resolution": resolution
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to convert video: {e}")
            raise

# Example usage
if __name__ == "__main__":
    analyzer = FFmpegAnalyzer()
    
    # Analyze video
    results = analyzer.analyze_video("sample_video.mp4")
    print(json.dumps(results, indent=2))
    
    # Extract frames
    frames = analyzer.extract_frames("sample_video.mp4", "extracted_frames", interval=2.0)
    print(f"Extracted {len(frames)} frames")
    
    # Generate thumbnail
    thumbnail = analyzer.generate_thumbnail("sample_video.mp4", "thumbnail.jpg")
    print(f"Generated thumbnail: {thumbnail}")