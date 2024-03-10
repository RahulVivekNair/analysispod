import json
import ffmpeg

def extract_metadata(video_path, output_path):
  try:
    print(f"Extracting metadata for {video_path}...")
    # Extract metadata using ffprobe
    probe = ffmpeg.probe(video_path)

    # Extract video stream information
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)

    # Extract audio stream information
    audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)

    # Extract metadata with units
    metadata = {
      'format': probe['format']['format_name'],
      'duration': f"{float(probe['format']['duration']):.2f} seconds",  # Format duration with 2 decimals
      'size': f"{int(probe['format']['size']) / (1024 * 1024):.2f} MB",  # Convert size to MB with 2 decimals
      'bit_rate': f"{int(probe['format']['bit_rate']) >> 3} kbps",  # Convert bit_rate to kbps
      'video': {
        'codec': video_stream['codec_name'],
        'width': f"{int(video_stream['width'])} pixels",
        'height': f"{int(video_stream['height'])} pixels",
        'frame_rate': f"{eval(video_stream['avg_frame_rate'])} fps",
        'bit_rate': f"{int(video_stream['bit_rate']) >> 3} kbps",  # Convert bit_rate to kbps
      } if video_stream else None,
      'audio': {
        'codec': audio_stream['codec_name'],
        'sample_rate': f"{int(audio_stream['sample_rate'])} Hz",
        'channels': int(audio_stream['channels']),
        'bit_rate': f"{int(audio_stream['bit_rate']) >> 3} kbps",  # Convert bit_rate to kbps
      } if audio_stream else None,
    }

    # Save metadata to a JSON file
    with open(output_path, 'w') as f:
      json.dump(metadata, f, indent=4)

    return metadata

  except Exception as e:
    print(f"Error extracting metadata: {str(e)}")
    return None
