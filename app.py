import json

from video_generator import VideoGenerator
from settings import ConfigLoader, VideoConfig

class VideoGeneratorApp:
    """Main application for video generation."""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the application with the specified configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        
    def run(self) -> None:
        """Run the video generation process."""
        raw_config = ConfigLoader.load_config(self.config_path)
        config = VideoConfig.from_dict(raw_config)
        generator = VideoGenerator(temperature=config.script.temperature)
        generator.generate_videos(video_settings=config)
        
