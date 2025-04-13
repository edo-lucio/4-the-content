import json
from typing import Dict, List, Union, Optional ,Any
from dataclasses import dataclass, field
from pathlib import Path


class VoiceSettings:
    """Voice configuration settings for video generation."""
    
    def __init__(self):
        self.use = None
    
    def __setattr__(self, name, value):
        # Dynamically assign the attribute
        super().__setattr__(name, value)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoiceSettings":
        # Create a new VoiceSettings instance
        instance = cls()
        
        # Get the selected voice type
        voice_type = data.get("use")
        instance.use = voice_type
        
        # If the voice type exists in the data, add its attributes to the instance
        if voice_type and voice_type in data:
            voice_data = data.get(voice_type, {})
            for key, value in voice_data.items():
                setattr(instance, key, value)
        
        return instance


@dataclass
class ImageSettings:
    """Image configuration settings for video generation."""
    from_scenes: bool = False
    n_images: int = 4
    height: int = 1080
    width: int = 1920
    sections: List[str] = field(default_factory=lambda: ["introduction", "core explanation", "conclusion"])
    lora: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageSettings":
        """Create ImageSettings from a dictionary, using default values where needed."""
        defaults = cls()
        return cls(
            from_scenes=data.get("from_scenes", defaults.from_scenes),
            n_images=data.get("n_images", defaults.n_images),
            height=data.get("height", defaults.height),
            width=data.get("width", defaults.width),
            sections=data.get("sections", defaults.sections),
            lora=data.get("lora", defaults.lora)
        )


@dataclass
class ScriptSettings:
    """Script configuration settings for video generation."""
    topic: str = ""
    title_description: str = ""
    script_description: str = ""
    images_description: str = ""
    thumbnail_description: str = ""
    temperature: float = 1.0
    handle: str = ""
    n_output_videos: int = 1
    max_reference_titles: int = 10
    max_reference_transcripts: int = 5
    generate_scripts: bool = True
    generate_images: bool = True
    generate_audios: bool = True
    from_scenes: bool = False
    n_images: int = 1
    sections: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScriptSettings":
        """Create ScriptSettings from a dictionary."""
        defaults = cls()
        return cls(
            topic=data.get("topic", defaults.topic),
            title_description=data.get("title_description", defaults.title_description),
            script_description=data.get("script_description", defaults.script_description),
            images_description=data.get("images_description", defaults.images_description),
            thumbnail_description=data.get("thumbnail_description", defaults.thumbnail_description),
            temperature=data.get("temperature", defaults.temperature),
            handle=data.get("handle", defaults.handle),
            n_output_videos=data.get("n_output_videos", defaults.n_output_videos),
            max_reference_titles=data.get("max_reference_titles", defaults.max_reference_titles),
            max_reference_transcripts=data.get("max_reference_transcripts", defaults.max_reference_transcripts),
            generate_scripts=data.get("generate_scripts", defaults.generate_scripts),
            generate_images=data.get("generate_images", defaults.generate_images),
            generate_audios=data.get("generate_audios", defaults.generate_audios),
            from_scenes=data.get("from_scenes", defaults.from_scenes),
            n_images=data.get("n_images", defaults.n_images),
            sections=data.get("sections", defaults.sections),
        )


@dataclass
class VideoConfig:
    """Complete video generation configuration."""
    script: ScriptSettings
    images: ImageSettings
    voice: VoiceSettings

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoConfig":
        """Create VideoConfig from a dictionary."""
        return cls(
            script=ScriptSettings.from_dict(data.get("script", {})),
            images=ImageSettings.from_dict(data.get("images", {})),
            voice=VoiceSettings.from_dict(data.get("voice", {}))
        )


class ConfigLoader:
    """Handles loading and validating configuration files."""

    @staticmethod
    def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from a JSON file.

        Args:
            config_path: Path to the configuration file

        Returns:
            Dictionary containing the configuration

        Raises:
            FileNotFoundError: If the config file doesn't exist
            json.JSONDecodeError: If the config file contains invalid JSON
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON format in {config_path}: {str(e)}",
                e.doc,
                e.pos
            )
