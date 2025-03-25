import json
from video_generator import VideoGenerator, VideoSettings

def load_config(config_file="config.json"):
    """Load configuration from a JSON file."""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file '{config_file}' not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{config_file}'.")
        exit(1)
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        exit(1)

if __name__ == "__main__":
    config = load_config()
    
    topic = config.get("topic", "")
    title_description = config.get("title_description", "")
    script_description = config.get("script_description", "")
    images_description = config.get("images_description", "")
    thumbnail_description = config.get("thumbnail_description", "")
    temperature = config.get("temperature", 1.0)
    generate_scripts = config.get("generate_scripts", True)
    generate_images = config.get("generate_images", True)
    generate_audios = config.get("generate_audios", True)
    handle = config.get("handle", "")
    n_output_videos = config.get("n_output_videos", 1)
    sections = config.get("sections", ["introduction", "core explanation", "conclusion"])
    max_reference_titles = config.get("max_reference_titles", 10)
    max_reference_transcripts = config.get("max_reference_transcripts", 10)
    from_scenes = config.get("from_scenes", False)
    n_images = config.get("n_images", 1)
    height = config.get("height", 1024)
    width = config.get("width", 1024)
    voice = config.get("voice", "am_bella+am_michael")
    pitch_shift = config.get("pitch_shift", 0)
    speed = config.get("speed", 1)
    stages = config.get("stages", {})
    
    generator = VideoGenerator(temperature=temperature)
    
    video_settings = VideoSettings(
        handle=handle,
        title_description=title_description,
        script_description=script_description,
        from_scenes=from_scenes,
        images_description=images_description,
        n_output_videos=n_output_videos,
        topic=topic,
        n_images=n_images,
        generate_scripts=generate_scripts,
        generate_images=generate_images,
        generate_audios=generate_audios,
        max_reference_titles=max_reference_titles,
        sections=sections,
        max_reference_transcripts=max_reference_transcripts,
        stages=stages,
        voice=voice,
        pitch_shift=pitch_shift, 
        speed=speed
    )
    
    generator.generate_videos(video_settings=video_settings)