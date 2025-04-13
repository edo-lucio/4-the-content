from get_videos import YouTubeClient, Video
from generators import TextGeneratorAgent, TensorArtGenerator, AudioGeneratorAgent
from settings import VideoConfig
from typing import List, Dict, Tuple
import os
from dataclasses import asdict

import pandas as pd

from dotenv import load_dotenv
load_dotenv()

class VideoSettings:
    def __init__(
        self,
        handle: str,
        title_description: str,
        script_description: str,
        images_description: str = "",
        thumbnail_description: str = "",
        topic: str = "",
        n_output_videos: int = 8,
        n_images: int = 10,
        max_reference_titles: int = 10,
        max_reference_transcripts: int = 1,
        sections: list = ["introduction", "core explanation", "conclusion"],
        from_scenes: bool = True,
        generate_scripts: bool = True,
        generate_images: bool = True,
        generate_audios: bool = True,

        # Image-related parameters
        stages: List[Dict] = [],

        # Audio-related parameters
        voice: str = "af_sky+af_bella",
        pitch_shift: float = 0,
        speed: float = 1

    ):
        # Create text settings dictionary
        self.text_settings = {
            "handle": handle,
            "n_images": n_images,
            "title_description": title_description,
            "script_description": script_description,
            "images_description": images_description,
            "thumbnail_description": thumbnail_description,
            "topic": topic,
            "n_output_videos": n_output_videos,
            "max_reference_titles": max_reference_titles,
            "max_reference_transcripts": max_reference_transcripts,
            "sections": sections,
            "from_scenes": from_scenes
        }
        
        # Create image settings dictionary
        self.image_settings = { "stages": stages }

        self.audio_settings = {
            "voice": voice,
            "pitch_shift": pitch_shift,
            "speed": speed
        }

        self.generate_scripts = generate_scripts
        self.generate_images = generate_images
        self.generate_audios = generate_audios
        
        for key, value in {**self.text_settings, **self.image_settings}.items():
            setattr(self, key, value)

class VideoGenerator:
    """Class to handle copying channel content and generating scripts."""
    
    def __init__(
            self, yt_api_key: str = os.getenv("GOOGLE_API_KEY"), output_folder: str = "output", temperature: float=0.8):
        """
        Initialize the ChannelCopier with YouTube client.
        
        Args:
            api_key: Optional YouTube API key. If not provided, uses environment variable.
        """
        self.client = YouTubeClient(yt_api_key)
        self.text_generator = TextGeneratorAgent(temperature=temperature, output_folder=output_folder)
        self.image_generator = TensorArtGenerator(output_folder=output_folder)
        self.audio_generator = AudioGeneratorAgent(output_folder=output_folder)
        self.text_contents = None
        self.output_folder = output_folder # e.g.: /output/topic/title/scripts

    def _get_channel_videos(self, handle: str, max_videos: int = 10) -> List[Video]:
        """
        Retrieve video titles from a channel.
        
        Args:
            handle: YouTube channel handle
            max_videos: Maximum number of video titles to retrieve
            
        Returns:
            List of video titles
        """
        channel = self.client.get_channel_by_handle(handle)
        videos = channel.get_videos(n=max_videos)

        return videos

    def _get_channel(self, handle: str, max_videos: int= 10):
        channel = self.client.get_channel_by_handle(handle)
        videos = channel.get_videos(n=max_videos)
        description = channel.description

        return videos, description

    def _get_content(self, content) -> Tuple[List[List], List[str]]:
        prompts, paths = [], []
        content_folder = "images" if content == ("image_prompts" or "thumbnail_image") else "audio"
        
        for root, dirs, files in os.walk(self.output_folder):
            if "scripts.csv" in files and not content_folder in dirs:
                file_path = os.path.join(root, "scripts.csv")
                try:
                    video_content = pd.read_csv(file_path, sep="\t")[content].iloc[0]
                    video_content = video_content.split("\n") if content_folder == "images" else video_content
                    prompts.append(video_content)
                    paths.append(root)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
        
        return prompts, paths

    def generate_text_content(self, text_settings) -> List[Dict]:
        """
        Generate scripts based on channel content.
        
        Args:
            handle: YouTube channel handle
            title_description: Description for title generation
            script_description: Description for script generation
            topic: Content topic
            n_videos: Number of scripts to generate
            max_reference_videos: Maximum reference videos to use
            
        Returns:
            List of generated scripts
        """

        titles_references = []
        transcripts_references = []
        n_video_examples = max(text_settings.max_reference_titles, text_settings.max_reference_transcripts)

        if text_settings.handle:
            video_examples, channel_description = self._get_channel(text_settings.handle, n_video_examples)
            titles_references = [
                f"Title: {video.title}, Views: {video.views}" for video in video_examples][:text_settings.max_reference_titles]
            transcripts_references = [video.transcript for video in video_examples][:text_settings.max_reference_transcripts]

        self.text_contents = self.text_generator.generate(
            n_output_scripts=text_settings.n_output_videos,
            topic=text_settings.topic,
            channel_description=channel_description,
            thumbnail_description=text_settings.thumbnail_description,
            titles_examples=titles_references,
            title_description=text_settings.title_description,
            script_description=text_settings.script_description,
            images_description=text_settings.images_description,
            n_images=text_settings.n_images,
            from_scenes=text_settings.from_scenes,
            transcripts=transcripts_references,
            sections=text_settings.sections
        )

        return self.text_contents

    def generate_image_content(self, image_settings) -> List[str]:
        """ 
            height: int = 1024,
            width: int = 1024,
            guidance_scale: float = 3.5,
            num_inference_steps: int = 20,
            max_sequence_length: int = 512,
            generator: Generator = torch.Generator("cpu").manual_seed(0)
        """

        prompts, paths = self._get_content("image_prompts")

        for prompt_list, path in zip(prompts, paths):
            self.image_generator.generate(prompts_list=prompt_list, path=path, image_settings=image_settings)

    def generate_audio_content(self, audio_settings):
        scripts, paths = self._get_content("scripts")

        for script, path in zip(scripts, paths):
            self.audio_generator.generate(script=script, path=path, audio_settings=audio_settings)

    def generate_videos(self, video_settings: VideoConfig) -> None:

        """
        Complete pipeline for copying channel content and generating scripts.
        
        Args:
            handle: YouTube channel handle
            title_description: Description for title generation
            script_description: Description for script generation
            topic: Content topic
            n_videos: Number of scripts to generate
            output_path: Path to output file
            
        Returns:
            List of generated scripts
        """

        if video_settings.script.generate_scripts:
            self.generate_text_content(video_settings.script)
        
        if video_settings.script.generate_images:
            self.generate_image_content(video_settings.images)

        if video_settings.script.generate_audios:
            self.generate_audio_content(video_settings.voice)

