from get_videos import YouTubeClient, Video
from generators import TextGeneratorAgent, ImageGeneratorAgent
from typing import List, Dict, Tuple
import os

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
        topic: str = "",
        n_output_videos: int = 8,
        n_images: int = 10,
        max_reference_titles: int = 10,
        max_reference_transcripts: int = 1,
        sections: list = ["introduction", "core explanation", "conclusion"],
        from_scenes: bool = True,
        generate_scripts: bool = True,
        # Image-related parameters
        height: int = 1024,
        width: int = 1024,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 20,
        max_sequence_length: int = 512
    ):
        # Create text settings dictionary
        self.text_settings = {
            "handle": handle,
            "n_images": n_images,
            "title_description": title_description,
            "script_description": script_description,
            "images_description": images_description,
            "topic": topic,
            "n_output_videos": n_output_videos,
            "max_reference_titles": max_reference_titles,
            "max_reference_transcripts": max_reference_transcripts,
            "sections": sections,
            "from_scenes": from_scenes
        }
        
        # Create image settings dictionary
        self.image_settings = {
            "height": height,
            "width": width,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "max_sequence_length": max_sequence_length
        }

        self.generate_scripts = generate_scripts
        
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
        self.image_generator = ImageGeneratorAgent(output_folder=output_folder)
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

    def _get_prompts(self) -> Tuple[List[List], List[str]]:
        prompts, paths = [], []
        
        for root, dirs, files in os.walk(self.output_folder):
            if "scripts.csv" in files and not "images" in dirs:
                file_path = os.path.join(root, "scripts.csv")
                try:
                    video_prompts = pd.read_csv(file_path, sep="\t")["image_prompts"].iloc[0].split("\n")
                    prompts.append(video_prompts)
                    paths.append(root)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
        
        return prompts, paths

    def generate_text_content(self, **kwargs) -> List[Dict]:
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
        handle = kwargs["handle"]
        max_reference_titles = kwargs["max_reference_titles"]
        max_reference_transcripts = kwargs["max_reference_transcripts"]
        n_output_videos = kwargs["n_output_videos"]
        topic = kwargs["topic"]
        title_description = kwargs["title_description"]
        script_description = kwargs["script_description"]
        images_description = kwargs["images_description"]
        sections = kwargs["sections"]
        from_scenes = kwargs["from_scenes"]
        n_images = kwargs["n_images"]

        titles_references = []
        transcripts_references = []
        n_video_examples = max(max_reference_titles, max_reference_transcripts)

        if handle:
            video_examples, channel_description = self._get_channel(handle, n_video_examples)
            titles_references = [f"Title: {video.title}, Views: {video.views}" for video in video_examples][:max_reference_titles]
            transcripts_references = [video.transcript for video in video_examples][:max_reference_transcripts]

        self.text_contents = self.text_generator.generate(
            n_output_scripts=n_output_videos,
            topic=topic,
            channel_description=channel_description,
            titles_examples=titles_references,
            title_description=title_description,
            script_description=script_description,
            images_description=images_description,
            n_images=n_images,
            from_scenes=from_scenes,
            transcripts=transcripts_references,
            sections=sections
        )

        return self.text_contents

    def generate_image_content(self, **kwargs) -> List[str]:
        """ 
            height: int = 1024,
            width: int = 1024,
            guidance_scale: float = 3.5,
            num_inference_steps: int = 20,
            max_sequence_length: int = 512,
            generator: Generator = torch.Generator("cpu").manual_seed(0)
        """

        prompts, paths = self._get_prompts()

        for prompt_list, path in zip(prompts, paths):
            self.image_generator.generate(prompts_list=prompt_list, path=path, **kwargs)
    
    def generate_videos(self, video_settings: VideoSettings) -> None:

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

        if video_settings.generate_scripts:
            self.generate_text_content(**video_settings.text_settings)
        
        self.generate_image_content(**video_settings.image_settings)

