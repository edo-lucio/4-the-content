from get_videos import YouTubeClient, Video
from generators import TextGeneratorAgent, ImageGeneratorAgent
from typing import List, Dict
import csv
import torch
import os

from dotenv import load_dotenv
load_dotenv()

class VideoGenerator:
    """Class to handle copying channel content and generating scripts."""
    
    def __init__(
            self, api_key: str = None, text_output_folder: str = "./scripts", 
            image_output_folder: str="./images", temperature: float=0.8):
        """
        Initialize the ChannelCopier with YouTube client.
        
        Args:
            api_key: Optional YouTube API key. If not provided, uses environment variable.
        """
        self.client = YouTubeClient(api_key or os.getenv("GOOGLE_API_KEY"))
        self.text_generator = TextGeneratorAgent(temperature=temperature, output_folder=text_output_folder)
        self.image_generator = ImageGeneratorAgent(output_folder=image_output_folder)
        self.text_contents = None
        self.text_output_folder = text_output_folder
        self.image_output_folder = image_output_folder

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

    def _get_videos(
        self,
        handle: str,
        title_description: str,
        script_description: str,
        topic: str = "",
        n_output_videos: int = 8,
        max_reference_titles: int = 10,
        max_reference_transcripts: int = 1,
        sections: list = ["introduction", "core part", "conclusion"]

    ) -> List[Dict]:
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
        n_video_examples = max(max_reference_titles, max_reference_transcripts)

        if handle:
            video_examples = self._get_channel_videos(handle, n_video_examples)
            titles_references = [video.title for video in video_examples][:max_reference_titles]
            transcripts_references = [video.transcript for video in video_examples][:max_reference_transcripts]


        return self.text_generator.generate(
            n_output_scripts=n_output_videos,
            topic=topic,
            titles_examples=titles_references,
            title_description=title_description,
            script_description=script_description,
            transcripts=transcripts_references,
            sections=sections
        )
    
    def generate_text_content(self, **kwargs) -> List[Dict]:
        self.text_contents = self._get_videos(**kwargs)
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
        if self.text_contents:
            contents = self.text_contents
        else:
            with open(f"{self.text_output_folder}/scripts.csv", newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                contents = [row for row in reader]
        
        prompts = [row["prompts"].split("\n") for row in contents]
        self.images_paths = self.image_generator.generate(prompts_list=prompts, **kwargs)

        return self.images_paths
    
    def generate_video(
        self, 
        handle: str,
        title_description: str,
        script_description: str,
        topic: str = "",
        n_output_videos: int = 8,
        max_reference_titles: int = 10,
        max_reference_transcripts: int = 1,
        sections: list = ["introduction", "core explanation", "conclusion"],
        generate_scripts: bool = True,
        height: int = 1024,
        width: int = 1024,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 20,
        max_sequence_length: int = 512,
        generator=torch.Generator("cpu").manual_seed(0)) -> None:

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

        if generate_scripts:
            self.generate_text_content(
                handle=handle, title_description=title_description, script_description=script_description,
                topic=topic, n_output_videos=n_output_videos, max_reference_titles=max_reference_titles,
                max_reference_transcripts=max_reference_transcripts, sections=sections)
        
        self.generate_image_content(
            height=height, width=width, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length, generator=generator)

