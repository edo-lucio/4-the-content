import os
from openai import OpenAI
from typing import List, Dict
from diffusers import FluxPipeline
import pandas as pd
import requests
import torch
import re

from prompts import prompts

from dotenv import load_dotenv
load_dotenv()

class ContentGenerator: 
    class Folder:
        def __init__(self, path: str):
            self.path = path
            os.makedirs(self.path, exist_ok=True)

    def __init__(self, output_folder: str, client: OpenAI):
        self.output_folder = output_folder
        self.client = client
        os.makedirs(output_folder, exist_ok=True)
    
    def _save_output(self):
        pass

    def _request(self):
        pass

    def generate():
        raise NotImplementedError("Subclasses must implement generate()")

class TextGeneratorAgent(ContentGenerator): 
    prompts = prompts

    def __init__(
            self, 
            api_key: str = os.getenv("DEEPSEEK_API"), 
            output_folder: str = "./scripts", 
            temperature: float = 0.8, 
            model: str="deepseek-chat"):
        
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.title_generation_prompt = prompts["title_generation_prompt"]
        self.script_generation_prompt = prompts["script_generation_prompt"]
        self.scenes_generation_prompt = prompts["scenes_generation_prompt"] 
        self.image_prompt_generation_prompt = prompts["image_prompt_generation_prompt"] 
        self.image_prompt_generation_prompt_with_scenes = prompts["image_prompt_generation_prompt_with_scenes"]
        self.temperature = temperature
        self.model = model
        super().__init__(output_folder, client)

    def _save_output(self, output_path: str, video: Dict) -> None:
        """
        Save generated scripts to CSV file.
        
        Args:
            scripts: List of scripts to save
            output_path: Path to output file
        """
        try:
            pd.DataFrame([video]).to_csv(output_path, index=False, sep="\t")
            print(f"Scripts successfully saved to {output_path}")
        except Exception as e:
            print(f"Error saving scripts: {str(e)}")
            raise

    def _request(self, prompt: str, max_tokens: int = 1000) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful YouTube script writer"},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            max_tokens=max_tokens, 
            temperature=self.temperature
        )

        return response.choices[0].message.content
    
    def _generate_titles(self) -> str:
        print("Generating Titles ...")
        prompt = self.title_generation_prompt.format(
            n=self.n_output_scripts, titles="\n".join(self.titles_examples), description=self.title_description, topic=self.topic)
        return self._request(prompt)

    def _generate_section(self, title: str, section: str) -> str:
        print(f"            - Generating {section}")
        prompt = self.script_generation_prompt.format(
            title=title, description=self.script_description, transcripts=self.transcripts_joined, section=section)
        return self._request(prompt, max_tokens=self.avg_transcript_length)
    
    def _generate_script(self, title: str):
        print(f"        - Generating script")
        script  = ""

        for section in self.sections:
            section_script = self._generate_section(title=title, section=section)
            script += f"\n {section_script}"

        return script
    
    def _generate_scenes(self, script: str) -> str:
        print(f"        - Generating scenes")
        prompt = self.scenes_generation_prompt.format(script=script)
        return self._request(prompt, max_tokens=self.avg_transcript_length)
    
    def _generate_image_prompts(self, scenes: str, script: str) -> str:
        print(f"        - Generating Image prompts")
        if not scenes:
            prompt = self.image_prompt_generation_prompt.format(
                topic=self.topic, script=script, images_description=self.images_description)
        else:
            prompt = self.image_prompt_generation_prompt_with_scenes.format(
                topic=self.topic, scenes=scenes, images_description=self.images_description)
        return self._request(prompt, max_tokens=self.avg_transcript_length)

    def generate(
            self,
            n_output_scripts: int,
            topic: str = "",
            titles_examples: str = "", 
            title_description: str = "", 
            script_description: str = "",
            images_description: str = "",
            from_scenes: bool = True,
            transcripts: str = "", 
            sections: list = ["introduction", "core part", "conclusion"]) -> dict: 
        
        self.n_output_scripts = n_output_scripts
        self.topic = topic
        self.titles_examples = titles_examples
        self.title_description = title_description
        self.script_description = script_description
        self.images_description = images_description
        self.transcripts_joined = "\n".join(transcripts)
        self.sections = sections
        self.avg_transcript_length = round(sum(
            [len(transcript) for transcript in transcripts]) / len(transcripts) if transcripts else 1000)
        self.avg_transcript_length = self.avg_transcript_length if self.avg_transcript_length <= 8192 else 8192
        self.from_scenes = from_scenes

        videos = []
        titles = self._generate_titles().split("\n")

        for i in range(self.n_output_scripts):
            video = {"title": "", "script": "", "scenes": "", "prompts": ""}

            title = titles[i]
            print(f"    Generating {title}")
            title = title.strip()
            script = self._generate_script(title)
            scenes = self._generate_scenes(script) if self.from_scenes  else ""
            image_prompts = self._generate_image_prompts(scenes=scenes, script=script)

            video["titles"] = title
            video["scripts"] = script
            video["scenes"] = scenes
            video["prompts"] = image_prompts

            topic_name = topic.replace(" ", "-")
            title_name = "-".join((re.sub(r'[^A-Za-z\s]', '', title).strip().split(" ")))
            folder_name = self.Folder(f"./{self.output_folder}/{topic_name}/{title_name}").path
            output_path = f"{folder_name}/scripts.csv"
            self._save_output(output_path=output_path, video=video)

        return videos

class ImageGeneratorAgent(ContentGenerator):
    def __init__(
            self, 
            api_key: str = os.getenv("NEBIUS_API_KEY"), 
            output_folder: str="./images", 
            output_format: str = "jpeg",
            model: str = "black-forest-labs/flux-dev"):
        
        client = OpenAI(base_url="https://api.studio.nebius.com/v1/", api_key=api_key)
        self.output_format = output_format
        self.model = model
        super().__init__(output_folder, client)

    def _save_output(self, image_url: str, image_path: str):
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            with open(image_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            print(f"Image downloaded successfully: {image_path}")
        else:
            print(f"Failed to download image. Status code: {response.status_code}")

    def _request(self, prompt: str, **kwargs) -> str:
        extra_body = {
            "response_extension": "webp",
            **kwargs,
            "seed": -1,
        }
        
        response = self.client.images.generate(
            model=self.model,
            prompt=prompt,
            response_format="url",
            extra_body=extra_body
        )

        return response.data[0].url
        
    def generate_image(
        self, 
        prompt: str, 
        image_path: str,
        height: int=1024, 
        width: int=1024, 
        guidance_scale: float = 3.5, 
        num_inference_steps: int = 20, 
        max_sequence_length: int = 512) -> str:
        
        image_url = self._request(
            prompt=prompt, height=height, width=width, guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps, max_sequence_length=max_sequence_length)

        self._save_output(image_url, image_path)

    def generate(
        self, 
        prompts_list: List[str], 
        path: str,
        height: int=1024, 
        width: int=1024, 
        guidance_scale: float = 3.5, 
        num_inference_steps: int = 20, 
        max_sequence_length: int = 512):

        prompts_list = [prompt for prompt in prompts_list if prompt.strip()]
        folder_path = self.Folder(f"{path}/images").path

        for i, prompt in enumerate(prompts_list):
            image_path = f"{folder_path}/{i}.{self.output_format}"
            self.generate_image(
                prompt=prompt, image_path=image_path,
                height=height, width=width,
                guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, 
                max_sequence_length=max_sequence_length)
            
class AudioGeneratorAgent(ContentGenerator):
    def __init__(self, output_folder: str="./images", output_format: str = "jpeg"):
        super().__init__(output_folder)
        self.pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
        self.pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
        self.output_format = output_format
        