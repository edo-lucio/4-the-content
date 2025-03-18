import os
from openai import OpenAI
from typing import List, Dict
import torch
from diffusers import FluxPipeline
import pandas as pd

from prompts import prompts

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API")

class ContentGenerator: 
    def __init__(self, output_folder: str):
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
    
    def _save_output(self):
        pass

    def generate():
        raise NotImplementedError("Subclasses must implement generate()")

class TextGeneratorAgent(ContentGenerator): 
    prompts = prompts

    def __init__(self, output_folder: str = "./scripts", temperature: float = 0.8):
        super().__init__(output_folder)
        self.title_generation_prompt = prompts["title_generation_prompt"]
        self.script_generation_prompt = prompts["script_generation_prompt"]
        self.scenes_generation_prompt = prompts["scenes_generation_prompt"] 
        self.image_prompt_generation_prompt = prompts["image_prompt_generation_prompt"] 
        self.temperature = temperature
    
    def _save_output(self, output_path: str, videos: List[Dict]) -> None:
        """
        Save generated scripts to CSV file.
        
        Args:
            scripts: List of scripts to save
            output_path: Path to output file
        """
        try:
            pd.DataFrame(videos).to_csv(output_path, index=False, sep="\t")
            print(f"Scripts successfully saved to {output_path}")
        except Exception as e:
            print(f"Error saving scripts: {str(e)}")
            raise

    def _deepseek_request(self, prompt: str, max_tokens: int = 1000) -> str:
        client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",
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
        return self._deepseek_request(prompt)

    def _generate_section(self, title: str, section: str) -> str:
        print(f"            - Generating {section}")
        prompt = self.script_generation_prompt.format(
            title=title, description=self.script_description, transcripts=self.transcripts_joined, section=section)
        return self._deepseek_request(prompt, max_tokens=self.avg_transcript_length)
    
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
        return self._deepseek_request(prompt, max_tokens=self.avg_transcript_length)
    
    def _generate_image_prompts(self, scenes: str) -> str:
        print(f"        - Generating Image prompts")

        prompt = self.image_prompt_generation_prompt.format(scenes=scenes)
        return self._deepseek_request(prompt, max_tokens=self.avg_transcript_length)

    def generate(
            self,
            n_output_scripts: int,
            topic: str = "",
            titles_examples: str = "", 
            title_description: str = "", 
            script_description: str = "",
            transcripts: str = "", 
            sections: list = ["introduction", "core part", "conclusion"]) -> dict: 
        
        self.n_output_scripts = n_output_scripts
        self.topic = topic
        self.titles_examples = titles_examples
        self.title_description = title_description
        self.script_description = script_description
        self.transcripts_joined = "\n".join(transcripts)
        self.sections = sections
        self.avg_transcript_length = round(sum(
            [len(transcript) for transcript in transcripts]) / len(transcripts) if transcripts else 1000)
        self.avg_transcript_length = self.avg_transcript_length if self.avg_transcript_length <= 8192 else 8192
        
        videos = {"titles": [], "scripts": [], "scenes": [], "prompts": []}
        titles = self._generate_titles().split("\n")

        print(self.n_output_scripts)
        print(len(titles))

        for i in range(self.n_output_scripts):
            title = titles[i]
            print(f"    Generating {title}")
            title = title.strip()
            script = self._generate_script(title)
            scenes = self._generate_scenes(script)
            image_prompts = self._generate_image_prompts(scenes)

            videos["titles"].append(title)
            videos["scripts"].append(script)
            videos["scenes"].append(scenes)
            videos["prompts"].append(image_prompts)

        output_path = f"./{self.output_folder}/scripts.csv"
        self._save_output(videos=videos, output_path=output_path)
        return videos

class ImageGeneratorAgent(ContentGenerator):
    def __init__(self, output_folder: str="./images", output_format: str = "jpeg"):
        super().__init__(output_folder)
        self.pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
        self.pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
        self.output_format = output_format

    def _save_output(self, image, image_id) -> str:
        output_path = f"{self.output_folder}/{image_id}.{self.output_format}"
        image.save(output_path)
        return output_path
        
    def generate_image(
        self, 
        prompt: str, 
        image_id: str,
        height: int=1024, 
        width: int=1024, 
        guidance_scale: float = 3.5, 
        num_inference_steps: int = 20, 
        max_sequence_length: int = 512, 
        generator=torch.Generator("cpu").manual_seed(0)) -> str:
        
        image = self.pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            generator=generator
        ).images[0]

        output_path = self._save_output(image, image_id)
        return output_path

    def generate(
        self, 
        prompts_list: str, 
        height: int=1024, 
        width: int=1024, 
        guidance_scale: float = 3.5, 
        num_inference_steps: int = 20, 
        max_sequence_length: int = 512, 
        generator=torch.Generator("cpu").manual_seed(0)) -> List[str]:

        prompts_list = prompts_list.split("\n")
        images_paths = []

        for i in range(len(prompts_list)):
            image_id = f"{i}"
            image_path = self.generate_image(
                prompt=prompts_list[i], image_id=image_id,
                height=height, width=width, 
                guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, 
                max_sequence_length=max_sequence_length, generator=generator)
            
            images_paths.append(image_path)

        return images_paths
            
class AudioGeneratorAgent(ContentGenerator):
    def __init__(self, output_folder: str="./images", output_format: str = "jpeg"):
        super().__init__(output_folder)
        self.pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
        self.pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
        self.output_format = output_format
        