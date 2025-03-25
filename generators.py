import os
from typing import List, Dict
import pandas as pd
import requests

from utils import process_script, process_title, process_topic, batched
from prompts import prompts

from clients import TensorArtClient, OpenAI

from dotenv import load_dotenv
load_dotenv()

class ContentGenerator: 
    class Folder:
        def __init__(self, path: str):
            self.path = path
            os.makedirs(self.path, exist_ok=True)

    def __init__(self, output_folder: str, client: object):
        self.output_folder = output_folder
        self.client = client
        os.makedirs(output_folder, exist_ok=True)
    
    def _save_output(self):
        raise NotImplementedError("Subclasses must implement generate()")

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
        self.script_adaptation_prompt = prompts["script_adaptation_prompt"]
        self.thumnbail_generation_prompt = prompts["thumnbail_generation_prompt"]

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
                {"role": "system", "content": f"You are a helpful YouTube script writer about the topic of {self.topic}"},
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
            n=self.n_output_scripts, titles="\n".join(self.titles_examples), 
            description=self.title_description, topic=self.topic, channel_description=self.channel_description)
        return self._request(prompt)

    def _generate_section(self, title: str, section: str, script: str) -> str:
        print(f"\t\t\t- Generating {section}")

        script_adaptation_prompt = ""
        if script: 
            script_adaptation_prompt = self.script_adaptation_prompt.format(previous_script=f"{script}")
            
        prompt = self.script_generation_prompt.format(
            title=title, description=self.script_description, transcripts=self.transcripts_joined, 
            section=section, script_adaptation_prompt=script_adaptation_prompt)
        return self._request(prompt, max_tokens=self.avg_transcript_length)
    
    def _generate_script(self, title: str):
        print(f"\t\t- Generating script")
        script  = ""

        for section in self.sections:
            section_script = self._generate_section(title=title, section=section, script=script)
            script += f"\n {section_script}"

        return script
    
    def _generate_scenes(self, script: str) -> str:
        print(f"\t\t- Generating scenes")
        prompt = self.scenes_generation_prompt.format(script=script)
        return self._request(prompt, max_tokens=self.avg_transcript_length)
    
    def _generate_image_prompts(self, scenes: str, script: str) -> str:
        print(f"\t\t- Generating Image prompts")
        if not scenes.strip():
            prompt = self.image_prompt_generation_prompt.format(
                topic=self.topic, images_description=self.images_description, 
                n_images=self.n_images)
            print(prompt)
        else:
            prompt = self.image_prompt_generation_prompt_with_scenes.format(
                topic=self.topic, scenes=scenes, images_description=self.images_description)
            
        return self._request(prompt, max_tokens=self.avg_transcript_length)
    
    def _generate_thumbnail_prompts(self, script: str):
        prompt = self.thumnbail_generation_prompt.format(
            script=script, thumbnail_description=self.thumbnail_description)
        return self._request(prompt, max_tokens=self.avg_transcript_length).split("\n")

    def generate(
            self,
            n_output_scripts: int,
            topic: str = "",
            channel_description: str = "",
            titles_examples: str = "", 
            title_description: str = "", 
            script_description: str = "",
            images_description: str = "",
            thumbnail_description: str = "",
            n_images: int = 10,
            from_scenes: bool = True,
            transcripts: str = "", 
            sections: list = ["introduction", "core part", "conclusion"]) -> dict: 
        
        self.n_output_scripts = n_output_scripts
        self.topic = topic
        self.channel_description = channel_description
        self.titles_examples = titles_examples
        self.title_description = title_description
        self.script_description = script_description
        self.images_description = images_description
        self.thumbnail_description = thumbnail_description
        self.transcripts_joined = "\n**".join(transcripts)
        self.n_images = n_images
        self.sections = sections
        self.avg_transcript_length = round(sum(
            [len(transcript) for transcript in transcripts]) / len(transcripts) if transcripts else 1000)
        self.avg_transcript_length = self.avg_transcript_length if self.avg_transcript_length <= 8192 else 8192
        self.from_scenes = from_scenes

        videos = []
        titles = self._generate_titles().split("\n")

        for i in range(self.n_output_scripts):
            video = {"title": "", "script": "", "scenes": "", "image_prompts": ""}

            title = titles[i]
            print(f"\tGenerating {title}")
            title = title.strip()
            script = self._generate_script(title)
            scenes = self._generate_scenes(script) if self.from_scenes  else " "
            image_prompts = self._generate_image_prompts(scenes=scenes, script=script)
            # thumbnail_prompt = self._generate_thumbnail_prompts(script)

            video["titles"] = title
            video["scripts"] = script
            video["scenes"] = scenes
            video["image_prompts"] = image_prompts
            # video["thumbnail_image"] = thumbnail_prompt[0]
            # video["thumbnail_text"] = thumbnail_prompt[1]

            topic_name = process_topic(topic)
            title_name = process_title(title)
            folder_name = self.Folder(f"./{self.output_folder}/{topic_name}/{title_name}").path
            output_path = f"{folder_name}/scripts.csv"
            self._save_output(output_path=output_path, video=video)

        return videos

class ImageGeneratorAgent(ContentGenerator):
    def __init__(
            self, 
            api_key: str = os.getenv("NEBIUS_API_KEY"), 
            output_folder: str="./images", 
            output_format: str = "png",
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
            extra_body=extra_body,
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
        num_inference_steps: int = 30, 
        max_sequence_length: int = 512):

        print(f"Generating Images using {self.model}")

        prompts_list_filtered = [prompt for prompt in prompts_list if prompt.strip()]
        folder_path = self.Folder(f"{path}/images").path

        for i, prompt in enumerate(prompts_list_filtered):
            image_path = f"{folder_path}/{i}.{self.output_format}"
            self.generate_image(
                prompt=prompt, image_path=image_path,
                height=height, width=width,
                guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, 
                max_sequence_length=max_sequence_length)
            
class TensorArtGenerator(ContentGenerator): 
    def __init__(
        self, 
        api_key: str = os.getenv("TENSOR_ART_API_KEY"), 
        app_id: str = os.getenv("TENSOR_ART_APP_ID"), 
        output_folder: str="./images", 
        output_format: str = "png",):

        client = TensorArtClient(app_id=app_id, api_key=api_key)

        self.output_format = output_format

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

    def _request(self, prompt, stages): 
            response = self.client.generate(prompt, stages)

            return response["job"]["successInfo"]["images"][0]["url"]

    def generate(self, prompts_list: List[str], path: str, **kwargs):
        print("Generating images ...") 
        prompts_list_filtered = [prompt for prompt in prompts_list if prompt.strip()]
        folder_path = self.Folder(f"{path}/images").path
        stages = kwargs["stages"]
        image_urls = []

        for prompt_batch in batched(prompts_list_filtered, 4):  # Process in chunks of 4
            print(f"Generating image for batch")
            url_results = self.client.generate(prompt_batch, stages)
            image_urls += url_results

        for i, url in enumerate(image_urls):
            image_path = f"{folder_path}/{i}.{self.output_format}"
            self._save_output(image_url=url, image_path=image_path)

class AudioGeneratorAgent(ContentGenerator):
    def __init__(self, output_folder: str="./audios", output_format: str = "mp3"):
        client = OpenAI(base_url="http://localhost:8880/v1", api_key="not-needed")
        self.output_format = output_format
        super().__init__(output_folder, client)

    def _lower_pitch(self, audio_path: str, pitch_shift: float) -> None: 
        from pydub import AudioSegment
        file_name = audio_path.split(".")[0]
        vanilla_audio = AudioSegment.from_mp3(audio_path)
        overrides = {"frame_rate": int(vanilla_audio.frame_rate * (2.0 ** (pitch_shift / 12.0)))}
        pitched_audio = vanilla_audio._spawn(vanilla_audio.raw_data, overrides=overrides)
        pitch_shifted_filename = f'{file_name}.{self.output_format}'
        pitched_audio.export(pitch_shifted_filename, format=self.output_format)

    def generate(self, script: str, path: str, **kwargs) -> None:
        script_processed = process_script(script)
        folder_path = self.Folder(f"{path}/audio").path
        audio_path = f"{folder_path}/audio.{self.output_format}"

        voice = kwargs["voice"]
        pitch_shift = kwargs["pitch_shift"]
        speed = kwargs["speed"]
        
        print(f"Generating Audios using Kokoro...")
        with self.client.audio.speech.with_streaming_response.create(
            model="kokoro",
            voice=voice, # single or multiple voicepack combo
            input=script_processed,
            speed=speed
        ) as response:
            response.stream_to_file(audio_path)

        if pitch_shift:
            self._lower_pitch(audio_path, pitch_shift)