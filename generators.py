import os
from typing import List, Dict
import pandas as pd
import requests
import whisper

from utils import process_script, process_title, process_topic, batched
from prompts import prompts

from clients import TensorArtClient, OpenAI
from zyphra import ZyphraClient

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
    # [Original TextGeneratorAgent code remains unchanged]
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

        created_sections = []

        for section in self.sections:
            section_script = self._generate_section(title=title, section=section, script=script)
            created_sections.append(section_script)

        return " ".join(created_sections)
    
    def _generate_scenes(self, script: str) -> str:
        print(f"\t\t- Generating scenes")
        prompt = self.scenes_generation_prompt.format(script=script)
        return self._request(prompt, max_tokens=self.avg_transcript_length)
    
    def generate_image_prompts(self, scenes: str, script: str) -> str:
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
        self.transcripts_joined = transcripts
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
    # [Original ImageGeneratorAgent code remains unchanged]
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
    # [Original TensorArtGenerator code remains unchanged]
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

    def generate(self, prompts_list: List[str], path: str, image_settings):
        print("Generating images ...") 
        prompts_list_filtered = [prompt for prompt in prompts_list if prompt.strip()]
        folder_path = self.Folder(f"{path}/images").path
        stages = image_settings.lora["stages"]
        image_urls = []

        for i, prompt in enumerate(prompts_list_filtered):
            print(f"Generating image for {prompt}")
            urls_list = self.client.generate([prompt], stages)[0]
            image_path = f"{folder_path}/{i}.{self.output_format}"
            self._save_output(image_url=urls_list, image_path=image_path)

class AudioGeneratorAgent(ContentGenerator):
    # [Original AudioGeneratorAgent code remains unchanged]
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

    def generate(self, script: str, path: str, audio_settings) -> None:
        script_processed = process_script(script)
        folder_path = self.Folder(f"{path}/audio").path
        audio_path = f"{folder_path}/audio.{self.output_format}"

        print(audio_settings)

        voice = audio_settings.voice
        pitch_shift = audio_settings.pitch_shift
        speed = audio_settings.speed
        
        print(f"Generating Audios using Kokoro {script_processed}")
        
        with self.client.audio.speech.with_streaming_response.create(
            model="kokoro",
            voice=voice, # single or multiple voicepack combo
            input=script_processed,
            speed=speed
        ) as response:
            response.stream_to_file(audio_path)

        if pitch_shift:
            self._lower_pitch(audio_path, pitch_shift)

class ZonosAudioGenerator(ContentGenerator):
    # [Original ZonosAudioGenerator code remains unchanged]
    def __init__(self, api_key:str, output_folder: str="./audios", output_format: str = "mp3"):
        client = ZyphraClient(api_key=api_key)
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

        clone_path = kwargs["clone_path"]

        if clone_path:
            with open(clone_path, "rb") as f:
                import base64
                audio_base64 = base64.b64encode(f.read()).decode('utf-8')

            audio_data = self.client.audio.speech.create(
                text="This will use the cloned voice",
                speaker_audio=audio_base64,
                speaking_rate=15,
                output_path = audio_path
            )

        print(f"Generating Audios using Kokoro {script_processed}")
        
        # Text-to-speech
        audio_data = self.client.audio.speech.create(
            text="Hello, world!",
            speaking_rate=15,
            model="zonos-v0.1-transformer"  # Default model
        )

# New SRTGeneratorAgent class for audio-to-SRT functionality
class SRTGeneratorAgent(ContentGenerator):
    """
    Agent for generating SRT subtitle files from audio files using Whisper
    """
    
    def __init__(self, output_folder: str="./subtitles", model_size: str="base"):
        """
        Initialize the SRT generator
        
        Parameters:
        - output_folder: Directory where SRT files will be saved
        - model_size: Whisper model size (tiny, base, small, medium, large)
        """
        # Note: Whisper doesn't need a client like the other generators, so we pass None
        super().__init__(output_folder, None)
        self.model_size = model_size
        self.model = None  # Will be loaded on first use
    
    def _format_time(self, seconds: float) -> str:
        """
        Convert seconds to SRT timestamp format (HH:MM:SS,mmm)
        """
        milliseconds = int((seconds % 1) * 1000)
        seconds = int(seconds)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    
    def _save_output(self, segments, output_path: str) -> None:
        """
        Save transcribed segments to an SRT file
        """
        try:
            with open(output_path, "w", encoding="utf-8") as srt_file:
                for i, segment in enumerate(segments, 1):
                    # Format as SRT entry
                    srt_file.write(f"{i}\n")
                    srt_file.write(f"{self._format_time(segment['start'])} --> {self._format_time(segment['end'])}\n")
                    srt_file.write(f"{segment['text'].strip()}\n\n")
            print(f"SRT file successfully saved to {output_path}")
        except Exception as e:
            print(f"Error saving SRT file: {str(e)}")
            raise
    
    def _load_model(self):
        """
        Load the Whisper model if not already loaded
        """
        if self.model is None:
            print(f"Loading Whisper {self.model_size} model...")
            self.model = whisper.load_model(self.model_size)
        return self.model
    
    def generate(self, path: str, language: str=None, audio_format: str="mp3") -> str:
        """
        Generate an SRT file from an audio file
        
        Parameters:
        - audio_path: Path to the audio file
        - output_path: Path where the SRT file should be saved (default: same as input with .srt extension)
        - language: Language code (e.g., 'en', 'fr') or None for auto-detection
        
        Returns:
        - Path to the generated SRT file
        """

        audio_path = f"{path}/audio/audio.{audio_format}"
        folder_path = self.Folder(f"{path}/subs").path
        output_path = f"{folder_path}/sub.srt"

        model = self._load_model()
        
        print(f"Transcribing audio file: {audio_path}")
        transcribe_options = {"task": "transcribe"}
        if language:
            transcribe_options["language"] = language
        
        # Run transcription
        result = model.transcribe(audio_path, **transcribe_options)
        
        # Save the result
        self._save_output(result["segments"], output_path)
        
        print(f"Successfully created SRT file with {len(result['segments'])} entries.")
        print(f"Detected language: {result.get('language', 'unknown')}")
        
        return output_path
    
    def generate_batch(self, audio_paths: List[str], output_folder: str=None, language: str=None) -> List[str]:
        """
        Generate SRT files for multiple audio files
        
        Parameters:
        - audio_paths: List of paths to audio files
        - output_folder: Folder where SRT files should be saved (default: same folder as each audio)
        - language: Language code (e.g., 'en', 'fr') or None for auto-detection
        
        Returns:
        - List of paths to the generated SRT files
        """
        output_paths = []
        
        for audio_path in audio_paths:
            # Set output path
            if output_folder:
                filename = os.path.basename(audio_path)
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_folder, f"{base_name}.srt")
            else:
                output_path = None  # Will use default
            
            # Generate SRT
            srt_path = self.generate(audio_path, output_path, language)
            output_paths.append(srt_path)
        
        return output_paths
    
    def generate_for_video_project(self, path: str, audio_filename: str="audio.mp3", language: str=None) -> str:
        """
        Generate an SRT file for a video project, following the project structure
        
        Parameters:
        - path: Base path to the project
        - audio_filename: Name of the audio file (default: "audio.mp3")
        - language: Language code (e.g., 'en', 'fr') or None for auto-detection
        
        Returns:
        - Path to the generated SRT file
        """
        # Construct paths
        audio_folder = os.path.join(path, "audio")
        audio_path = os.path.join(audio_folder, audio_filename)
        subtitles_folder = os.path.join(path, "subtitles")
        output_path = os.path.join(subtitles_folder, "subtitles.srt")
        
        # Create subtitles folder
        os.makedirs(subtitles_folder, exist_ok=True)
        
        # Generate SRT
        if os.path.exists(audio_path):
            return self.generate(audio_path, output_path, language)
        else:
            print(f"Audio file not found: {audio_path}")
            return None