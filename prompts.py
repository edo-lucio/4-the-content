title_generation_prompt  = """
    You are a talented YouTube script writer for my youtube channel which delves into the topic of {topic}. \n
    Generate {n} YouTube video titles taking inspiration from the following list as examples: {titles}. \n
    {description} \n
    Please, report each title separated by a newline. \n
    Include ONLY video titles and nothing else."""

script_generation_prompt = """
    Write a detailed {section} part of a YouTube video script from the following title: {title}. \n
    {description} \n

    I'm going to give you the video's script of a channel I want to take inspiration from.
    Use a similar tone script structure and length as the following: \n
    {transcripts}

    Include only the script that the narrator should play.
    Do not include informations about the scenes and the transitions.
    """

scenes_generation_prompt = """ 
    You're a talented YouTube video editor.
    I'm going to provide you a video script.
    Please, give me a list of scenes that suits the topic of the script. 
    Return only the scenes separated by a newline. Here is the script: \n 
    {script} 
    \n
    """

image_prompt_generation_prompt = """ 
    Generate highly detailed image prompts based on the full script of the video about the following topic: {topic}.
    Here is the full script of the video : {script}
    Each image should follow the specific artistic and stylistic elements below, ensuring a coherent visual aesthetic:
    {images_description}.
    Return only the image prompts separated by a newline.
    """

image_prompt_generation_prompt_with_scenes = """ 
    Generate highly detailed image prompts based on the scenes of the video about the following topic: {topic}.
    Here are the scenes of the video : {scenes}
    Each image should follow the specific artistic and stylistic elements below, ensuring a coherent visual aesthetic:
    {images_description}.
    Mantain a coherent style among the image prompts, making recurrent objects, characters and scenarios have the same 
    characteristics among every prompt. 
    Return only the image prompts separated by a newline.
    """

prompts = {
    "title_generation_prompt": title_generation_prompt, 
    "script_generation_prompt": script_generation_prompt, 
    "scenes_generation_prompt": scenes_generation_prompt,
    "image_prompt_generation_prompt": image_prompt_generation_prompt,
    "image_prompt_generation_prompt_with_scenes": image_prompt_generation_prompt_with_scenes
}