title_generation_prompt  = """
    You are a talented YouTube script writer for my youtube channel which delves into the topic of {topic}. \n
    You have to help me writing scripts for a youtube channel with the following aim: {channel_description}.
    Generate {n} YouTube video titles taking inspiration from the following list as examples: ** {titles} **. \n
    Take into consideration also the views to maximize the virality of the video.
    {description} \n
    Please, report each title separated by a newline. \n
    Include ONLY video titles and nothing else."""

script_generation_prompt = """
    Write a detailed {section} part of a YouTube video script from the following title: {title}.
    {script_adaptation_prompt}.

    Use the following general writing guidelines: \n
    GUIDELINES : **{description}** 

    I'm going to give you a list of video's scripts of a channel I want to take inspiration from.
    Use a similar tone and structure as the following. I'm going to separate each different script by using "**": \n
    SCRIPT REFERENCES: {transcripts}

    Include only the script that the narrator should play.
    Do not include informations about the scenes, sections or transitions.
    """

script_adaptation_prompt = """
    Contextualize and adapt the section that you are going to write considering that it comes after the following: 
    **{previous_script}**
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
    Generate {n_images} highly detailed image prompts based on the full script of the video about the following topic: {topic}.
    Each prompt must contain highly descriptive informations about the content of the image and exhaustive guidelines on the artistic style.
    Each image should follow the specific artistic and stylistic elements below, ensuring a coherent visual aesthetic:
    {images_description}.
    Return only the image prompts separated by a newline.
    """

image_prompt_generation_prompt_with_scenes = """ 
    Generate highly detailed image prompts based on the scenes of the video about the following topic: {topic}.
    Here are the scenes of the video: {scenes}
    Each image should follow the specific artistic and stylistic elements below, ensuring a coherent visual aesthetic:
    {images_description}.
    Mantain a coherent style among the image prompts, making recurrent objects, characters and scenarios have the same 
    characteristics among every prompt. 
    Return only the image prompts separated by a newline.
    """

thumnbail_generation_prompt = """
    Generate a highly detailed image prompt to generate a catchy thumbnail for my YouTube video.
    It must be attention grabber.
    I'm going to provide you with both the full script of the video and the description of the visuals.
    SCRIPT: ** {script} **
    VISUALS DESCRIPTION: {thumbnail_description}
    Return only the image prompt on the first line and the text overlay on the second.
    Do not add any other text.
    Separate the output with newlines. 
"""

prompts = {
    "title_generation_prompt": title_generation_prompt, 
    "script_generation_prompt": script_generation_prompt, 
    "scenes_generation_prompt": scenes_generation_prompt,
    "image_prompt_generation_prompt": image_prompt_generation_prompt,
    "image_prompt_generation_prompt_with_scenes": image_prompt_generation_prompt_with_scenes,
    "script_adaptation_prompt": script_adaptation_prompt,
    "thumnbail_generation_prompt": thumnbail_generation_prompt
}