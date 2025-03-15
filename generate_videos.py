from video_generator import VideoGenerator

if __name__ == "__main__":
    generator = VideoGenerator(temperature=1.5)
    handle = "thehiddenlibrary33"

    title_description = """
        Titles must show the topic of the video, be catchy and elicit curiosity"""

    script_description = """
        Script Structure & Flow: \n
        1. Introduction (300-400 words)
        Open with a powerful hook that immediately captures attention.
        Introduce the topic in a mystical and immersive manner—avoid generic explanations.
        Briefly outline what the video will explore, teasing key insights without giving everything away upfront. \n
        2. Historical & Philosophical Context (500-600 words)
        Provide a historical background linked to the topic, referencing ancient texts, lost civilizations, or spiritual figures.
        Explain the esoteric or philosophical significance behind the subject.
        Show why this knowledge has been hidden, forgotten, or misunderstood over time. \n
        3. Core Explanation (Part 1) (600-700 words)
        Begin breaking down the topic into key concepts, making them digestible yet profound.
        Use storytelling, analogies, and metaphorical descriptions to simplify complex ideas.
        Build intrigue by leaving some questions unanswered to maintain curiosity. \n
        4. Core Explanation (Part 2) (600-700 words)
        Dive deeper into the topic, exploring hidden meanings, paradoxes, or advanced concepts.
        Address myths, misconceptions, or alternative perspectives.
        Provide little-known facts or insights to keep viewers engaged.\n
        5. Practical Application (500-600 words)
        Share real-world techniques, exercises, or rituals that viewers can try.
        Explain how these insights can transform their mindset, consciousness, or daily life.
        Offer guidance on overcoming obstacles or common pitfalls when applying the knowledge. \n
        6. Conclusion (300-400 words)
        Summarize key points in a compelling and thought-provoking way.
        End with a powerful statement or open-ended question to leave viewers reflecting.
        Include a subtle yet effective call to action, encouraging engagement or further exploration."""
    
    topic = "esotherism and ancient knowledge"

    videos = generator.generate_text_content(
        handle=handle, title_description=title_description, script_description=script_description, n_videos=1)
    print(videos)