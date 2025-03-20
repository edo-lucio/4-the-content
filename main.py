from video_generator import VideoGenerator, VideoSettings

if __name__ == "__main__":

    topic = "esotherism and ancient knowledge"

    title_description = """
        Titles must hint the topic of the video, be catchy, attention grabber and easily accessible without much knowledge of the topic"""

    script_description = """
        Script Structure & Flow: \n
        Address an audience of laymen but curious viewers.
        Open with a powerful hook that immediately captures attention.
        Introduce the topic in a mystical and immersive manner—avoid generic explanations.
        Briefly outline what the video will explore, teasing key insights without giving everything away upfront. \n
        Provide a historical background linked to the topic, referencing ancient texts, lost civilizations, or spiritual figures.
        Explain the esoteric or philosophical significance behind the subject.
        Show why this knowledge has been hidden, forgotten, or misunderstood over time. \n
        Begin breaking down the topic into key concepts, making them digestible yet profound.
        Use storytelling, analogies, and metaphorical descriptions to simplify complex ideas.
        Build intrigue by leaving some questions unanswered to maintain curiosity. \n
        Dive deeper into the topic, exploring hidden meanings, paradoxes, or advanced concepts.
        Address myths, misconceptions, or alternative perspectives.
        Provide little-known facts or insights to keep viewers engaged.\n
        Share real-world techniques, exercises, or rituals that viewers can try.
        Explain how these insights can transform their mindset, consciousness, or daily life.
        Offer guidance on overcoming obstacles or common pitfalls when applying the knowledge. \n
        Summarize key points in a compelling and thought-provoking way.
        End with a powerful statement or open-ended question to leave viewers reflecting.
        Include a subtle yet effective call to action, encouraging engagement or further exploration."""
    
    images_description = f"""
        Style & Theme
        A highly detailed esoteric manuscript illustration resembling medieval alchemical blueprints, Leonardo da Vinci’s notebooks, or ancient mystical scrolls.
        The artwork should be flat and symmetrical, viewed from a straight-on perspective, ensuring no background depth, page curvature, or book/table perspective.
        Cryptic handwritten text, runes, and sigils should be incorporated, appearing like sacred knowledge inscribed by an ancient mystic.
        Sacred geometry, planetary alignments, mystical symbols, and interconnected diagrams should dominate the design, reflecting deep metaphysical insights.
        Glowing mystical energy flows should be subtly woven into the design, enhancing the feeling of hidden wisdom.
        Visual Composition
        No physical edges or worn-out textures – the entire image should appear as a timeless, pristine esoteric blueprint rather than an aged document.
        The central human figure (if applicable) should be depicted as an illuminated silhouette, integrated into the diagram, symbolizing transformation and enlightenment.
        The overall color scheme should be warm sepia, resembling aged ink and parchment, with golden highlights for mystical energy.
        No modern elements—strictly use medieval alchemical symbolism, ancient script, and cosmic patterns.
        Examples of Image Descriptions (Modify Based on Topic):
        The Philosopher’s Awakening – A detailed alchemical illustration depicting a human figure standing at the center of a golden triangle, surrounded by celestial symbols, ancient runes, and sacred geometry. The body is outlined with glowing energy, indicating spiritual transformation. Around the edges, handwritten mystical texts and equations hint at hidden knowledge.
        The Cosmic Laboratory – An intricate medieval alchemist’s lab, filled with mystical glass apparatus, floating sigils, and interconnected geometric blueprints. Golden energy radiates from an elixir-filled flask, symbolizing enlightenment. Cryptic diagrams and planetary motions are drawn in the background.
        The Gate to Higher Consciousness – A massive ancient staircase leading to an open portal, surrounded by concentric circles of sacred symbols, runes, and metaphysical equations. The atmosphere glows with a divine golden aura, emphasizing a transition beyond material existence.
        The Master of Hidden Wisdom – A seated mystical sage, depicted in a detailed, symmetrical esoteric drawing, surrounded by cosmic inscriptions and interwoven celestial symbols. Their hands radiate glowing energy, revealing the mastery of secret teachings.
"""
    
    temperature = 1.5
    generate_scripts = True
    handle = "thehiddenlibrary33"
    n_output_videos = 2
    from_scenes = False
    n_images = 4
    max_reference_titles = 10
    max_reference_transcripts = 1
    sections = ["introduction", "core explanation", "conclusion"]
    height = 1024
    width = 1024

    generator = VideoGenerator(temperature=temperature)
    video_settings = VideoSettings(
        handle=handle, 
        title_description=title_description, 
        script_description=script_description, 
        images_description=images_description,
        topic=topic,
        n_output_videos=n_output_videos, 
        n_images = n_images,
        max_reference_titles=max_reference_titles,
        max_reference_transcripts=max_reference_transcripts,
        sections=sections,
        from_scenes=from_scenes,
        generate_scripts=generate_scripts,
        height=height,
        width=width)

    generator.generate_videos(video_settings=video_settings)