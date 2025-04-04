import re 
from itertools import islice

def process_script(script: str):
    return re.sub(r"\[[^\]]*\]|\(.*?\)|\*\*.*?\*\*|\*", "", script)

def process_topic(topic: str):
    return topic.replace(" ", "-")

def process_title(title: str):
    return "-".join((re.sub(r'[^A-Za-z\s]', '', title).strip().split(" ")))

def batched(iterable, n):
    """Helper function to yield n-sized chunks from an iterable."""
    it = iter(iterable)
    while chunk := list(islice(it, n)):
        yield chunk