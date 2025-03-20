import os
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi

from dotenv import load_dotenv
load_dotenv()

class YouTubeClient:
    """Handles all YouTube API interactions."""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key)

    def get_channel_by_handle(self, handle):
        """Get a Channel object by YouTube handle."""
        response = self.youtube.channels().list(
            part='id',
            forHandle=handle
        ).execute()
        
        if not response.get('items'):
            raise ValueError(f"Channel handle '{handle}' not found.")
            
        return Channel(self, response['items'][0]['id'])

    def get_uploads_playlist_id(self, channel_id):
        """Get the uploads playlist ID for a channel."""
        response = self.youtube.channels().list(
            part='contentDetails',
            id=channel_id
        ).execute()
        return response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

    def get_playlist_items(self, playlist_id):
        """Get all items from a playlist."""
        items = []
        next_page_token = None
        
        while True:
            response = self.youtube.playlistItems().list(
                part='snippet',
                playlistId=playlist_id,
                maxResults=50,
                pageToken=next_page_token
            ).execute()
            
            items.extend(response.get('items', []))
            next_page_token = response.get('nextPageToken')
            
            if not next_page_token:
                break
                
        return items

    def get_video_statistics(self, video_ids):
        """Get view statistics for multiple videos."""
        stats = {}
        
        for i in range(0, len(video_ids), 50):
            chunk = video_ids[i:i+50]
            response = self.youtube.videos().list(
                part='statistics',
                id=','.join(chunk)
            ).execute()
            
            for item in response.get('items', []):
                vid = item['id']
                stats[vid] = int(item['statistics'].get('viewCount', 0))
                
        return stats

    def get_transcript(self, video_id):
        """Get the transcript for a video using youtube-transcript-api."""
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            return ' '.join([entry['text'] for entry in transcript_list])
        except Exception as e:
            print(f"Error fetching transcript for video {video_id}: {e}")
            return None
        
    def get_channel_info(self, channel_id):
        """Get channel information including description."""
        response = self.youtube.channels().list(
            part='snippet,statistics',
            id=channel_id
        ).execute()
        
        if not response.get('items'):
            raise ValueError(f"Channel ID '{channel_id}' not found.")
            
        return response['items'][0]
    
class Channel:
    """Represents a YouTube channel and its content."""
    def __init__(self, client, channel_id):
        self.client = client
        self.id = channel_id
        self._uploads_playlist_id = None
        self._info = None  # Add this line to initialize _info

    @property
    def uploads_playlist_id(self):
        """Lazy-load the uploads playlist ID."""
        if not self._uploads_playlist_id:
            self._uploads_playlist_id = self.client.get_uploads_playlist_id(self.id)
        return self._uploads_playlist_id
    
    @property
    def info(self):
        """Lazy-load channel information."""
        if not self._info:
            self._info = self.client.get_channel_info(self.id)
        return self._info
        
    @property
    def description(self):
        """Get channel description."""
        return self.info['snippet'].get('description', '')
        
    @property
    def title(self):
        """Get channel title."""
        return self.info['snippet'].get('title', '')
        
    @property
    def subscriber_count(self):
        """Get channel subscriber count."""
        return int(self.info['statistics'].get('subscriberCount', 0))
    
    def get_videos(self, transcript=True, n=100):
        """Get all videos from the channel sorted by views descending."""
        playlist_items = self.client.get_playlist_items(self.uploads_playlist_id)
        videos = []
        video_ids = []

        for item in playlist_items:
            video_id = item['snippet']['resourceId']['videoId']
            title = item['snippet']['title']
            description = item['snippet']['description']
            thumbnails = item['snippet']['thumbnails']
            videos.append(Video(video_id, title, description, thumbnails))
            video_ids.append(video_id)
            
        stats = self.client.get_video_statistics(video_ids)
        
        for video in videos:
            video.views = stats.get(video.id, 0)
            if transcript:
                video.transcript = self.client.get_transcript(video.id)
            
        videos.sort(key=lambda v: v.views, reverse=True)
        return videos[:n]

class Video:
    """Represents a YouTube video with basic information."""
    
    def __init__(self, video_id, title, description, thumbnails, views=0, transcript=""):
        self.id = video_id
        self.title = title
        self.views = views
        self.description = description
        self.thumbnails = thumbnails
        self.transcript = transcript

    def __repr__(self):
        return f"Video(id='{self.id}', title='{self.title}', views={self.views})"

if __name__ == "__main__":
    API_KEY = os.getenv("GOOGLE_API_KEY")
    client = YouTubeClient(API_KEY)
    
    try:
        channel = client.get_channel_by_handle("@DailyJesusDevotional")
        videos = channel.get_videos()
        
        print(f"Found {len(videos)} videos for channel (sorted by views):")
        for video in videos[:10]:
            print(f"- {video.title} (Views: {video.views}, Transcript: {video.transcript})")
    except Exception as e:
        print(f"Error: {e}")