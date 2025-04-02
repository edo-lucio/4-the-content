import time
import hashlib
import hmac
import json
import requests
from typing import Dict, Any, Optional, List
from openai import OpenAI

class TensorArtClient:
    def __init__(self, app_id: str, api_key: str):
        self.app_id = app_id
        self.api_key = api_key
        self.BASE_URL = 'https://ap-east-1.tensorart.cloud/v1/jobs'
        
    def _generate_signature(self, timestamp: str) -> str:
        """Generate HMAC SHA256 signature for API authentication."""
        message = f'appId={self.app_id}&timestamp={timestamp}'
        return hmac.new(
            self.api_key.encode('utf-8'), 
            message.encode('utf-8'), 
            hashlib.sha256
        ).hexdigest()

    def _prepare_headers(self) -> Dict[str, str]:
        """Prepare authenticated API request headers."""
        timestamp = str(int(time.time() * 1000))
        return {
            'Content-Type': 'application/json',
            'appId': self.app_id,
            'timestamp': timestamp,
            'signature': self._generate_signature(timestamp),
            'Authorization': f"Bearer {self.api_key}"
        }

    def _prepare_stages(self, stages, prompts):
        for stage in stages: 
            if stage["type"] == "INPUT_INITIALIZE":
                stage["inputInitialize"]["count"] = len(prompts)
            if stage["type"] == "DIFFUSION":
                stage["diffusion"]["prompts"] = [{"text": prompt} for prompt in prompts]
        return stages
    
    def generate(self, prompts: List[str], stages: Dict) -> List[str]:
        """
        Synchronously generate an image and wait for its completion.
        
        Args:
            prompt (str): Text description for image generation
            width (int, optional): Image width. Defaults to 1920.
            height (int, optional): Image height. Defaults to 1080.
            steps (int, optional): Diffusion steps. Defaults to 25.
            cfg_scale (float, optional): Classifier-free guidance scale. Defaults to 8.0.
            max_wait_time (int, optional): Maximum time to wait for job completion. Defaults to 600 seconds.
            poll_interval (int, optional): Time between status checks. Defaults to 5 seconds.
        
        Returns:
            Dict with completed job details or None if request fails
        """
        headers = self._prepare_headers()
        request_id = str(int(time.time() * 1000))
        payload = {"requestId": request_id, "stages": self._prepare_stages(stages, prompts)}

        try:
            post_response = requests.post(self.BASE_URL, headers=headers, data=json.dumps(payload))
            if post_response.status_code != 200:
                raise Exception(f"Job submission failed. Status: {post_response.status_code}")
            
            job_response = post_response.json()
            job_id = job_response['job']['id']
            
            start_time = time.time()
            while time.time() - start_time < 300:
                status_response = requests.get(f"{self.BASE_URL}/{job_id}", headers=headers)
                status_data = status_response.json()
                job_status = status_data.get('job', {}).get('status')
                
                if job_status == 'SUCCESS':
                    print("Job completed successfully!")
                    images = status_data["job"]["successInfo"]["images"]
                    return [image["url"] for image in images]
                
                elif job_status == 'FAILED':
                    print("Job failed.")
                    return None
                
                time.sleep(15)
            
            print("Job timed out.")
            return self.generate(prompts=prompts, stages=stages)
        
        except Exception as e:
            print(f"Error in image generation: {e}")
            return None