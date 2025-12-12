from dotenv import load_dotenv
from pathlib import Path


def load_api_keys():
    """Load API keys"""
    env_file = Path(__file__).parent / "api_keys.env"
    load_dotenv(env_file)
