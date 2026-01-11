import os
from dotenv import load_dotenv

load_dotenv()

POLYGON_API_KEY = os.getenv('POLYGON_API_KEY', '')
PORT = int(os.getenv('PORT', 5000))
