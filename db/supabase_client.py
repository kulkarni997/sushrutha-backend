import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL:
    raise EnvironmentError("SUPABASE_URL is not set in environment variables.")
if not SUPABASE_KEY:
    raise EnvironmentError("SUPABASE_KEY is not set in environment variables.")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
