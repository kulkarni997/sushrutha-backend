"""
Supabase client — lazy initialization.

Defers create_client() to first actual use so backend can boot even if
DNS/network is flaky (e.g., phone hotspot, cold-start on HuggingFace Spaces).

Public API unchanged: `from db.supabase_client import supabase` still works.
"""
import os
from dotenv import load_dotenv

load_dotenv(override=True)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL:
    raise EnvironmentError("SUPABASE_URL is not set in environment variables.")
if not SUPABASE_KEY:
    raise EnvironmentError("SUPABASE_KEY is not set in environment variables.")


class _LazySupabase:
    """Proxy that creates the real client on first attribute access."""
    def __init__(self):
        self._client = None
        self._init_attempted = False
        self._init_error = None

    def _init_client(self):
        if self._client is not None:
            return self._client
        if self._init_attempted and self._init_error:
            # Retry once per request — transient DNS/network failures are common
            self._init_attempted = False

        self._init_attempted = True
        try:
            from supabase import create_client
            self._client = create_client(SUPABASE_URL, SUPABASE_KEY)
            self._init_error = None
            print("[supabase_client] Connection established")
            return self._client
        except Exception as e:
            self._init_error = e
            print(f"[supabase_client] Connection failed: {e}")
            raise

    def __getattr__(self, name):
        client = self._init_client()
        return getattr(client, name)


supabase = _LazySupabase()