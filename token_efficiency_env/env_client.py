import sys, os
sys.path.insert(0, os.path.abspath('../OpenEnv/src'))

from openenv.core import EnvClient

class TokenEfficiencyClient(EnvClient):
    base_url: str = "https://your-hf-space-url.hf.space"