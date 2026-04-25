"""
Back-compat shim — the real client lives in ``token_efficiency_env.client``.

Older notebooks and READMEs reference ``TokenEfficiencyClient`` from this
module. New code should use ``TokenEfficiencyEnv`` from
``token_efficiency_env.client`` directly:

    from token_efficiency_env.client import TokenEfficiencyEnv

    async with TokenEfficiencyEnv(base_url="https://<your-space>.hf.space") as env:
        ...
"""

from .client import TokenEfficiencyEnv as TokenEfficiencyClient
from .client import TokenEfficiencyEnv

__all__ = ["TokenEfficiencyEnv", "TokenEfficiencyClient"]
