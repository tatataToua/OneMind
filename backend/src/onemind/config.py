"""Runtime configuration.

Every knob that differs between the demo laptop and a production deployment lives
here, so swapping local Ollama for AWS Bedrock is an environment change rather
than a code change.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ONEMIND_", env_file=".env", extra="ignore")

    # --- provider selection -------------------------------------------------
    llm_provider: str = "ollama"

    # --- ollama -------------------------------------------------------------
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3.5:4b"
    # 256K is the model's ceiling; we cap far lower so four concurrent agent
    # slots fit alongside the weights in 8GB of VRAM.
    ollama_num_ctx: int = 16384
    ollama_timeout_s: float = 120.0

    # --- bedrock (production path, unused on the demo box) ------------------
    bedrock_region: str = "us-east-1"
    bedrock_model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"

    # --- orchestration ------------------------------------------------------
    # Ollama serves this many requests concurrently; fan-out wider than this
    # queues rather than parallelises.
    max_parallel_agents: int = 4
    agent_timeout_s: float = 90.0

    # --- guardrails ---------------------------------------------------------
    phi_redaction_enabled: bool = True

    # --- data ---------------------------------------------------------------
    fixtures_dir: str = "fixtures"


settings = Settings()
