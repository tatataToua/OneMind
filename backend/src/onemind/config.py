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
    # Bearer token the provider presents to `llm/gateway.py`. Empty for an
    # ordinary local run, where nothing sits between this process and loopback
    # and there is nothing to authenticate to. Set on both ends only for the
    # tunnelled demo, where the hosted container reaches back to a laptop GPU.
    ollama_auth_token: str = ""
    # Port the gateway listens on. Ollama keeps 11434 and stays bound to
    # loopback; the tunnel is pointed here, so what reaches the internet is the
    # allowlist rather than the model server.
    ollama_gateway_port: int = 11435

    # --- bedrock (production path, unused on the demo box) ------------------
    bedrock_region: str = "us-east-1"
    bedrock_model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"

    # --- groq (what the hosted deployment runs on) --------------------------
    # No default key: an unset key must fail loudly at construction rather than
    # send an unauthenticated request from a deployed container.
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    # Must be a model offering strict-mode structured outputs, or routing loses
    # its decode-time guarantee and falls back to hoping. As of this writing
    # that is qwen3.8-27b and the two gpt-oss sizes. Kept in the Qwen family on
    # purpose so the hosted eval stays comparable to the local one.
    groq_model: str = "qwen/qwen3.8-27b"
    # Groq's answer to Ollama's `think: false`. Empty string omits the
    # parameter entirely, for models that reject it.
    groq_reasoning_format: str = "hidden"
    groq_timeout_s: float = 120.0
    # The free tier's binding limit is 8000 tokens/minute, not a request count
    # (the request cap is 1000/day). One turn is a routing call, two calls per
    # specialist per wave, and a synthesis stream, so a single turn can spend a
    # meaningful slice of a minute's token budget and two in quick succession
    # cross it. An unretried 429 reaches the clinician as a 500. Retried with
    # exponential backoff, or with whatever `Retry-After` says when Groq sends
    # one. Six attempts because the backoff has to outlast a token-per-minute
    # window refilling, which three (~7s) does not.
    groq_max_retries: int = 6
    groq_retry_base_delay_s: float = 1.0

    # --- orchestration ------------------------------------------------------
    # Ollama serves this many requests concurrently; fan-out wider than this
    # queues rather than parallelises.
    max_parallel_agents: int = 4
    agent_timeout_s: float = 90.0
    # Waves of dispatch per request. Two means a specialist blocked for want of
    # an identifier gets one retry once a sibling establishes it. Raising this
    # reintroduces the unbounded plan-act-observe loop `agents/base.py` exists
    # to avoid, so it is a cap rather than a starting point.
    max_waves: int = 2

    # --- session memory -----------------------------------------------------
    # How long an idle conversation - and the PHI vocabulary it holds - stays
    # in memory. Nothing is written to disk; this is the whole retention story.
    session_ttl_s: float = 1800.0
    max_sessions: int = 200
    # Prior turns shown to the router. Only the router sees history at all.
    history_turns: int = 3
    # Retrieved evidence carried between turns for cross-turn reconciliation.
    max_retained_results: int = 8

    # --- guardrails ---------------------------------------------------------
    phi_redaction_enabled: bool = True
    # Flag instruction-shaped text arriving in tool results. The fence around
    # retrieved data is unconditional; this only controls the audit signal.
    injection_detection_enabled: bool = True

    # --- request limits -----------------------------------------------------
    # Sustained rate and burst for the two endpoints that run inference.
    # Deliberately generous: the control that actually bounds a caller's share
    # of the GPU is the concurrency cap below, not this.
    rate_limit_per_minute: float = 20.0
    rate_limit_burst: int = 10
    # Ceiling on distinct callers tracked at once. The limiter allocates per
    # source address and the attacker picks the addresses, so the map is
    # bounded the same way `max_sessions` bounds conversations.
    rate_limit_max_keys: int = 1024
    # In-flight requests per caller. One request can occupy every one of
    # `max_parallel_agents` inference slots for `agent_timeout_s`, so this is
    # what stops a single client holding the whole model.
    max_concurrent_per_client: int = 2
    # Bytes accepted on a request body. `message` is capped at 4000 characters
    # by the schema; this refuses the oversized payload before it is parsed.
    max_request_bytes: int = 64 * 1024

    # --- data ---------------------------------------------------------------
    fixtures_dir: str = "fixtures"

    # --- serving ------------------------------------------------------------
    # Built frontend for the API to serve at "/". Empty in development, where
    # Vite owns the UI and proxies /api here. The container sets it, which is
    # what collapses the two origins into one and retires CORS in production.
    static_dir: str = ""


settings = Settings()
