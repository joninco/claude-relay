"""Entry point: python -m claude_relay"""

import argparse
import logging

from aiohttp import web

from .config import ProxyConfig
from .backend import init_state
from .server import create_app


def main():
    parser = argparse.ArgumentParser(description="Claude Relay")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5021)
    parser.add_argument("--backend", default="http://localhost:30000", help="sglang/vLLM backend URL")
    parser.add_argument("--vision-url", default="", help="Vision model endpoint (enables image agent)")
    parser.add_argument("--vision-model", default="", help="Vision model name (required with --vision-url)")
    parser.add_argument("--ttl", type=int, default=30, help="Backend detect cache TTL")
    parser.add_argument("--no-image-agent", action="store_true", help="Disable image agent")
    parser.add_argument("--force-vision", action="store_true", help="Force analyzeImage on every image")
    parser.add_argument("--no-sort-tools", action="store_true", help="Disable tool sorting for KV cache")
    parser.add_argument("--no-strip-billing", action="store_true", help="Keep billing nonce in system prompt")
    parser.add_argument("--no-strip-cache-control", action="store_true", help="Keep cache_control fields")
    parser.add_argument("--no-strip-date", action="store_true", help="Keep date injection in user messages")
    parser.add_argument("--dump-requests", action="store_true", help="Dump request bodies to debug/ directory")
    parser.add_argument("--tool-debug", action="store_true", help="Dump raw backend SSE for tool requests and log tool-call diagnostics")
    parser.add_argument("--max-image-b64-chars", type=int, default=10_000_000, help="Maximum base64 chars per image (default: 10MB)")
    parser.add_argument("--max-tools", type=int, default=500, help="Maximum tools array length (default: 500)")
    parser.add_argument("--request-timeout", type=int, default=1800, help="Total request timeout in seconds (default 1800)")
    parser.add_argument("--sock-read-timeout", type=int, default=60, help="Max silence between chunks in seconds (default 60)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = ProxyConfig(
        host=args.host,
        port=args.port,
        backend_url=args.backend,
        backend_detect_ttl=args.ttl,
        vision_url=args.vision_url,
        vision_model=args.vision_model,
        image_agent_enabled=bool(args.vision_url) and not args.no_image_agent,
        force_vision=args.force_vision,
        sort_tools=not args.no_sort_tools,
        strip_billing_nonce=not args.no_strip_billing,
        strip_cache_control=not args.no_strip_cache_control,
        strip_date_injection=not args.no_strip_date,
        dump_requests=args.dump_requests,
        tool_debug=args.tool_debug,
        max_image_b64_chars=args.max_image_b64_chars,
        max_tools=args.max_tools,
        request_timeout=args.request_timeout,
        sock_read_timeout=args.sock_read_timeout,
    )

    init_state(config.backend_detect_ttl)

    print(f"Claude relay: {config.host}:{config.port} -> {config.backend_url}")
    print(f"Vision: {config.vision_url} ({config.vision_model})")
    print(f"Image agent: {'enabled' if config.image_agent_enabled else 'disabled'}{' (force-vision)' if config.force_vision else ''}")
    print(f"Backend detect TTL: {config.backend_detect_ttl}s")
    norm_flags = [
        f"sort_tools={config.sort_tools}",
        f"strip_billing={config.strip_billing_nonce}",
        f"strip_cache_control={config.strip_cache_control}",
        f"strip_date={config.strip_date_injection}",
    ]
    print(f"KV normalization: {', '.join(norm_flags)}")
    print(f"Tool debug: {'enabled' if config.tool_debug else 'disabled'}")

    app = create_app(config)
    web.run_app(app, host=config.host, port=config.port, print=lambda _: None)


if __name__ == "__main__":
    main()
