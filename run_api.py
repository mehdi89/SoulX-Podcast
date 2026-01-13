"""
Quick start script for SoulX-Podcast API

Usage:
    python run_api.py
    python run_api.py --port 8080
    python run_api.py --model pretrained_models/SoulX-Podcast-1.7B
"""
import os
import sys
import argparse
import signal
import time


def main():
    parser = argparse.ArgumentParser(description="Start SoulX-Podcast API server")
    parser.add_argument(
        "--model",
        type=str,
        default="pretrained_models/SoulX-Podcast-1.7B",
        help="Model path (default: pretrained_models/SoulX-Podcast-1.7B)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API port (default: 8000)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="API host address (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["hf", "vllm"],
        default="hf",
        help="LLM engine (default: hf)"
    )
    parser.add_argument(
        "--fp16-flow",
        action="store_true",
        help="Use FP16 precision for Flow model (faster but slightly lower quality)"
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=2,
        help="Max concurrent tasks (default: 2)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable hot reload (development mode)"
    )

    args = parser.parse_args()

    # Set environment variables
    os.environ["MODEL_PATH"] = args.model
    os.environ["API_HOST"] = args.host
    os.environ["API_PORT"] = str(args.port)
    os.environ["LLM_ENGINE"] = args.engine
    os.environ["FP16_FLOW"] = "true" if args.fp16_flow else "false"
    os.environ["MAX_CONCURRENT_TASKS"] = str(args.max_tasks)
    os.environ["API_RELOAD"] = "true" if args.reload else "false"

    # Check model path
    if not os.path.exists(args.model):
        print(f"Error: Model path does not exist: {args.model}")
        print("\nPlease download the model first:")
        print(f"huggingface-cli download --resume-download Soul-AILab/SoulX-Podcast-1.7B --local-dir {args.model}")
        sys.exit(1)

    # Print startup info
    print("=" * 60)
    print("SoulX-Podcast API Server Starting...")
    print("=" * 60)
    print(f"Model path:    {args.model}")
    print(f"Server URL:    http://{args.host}:{args.port}")
    print(f"API docs:      http://localhost:{args.port}/docs")
    print(f"LLM engine:    {args.engine}")
    print(f"FP16 Flow:     {'Yes' if args.fp16_flow else 'No'}")
    print(f"Max tasks:     {args.max_tasks}")
    print("=" * 60)
    print("\nVoices available:")
    print("  S1: Alice (female host)")
    print("  S2: Frank (male guest)")
    print("=" * 60)
    print("\nLoading model, please wait...\n")
    print("Tip: Press Ctrl+C to stop (press twice to force quit)\n")

    # Set signal handler for quick exit
    shutdown_count = 0

    def signal_handler(signum, frame):
        nonlocal shutdown_count
        shutdown_count += 1
        if shutdown_count == 1:
            print("\n\nGracefully shutting down... (press Ctrl+C again to force quit)")
        else:
            print("\n\nForce quit!")
            # Clean up GPU memory
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            os._exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

    # Start API
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
