import sys
import waitress
import main

sys.stdout.reconfigure(encoding='utf-8')

if __name__ == '__main__':
    # Preload all vector artifacts
    main.warmup_all_models()
    
    print("=" * 80)
    print(" STARTING PRODUCTION WSGI SERVER (Waitress)")
    print(" - Port: 7500")
    print(" - Threads: 64 concurrent async workers")
    print(" - Listen Backlog: 2048 TCP connections")
    print(" - Connection Limit: 1000 simultaneous sockets")
    print("=" * 80)
    
    waitress.serve(
        main.app,
        host='0.0.0.0',
        port=7500,
        threads=64,
        backlog=2048,
        connection_limit=1000,
        channel_timeout=30
    )
