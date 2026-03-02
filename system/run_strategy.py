import argparse
import sys

from live_database_setup import setup_signal_database

from live_strategies.live_donchian_strategy import LiveDonchianBtcusd


def main():
    # 1. Setup the command line argument parser
    parser = argparse.ArgumentParser(
        description="Run a specific live trading strategy."
    )
    parser.add_argument("strategy_name", type=str)
    args = parser.parse_args()

    # 2. Map the text argument to the actual Python class
    strategy_map = {
        "donchian_BtcUsd": LiveDonchianBtcusd,
        # "momentum": LiveMomentum,
    }

    # 3. Instantiate and run
    if args.strategy_name not in strategy_map:
        print(f"❌ Error: Unknown strategy '{args.strategy_name}'.")
        print(f"Available strategies: {list(strategy_map.keys())}")
        sys.exit(1)

    setup_signal_database(db_path="system/live_database.db")

    print(f"Booting up sequence for: {args.strategy_name.upper()}")

    # Initialize the class and trigger its base run() method
    active_strategy = strategy_map[args.strategy_name]()
    active_strategy.run()


if __name__ == "__main__":
    main()
