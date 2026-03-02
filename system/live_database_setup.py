import sqlite3


def setup_signal_database(db_path="system/live_database.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. CONTROL PANEL (The Kill Switches)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS strategy_control (
            strategy_name TEXT PRIMARY KEY,
            is_active BOOLEAN,
            allocated_capital REAL
        )
    """
    )

    # 2. VIRTUAL POSITIONS (What each strategy "owns")
    # This allows performance tracking per strategy.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS virtual_positions (
            strategy_name TEXT,
            ticker TEXT,
            quantity REAL,
            entry_price REAL,
            last_updated TIMESTAMP,
            PRIMARY KEY (strategy_name, ticker)
        )
    """
    )

    # 3. SIGNAL BUS (The mailbox where instructions are left for the Executor)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS target_signals (
            strategy_name TEXT,
            ticker TEXT,
            target_quantity REAL,
            signal_price REAL,
            timestamp TIMESTAMP,
            PRIMARY KEY (strategy_name, ticker)
        )
    """
    )

    # 4. TRADE HISTORY (For your performance analysis)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS virtual_trade_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP,
            strategy_name TEXT,
            ticker TEXT,
            action TEXT,         -- 'BUY' or 'SELL'
            quantity REAL,
            price REAL
        )
    """
    )

    conn.commit()
    conn.close()
    print("✅ Signal Database Initialized.")


def register_strategy(db_path, strategy_name, capital):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Insert or update the strategy to be ACTIVE (1) with the specified capital
    cursor.execute(
        """
        INSERT OR REPLACE INTO strategy_control (strategy_name, is_active, allocated_capital) 
        VALUES (?, 1, ?)
    """,
        (strategy_name, capital),
    )

    conn.commit()
    conn.close()
    print(f"✅ Registered '{strategy_name}' with ${capital} and set to ACTIVE.")
