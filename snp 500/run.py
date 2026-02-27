#!/usr/bin/env python3
"""
run.py — Unified launcher for the S&P 500 ML Prediction System

    python run.py              # Interactive menu
    python run.py train        # Train models (first time)
    python run.py daily        # Daily update cycle (for cron)
    python run.py dashboard    # Open live performance dashboard
    python run.py evolve       # Run full evolution cycle
    python run.py report       # Print performance to terminal
    python run.py setup        # Show automation setup guide
"""

import sys
import os
import subprocess

SCRIPTS = {
    "predictor": "sp500_predictor.py",
    "evolution": "sp500_evolution.py",
    "tracker":   "sp500_live_tracker.py",
    "backtester": "sp500_backtester.py",
    "advisor":   "sp500_advisor.py",
    "daily":     "sp500_daily.py",
}

def check_deps():
    """Check and report dependency status."""
    deps = {}
    for name, pkg in [("pandas", "pandas"), ("sklearn", "scikit-learn"),
                       ("numpy", "numpy"), ("matplotlib", "matplotlib"),
                       ("pygame", "pygame"), ("torch", "torch"),
                       ("yfinance", "yfinance"), ("openpyxl", "openpyxl")]:
        try:
            __import__(name)
            deps[name] = True
        except ImportError:
            deps[name] = False

    required = ["pandas", "sklearn", "numpy", "matplotlib", "openpyxl"]
    missing = [d for d in required if not deps[d]]
    if missing:
        print(f"\n❌ Missing required packages: {', '.join(missing)}")
        pkgs = " ".join({"sklearn": "scikit-learn"}.get(m, m) for m in missing)
        print(f"   Install: pip install {pkgs}\n")
        return False

    optional_missing = []
    if not deps["pygame"]:
        optional_missing.append("pygame (for GUI dashboards)")
    if not deps["torch"]:
        optional_missing.append("torch (for LSTM model)")
    if not deps["yfinance"]:
        optional_missing.append("yfinance (for auto data updates)")

    if optional_missing:
        print(f"\n⚠️  Optional packages not installed:")
        for m in optional_missing:
            print(f"   • {m}")
        print()

    return True

def find_data():
    """Locate the Excel data file."""
    candidates = [
        "SP500_Analysis.xlsx",
        os.path.expanduser("~/Downloads/SP500_Analysis.xlsx"),
        os.path.expanduser("~/Desktop/SP500_Analysis.xlsx"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def run_script(script_key, args=None):
    """Run one of the component scripts."""
    script = SCRIPTS.get(script_key)
    if not script or not os.path.exists(script):
        print(f"Script not found: {script}")
        print(f"Make sure all files are in: {os.getcwd()}")
        return

    cmd = [sys.executable, script] + (args or [])
    subprocess.run(cmd)

def interactive_menu():
    """Show interactive menu."""
    print("\n" + "=" * 56)
    print("   S&P 500 ML Prediction System")
    print("=" * 56)

    data_path = find_data()
    if data_path:
        print(f"   📊 Data found: {data_path}")
    else:
        print("   ⚠️  No data file found (SP500_Analysis.xlsx)")

    # Check for existing models/trades
    has_models = os.path.exists("sp500_models/live/live_models.pkl")
    has_trades = os.path.exists("sp500_logs/paper_trades.json")
    has_evo = os.path.exists("sp500_logs/evolution_history.json")

    status = []
    if has_models: status.append("Models ✓")
    if has_trades: status.append("Trades ✓")
    if has_evo: status.append("Evolution ✓")
    if status:
        print(f"   Status: {' | '.join(status)}")

    print("""
   ── Getting Started ──────────────────────────
   1. Train models           First-time setup
   2. Run daily pipeline     Data → predict → signal → email
   3. View performance       Terminal report

   ── Market Advisor ──────────────────────────
   4. Today's signal         Buy/sell/hold recommendation
   5. Backtest signals       How signals performed historically

   ── Dashboards (pygame) ──────────────────────
   6. Training dashboard     Watch models train
   7. Live tracker           Paper trading + charts

   ── Backtesting & Analysis ───────────────────
   8. Full backtest + report 10 strategies + Monte Carlo → HTML
   9. Full evolution cycle   Walk-forward + tournament

   ── Setup ────────────────────────────────────
   e. Setup email alerts     Configure daily email signal
   s. Setup automation       Cron / scheduler guide
   0. Quit
    """)

    choice = input("   Select (0-9/e/s): ").strip()

    data_arg = ["--data", data_path] if data_path else []

    if choice == "1":
        run_script("predictor")
    elif choice == "2":
        run_script("daily", data_arg)
    elif choice == "3":
        run_script("tracker", ["--report"])
    elif choice == "4":
        run_script("daily", ["--signal-only"] + data_arg)
    elif choice == "5":
        run_script("advisor", ["--backtest"] + data_arg)
    elif choice == "6":
        run_script("predictor")
    elif choice == "7":
        run_script("tracker", data_arg)
    elif choice == "8":
        run_script("backtester", ["--full"] + data_arg)
    elif choice == "9":
        run_script("evolution", ["--update"] + data_arg)
    elif choice == "e":
        run_script("daily", ["--setup-email"])
    elif choice == "s":
        run_script("evolution", ["--setup"])
    elif choice == "0":
        return
    else:
        print("   Invalid choice.")

    # Loop back
    interactive_menu()

if __name__ == "__main__":
    if not check_deps():
        sys.exit(1)

    args = sys.argv[1:]

    if not args:
        interactive_menu()
    elif args[0] == "train":
        run_script("predictor")
    elif args[0] == "daily":
        data = find_data()
        run_script("daily", (["--data", data] if data else []))
    elif args[0] == "dashboard":
        run_script("tracker")
    elif args[0] == "evolve":
        data = find_data()
        run_script("evolution", ["--update"] + (["--data", data] if data else []))
    elif args[0] == "report":
        run_script("tracker", ["--report"])
    elif args[0] == "backtest":
        data = find_data()
        run_script("backtester", ["--full"] + (["--data", data] if data else []))
    elif args[0] == "signal":
        data = find_data()
        run_script("daily", ["--signal-only"] + (["--data", data] if data else []))
    elif args[0] == "setup":
        run_script("evolution", ["--setup"])
    else:
        print(f"Unknown command: {args[0]}")
        print("Usage: python run.py [train|daily|dashboard|evolve|report|backtest|signal|setup]")
