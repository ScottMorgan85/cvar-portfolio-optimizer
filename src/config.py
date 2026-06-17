from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
BENCHMARKS_DIR = ROOT / "benchmarks" / "results"

DATA_DIR.mkdir(exist_ok=True)
BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TICKERS = ["SPY", "QQQ", "IEF", "GLD", "VNQ", "EEM", "HYG", "LQD"]

SP500_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "UNH", "XOM", "JNJ", "JPM", "V",
    "PG", "MA", "HD", "CVX", "MRK", "ABBV", "LLY", "PEP", "KO", "COST",
    "AVGO", "WMT", "MCD", "CSCO", "TMO", "ABT", "CRM", "ACN", "DHR", "NKE",
    "TXN", "NEE", "PM", "UNP", "AMD", "HON", "ORCL", "LOW", "INTC", "UPS",
    "IBM", "QCOM", "BA", "CAT", "SPGI", "GE", "DIS", "AMGN", "INTU", "DE",
    "SBUX", "ISRG", "AMAT", "PLD", "GS", "MDLZ", "ADI", "BKNG", "AXP", "GILD",
    "SYK", "TJX", "BLK", "MMC", "VRTX", "REGN", "CVS", "SCHW", "ADP", "C",
    "PNC", "ZTS", "LRCX", "SO", "MO", "CI", "BDX", "DUK", "CME", "TMUS",
    "BSX", "CL", "EOG", "CB", "NOC", "MMM", "ITW", "SLB", "USB", "EQIX",
    "APD", "WM", "MU", "FCX", "CSX",
]

DEFAULT_START = "2007-01-01"
DEFAULT_N_SIMULATIONS = 10_000
DEFAULT_CONFIDENCE = 0.95

SCENARIOS = {
    "COVID Crash": {
        "start": "2020-02-19",
        "end": "2020-03-23",
        "description": "Peak to trough: S&P 500 -34% in 33 days",
    },
    "2022 Rates Shock": {
        "start": "2022-01-01",
        "end": "2022-12-31",
        "description": "Fed hiking cycle: bonds and equities fell together",
    },
    "GFC Peak": {
        "start": "2008-09-15",
        "end": "2009-03-09",
        "description": "Lehman to S&P trough: -47%",
    },
    "Full History": {
        "start": "2015-01-01",
        "end": None,
        "description": "Full available history",
    },
}
