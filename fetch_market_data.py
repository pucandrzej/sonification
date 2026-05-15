# DODATKOWO ZAKODUJ SPRAWDZENIE, CZY API ZWRÓCIŁO POPRAWNĄ ODPOWIEDŹ
# PODPOWIEDŹ: response.status_code

# CZY PRZECHOWYWANIE DANYCH W JEDNYM NADPISYWANYM PLIKU TO DOBRE WYJŚCIE? ZAPROPONUJ INNE ROZWIĄZANIE
# PODPOWIEDŹ: https://docs.python.org/3/library/sqlite3.html, Pandas, inne

import json
import time
import requests
import numpy as np

from datetime import datetime

from utils import (
    VS_CURRENCY,
    DAYS,
    INTERVAL,
    TOP_N,
    API_TIMEOUT,
    API_SLEEP_TIME_BETWEEN_CALLS,
    normalize_minmax,
    ms_to_datetime,
    MARKET_DATA_FILE,
    SECRETS_FILE,
)

# load the API key from secrets
with open(SECRETS_FILE, "r", encoding="utf-8") as f:
    API_KEY = json.load(f)["API_KEY"]

HEADERS = {"x-cg-demo-api-key": API_KEY}


def fetch_coin(coin):
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"

    params = {
        "vs_currency": VS_CURRENCY,
        "days": DAYS,
    }

    if INTERVAL is not None:  # in free plan we can't load minutely prices :')
        params["interval"] = INTERVAL

    response = requests.get(url, params=params, headers=HEADERS, timeout=API_TIMEOUT)

    response.raise_for_status()

    raw_prices = response.json()["prices"]

    times_ms = [x[0] for x in raw_prices]
    prices = [x[1] for x in raw_prices]

    times_human = [ms_to_datetime(t) for t in times_ms]

    return times_ms, times_human, prices


def main():
    url = "https://api.coingecko.com/api/v3/coins/markets"

    params = {
        "vs_currency": VS_CURRENCY,
        "order": "market_cap_desc",
        "per_page": TOP_N,
        "page": 1,
        "sparkline": False,
    }

    response = requests.get(url, params=params, headers=HEADERS, timeout=API_TIMEOUT)

    response.raise_for_status()

    coins = [coin["id"] for coin in response.json()]

    print("Top coins:")
    print(coins)

    result = {"created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "coins": {}}

    common_len = None

    for coin in coins:
        print(f"Pobieram {coin}...")

        times_ms, times_human, prices = fetch_coin(coin)

        result["coins"][coin] = {
            "times_ms": times_ms,
            "times_human": times_human,
            "prices": prices,
        }

        common_len = len(prices) if common_len is None else min(common_len, len(prices))

        time.sleep(API_SLEEP_TIME_BETWEEN_CALLS)

    for coin in coins:
        data = result["coins"][coin]

        data["times_ms"] = data["times_ms"][-common_len:]
        data["times_human"] = data["times_human"][-common_len:]

        prices = np.asarray(data["prices"][-common_len:], dtype=float)

        data["prices"] = prices.tolist()

        data["normalized"] = normalize_minmax(prices).tolist()

        data["normalized_derivative"] = normalize_minmax(
            np.diff(prices, prepend=prices[0])
        ).tolist()

    with open(MARKET_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Zapisano {MARKET_DATA_FILE}")


if __name__ == "__main__":
    main()
