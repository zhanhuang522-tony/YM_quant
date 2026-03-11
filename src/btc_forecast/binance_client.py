from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import httpx
import pandas as pd

logger = logging.getLogger(__name__)


class BinanceClient:
    def __init__(self, base_url: str = "https://api.binance.com", timeout: float = 20.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @staticmethod
    def _to_ms(ts: datetime) -> int:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return int(ts.timestamp() * 1000)

    @staticmethod
    def _interval_ms(interval: str) -> int:
        mapping = {
            "1m": 60_000,
            "5m": 300_000,
            "15m": 900_000,
            "30m": 1_800_000,
            "1h": 3_600_000,
            "4h": 14_400_000,
            "1d": 86_400_000,
        }
        if interval not in mapping:
            raise ValueError(f"Unsupported interval: {interval}")
        return mapping[interval]

    def _request_klines(self, symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000) -> list[list]:
        endpoint = f"{self.base_url}/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": limit,
        }
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.get(endpoint, params=params)
                if resp.status_code >= 500:
                    raise httpx.HTTPStatusError("server error", request=resp.request, response=resp)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                sleep_s = min(2**attempt, 16)
                logger.warning("Kline request failed (%s) %s", symbol, exc)
                time.sleep(sleep_s)

        assert last_err is not None
        raise last_err

    def fetch_ohlcv(self, symbol: str, interval: str, start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
        start_ms = self._to_ms(start_utc)
        end_ms = self._to_ms(end_utc)
        step_ms = self._interval_ms(interval)
        cursor = start_ms
        rows: list[list] = []

        while cursor <= end_ms:
            payload = self._request_klines(symbol=symbol, interval=interval, start_ms=cursor, end_ms=end_ms, limit=1000)
            if not payload:
                break
            rows.extend(payload)
            last_open_ms = int(payload[-1][0])
            cursor = last_open_ms + step_ms
            time.sleep(0.05)
            if len(payload) < 1000:
                break

        if not rows:
            raise RuntimeError(f"No klines returned for {symbol}")

        columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "trades",
            "taker_base_vol",
            "taker_quote_vol",
            "ignore",
        ]
        df = pd.DataFrame(rows, columns=columns).drop(columns=["ignore"])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

        for col in ["open", "high", "low", "close", "volume", "quote_asset_volume", "taker_base_vol", "taker_quote_vol"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["trades"] = pd.to_numeric(df["trades"], errors="coerce").fillna(0).astype(int)
        return df.sort_values("open_time").reset_index(drop=True)
