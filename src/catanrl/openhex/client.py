"""HTTP client for OpenHex API (https://open-hex.web.app/api)."""

from __future__ import annotations

from typing import Any, Dict, List

import requests

DEFAULT_BASE_URL = "https://open-hex.web.app/api"


class OpenHexClient:
    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def register_bot(
        self,
        name: str,
        mode: str = "1v1",
        coordinate_format: str = "coordinates",
    ) -> Dict[str, Any]:
        r = self.session.post(
            f"{self.base_url}/bots",
            json={
                "name": name,
                "mode": mode,
                "coordinateFormat": coordinate_format,
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def queue_join(self, mode: str = "1v1") -> None:
        r = self.session.post(
            f"{self.base_url}/queue/join",
            json={"mode": mode},
            timeout=self.timeout,
        )
        r.raise_for_status()

    def queue_leave(self, mode: str = "1v1") -> None:
        r = self.session.post(
            f"{self.base_url}/queue/leave",
            json={"mode": mode},
            timeout=self.timeout,
        )
        r.raise_for_status()

    def queue_status(self) -> Dict[str, Any]:
        r = self.session.get(f"{self.base_url}/queue/status", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def games_active(self) -> List[Dict[str, Any]]:
        r = self.session.get(
            f"{self.base_url}/games",
            params={"status": "active"},
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            return data
        return data.get("games", data)

    def get_state(self, game_id: str) -> Dict[str, Any]:
        r = self.session.get(
            f"{self.base_url}/games/{game_id}/state",
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def get_events(self, game_id: str, after: int = -1) -> Dict[str, Any]:
        r = self.session.get(
            f"{self.base_url}/games/{game_id}/events",
            params={"after": after},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def post_action(
        self,
        game_id: str,
        body: Dict[str, Any],
    ) -> Dict[str, Any]:
        r = self.session.post(
            f"{self.base_url}/games/{game_id}/actions",
            json=body,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()
