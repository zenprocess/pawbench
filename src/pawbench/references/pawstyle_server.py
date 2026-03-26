"""Gold standard PawStyle by Lola REST API server.

A fully working Python stdlib http.server implementation that serves all 6 Lola
products and handles every endpoint referenced across the PawStyle benchmark
scenarios. This is the canonical "correct output" that agent-generated code is
compared against.

Endpoints
---------
GET  /api/products              — all 6 products
GET  /api/products/<id>         — single product (404 if missing)
GET  /api/health                — server health + uptime + counters
POST /api/orders                — place an order (validates items)
POST /api/wishlist              — toggle product in Lola's Favorites
GET  /api/wishlist              — current wishlist
GET  /api/size-guide/<breed>    — Lola-tested breed-specific sizing

All responses include CORS headers.

Usage::

    python -m pawbench.references.pawstyle_server          # port 8080
    python -m pawbench.references.pawstyle_server --port 9090
"""

from __future__ import annotations

import json
import random
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Product catalog
# ---------------------------------------------------------------------------

PRODUCTS: list[dict[str, Any]] = [
    {
        "id": 1,
        "name": "Lola's Signature Bandana",
        "price": 12.00,
        "sizes": ["XS", "S", "M", "L", "XL"],
        "image": "/img/bandana.jpg",
        "lola_pick": True,
    },
    {
        "id": 2,
        "name": "Cozy Knit Sweater",
        "price": 35.00,
        "sizes": ["XS", "S", "M", "L", "XL"],
        "image": "/img/sweater.jpg",
        "lola_pick": False,
    },
    {
        "id": 3,
        "name": "Rainy Day Raincoat",
        "price": 45.00,
        "sizes": ["XS", "S", "M", "L", "XL"],
        "image": "/img/raincoat.jpg",
        "lola_pick": False,
    },
    {
        "id": 4,
        "name": "Adventure Booties (4-pack)",
        "price": 28.00,
        "sizes": ["XS", "S", "M", "L", "XL"],
        "image": "/img/booties.jpg",
        "lola_pick": False,
    },
    {
        "id": 5,
        "name": "Dapper Bow Tie",
        "price": 15.00,
        "sizes": ["XS", "S", "M", "L", "XL"],
        "image": "/img/bowtie.jpg",
        "lola_pick": True,
    },
    {
        "id": 6,
        "name": "Walk-in-Style Harness",
        "price": 40.00,
        "sizes": ["XS", "S", "M", "L", "XL"],
        "image": "/img/harness.jpg",
        "lola_pick": False,
    },
]

VALID_SIZES = {"XS", "S", "M", "L", "XL"}

# ---------------------------------------------------------------------------
# Size guide data (Lola-tested)
# ---------------------------------------------------------------------------

GENERIC_SIZES: dict[str, dict[str, str]] = {
    "XS": {
        "breed_example": "Chihuahua",
        "chest": "10-13in",
        "neck": "8-10in",
        "weight": "2-6lbs",
    },
    "S": {
        "breed_example": "Pomeranian",
        "chest": "13-16in",
        "neck": "10-12in",
        "weight": "6-12lbs",
    },
    "M": {
        "breed_example": "Beagle",
        "chest": "16-22in",
        "neck": "12-16in",
        "weight": "12-35lbs",
    },
    "L": {
        "breed_example": "Labrador",
        "chest": "22-28in",
        "neck": "16-20in",
        "weight": "35-70lbs",
    },
    "XL": {
        "breed_example": "Golden Retriever",
        "chest": "28-34in",
        "neck": "20-24in",
        "weight": "70-100lbs",
    },
}

BREED_ADJUSTMENTS: dict[str, dict[str, dict[str, str]]] = {
    "chihuahua": {
        "XS": {
            "breed_example": "Chihuahua",
            "chest": "9-12in",
            "neck": "7-9in",
            "weight": "2-5lbs",
        },
        "S": {
            "breed_example": "Large Chihuahua",
            "chest": "12-14in",
            "neck": "9-11in",
            "weight": "5-8lbs",
        },
    },
    "pomeranian": {
        "XS": {
            "breed_example": "Toy Pomeranian",
            "chest": "10-13in",
            "neck": "8-10in",
            "weight": "3-5lbs",
        },
        "S": {
            "breed_example": "Pomeranian",
            "chest": "13-16in",
            "neck": "10-12in",
            "weight": "5-7lbs",
        },
    },
    "beagle": {
        "M": {
            "breed_example": "Beagle",
            "chest": "17-21in",
            "neck": "13-16in",
            "weight": "18-30lbs",
        },
        "L": {
            "breed_example": "Large Beagle",
            "chest": "21-24in",
            "neck": "16-18in",
            "weight": "30-40lbs",
        },
    },
    "labrador": {
        "L": {
            "breed_example": "Labrador",
            "chest": "24-28in",
            "neck": "18-22in",
            "weight": "55-75lbs",
        },
        "XL": {
            "breed_example": "Large Labrador",
            "chest": "28-32in",
            "neck": "22-26in",
            "weight": "75-100lbs",
        },
    },
    "golden": {
        "L": {
            "breed_example": "Golden Retriever",
            "chest": "24-30in",
            "neck": "18-22in",
            "weight": "55-75lbs",
        },
        "XL": {
            "breed_example": "Large Golden Retriever",
            "chest": "30-36in",
            "neck": "22-26in",
            "weight": "75-100lbs",
        },
    },
}

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

_start_time: float = 0.0
_orders_count: int = 0
_wishlist: set[int] = set()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _product_by_id(pid: int) -> dict[str, Any] | None:
    for p in PRODUCTS:
        if p["id"] == pid:
            return p
    return None


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def _size_guide_for(breed: str) -> dict[str, Any]:
    breed_lower = breed.lower()
    sizes = dict(GENERIC_SIZES)  # shallow copy of top-level keys
    if breed_lower in BREED_ADJUSTMENTS:
        for size_key, measurements in BREED_ADJUSTMENTS[breed_lower].items():
            sizes[size_key] = measurements
    return {
        "breed": breed,
        "tagline": "Lola-tested sizing for every breed",
        "sizes": sizes,
    }


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------


class PawStyleHandler(BaseHTTPRequestHandler):
    """HTTP request handler for PawStyle by Lola."""

    # -- CORS helper --------------------------------------------------------

    def _set_cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    # -- Response helpers ---------------------------------------------------

    def _respond_json(self, code: int, body: Any) -> None:
        payload = _json_bytes(body)
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self._set_cors()
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _respond_error(self, code: int, message: str) -> None:
        self._respond_json(code, {"error": message})

    # -- Logging ------------------------------------------------------------

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        elapsed = (time.time() - _start_time) * 1000
        print(  # noqa: T201
            f"[{elapsed:>10.1f}ms] {self.command} {self.path} -> {args[1] if len(args) > 1 else '?'}"
        )

    # -- OPTIONS (CORS preflight) ------------------------------------------

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._set_cors()
        self.end_headers()

    # -- GET ----------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/api/products":
            self._respond_json(200, PRODUCTS)
            return

        if path.startswith("/api/products/"):
            raw_id = path[len("/api/products/") :]
            try:
                pid = int(raw_id)
            except ValueError:
                self._respond_error(400, f"Invalid product ID: {raw_id}")
                return
            product = _product_by_id(pid)
            if product is None:
                self._respond_error(404, f"Product {pid} not found")
                return
            self._respond_json(200, product)
            return

        if path == "/api/health":
            self._respond_json(
                200,
                {
                    "status": "ok",
                    "store": "PawStyle by Lola",
                    "uptime": round(time.time() - _start_time, 2),
                    "orders_count": _orders_count,
                    "products_count": len(PRODUCTS),
                },
            )
            return

        if path.startswith("/api/size-guide/"):
            breed = path[len("/api/size-guide/") :]
            if not breed:
                breed = "generic"
            self._respond_json(200, _size_guide_for(breed))
            return

        if path == "/api/size-guide":
            self._respond_json(200, _size_guide_for("generic"))
            return

        if path == "/api/wishlist":
            self._respond_json(
                200,
                {
                    "wishlist": sorted(_wishlist),
                    "count": len(_wishlist),
                },
            )
            return

        self._respond_error(404, "Not found")

    # -- POST ---------------------------------------------------------------

    def do_POST(self) -> None:  # noqa: N802
        global _orders_count  # noqa: PLW0603

        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length) if content_length else b""

        try:
            body = json.loads(raw_body) if raw_body else {}
        except json.JSONDecodeError:
            self._respond_error(400, "Invalid JSON body")
            return

        # -- POST /api/orders -----------------------------------------------
        if path == "/api/orders":
            items = body.get("items")
            if not isinstance(items, list) or len(items) == 0:
                self._respond_error(400, "Request body must contain a non-empty 'items' array")
                return

            total = 0.0
            for idx, item in enumerate(items):
                pid = item.get("product_id")
                size = item.get("size")
                qty = item.get("quantity")

                if not isinstance(pid, int) or _product_by_id(pid) is None:
                    self._respond_error(400, f"Item {idx}: invalid product_id {pid}")
                    return
                if size not in VALID_SIZES:
                    self._respond_error(
                        400,
                        f"Item {idx}: invalid size '{size}'. Must be one of {sorted(VALID_SIZES)}",
                    )
                    return
                if not isinstance(qty, int) or qty < 1:
                    self._respond_error(400, f"Item {idx}: quantity must be a positive integer")
                    return

                product = _product_by_id(pid)
                assert product is not None  # guarded above
                total += product["price"] * qty

            order_id = f"{random.randint(100000, 999999)}"  # noqa: S311
            _orders_count += 1

            self._respond_json(
                201,
                {
                    "order_id": order_id,
                    "total": round(total, 2),
                    "items_count": len(items),
                    "message": "Lola approves your style!",
                },
            )
            return

        # -- POST /api/wishlist ---------------------------------------------
        if path == "/api/wishlist":
            pid = body.get("product_id")
            if not isinstance(pid, int) or _product_by_id(pid) is None:
                self._respond_error(
                    400,
                    f"Invalid product_id: {pid}. Must be 1-{len(PRODUCTS)}.",
                )
                return

            if pid in _wishlist:
                _wishlist.discard(pid)
                wishlisted = False
                message = "Removed from Lola's Favorites"
            else:
                _wishlist.add(pid)
                wishlisted = True
                message = "Added to Lola's Favorites!"

            self._respond_json(
                200,
                {
                    "wishlisted": wishlisted,
                    "wishlist": sorted(_wishlist),
                    "message": message,
                },
            )
            return

        self._respond_error(404, "Not found")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(host: str = "0.0.0.0", port: int = 8080) -> None:  # noqa: S104
    """Start the PawStyle server."""
    global _start_time  # noqa: PLW0603
    _start_time = time.time()

    server = HTTPServer((host, port), PawStyleHandler)
    print(f"PawStyle by Lola API running on http://{host}:{port}")  # noqa: T201
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down PawStyle server.")  # noqa: T201
        server.server_close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PawStyle by Lola API server")
    parser.add_argument("--port", type=int, default=8080, help="Port (default 8080)")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default 0.0.0.0)")  # noqa: S104
    args = parser.parse_args()
    run(host=args.host, port=args.port)
