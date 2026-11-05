"""Table viewer web server (run with ``python -m xpm_mlboard.tables``).

Serves the logical tables of a :class:`~xpm_mlboard.tables.TableViewerService`
as Perspective tables over a websocket, plus a small page driving the
``<perspective-viewer>`` bundled in ``perspective-python`` (so the browser
always runs the client matching the server version, without any CDN).

Built on FastAPI/uvicorn, which experimaestro already depends on.
"""

import argparse
import asyncio
import contextlib
import json
import logging
import socket
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import perspective
import polars as pl
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from perspective.handlers.starlette import PerspectiveStarletteHandler

from . import READY_MESSAGE, VIEWS_DIR
from .loader import TableRegistry, to_arrow, to_perspective

logger = logging.getLogger("xpm.mlboard.tables")

STATIC = Path(__file__).parent / "static"
PERSPECTIVE_STATIC = Path(perspective.__file__).parent / "widget" / "static"


@dataclass
class PublishedTable:
    #: Name of the Perspective table (changes when the schema changes)
    hosted_name: str
    schema: List[str]
    table: Any
    generation: int
    rows: int
    #: Perspective column types (``float``, ``string``, ...)
    types: Dict[str, str]


class Publisher:
    """Keeps one Perspective table per logical table.

    Rows are replaced in place; when the columns change, a new Perspective table
    (next *generation*) is hosted, since a table cannot change its schema. The
    page notices the new name and reopens it; old generations are deleted once
    no view uses them anymore.
    """

    def __init__(self, client):
        self.client = client
        self.tables: Dict[str, PublishedTable] = {}
        self.retired: List[Any] = []

    def publish(self, name: str, df: pl.DataFrame):
        current = self.tables.get(name)
        if df.width == 0:
            # No rows loaded (yet / anymore)
            if current is not None:
                self.retired.append(self.tables.pop(name).table)
            return

        df = to_perspective(df)
        schema = [f"{column}:{dtype}" for column, dtype in df.schema.items()]
        data = to_arrow(df)
        if current is not None and current.schema == schema:
            current.table.replace(data)
            current.rows = df.height
            return

        generation = 0
        if current is not None:
            generation = current.generation + 1
            self.retired.append(current.table)
        hosted_name = f"{name}#{generation}"
        table = self.client.table(data, name=hosted_name)
        self.tables[name] = PublishedTable(
            hosted_name=hosted_name,
            schema=schema,
            table=table,
            generation=generation,
            rows=df.height,
            types=table.schema(),
        )

    def collect(self):
        """Delete the retired tables that have no view left."""
        remaining = []
        for table in self.retired:
            try:
                table.delete()
            except Exception:
                remaining.append(table)
        self.retired = remaining


class TablesApp:
    """Watches the registered tables and publishes them to Perspective."""

    def __init__(self, root: Path, poll_interval: float = 2.0):
        self.root = root
        self.poll_interval = poll_interval
        self.registry = TableRegistry(root)
        self.server = perspective.Server()
        self.publisher = Publisher(self.server.new_local_client())
        self._lock = threading.Lock()

    def sync(self):
        with self._lock:
            for name in self.registry.refresh():
                try:
                    self.publisher.publish(name, self.registry.frame(name))
                except Exception:
                    logger.exception("Could not publish table %s", name)
            self.publisher.collect()

    def tables(self) -> List[Dict[str, Any]]:
        with self._lock:
            result = []
            for table in self.registry.summary():
                published = self.publisher.tables.get(table["name"])
                table["table"] = published.hosted_name if published else None
                table["rows"] = published.rows if published else 0
                table["types"] = published.types if published else {}
                result.append(table)
            return result

    def view_path(self, name: str) -> Path:
        return self.root / VIEWS_DIR / f"{quote(name, safe='')}.json"

    async def watch(self):
        while True:
            try:
                await asyncio.to_thread(self.sync)
            except Exception:
                logger.exception("Error while refreshing the tables")
            await asyncio.sleep(self.poll_interval)


def create_app(tables: TablesApp) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(tables.watch())
        try:
            yield
        finally:
            task.cancel()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/static/app.js")
    async def app_js():
        return FileResponse(STATIC / "app.js", media_type="text/javascript")

    @app.get("/static/perspective.js")
    async def perspective_js():
        return FileResponse(
            PERSPECTIVE_STATIC / "perspective-anywidget.js",
            media_type="text/javascript",
        )

    @app.get("/static/perspective.css")
    async def perspective_css():
        return FileResponse(
            PERSPECTIVE_STATIC / "perspective-anywidget.css", media_type="text/css"
        )

    @app.get("/api/tables")
    async def list_tables():
        return {"tables": tables.tables()}

    @app.get("/api/views/{name:path}")
    async def get_view(name: str):
        path = tables.view_path(name)
        if not path.is_file():
            return JSONResponse(None)
        return JSONResponse(json.loads(path.read_text()))

    @app.put("/api/views/{name:path}")
    async def put_view(name: str, request: Request):
        try:
            config = await request.json()
        except Exception:
            raise HTTPException(400, "Invalid JSON") from None
        path = tables.view_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(config, indent=2, sort_keys=True))
        tmp.replace(path)
        return {"saved": True}

    @app.websocket("/websocket")
    async def websocket(websocket: WebSocket):
        handler = PerspectiveStarletteHandler(
            perspective_server=tables.server, websocket=websocket
        )
        with contextlib.suppress(Exception):
            await handler.run()

    return app


def main(argv: Optional[List[str]] = None):
    import uvicorn

    parser = argparse.ArgumentParser(description="xpm-mlboard table viewer")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    tables = TablesApp(args.root, poll_interval=args.poll_interval)

    # Bind ourselves so that an OS-assigned port (--port 0) can be reported
    sock = socket.create_server((args.host, args.port))
    port = sock.getsockname()[1]
    print(f"{READY_MESSAGE} http://{args.host}:{port}/", file=sys.stderr, flush=True)

    config = uvicorn.Config(create_app(tables), log_level="warning", ws_max_size=2**30)
    uvicorn.Server(config).run(sockets=[sock])
