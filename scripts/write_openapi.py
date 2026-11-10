"""Write the dashboard API's OpenAPI document to a JSON file, for the site's generated types.

Lane: Ilan Danial (machinery).

    python -m scripts.write_openapi web/node_modules/.cache/optspread/openapi.json

The document is the one ``optspread.server`` serves at ``/openapi.json``, taken from
``create_app`` without serving it: no server, no data and no network. The settings
point into an empty temporary folder, so nothing on disk can change the document. The
site folder exists there, so ``/`` is the static site rather than the stand-in route
a missing build gets, and the document describes the API alone. Keys are sorted and
the file is ASCII with LF line ends and a final newline, so the same code always
writes the same bytes.

``make web`` runs it, then ``npm run api-types`` (openapi-typescript) turns the file
into ``web/lib/schema.d.ts``, which ``web/lib/api.ts`` and its siblings read their
types from. Needs the ``dashboard`` extra.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any


def openapi_document() -> dict[str, Any]:
    """``create_app(...).openapi()`` for settings in an empty temporary folder."""
    from optspread.server import Settings, create_app  # the dashboard extra

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "site").mkdir()
        settings = Settings(
            runs=root / "runs",
            docs=root / "docs",
            data=root / "data",
            web=root / "site",
            readme=root / "README.md",
            attribution=root / "attribution",
            actionmix=root / "actionmix",
        )
        document: dict[str, Any] = create_app(settings).openapi()
    return document


def render(document: dict[str, Any]) -> str:
    """The document as the file holds it: keys sorted, two-space indents, ASCII (other
    characters escaped), a final newline."""
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the dashboard's OpenAPI document.")
    parser.add_argument("out", type=Path, help="the JSON file to write")
    args = parser.parse_args(argv)
    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(openapi_document()), encoding="utf-8", newline="\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
