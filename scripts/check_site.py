#!/usr/bin/env python3
"""Check local website references, accessibility labels and asset integrity."""
from __future__ import annotations

import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docs"


class Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.controls: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag: str, attributes: list[tuple[str, str | None]]) -> None:
        attrs = dict(attributes)
        if node_id := attrs.get("id"):
            if node_id in self.ids:
                self.errors.append(f"Duplicate id: {node_id}")
            self.ids.add(node_id)
        for key in ("href", "src", "poster"):
            if target := attrs.get(key):
                self.links.append(target)
        if target := attrs.get("aria-controls"):
            self.controls.extend(target.split())
        if tag == "img" and not attrs.get("alt"):
            self.errors.append("Image without alt text")
        if tag == "video":
            if "controls" not in attrs or not attrs.get("aria-label"):
                self.errors.append("Video needs controls and an accessible label")
            if not attrs.get("width") or not attrs.get("height"):
                self.errors.append("Video needs intrinsic dimensions")


def main() -> None:
    page = Page()
    page.feed((SITE / "index.html").read_text(encoding="utf-8"))
    errors = page.errors
    for target in page.links:
        url = urlsplit(target)
        if url.scheme or url.netloc:
            continue
        if url.path:
            path = (SITE / unquote(url.path)).resolve()
            if not path.is_relative_to(SITE.resolve()) or not path.is_file():
                errors.append(f"Missing or out-of-site asset: {target}")
        elif url.fragment and url.fragment not in page.ids:
            errors.append(f"Missing anchor: {target}")
    for target in page.controls:
        if target not in page.ids:
            errors.append(f"Missing aria-controls target: {target}")
    manifest = json.loads((SITE / "assets/manifest.json").read_text())
    for asset in manifest["assets"]:
        path = SITE / asset["path"]
        if not path.is_file():
            errors.append(f"Missing manifest asset: {path}")
            continue
        payload = path.read_bytes()
        if len(payload) != asset["bytes"] or hashlib.sha256(payload).hexdigest() != asset["sha256"]:
            errors.append(f"Asset changed without manifest update: {asset['path']}")
    for required in ("LICENSE", "NOTICE", "CITATION.cff", "code/LICENSE", "code/NOTICE"):
        if not (ROOT / required).is_file():
            errors.append(f"Missing repository metadata: {required}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Checked {len(page.links)} references, {len(page.ids)} IDs and {len(manifest['assets'])} assets.")


if __name__ == "__main__":
    main()
