# SPDX-FileCopyrightText: Copyright 2026 bbport contributors
# SPDX-License-Identifier: GPL-2.0-or-later
"""launcher/trophy_list.py on synthetic files (no game files, no key): the folder, the XML, the
unlocked ids, the times from trophies.jsonl, the page's counts."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "launcher"))
import trophy_list  # noqa: E402

XML = """<!--Sce-Np-Trophy-Signature: abc-->
<?xml version="1.0" encoding="UTF-8"?>
<trophyconf version="1.1" platform="ps4">
 <title-name>Test</title-name>
 <trophy id="000" hidden="no" ttype="P" pid="-1"><name>All of it</name><detail>Everything.</detail></trophy>
 <trophy id="001" hidden="yes" ttype="G" pid="000"><name>Fish &amp; Chips</name><detail>Secret.</detail></trophy>
 <trophy id="002" hidden="no" ttype="B" pid="000"><name>Bronze</name><detail>Small.</detail></trophy>
</trophyconf>
"""


def main():
    with tempfile.TemporaryDirectory() as tmp:
        user = Path(tmp)
        # Nothing kept yet: an empty page.
        page = trophy_list.load(user)
        assert page["folder"] is None and page["total"] == 0 and not page["named"]

        # Unlocks without the key: numbers only, from unlocked.txt and trophies.jsonl.
        folder = user / "trophy" / "NPWR05818_00"
        folder.mkdir(parents=True)
        (folder / "unlocked.txt").write_text("# the trophies unlocked here, one id a line\n2\n", encoding="utf-8")
        (user / "trophies.jsonl").write_text(
            '{"v":1,"title_id":"CUSA03173","trophy_id":2,"unlocked_at":"2026-10-26T21:05:00Z"}\n'
            '{"v":1,"title_id":"CUSA03173","trophy_id":2,"unlocked_at":"2026-10-27T10:00:00Z"}\n'
            '{"v":1,"title_id":"CUSA03173","trophy_id":1,"unlocked_at":"2026-10-26T22:00:00Z"}\n'
            'not json\n', encoding="utf-8")
        page = trophy_list.load(user)
        assert page["folder"] == folder and not page["named"]
        assert [t["id"] for t in page["trophies"]] == [1, 2] and page["unlocked"] == 2
        assert page["trophies"][1]["time"] == "2026-10-26T21:05:00Z"  # the first one

        # With the decrypted XML: every trophy, names, grades, the language's file first.
        (folder / "TROP.XML").write_text(XML, encoding="utf-8")
        (folder / "TROP_08.XML").write_text(XML.replace("All of it", "Всё"), encoding="utf-8")
        (folder / "TROP000.PNG").write_bytes(b"\x89PNG")
        page = trophy_list.load(user, "8")
        assert page["named"] and page["total"] == 3 and page["unlocked"] == 2
        first = page["trophies"][0]
        assert first["name"] == "Всё" and first["grade"] == "P" and not first["unlocked"]
        assert first["icon"] == folder / "TROP000.PNG"
        assert page["trophies"][1]["name"] == "Fish & Chips" and page["trophies"][1]["hidden"]
        assert page["grades"] == {"P": [0, 1], "G": [1, 1], "B": [1, 1]}
        assert trophy_list.load(user, "1")["trophies"][0]["name"] == "All of it"

        assert trophy_list.valid_key("00112233445566778899aabbccddEEFF")
        assert not trophy_list.valid_key("0011") and not trophy_list.valid_key("zz" * 16)
    print("trophy-list: ok")


if __name__ == "__main__":
    main()
