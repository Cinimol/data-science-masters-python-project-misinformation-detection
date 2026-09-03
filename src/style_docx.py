"""
Post-processing pass that applies a consistent, publication-style table design
to a generated .docx.

pandoc emits tables with a plain uniform grid, which reads as unformatted in a
submitted report. This script rewrites every table in ``word/document.xml`` to
a house style:

*   a filled header row in the document accent colour with reversed-out bold
    type, marked to repeat if the table breaks across a page;
*   subtle row banding so long numeric tables remain readable across a row;
*   horizontal rules only, with heavier rules above and below the table and
    beneath the header, and no vertical rules (the convention in scientific
    typesetting, following the booktabs style);
*   generous cell padding and a slightly reduced type size so wide tables fit
    the text block;
*   right-aligned numeric columns, detected automatically.

Table captions, written in the source as a bold "Table n:" paragraph, are
reduced to caption size and kept with the table that follows.

Usage
-----
    python -m src.style_docx "report.docx"            # rewrite in place
    python -m src.style_docx in.docx -o out.docx
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}


def q(tag: str) -> str:
    return f"{{{W}}}{tag}"


# --------------------------------------------------------------------------- #
# House palette
# --------------------------------------------------------------------------- #
ACCENT = "1F3864"        # header fill, heavy rules
BAND = "EFF3F8"          # alternating row fill
RULE = "BFCBD9"          # light inner rule
HEADER_TEXT = "FFFFFF"
TABLE_PT = 18            # half-points, i.e. 9 pt
CAPTION_PT = 17          # half-points, i.e. 8.5 pt

# Schema-mandated child ordering for the elements we touch.
TBLPR_ORDER = ["tblStyle", "tblpPr", "tblOverlap", "bidiVisual",
               "tblStyleRowBandSize", "tblStyleColBandSize", "tblW", "jc",
               "tblCellSpacing", "tblInd", "tblBorders", "shd", "tblLayout",
               "tblCellMar", "tblLook", "tblCaption", "tblDescription"]
TCPR_ORDER = ["cnfStyle", "tcW", "gridSpan", "hMerge", "vMerge", "tcBorders",
              "shd", "noWrap", "tcMar", "textDirection", "tcFitText",
              "vAlign", "hideMark"]
RPR_ORDER = ["rStyle", "rFonts", "b", "bCs", "i", "iCs", "caps", "smallCaps",
             "strike", "dstrike", "outline", "shadow", "emboss", "imprint",
             "noProof", "snapToGrid", "vanish", "webHidden", "color",
             "spacing", "w", "kern", "position", "sz", "szCs", "highlight",
             "u", "effect", "bdr", "shd", "fitText", "vertAlign", "rtl",
             "cs", "em", "lang", "eastAsianLayout", "specVanish", "oMath"]
PPR_ORDER = ["pStyle", "keepNext", "keepLines", "pageBreakBefore",
             "framePr", "widowControl", "numPr", "suppressLineNumbers",
             "pBdr", "shd", "tabs", "suppressAutoHyphens", "kinsoku",
             "wordWrap", "overflowPunct", "topLinePunct", "autoSpaceDE",
             "autoSpaceDN", "bidi", "adjustRightInd", "snapToGrid",
             "spacing", "ind", "contextualSpacing", "mirrorIndents",
             "suppressOverlap", "jc", "textDirection", "textAlignment",
             "textboxTightWrap", "outlineLvl", "divId", "cnfStyle", "rPr"]


def _insert_ordered(parent: etree._Element, child: etree._Element,
                    order: list[str]) -> None:
    """Insert ``child`` into ``parent`` at the position the schema requires."""
    name = etree.QName(child).localname
    if name in order:
        idx = order.index(name)
        for existing in parent:
            ename = etree.QName(existing).localname
            if ename in order and order.index(ename) > idx:
                existing.addprevious(child)
                return
    parent.append(child)


def _ensure(parent: etree._Element, tag: str,
            order: list[str]) -> etree._Element:
    """Return ``parent``'s child ``tag``, creating it in schema order."""
    found = parent.find(q(tag), NS)
    if found is None:
        found = etree.SubElement(parent, q(tag))
        parent.remove(found)
        _insert_ordered(parent, found, order)
    return found


def _el(tag: str, **attrs) -> etree._Element:
    e = etree.Element(q(tag))
    for k, v in attrs.items():
        e.set(q(k), str(v))
    return e


# --------------------------------------------------------------------------- #
def _table_properties() -> etree._Element:
    """Borders, width and cell padding for the whole table."""
    pr = etree.Element(q("tblPr"))
    pr.append(_el("tblW", w="5000", type="pct"))

    borders = etree.SubElement(pr, q("tblBorders"))
    # CT_TblBorders enforces the order top, left, bottom, right, insideH,
    # insideV; emitting them out of order fails schema validation in Word.
    for side, sz, colour in (("top", 12, ACCENT), ("left", 0, "auto"),
                             ("bottom", 12, ACCENT), ("right", 0, "auto"),
                             ("insideH", 4, RULE), ("insideV", 0, "auto")):
        b = etree.SubElement(borders, q(side))
        b.set(q("val"), "single" if sz else "none")
        b.set(q("sz"), str(sz))
        b.set(q("space"), "0")
        b.set(q("color"), colour)

    pr.append(_el("tblLayout", type="autofit"))
    mar = etree.SubElement(pr, q("tblCellMar"))
    for side, val in (("top", 70), ("left", 110), ("bottom", 70),
                      ("right", 110)):
        m = etree.SubElement(mar, q(side))
        m.set(q("w"), str(val))
        m.set(q("type"), "dxa")
    pr.append(_el("tblLook", val="04A0", firstRow="1", lastRow="0",
                  firstColumn="1", lastColumn="0", noHBand="0", noVBand="1"))
    return pr


def _shade(tc: etree._Element, fill: str) -> None:
    tcpr = _ensure(tc, "tcPr", [])
    for old in tcpr.findall(q("shd"), NS):
        tcpr.remove(old)
    shd = _el("shd", val="clear", color="auto", fill=fill)
    _insert_ordered(tcpr, shd, TCPR_ORDER)


def _style_runs(cell: etree._Element, *, bold: bool | None = None,
                colour: str | None = None, size: int = TABLE_PT) -> None:
    for run in cell.iter(q("r")):
        rpr = run.find(q("rPr"), NS)
        if rpr is None:
            rpr = etree.Element(q("rPr"))
            run.insert(0, rpr)
        if bold is not None:
            for old in rpr.findall(q("b"), NS):
                rpr.remove(old)
            if bold:
                _insert_ordered(rpr, _el("b"), RPR_ORDER)
        if colour is not None:
            for old in rpr.findall(q("color"), NS):
                rpr.remove(old)
            _insert_ordered(rpr, _el("color", val=colour), RPR_ORDER)
        for tag in ("sz", "szCs"):
            for old in rpr.findall(q(tag), NS):
                rpr.remove(old)
            _insert_ordered(rpr, _el(tag, val=size), RPR_ORDER)


def _cell_text(tc: etree._Element) -> str:
    return "".join(t.text or "" for t in tc.iter(q("t"))).strip()


NUMERIC = re.compile(r"^[+\-−]?[\d,]*\.?\d+\s*%?$")


def _align_cell(tc: etree._Element, how: str) -> None:
    for para in tc.findall(q("p"), NS):
        ppr = _ensure(para, "pPr", PPR_ORDER)
        for old in ppr.findall(q("jc"), NS):
            ppr.remove(old)
        _insert_ordered(ppr, _el("jc", val=how), PPR_ORDER)


def style_table(tbl: etree._Element) -> None:
    rows = tbl.findall(q("tr"), NS)
    if not rows:
        return

    for old in tbl.findall(q("tblPr"), NS):
        tbl.remove(old)
    tbl.insert(0, _table_properties())

    # --- header row -------------------------------------------------------- #
    head = rows[0]
    trpr = head.find(q("trPr"), NS)
    if trpr is None:
        trpr = etree.Element(q("trPr"))
        head.insert(0, trpr)
    if trpr.find(q("tblHeader"), NS) is None:
        trpr.append(_el("tblHeader"))
    if trpr.find(q("cantSplit"), NS) is None:
        trpr.append(_el("cantSplit"))
    for tc in head.findall(q("tc"), NS):
        _shade(tc, ACCENT)
        _style_runs(tc, bold=True, colour=HEADER_TEXT)

    # --- body rows: banding ------------------------------------------------ #
    body = rows[1:]
    for i, tr in enumerate(body):
        for tc in tr.findall(q("tc"), NS):
            if i % 2 == 1:
                _shade(tc, BAND)
            _style_runs(tc)

    # --- numeric columns are right aligned --------------------------------- #
    if body:
        n_cols = max(len(tr.findall(q("tc"), NS)) for tr in body)
        for col in range(n_cols):
            vals = []
            for tr in body:
                cells = tr.findall(q("tc"), NS)
                if col < len(cells):
                    vals.append(_cell_text(cells[col]))
            vals = [v for v in vals if v]
            if not vals:
                continue
            if sum(bool(NUMERIC.match(v)) for v in vals) / len(vals) >= 0.7:
                for tr in rows:
                    cells = tr.findall(q("tc"), NS)
                    if col < len(cells):
                        _align_cell(cells[col], "right")


CAPTION = re.compile(r"^Table\s+[A-Z]?\d+\s*[:.]")


def style_captions(root: etree._Element) -> int:
    """Reduce 'Table n: ...' paragraphs to caption size and keep with table."""
    n = 0
    body = root.find(q("body"), NS)
    if body is None:
        return 0
    for para in body.findall(q("p"), NS):
        text = "".join(t.text or "" for t in para.iter(q("t"))).strip()
        if not CAPTION.match(text):
            continue
        ppr = _ensure(para, "pPr", PPR_ORDER)
        if ppr.find(q("keepNext"), NS) is None:
            _insert_ordered(ppr, _el("keepNext"), PPR_ORDER)
        for old in ppr.findall(q("spacing"), NS):
            ppr.remove(old)
        sp = _el("spacing", before="180", after="60")
        _insert_ordered(ppr, sp, PPR_ORDER)
        _style_runs(para, size=CAPTION_PT)
        n += 1
    return n


# --------------------------------------------------------------------------- #
def restyle(src: Path, dst: Path) -> tuple[int, int]:
    tmp = Path(tempfile.mkdtemp())
    try:
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp)
        doc = tmp / "word" / "document.xml"
        tree = etree.parse(str(doc))
        root = tree.getroot()

        tables = root.iter(q("tbl"))
        n_tables = 0
        for tbl in list(tables):
            style_table(tbl)
            n_tables += 1
        n_caps = style_captions(root)
        tree.write(str(doc), xml_declaration=True, encoding="UTF-8",
                   standalone=True)

        if dst.exists():
            dst.unlink()
        # The OPC specification requires [Content_Types].xml to be the first
        # entry in the package, so it is added before the remaining parts.
        subprocess.run(["zip", "-Xq", str(dst.resolve()),
                        "[Content_Types].xml"], cwd=tmp, check=True)
        subprocess.run(["zip", "-Xrq", str(dst.resolve()), ".",
                        "-x", "[Content_Types].xml"], cwd=tmp, check=True)
        return n_tables, n_caps
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("docx")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()
    src = Path(args.docx)
    dst = Path(args.out) if args.out else src
    if dst == src:
        tmp = src.with_suffix(".styling.docx")
        n_t, n_c = restyle(src, tmp)
        tmp.replace(src)
    else:
        n_t, n_c = restyle(src, dst)
    print(f"[style] restyled {n_t} tables and {n_c} captions -> {dst}")


if __name__ == "__main__":
    main()
