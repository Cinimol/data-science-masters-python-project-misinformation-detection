"""
Stage 13 -- the project's flowcharts, as editable draw.io files and as figures.

Each diagram is declared once as nodes, edges and geometry, and is then emitted
twice: as a ``.drawio`` file the reader can open and edit, and as a PNG the
report embeds. Because both outputs derive from the same declaration, the
editable file and the printed figure cannot drift apart, which is the same
discipline applied to the numeric results elsewhere in this artefact.

    python -m src.flowcharts                 # every diagram
    python -m src.flowcharts --only pipeline

Design system. Six semantic roles carry the whole scheme, so colour means
something rather than decorating: a warm sand terminator opens and closes a
flow, soft blue is an ordinary process step, white is a stored artefact, gold
is a decision, deeper amber marks the element this project contributes, and
muted red marks an adversarial or corrective branch. Strokes are two shades
darker than their fill so the shapes survive greyscale printing.
"""
from __future__ import annotations

import argparse
import sys
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                      # noqa: E402

OUT_DRAWIO = Path(__file__).resolve().parents[1] / "diagrams"
INK = "#1F3864"

# --------------------------------------------------------------------------- #
# The palette. Each role is (fill, stroke, font).
# --------------------------------------------------------------------------- #
ROLE = {
    "terminator": ("#F7D6B0", "#C98B4E", "#5B3A17"),
    "process":    ("#D6E4F2", "#7BA0C4", INK),
    "artefact":   ("#FFFFFF", "#A3B1C2", "#42506A"),
    "decision":   ("#FBEBC8", "#D4A843", "#5B4413"),
    "proposed":   ("#FBDCC4", "#D98F4A", "#6B3C13"),
    "adversary":  ("#F3D2CE", "#BE7C74", "#6E2F28"),
}
EDGE = "#7A8CA3"
FONT = "Helvetica"


class Node:
    """One box in a flowchart, in draw.io page coordinates (y grows downward)."""

    def __init__(self, key, label, x, y, w, h, role="process", shape="rect"):
        self.key, self.label = key, label
        self.x, self.y, self.w, self.h = x, y, w, h
        self.role, self.shape = role, shape

    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def cy(self):
        return self.y + self.h / 2


class Edge:
    """A directed connector, optionally labelled and optionally dashed."""

    def __init__(self, a, b, label="", dashed=False, mid=None):
        self.a, self.b, self.label, self.dashed = a, b, label, dashed
        self.mid = mid          # optional x or y of the elbow, to avoid overlaps


class Chart:
    def __init__(self, key, title, nodes, edges, width=900, height=1160, note=""):
        self.key, self.title = key, title
        self.nodes = {n.key: n for n in nodes}
        self.order = [n.key for n in nodes]
        self.edges, self.width, self.height, self.note = edges, width, height, note


# --------------------------------------------------------------------------- #
# draw.io emission
# --------------------------------------------------------------------------- #
def _node_style(n: Node) -> str:
    fill, stroke, font = ROLE[n.role]
    base = (f"whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
            f"fontColor={font};fontFamily={FONT};fontSize=12;strokeWidth=1.5;"
            f"verticalAlign=middle;align=center;spacing=4;")
    if n.shape == "stadium":
        return "rounded=1;arcSize=50;" + base
    if n.shape == "diamond":
        return "rhombus;" + base
    if n.shape == "label":
        return (f"text;html=1;align=left;verticalAlign=middle;"
                f"fontFamily={FONT};fontSize=15;fontStyle=1;fontColor={INK};")
    return "rounded=1;arcSize=10;" + base


def _edge_style(e: Edge, a: Node, b: Node) -> str:
    _, _, es, en = _anchor(a, b)
    ex, ey = SIDE_XY[es]
    nx, ny = SIDE_XY[en]
    s = ("edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;jettySize=auto;"
         f"strokeColor={EDGE};strokeWidth=1.6;endArrow=blockThin;endFill=1;"
         f"fontFamily={FONT};fontSize=10;fontColor=#42506A;"
         "labelBackgroundColor=#FFFFFF;"
         f"exitX={ex};exitY={ey};exitDx=0;exitDy=0;"
         f"entryX={nx};entryY={ny};entryDx=0;entryDy=0;")
    if e.dashed:
        s += "dashed=1;dashPattern=6 4;"
    return s


def to_drawio(charts: list[Chart]) -> str:
    """Serialise one or more charts as a multi-page draw.io document."""
    mxfile = ET.Element("mxfile", {"host": "app.diagrams.net",
                                   "type": "device", "version": "24.0.0"})
    for ch in charts:
        dia = ET.SubElement(mxfile, "diagram", {"id": ch.key, "name": ch.title})
        model = ET.SubElement(dia, "mxGraphModel", {
            "dx": "1100", "dy": "800", "grid": "1", "gridSize": "10",
            "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1",
            "fold": "1", "page": "1", "pageScale": "1",
            "pageWidth": str(ch.width), "pageHeight": str(ch.height),
            "math": "0", "shadow": "0", "background": "#FFFFFF"})
        root = ET.SubElement(model, "root")
        ET.SubElement(root, "mxCell", {"id": "0"})
        ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

        title = ET.SubElement(root, "mxCell", {
            "id": f"{ch.key}_title", "value": ch.title,
            "style": (f"text;html=1;align=center;verticalAlign=middle;"
                      f"fontFamily={FONT};fontSize=20;fontStyle=1;"
                      f"fontColor={INK};"),
            "vertex": "1", "parent": "1"})
        ET.SubElement(title, "mxGeometry", {
            "x": "40", "y": "28", "width": str(ch.width - 80), "height": "36",
            "as": "geometry"})

        if ch.note:
            note = ET.SubElement(root, "mxCell", {
                "id": f"{ch.key}_note", "value": ch.note,
                "style": (f"text;html=1;align=left;verticalAlign=top;"
                          f"fontFamily={FONT};fontSize=10;fontStyle=2;"
                          f"fontColor=#5A6B84;"),
                "vertex": "1", "parent": "1"})
            ET.SubElement(note, "mxGeometry", {
                "x": "48", "y": str(ch.height - 78),
                "width": str(ch.width - 96), "height": "44", "as": "geometry"})

        for key in ch.order:
            n = ch.nodes[key]
            cell = ET.SubElement(root, "mxCell", {
                "id": f"{ch.key}_{n.key}", "value": n.label,
                "style": _node_style(n), "vertex": "1", "parent": "1"})
            ET.SubElement(cell, "mxGeometry", {
                "x": str(n.x), "y": str(n.y), "width": str(n.w),
                "height": str(n.h), "as": "geometry"})

        for i, e in enumerate(ch.edges):
            cell = ET.SubElement(root, "mxCell", {
                "id": f"{ch.key}_e{i}", "value": e.label,
                "style": _edge_style(e, ch.nodes[e.a], ch.nodes[e.b]),
                "edge": "1", "parent": "1",
                "source": f"{ch.key}_{e.a}", "target": f"{ch.key}_{e.b}"})
            geo = ET.SubElement(cell, "mxGeometry",
                                {"relative": "1", "as": "geometry"})
            pts, _, _ = _route(ch.nodes[e.a], ch.nodes[e.b], e.mid)
            if len(pts) > 2:
                arr = ET.SubElement(geo, "Array", {"as": "points"})
                for q in pts[1:-1]:
                    ET.SubElement(arr, "mxPoint",
                                  {"x": str(round(q[0])), "y": str(round(q[1]))})

    ET.indent(mxfile, space="  ")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            + ET.tostring(mxfile, encoding="unicode"))


# --------------------------------------------------------------------------- #
# Matplotlib rendering of the same geometry
# --------------------------------------------------------------------------- #
def _anchor(a: Node, b: Node):
    """Exit and entry points, with the side each uses.

    Returns ``(p0, p1, exit_side, entry_side)`` where a side is one of
    ``n``, ``s``, ``e`` or ``w``. The same choice drives both the PNG render
    and the exit/entry hints written into the draw.io edge, so a connector
    leaves and arrives at the same place in both outputs.
    """
    if abs(a.cx - b.cx) < max(a.w, b.w) * 0.55:
        if a.cy < b.cy:
            return (a.cx, a.y + a.h), (b.cx, b.y), "s", "n"
        return (a.cx, a.y), (b.cx, b.y + b.h), "n", "s"
    # A tall target is entered at the source's own height, which turns four
    # separate inputs into four straight lines rather than four elbows.
    y = a.cy if b.y <= a.cy <= b.y + b.h else b.cy
    if a.cx < b.cx:
        return (a.x + a.w, a.cy), (b.x, y), "e", "w"
    return (a.x, a.cy), (b.x + b.w, y), "w", "e"


SIDE_XY = {"n": (0.5, 0), "s": (0.5, 1), "e": (1, 0.5), "w": (0, 0.5)}

def _route(a: Node, b: Node, mid=None):
    """Orthogonal waypoints from a to b, with the sides used at each end.

    Three cases cover every connector in these diagrams: a straight drop
    between vertically stacked boxes, an L between boxes offset in both axes,
    and a Z between boxes side by side at different heights. Returning explicit
    points lets the PNG render and the draw.io waypoints follow the same path.
    """
    p0, p1, es, en = _anchor(a, b)
    if abs(p0[0] - p1[0]) < 1.5 or abs(p0[1] - p1[1]) < 1.5:
        return [p0, p1], es, en
    if es in "ns":                                   # leave vertically: Z through y
        m = mid if mid is not None else (p0[1] + p1[1]) / 2
        return [p0, (p0[0], m), (p1[0], m), p1], es, en
    m = mid if mid is not None else (p0[0] + p1[0]) / 2   # leave horizontally
    return [p0, (m, p0[1]), (m, p1[1]), p1], es, en



def render(ch: Chart, path: Path, dpi: int = 190) -> None:
    """Draw one chart to PNG using the coordinates the draw.io file uses."""
    fig_w, fig_h = ch.width / 100, ch.height / 100
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, ch.width)
    ax.set_ylim(ch.height, 0)                 # y grows downward, as in draw.io
    ax.axis("off")
    fig.patch.set_facecolor("white")

    for e in ch.edges:
        a, b = ch.nodes[e.a], ch.nodes[e.b]
        pts, es, _ = _route(a, b, e.mid)
        xs = [q[0] for q in pts]
        ys = [q[1] for q in pts]
        ax.plot(xs, ys, color=EDGE, linewidth=1.6, zorder=1, solid_joinstyle="round",
                linestyle=(0, (5, 3)) if e.dashed else "solid")
        ax.add_patch(FancyArrowPatch(
            pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=11,
            linewidth=1.6, color=EDGE, shrinkA=0, shrinkB=2, zorder=1,
            linestyle=(0, (5, 3)) if e.dashed else "solid"))
        if e.label:
            mx = (pts[0][0] + pts[1][0]) / 2 if es in "ew" else pts[0][0]
            my = pts[0][1] if es in "ew" else (pts[0][1] + pts[1][1]) / 2
            ax.text(mx, my, e.label, ha="center", va="center", fontsize=8.5,
                    color="#42506A", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none"))

    for key in ch.order:
        n = ch.nodes[key]
        fill, stroke, font = ROLE[n.role]
        if n.shape == "label":
            ax.text(n.x, n.cy, n.label, ha="left", va="center", fontsize=11.5,
                    weight="bold", color=INK, zorder=3)
            continue
        if n.shape == "diamond":
            pts = [(n.cx, n.y), (n.x + n.w, n.cy), (n.cx, n.y + n.h), (n.x, n.cy)]
            ax.add_patch(Polygon(pts, closed=True, facecolor=fill,
                                 edgecolor=stroke, linewidth=1.5, zorder=2))
        else:
            arc = 0.5 if n.shape == "stadium" else 0.06
            ax.add_patch(FancyBboxPatch(
                (n.x, n.y), n.w, n.h,
                boxstyle=f"round,pad=0,rounding_size={min(n.h * arc, n.h / 2)}",
                facecolor=fill, edgecolor=stroke, linewidth=1.5, zorder=2))
        ax.text(n.cx, n.cy, n.label.replace("<br>", "\n"), ha="center",
                va="center", fontsize=9.2, color=font, linespacing=1.45,
                zorder=3, wrap=True)

    ax.text(ch.width / 2, 46, ch.title, ha="center", va="center",
            fontsize=15.5, weight="bold", color=INK)
    if ch.note:
        wrapped = "\n".join(textwrap.wrap(ch.note, width=int(ch.width / 6.4)))
        ax.text(48, ch.height - 66, wrapped, ha="left", va="top",
                fontsize=8.4, style="italic", color="#5A6B84", linespacing=1.5)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# The diagrams
# --------------------------------------------------------------------------- #
def chart_research() -> Chart:
    """The project's own research process, from question to submission."""
    W, X, XR = 230, 100, 500
    n = [
        Node("start", "Identify the research problem", X, 96, W, 52,
             "terminator", "stadium"),
        Node("lit", "Literature review<br>and gap analysis", X, 186, W, 62),
        Node("q", "Formulate research question<br>and objectives", X, 284, W, 62),
        Node("corpus", "Build the corpus and<br>the evaluation protocols", X, 382, W, 62),
        Node("probe", "Run the content-free<br>shortcut probe", X, 480, W, 62, "proposed"),
        Node("gate", "Is the label a function<br>of the source?", X - 30, 578, W + 60, 96,
             "decision", "diamond"),
        Node("model", "Design and train the<br>detector and baselines", X, 716, W, 62),
        Node("eval", "Evaluate under all three<br>protocols", X, 814, W, 62),
        Node("reflect", "Interpret, evaluate the<br>approach, reflect", X, 946, W, 62),
        Node("end", "Submit report and artefact", X, 1044, W, 52,
             "terminator", "stadium"),
        Node("reframe", "Reframe: measure the<br>benchmark, not the model", XR, 596, W, 62,
             "proposed"),
        Node("addp", "Add the source-disjoint<br>protocol and the rotation", XR, 694, W, 62,
             "proposed"),
        Node("adv", "Add the adversarial<br>remedy", XR, 814, W, 62, "adversary"),
        Node("withdraw", "Withdraw the<br>architectural claim", XR, 926, W, 62, "adversary"),
    ]
    e = [Edge("start", "lit"), Edge("lit", "q"), Edge("q", "corpus"),
         Edge("corpus", "probe"), Edge("probe", "gate"),
         Edge("gate", "model", "no"), Edge("gate", "reframe", "yes"),
         Edge("reframe", "addp"), Edge("addp", "model"),
         Edge("model", "eval"), Edge("eval", "adv", "", True),
         Edge("adv", "withdraw", "", True), Edge("withdraw", "reflect", "", True),
         Edge("eval", "reflect"), Edge("reflect", "end")]
    return Chart("research", "Research Process", n, e, 880, 1200,
                 "Solid: the planned path. Dashed: the corrective branch taken "
                 "when the probe returned a perfect score and again when the "
                 "rotation showed the architectural margin was not "
                 "identifiable, recorded as SC-01 and SC-02 in Appendix K.")


def chart_pipeline() -> Chart:
    """The six processing stages and the artefact each one caches."""
    W, H, X, XA = 250, 60, 110, 500
    rows = [
        ("s1", "1. Build the corpus", "corpus.parquet<br>21,798 posts, three splits"),
        ("s2", "2. Frozen encoders", "feat_base.npz<br>CLIP, DistilRoBERTa, consistency"),
        ("s3", "3. Language model", "feat_llm.npz<br>hidden state, zero-shot P(false)"),
        ("s4", "4. Experiment grid", "runs.csv, preds/<br>21 systems, 165 runs"),
        ("s5", "5. Robustness suite", "robustness.csv<br>ablation, probe, cost"),
        ("s6", "6. Analysis and rotation", "results/, figures/<br>intervals, tests, rotation"),
    ]
    n = [Node("start", "Fakeddit metadata and images", X, 96, W, 50,
              "terminator", "stadium")]
    e = []
    y = 176
    for i, (k, lab, art) in enumerate(rows):
        n.append(Node(k, lab, X, y, W, H, "process"))
        n.append(Node(k + "a", art, XA, y, 290, H, "artefact"))
        e.append(Edge(k, k + "a", "", True))
        if i == 0:
            e.append(Edge("start", k))
        else:
            e.append(Edge(rows[i - 1][0], k))
        y += 108
    n.append(Node("end", "Report figures, tables<br>and triage queue", X, y, W, 58,
                  "terminator", "stadium"))
    e.append(Edge("s6", "end"))
    return Chart("pipeline", "Processing Pipeline", n, e, 880, y + 170,
                 "Each stage writes the cached artefact on its right and reads "
                 "only the one above, so any stage can be re-run in isolation.")


def chart_model() -> Chart:
    """Consistency-Gated Fusion, including the adversarial branch."""
    n = [
        Node("t", "Headline<br>DistilRoBERTa (768)", 60, 120, 190, 62),
        Node("v", "Image<br>CLIP ViT-B/32 (512)", 60, 226, 190, 62),
        Node("c", "Consistency scores<br>cosine, batch rank", 60, 352, 190, 62, "proposed"),
        Node("m", "Metadata<br>nine features", 60, 458, 190, 62),
        Node("pt", "Project to 256", 300, 120, 150, 62),
        Node("pv", "Project to 256", 300, 226, 150, 62),
        Node("inter", "Interaction<br>t · v,  |t - v|", 500, 120, 175, 62),
        Node("gate", "Consistency gate<br>sigmoid mask", 500, 226, 175, 62, "proposed"),
        Node("fuse", "Concatenate<br><br>fused<br>representation", 730, 120, 150, 400),
        Node("clf", "Classifier<br>P(false)", 950, 160, 180, 64, "proposed"),
        Node("adv", "Gradient reversal<br>source-community<br>adversary", 950, 340, 180, 84,
             "adversary"),
        Node("out", "Calibrated<br>probability and<br>readable gate", 1190, 152, 165, 80,
             "terminator", "stadium"),
    ]
    e = [Edge("t", "pt"), Edge("v", "pv"), Edge("pt", "inter"), Edge("pv", "gate"),
         Edge("c", "gate", "", True, mid=478),
         Edge("inter", "fuse"), Edge("gate", "fuse"),
         Edge("c", "fuse"), Edge("m", "fuse"),
         Edge("fuse", "clf", mid=905), Edge("fuse", "adv", "", True, mid=920),
         Edge("clf", "out")]
    return Chart("model", "Consistency-Gated Fusion", n, e, 1410, 640,
                 "Dashed amber: the consistency scores also drive the gate, "
                 "which rescales the visual stream element-wise. Dashed red: "
                 "the adversarial variant only. The gate is a bounded per-item "
                 "quantity and is therefore readable at inference.")


def chart_protocols() -> Chart:
    """How the same corpus is split three ways, plus the rotation."""
    n = [
        Node("corpus", "Corpus: 21,798 posts,<br>17 source communities", 330, 100, 260, 56,
             "terminator", "stadium"),
        Node("a", "Protocol A<br>stratified random 70/10/20", 60, 220, 230, 66),
        Node("b", "Protocol B<br>source-disjoint, 5 held out", 335, 220, 250, 66, "proposed"),
        Node("c", "Protocol C<br>temporal, earliest 70%", 630, 220, 230, 66),
        Node("aq", "Sources appear on<br>both sides of the split", 60, 330, 230, 60, "artefact"),
        Node("bq", "No source appears on<br>more than one side", 335, 330, 250, 60, "artefact"),
        Node("cq", "Sources persist,<br>time does not", 630, 330, 230, 60, "artefact"),
        Node("probe", "Content-free probe:<br>subreddit name only", 60, 440, 230, 62, "proposed"),
        Node("rot", "Leave-one-community-out<br>rotation, 15 folds", 335, 440, 250, 62, "proposed"),
        Node("prior", "Class-prior matched<br>resample, n = 5,786", 630, 440, 230, 62, "adversary"),
        Node("check", "Does the gap survive<br>the alternative explanation?", 300, 560, 320, 88,
             "decision", "diamond"),
        Node("metrics", "Bootstrap intervals, Holm-corrected tests,<br>"
                        "calibration, precision at k, per-community results",
             215, 700, 490, 66),
        Node("out", "Reported result with its own limits stated", 265, 806, 390, 50,
             "terminator", "stadium"),
    ]
    e = [Edge("corpus", "a"), Edge("corpus", "b"), Edge("corpus", "c"),
         Edge("a", "aq", "", True), Edge("b", "bq", "", True), Edge("c", "cq", "", True),
         Edge("aq", "probe"), Edge("bq", "rot"), Edge("cq", "prior"),
         Edge("probe", "check"), Edge("rot", "check"), Edge("prior", "check"),
         Edge("check", "metrics", "yes"), Edge("metrics", "out")]
    return Chart("protocols", "Evaluation Protocol Design", n, e, 920, 930,
                 "Amber marks the three diagnostics this project adds. Red "
                 "marks the test of the sceptical reading: prior shift "
                 "explains none of the gap.")


def chart_ethics() -> Chart:
    """Each ethical constraint traced to a code decision and a reported metric."""
    rows = [
        ("p", "Personal data<br>is re-identifiable",
         "Drop the author column<br>at build time", "Asserted by a unit test"),
        ("con", "Public posting<br>is not consent",
         "Aggregate reporting only;<br>no post reproduced", "Contextual integrity test"),
        ("lab", "Labels are proxies,<br>not adjudications",
         "Build protocols the proxy<br>cannot be exploited under", "Protocols B and rotation"),
        ("bias", "One aggregate<br>conceals disparity",
         "Report each held-out<br>community separately", "0.472 to 0.841 spread"),
        ("mis", "A detector can<br>become a censor",
         "Emit a calibrated score,<br>never a decision", "ECE 0.223 to 0.138"),
        ("env", "Compute has an<br>environmental cost",
         "Freeze the encoders", "Five CPU-hours, no GPU"),
    ]
    n = [Node("start", "Ethical consideration", 70, 96, 240, 46, "terminator", "stadium"),
         Node("mid", "Decision taken in the artefact", 380, 96, 260, 46,
              "terminator", "stadium"),
         Node("end", "Evidence in the report", 710, 96, 240, 46, "terminator", "stadium")]
    e = []
    y = 176
    for k, a, b, c in rows:
        n += [Node(k + "1", a, 70, y, 240, 66, "adversary"),
              Node(k + "2", b, 380, y, 260, 66, "process"),
              Node(k + "3", c, 710, y, 240, 66, "artefact")]
        e += [Edge(k + "1", k + "2"), Edge(k + "2", k + "3")]
        y += 96
    e += [Edge("start", "p1", "", True), Edge("mid", "p2", "", True),
          Edge("end", "p3", "", True)]
    return Chart("ethics", "Ethics to Artefact: Traceability", n, e, 1020, y + 130,
                 "Every ethical position on the left changes something in the "
                 "middle column, and the right column is where a marker can "
                 "check that it did.")


def chart_cycles() -> Chart:
    """The eight development cycles and the two scope decisions they forced."""
    steps = [
        ("c1", "Cycle 1<br>Can the corpus be built<br>reproducibly?", "process"),
        ("c2", "Cycle 2<br>Does the label depend<br>on the source?", "proposed"),
        ("c3", "Cycle 3<br>Does the gap survive<br>a corrected split?", "process"),
        ("c4", "Cycle 4<br>Does explicit consistency<br>help under shift?", "process"),
        ("c5", "Cycle 5<br>Why does consistency<br>hurt across communities?", "process"),
        ("c6", "Cycle 6<br>Does adversarial training<br>close the gap?", "process"),
        ("c7", "Cycle 7<br>Is the gap explained<br>by prior shift?", "adversary"),
        ("c8", "Cycle 8<br>Is the architectural<br>margin identifiable?", "adversary"),
    ]
    n = [Node("start", "Approved proposal", 90, 96, 240, 46, "terminator", "stadium")]
    e = []
    y = 172
    for i, (k, lab, role) in enumerate(steps):
        n.append(Node(k, lab, 90, y, 240, 74, role))
        e.append(Edge(steps[i - 1][0] if i else "start", k))
        y += 104
    n += [Node("sc1", "SC-01<br>Reframe from accuracy<br>to measurement", 430, 276, 240, 74,
               "proposed"),
          Node("sc2", "SC-02<br>Withdraw the<br>architectural claim", 430, 900, 240, 74,
               "adversary"),
          Node("drop", "Two planned fusion<br>variants dropped", 730, 276, 220, 74, "artefact"),
          Node("hon", "Section 7.2 states<br>the aim was not met", 730, 900, 220, 74,
               "artefact"),
          Node("end", "Submitted claim set", 90, y, 240, 46, "terminator", "stadium")]
    e += [Edge("c2", "sc1"), Edge("sc1", "drop"),
          Edge("c8", "sc2"), Edge("sc2", "hon"), Edge("c8", "end")]
    return Chart("cycles", "Development Cycles and Scope Decisions", n, e,
                 1010, y + 165,
                 "Each cycle ended in a result that determined the next. Two "
                 "results forced a documented change of scope rather than a "
                 "change of story.")


def chart_architecture() -> Chart:
    """Figure 1: the pipeline and the proposed model on a single page.

    Panel (a) runs left to right so the six stages read as a sequence, with the
    artefact each one caches directly beneath it. Panel (b) uses the same
    left-to-right reading order for the model, so a reader moves through both
    panels the same way.
    """
    n, e = [], []

    n.append(Node("la", "(a)  Processing pipeline", 60, 96, 400, 30,
                  "process", "label"))
    stages = [
        ("s1", "1. Build the<br>corpus", "corpus.parquet<br>21,798 posts"),
        ("s2", "2. Frozen<br>encoders", "feat_base.npz<br>CLIP, DistilRoBERTa"),
        ("s3", "3. Language<br>model", "feat_llm.npz<br>hidden state, P(false)"),
        ("s4", "4. Experiment<br>grid", "runs.csv, preds/<br>21 systems, 165 runs"),
        ("s5", "5. Robustness<br>suite", "robustness.csv<br>ablation, probe, cost"),
        ("s6", "6. Analysis and<br>rotation", "results/, figures/<br>intervals, tests"),
    ]
    for i, (k, lab, art) in enumerate(stages):
        x = 60 + i * 222
        n.append(Node(k, lab, x, 142, 196, 76))
        n.append(Node(k + "a", art, x, 262, 196, 62, "artefact"))
        e.append(Edge(k, k + "a", "", True))
        if i:
            e.append(Edge(stages[i - 1][0], k))

    n.append(Node("lb", "(b)  Consistency-Gated Fusion", 60, 372, 460, 30,
                  "process", "label"))
    n += [
        Node("t", "Headline<br>DistilRoBERTa (768)", 60, 424, 190, 62),
        Node("v", "Image<br>CLIP ViT-B/32 (512)", 60, 530, 190, 62),
        Node("c", "Consistency scores<br>cosine, batch rank", 60, 656, 190, 62, "proposed"),
        Node("m", "Metadata<br>nine features", 60, 762, 190, 62),
        Node("pt", "Project to 256", 300, 424, 150, 62),
        Node("pv", "Project to 256", 300, 530, 150, 62),
        Node("inter", "Interaction<br>t · v,  |t - v|", 500, 424, 175, 62),
        Node("gate", "Consistency gate<br>sigmoid mask", 500, 530, 175, 62, "proposed"),
        Node("fuse", "Concatenate<br><br>fused<br>representation", 730, 424, 150, 400),
        Node("clf", "Classifier<br>P(false)", 950, 464, 180, 64, "proposed"),
        Node("adv", "Gradient reversal<br>source-community<br>adversary", 950, 644, 180, 84,
             "adversary"),
        Node("out", "Calibrated<br>probability and<br>readable gate", 1190, 456, 165, 80,
             "terminator", "stadium"),
    ]
    e += [Edge("t", "pt"), Edge("v", "pv"), Edge("pt", "inter"), Edge("pv", "gate"),
          Edge("c", "gate", "", True, mid=478),
          Edge("inter", "fuse"), Edge("gate", "fuse"),
          Edge("c", "fuse"), Edge("m", "fuse"),
          Edge("fuse", "clf", mid=905), Edge("fuse", "adv", "", True, mid=920),
          Edge("clf", "out")]

    return Chart("architecture", "Artefact Architecture", n, e, 1440, 960,
                 "Panel (a): each stage writes the cached artefact beneath it "
                 "and reads only the one to its left, so any stage can be "
                 "re-run in isolation. Panel (b): dashed amber marks the "
                 "consistency scores driving the gate, which rescales the "
                 "visual stream element-wise; dashed red marks the adversarial "
                 "variant only.")


CHARTS = {"architecture": chart_architecture, "research": chart_research, "pipeline": chart_pipeline,
          "model": chart_model, "protocols": chart_protocols,
          "ethics": chart_ethics, "cycles": chart_cycles}


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="comma-separated subset")
    a = ap.parse_args()
    keys = [k.strip() for k in a.only.split(",") if k.strip()] or list(CHARTS)

    OUT_DRAWIO.mkdir(exist_ok=True)
    built = []
    for k in keys:
        ch = CHARTS[k]()
        built.append(ch)
        (OUT_DRAWIO / f"{k}.drawio").write_text(to_drawio([ch]))
        png = C.FIGURES / f"flow_{k}.png"
        render(ch, png)
        print(f"[flow] {k:10s} -> diagrams/{k}.drawio and {png.name}")

    combined = OUT_DRAWIO / "mmid-diagrams.drawio"
    combined.write_text(to_drawio(built))
    print(f"[flow] combined workbook -> {combined} ({len(built)} pages)")


if __name__ == "__main__":
    main()
