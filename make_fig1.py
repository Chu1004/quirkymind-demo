#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_fig1.py — stitch per-panel screenshots into one Figure 1.

    python make_fig1.py A.png B.png C.png D.png --layout app    # A B C in a row, D below
    python make_fig1.py A.png B.png C.png D.png --layout grid   # 2 x 2 (default)
    python make_fig1.py A.png B.png C.png D.png --layout wide   # 4 in a row
    python make_fig1.py A.png C.png --layout wide                # any subset

Why stitch rather than screenshot the whole screen: a full-screen capture at
one zoom level shrinks every panel equally, so the matrix ticks in C and the
bar labels in D come out unreadable in a two-column paper. Capturing each
panel separately (Figure mode on, browser zoom 150-200 %) and composing them
here keeps each panel at its own legible size, then scales the composite once.

Each tile can get a lettered badge and module title above it. Since the demo
now prints the same header inside every panel, --badge is off by default: the
screenshots already carry their own titles, and stacking a second row of them
just doubles the words. Turn it on when the tiles are cropped below their own
headers.

--layout app reproduces the screen: the first tiles side by side in one row
and the last one full width underneath. Within the top row the tiles keep
their relative widths and share one scale factor, and the bottom tile is
scaled to the same total width, so the type comes out close to the same size
everywhere — which the equal-width default does not manage when one panel is
much wider than the rest.

All tiles are scaled to the same width by default, which is what makes a
wide panel end up with smaller text than a narrow one: panel D spans the
whole app window, so squeezing it to one column width shrinks its type far
more than panel A's. --native turns the resizing off and composes the
screenshots at their captured size, so every panel keeps the text size it was
captured at. Capture all of them at the same browser zoom and device pixel
ratio and the type matches across the figure.

Panels have very different natural heights. A row of four therefore leaves a
lot of white under the short ones, and the obvious fix — cutting every tile to
the shortest height — slices content off the tall ones. So the default is to
keep every tile whole and align them to a common top edge; --crop is there
only for tiles you have already cropped yourself and know are safe, and it
now warns about how much it removes. When the heights differ a lot, the grid
layout or two separate figures work better than one wide row.
"""

import argparse
import os

from PIL import Image, ImageDraw, ImageFont

# Same four strings as the demo's PANEL_TITLES and the paper's System
# Overview bullets; change all three together.
TITLES = {
    "A": "Heterodox Persona Curation",
    "B": "Dual-Stream Trait Anchoring",
    "C": "Real-Time Geometric Tracking",
    "D": "Live Multi-Turn Auditing & Downstream Tasks",
}
INK, PANEL, GRID = (34, 33, 28), (255, 255, 255), (223, 220, 211)
# Outer margin and inter-tile gap, in px of the composite. The outer margin is
# wasted space once the figure is placed in LaTeX — the figure environment
# adds its own — so it is small by default and --pad 0 removes it entirely.
PAD, GAP, BADGE = 10, 26, 44


def font(size, bold=False):
    for name in (["DejaVuSans-Bold.ttf"] if bold else ["DejaVuSans.ttf"]):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def load(path, tol=8, frac_min=0.02):
    """Open a screenshot and trim the page background around the panel card.

    The exact-match trim this used to do finds nothing when the margin and the
    card are both near-white — which is the usual case here, since the page is
    #F7F6F2 and the card is #FFFFFF, four levels apart. So the scan is by
    tolerance: walk in from each edge while every pixel on that row or column
    is within `tol` of the corner colour. Whatever is left is the card, and
    tiles then butt together with no white band between them."""
    im = Image.open(path).convert("RGB")
    px = im.load()
    w, h = im.size
    bg = px[0, 0]

    def near(p):
        return all(abs(p[i] - bg[i]) <= tol for i in range(3))

    # Scanning inward until a line is *entirely* background stops at the first
    # stray mark — a drop shadow, a scrollbar sliver, the outermost dot of a
    # rounded window corner — and leaves the whole margin behind. Instead,
    # measure how much of each line is not background and keep the lines that
    # carry real content: the card's own border runs the full height, so it
    # clears the threshold, while a stray dot never does.
    def frac(samples):
        return sum(0 if near(p) else 1 for p in samples) / max(1, len(samples))

    step_x, step_y = max(1, w // 400), max(1, h // 400)
    col_f = [frac([px[x, y] for y in range(0, h, step_y)]) for x in range(w)]
    row_f = [frac([px[x, y] for x in range(0, w, step_x)]) for y in range(h)]
    thr = frac_min

    cols = [x for x, f in enumerate(col_f) if f >= thr]
    rows = [y for y, f in enumerate(row_f) if f >= thr]
    if not cols or not rows:
        print(f"  [trim] {os.path.basename(path)}: nothing above the content "
              f"threshold ({thr:.0%}); left as captured")
        return im
    left, right = cols[0], cols[-1]
    top, bot = rows[0], rows[-1]
    if right - left < 10 or bot - top < 10:          # nothing recognisable: keep it
        print(f"  [trim] {os.path.basename(path)}: nothing trimmed "
              f"(is the margin the same colour as the content?)")
        return im
    if (left, top, right, bot) != (0, 0, w - 1, h - 1):
        print(f"  [trim] {os.path.basename(path)}: {w}x{h} -> "
              f"{right - left + 1}x{bot - top + 1} "
              f"(l{left} t{top} r{w - 1 - right} b{h - 1 - bot})")
    return im.crop((left, top, right + 1, bot + 1))


def badge(draw, x, y, letter, title, f_letter, f_title):
    draw.rounded_rectangle([x, y, x + BADGE, y + BADGE], radius=9, fill=INK)
    w = draw.textlength(letter, font=f_letter)
    draw.text((x + (BADGE - w) / 2, y + 7), letter, fill=PANEL, font=f_letter)
    draw.text((x + BADGE + 12, y + 9), title, fill=INK, font=f_title)


def compose_app(a):
    """A B C in one row, D full width below — the screen's own layout.

    The top row is scaled by one factor so the panels keep their relative
    widths (the screen's 3:4:5 columns), and the bottom panel is scaled to the
    same total width. Both factors end up close, so text is a consistent size
    across the figure instead of the wide panel coming out tiny."""
    ims = [load(p, a.trim_tol, a.trim_frac) for p in a.images]
    letters = list(a.letters)[:len(ims)]
    top, bottom = ims[:-1], ims[-1]

    W = a.row_width
    gaps = GAP * (len(top) - 1)
    if a.equal_top:
        each = (W - gaps) // len(top)
        factors = [each / im.width for im in top]
        top = [im.resize((each, int(im.height * f)), Image.LANCZOS)
               for im, f in zip(top, factors)]
        f_top = sum(factors) / len(factors)
        for L, f in zip(letters, factors):
            if f > 1.25:
                print(f"  [warn] tile {L} is upscaled x{f:.2f}; recapture it wider "
                      f"for a sharp figure")
    else:
        f_top = (W - gaps) / sum(im.width for im in top)
        top = [im.resize((int(im.width * f_top), int(im.height * f_top)), Image.LANCZOS)
               for im in top]
    f_bot = W / bottom.width
    bottom = bottom.resize((W, int(bottom.height * f_bot)), Image.LANCZOS)
    print(f"  [app] top row x{f_top:.2f}, bottom x{f_bot:.2f} "
          f"(closer to each other = more consistent text size)")

    f_letter, f_title = font(26, bold=True), font(30, bold=True)
    head = (BADGE + 20) if a.badge else 0
    top_h = max(im.height for im in top)
    H = PAD * 2 + head + top_h + GAP + head + bottom.height
    canvas = Image.new("RGB", (PAD * 2 + W, H), PANEL)
    draw = ImageDraw.Draw(canvas)

    x, y = PAD, PAD
    for L, im in zip(letters, top):
        if a.badge:
            badge(draw, x, y, L, TITLES.get(L, ""), f_letter, f_title)
        canvas.paste(im, (x, y + head))
        if not a.no_frame:
            draw.rounded_rectangle([x - 1, y + head - 1, x + im.width, y + head + im.height],
                                   radius=10, outline=GRID, width=2)
        x += im.width + GAP
    y += head + top_h + GAP
    L = letters[-1]
    if a.badge:
        badge(draw, PAD, y, L, TITLES.get(L, ""), f_letter, f_title)
    canvas.paste(bottom, (PAD, y + head))
    if not a.no_frame:
        draw.rounded_rectangle([PAD - 1, y + head - 1,
                                PAD + bottom.width, y + head + bottom.height],
                               radius=10, outline=GRID, width=2)

    canvas.save(a.out, dpi=(300, 300))
    print(f"wrote {a.out}  ({canvas.width}x{canvas.height}px, {len(ims)} panels, layout=app)")
    if a.fit:
        report_fit(a.fit, canvas.width, canvas.height, a.ui_text_px)


def report_fit(spec: str, w: int, h: int, ui_px: float):
    # w, h: composite pixels. ui_px: measured height of a lowercase letter.
    """How big will the interface text be once this composite is scaled into
    a figure of the given size? A two-column AAAI page gives about 7.0 x 9.2
    inches of text, so one third of a page is roughly 7.0 x 3.0."""
    try:
        tw, th = (float(v) for v in spec.lower().split("x"))
    except ValueError:
        print(f"  [fit] could not read --fit {spec!r}; expected WIDTHxHEIGHT in inches")
        return
    s_w, s_h = tw / (w / 300), th / (h / 300)      # composite is saved at 300 dpi
    s = min(s_w, s_h)
    print(f"  [fit] composite {w}x{h}px = {w/300:.1f}x{h/300:.1f}in at 300dpi")
    print(f"  [fit] to fit {tw}x{th}in it is scaled x{s:.2f} "
          f"(limited by {'height' if s_h < s_w else 'width'}); "
          f"effective {300*s:.0f} dpi")
    if s_h < s_w * 0.8:
        print(f"  [fit] the composite is too tall for this box: it will only fill "
              f"{100*s/s_w:.0f}% of the width. Fewer rows, or shorter tiles, would use "
              f"the space better.")
    if ui_px > 0:
        pt = ui_px * s / 300 * 72
        verdict = ("legible" if pt >= 7 else
                   "too small — under 7pt it will not survive printing")
        print(f"  [fit] interface text ~{pt:.1f}pt in the paper ({verdict}); "
              f"a caption is about 9pt")
        # What the composite would have to look like for 7pt. When the box is
        # fixed, only two things move the point size: how many tiles share the
        # width, and how large the text is inside each capture. Both reduce to
        # one ratio — text height as a fraction of the composite's width.
        need_ratio = 7.0 / (tw * 72)
        have_ratio = ui_px / w
        factor = need_ratio / have_ratio
        print(f"  [fit] text is {100*have_ratio:.2f}% of the composite width; "
              f"7pt needs {100*need_ratio:.2f}%, i.e. {factor:.1f}x more")
        if factor > 1.05:
            print(f"  [fit] two ways to get there, and they multiply: put "
                  f"{factor:.1f}x fewer tiles in a row (a 4-wide row at "
                  f"{factor:.1f}x becomes {max(1, round(4/factor))}-wide), or raise the "
                  f"demo's Figure text scale by about {factor:.1f}x.")


def main():
    global PAD, GAP
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="screenshots, in panel order A B C D")
    ap.add_argument("--letters", default="ABCD", help="letters for the images, in order")
    ap.add_argument("--layout", choices=["grid", "wide", "tall", "app"], default="grid")
    ap.add_argument("--row-width", type=int, default=5200,
                    help="total content width of the composite in --layout app")
    ap.add_argument("--height", default="min",
                    help="with --equal-height, the height every tile is scaled to: "
                         "min (default, nothing is upscaled), max (nothing is "
                         "downscaled), mean, or a number of px. min shrinks the "
                         "tallest tile, which makes its text the smallest in the row; "
                         "mean or max raise the short tiles instead.")
    ap.add_argument("--equal-height", action="store_true",
                    help="scale every tile to a common height instead of a common "
                         "width, so a row of panels ends flush at the bottom. Widths "
                         "then differ, which is usually what a row of screenshots "
                         "wants: the panels were captured at different content "
                         "heights, and a ragged baseline reads as a mistake.")
    ap.add_argument("--equal-top", action="store_true",
                    help="in --layout app, give every tile in the top row the same "
                         "width instead of keeping the widths they were captured at. "
                         "Use when one of them was captured much narrower than the "
                         "others; it is upscaled, so its text gets bigger but softer.")
    ap.add_argument("--badge", action="store_true",
                    help="draw a lettered badge and module title above each tile "
                         "(off by default: the screenshots already have headers)")
    ap.add_argument("--crop", action="store_true",
                    help="CUTS CONTENT: trims every tile to the height of the shortest. "
                         "Only safe when the tall tiles end in blank space.")
    ap.add_argument("--max-aspect", type=float, default=0.0,
                    help="if >0, scale down any tile taller than this multiple of the "
                         "tile width instead of cropping it, so nothing is lost")
    ap.add_argument("--tile-width", type=int, default=1400,
                    help="each tile is scaled to this width (ignored with --native)")
    ap.add_argument("--native", action="store_true",
                    help="do not resize: compose the screenshots at their captured "
                         "size, so text is the same size in every panel")
    ap.add_argument("--widths", default="",
                    help="comma-separated per-tile widths in px, e.g. 1400,1400,1400,2000; "
                         "use when one panel is much wider than the others")
    ap.add_argument("-o", "--out", default="fig_interface.png")
    ap.add_argument("--pad", type=int, default=PAD,
                    help="outer margin in px (default 10; 0 for a flush edge)")
    ap.add_argument("--gap", type=int, default=GAP,
                    help="gap between tiles in px (default 26; 0 butts them together)")
    ap.add_argument("--inset", default="",
                    help="crop this many px from each tile AFTER trimming: one number "
                         "for all sides, or L,R,T,B. Use it when the captures are the "
                         "cards themselves — trimming then finds nothing to remove and "
                         "the space between panels is the cards' own padding, which "
                         "only an inset can take out. Try 40 first; the card's rounded "
                         "border goes with it.")
    ap.add_argument("--trim-frac", type=float, default=0.02,
                    help="a row or column counts as content when at least this "
                         "fraction of it is not background (default 0.02). Raise it "
                         "if a faint shadow keeps the margin; lower it if a thin "
                         "card border gets cut.")
    ap.add_argument("--trim-tol", type=int, default=8,
                    help="how close to the corner colour a pixel counts as page "
                         "background when trimming (default 8; raise it if a white "
                         "band survives, lower it if the card edge is cut)")
    ap.add_argument("--no-frame", action="store_true",
                    help="drop the thin outline drawn around each tile. With --gap 0 "
                         "the panels then read as one continuous strip, which is what "
                         "you want when each screenshot already has its own card "
                         "border — two borders a pixel apart look like a rendering "
                         "fault, not a layout.")
    ap.add_argument("--fit", default="",
                    help="target size in inches, e.g. 7x3 for a figure* that is one "
                         "third of a letterpaper page. Reports how far the composite "
                         "has to shrink and how large the interface text ends up in "
                         "points, so you know before compiling whether it is legible.")
    ap.add_argument("--ui-text-px", type=float, default=0.0,
                    help="height in px of a lowercase letter in the screenshots "
                         "(measure once in an image editor). Used by --fit.")
    a = ap.parse_args()

    PAD, GAP = a.pad, a.gap
    if a.layout == "app":
        compose_app(a)
        return

    if a.equal_height:
        ims = [load(p, a.trim_tol, a.trim_frac) for p in a.images]
        hs = [im.height for im in ims]
        if a.height == "min":
            h = min(hs)
        elif a.height == "max":
            h = max(hs)
        elif a.height == "mean":
            h = int(sum(hs) / len(hs))
        else:
            h = int(a.height)
        tiles = [(L, im.resize((max(1, int(im.width * h / im.height)), h), Image.LANCZOS))
                 for L, im in zip(a.letters, ims)]
        print(f"  [equal-height] target {h}px:",
              ", ".join(f"{L} {im0.height}->{im.height} (x{h / im0.height:.2f})"
                        for (L, im), im0 in zip(tiles, ims)))
        compose(a, tiles)
        return

    widths = [int(w) for w in a.widths.split(",")] if a.widths else []
    tiles = []
    for i, (path, letter) in enumerate(zip(a.images, a.letters)):
        im = load(path, a.trim_tol, a.trim_frac)
        if a.native:
            target = im.width
        elif i < len(widths):
            target = widths[i]
        else:
            target = a.tile_width
        if target != im.width:
            s_ = target / im.width
            im = im.resize((target, int(im.height * s_)), Image.LANCZOS)
        tiles.append((letter, im))
    if a.native:
        print("  [native] tile widths:",
              ", ".join(f"{L}={im.width}" for L, im in tiles),
              "- text matches only if every capture used the same zoom and DPR")

    if a.max_aspect > 0:
        # Shrink the tall tiles rather than cutting them: a panel that is
        # narrower than its neighbours still shows all of its content.
        cap = int(a.tile_width * a.max_aspect)
        out = []
        for L, im in tiles:
            if im.height > cap:
                s_ = cap / im.height
                im = im.resize((int(im.width * s_), cap), Image.LANCZOS)
                print(f"  [{L}] scaled to {im.width}x{im.height} to fit the aspect cap")
            out.append((L, im))
        tiles = out

    if a.crop and len(tiles) > 1:
        h = min(im.height for _, im in tiles)
        for L, im in tiles:
            if im.height > h:
                print(f"  [warn] --crop removes {im.height - h}px from the bottom of "
                      f"tile {L}; check that nothing of the panel is lost")
        tiles = [(L, im.crop((0, 0, im.width, h))) for L, im in tiles]

    compose(a, tiles)


def apply_inset(a, tiles):
    if not a.inset:
        return tiles
    v = [int(x) for x in a.inset.split(",")]
    l, r, t, b = (v * 4)[:4] if len(v) == 1 else (v + [0, 0, 0, 0])[:4]
    out = []
    for L, im in tiles:
        w, h = im.size
        if w - l - r < 20 or h - t - b < 20:
            print(f"  [inset] tile {L} too small for the inset; left as is")
            out.append((L, im))
            continue
        out.append((L, im.crop((l, t, w - r, h - b))))
    print(f"  [inset] cropped l{l} r{r} t{t} b{b} from every tile")
    return out


def compose(a, tiles):
    tiles = apply_inset(a, tiles)
    f_letter, f_title = font(26, bold=True), font(30, bold=True)
    head = (BADGE + 20) if a.badge else 0
    n = len(tiles)
    cols = {"grid": 2 if n > 2 else n, "wide": n, "tall": 1}[a.layout]
    rows = -(-n // cols)
    col_w = max(im.width for _, im in tiles)
    row_h = [max(im.height for _, im in tiles[r * cols:(r + 1) * cols]) + head
             for r in range(rows)]
    W = PAD * 2 + cols * col_w + (cols - 1) * GAP
    H = PAD * 2 + sum(row_h) + (rows - 1) * GAP
    canvas = Image.new("RGB", (W, H), PANEL)
    draw = ImageDraw.Draw(canvas)

    y = PAD
    for r in range(rows):
        x = PAD
        for c in range(cols):
            i = r * cols + c
            if i >= n:
                break
            letter, im = tiles[i]
            ox = x + (col_w - im.width) // 2          # centre a narrower tile
            if a.badge:
                badge(draw, ox, y, letter, TITLES.get(letter, ""), f_letter, f_title)
            canvas.paste(im, (ox, y + head))
            if not a.no_frame:
                draw.rounded_rectangle([ox - 1, y + head - 1,
                                        ox + im.width, y + head + im.height],
                                       radius=10, outline=GRID, width=2)
            x += col_w + GAP
        y += row_h[r] + GAP

    canvas.save(a.out, dpi=(300, 300))
    print(f"wrote {a.out}  ({W}x{H}px, {n} panel(s), layout={a.layout})")
    if a.fit:
        report_fit(a.fit, W, H, a.ui_text_px)


if __name__ == "__main__":
    main()