"""Shared matplotlib styling for nablachem figures.

Usage
-----
import sys; sys.path.insert(0, "..")  # or wherever this file lives
import plot_style

plot_style.load_style()

f, axs = plt.subplots(1, 4, figsize=(12, 3), dpi=600, constrained_layout=True)
for ax, letter in zip(axs, "ABCD"):
    plot_style.format_panel(ax, letter)
"""
from matplotlib import rcParams
from matplotlib import font_manager as fm


def load_style(scale: float = 1.4):
    """Set rcParams for Fira Sans Extra Condensed with scaled font sizes.

    Requires: brew install --cask font-fira-sans-extra-condensed

    Parameters
    ----------
    scale : float
        Multiplier applied to the default font.size (default 1.4).
    """
    for f in fm.findSystemFonts(fontpaths=None, fontext="ttf"):
        if "Fira" in f:
            fm.fontManager.addfont(f)
    fm._load_fontmanager(try_read_cache=False)

    rcParams["font.family"] = "Fira Sans Extra Condensed"
    rcParams["mathtext.fontset"] = "custom"
    rcParams["mathtext.it"] = "Fira Sans Extra Condensed:italic"
    rcParams["mathtext.default"] = "regular"

    base = 10.0  # matplotlib default
    rcParams["font.size"] = base * scale
    rcParams["axes.labelsize"] = base * scale
    rcParams["xtick.labelsize"] = base * scale
    rcParams["ytick.labelsize"] = base * scale
    rcParams["legend.fontsize"] = base * scale * 0.9
    rcParams["legend.frameon"] = False
    rcParams["legend.handletextpad"] = 0.1
    rcParams["legend.handlelength"] = 1.0

    rcParams["xtick.direction"] = "out"
    rcParams["ytick.direction"] = "out"
    rcParams["xtick.major.size"] = 4
    rcParams["ytick.major.size"] = 4
    rcParams["xtick.minor.size"] = 2
    rcParams["ytick.minor.size"] = 2


def format_panel(ax, letter: str = ""):
    """Despine and optionally add a bold panel letter to an axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    letter : str
        Panel label (e.g. "A"). If empty, only despining is applied.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", top=False, right=False)
    if letter:
        ax.set_title(letter, loc="left", fontweight="bold")
