"""One plotting style for every figure in the project."""
import matplotlib as mpl
import matplotlib.pyplot as plt

INK = "#1c2541"      # text and actual values
RUST = "#c8553d"     # pollution, main accent
BLUE = "#2a6f97"     # weather driven models
TEAL = "#3a9188"     # second model colour
GOLD = "#d9a441"     # highlights
GREY = "#8d99ae"     # baselines and secondary lines
LIGHT = "#e9ecef"    # shading
PAPER = "#fbfaf7"

MODEL_COLORS = {
    "Persistence": GREY, "Seasonal naive": "#b8c0cc", "ETS": "#a68a64",
    "SARIMA + Fourier": TEAL, "SARIMAX, ECMWF forecast": BLUE, "SARIMAX, GFS forecast": "#6a9fc0",
    "SARIMAX, actual weather": RUST, "VAR": "#7d6b91",
    "Random forest": "#5a8f69", "Gradient boosting": "#86b38f", "SVR": "#b5c99a",
}


def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "axes.labelcolor": INK, "axes.edgecolor": "#b0b7c3",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e3e6eb", "grid.linewidth": 0.7,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "legend.frameon": False, "lines.linewidth": 1.6,
    })


def save(fig, name):
    from config import FIGURES
    path = FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    print("saved", path.name)
