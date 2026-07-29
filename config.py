from pathlib import Path

# Paths  
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "rawdata"              # raw Google Trends CSVs (g1-g15)

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PANEL_DIR = OUTPUTS_DIR / "panel"                # stitched panel CSV
FIGURES_DIR = OUTPUTS_DIR / "figures"            # paper figures
TABLES_DIR = OUTPUTS_DIR / "tables"              # paper tables (CSV/LaTeX)

PANEL_CSV = PANEL_DIR / "agri_trends_panel_2016_2025_weekly_FL.csv"

# Group / crop configuration 
GROUPS = ["g1", "g2", "g3", "g4", "g5", "g6",
          "g7", "g8", "g9", "g10", "g11", "g12",
          "g13", "g14", "g15"]

REF_GROUP = "g1"          # all groups are rescaled to this group's anchor level

# Anchor crops per non-reference group (must be present in both that group AND
# the reference group). Cucumber is primary; strawberry is the second bridge.
GROUP_ANCHORS = {
    "g2":  ["cucumber", "strawberry"],
    "g3":  ["cucumber", "strawberry"],
    "g4":  ["cucumber", "strawberry"],
    "g5":  ["cucumber", "strawberry"],
    "g6":  ["cucumber", "strawberry"],
    "g7":  ["cucumber", "strawberry"],
    "g8":  ["cucumber", "strawberry"],
    "g9":  ["cucumber", "strawberry"],
    "g10": ["cucumber", "strawberry"],
    "g11": ["cucumber", "strawberry"],
    "g12": ["cucumber", "strawberry"],
    "g13": ["cucumber", "strawberry"],
    "g14": ["cucumber", "strawberry"],
    "g15": ["cucumber", "strawberry"],
}

# Crops that appear in more than one group. Kept from the first group seen
# (reference group first), dropped from later groups to avoid double-counting.
DUPLICATE_CROPS = {"cucumber", "strawberry"}

# constants
LT1_VALUE = 0.5     # Google Trends reports "<1" as this value
ZERO_GATE = 0.20    # flag crops with > 20 % zero / NaN weeks
SEAM_REL_GATE = 0.15  # flag stitch seams with > 15 % median overlap error (Gate A)

# ensure output directories exist 
for _d in (PANEL_DIR, FIGURES_DIR, TABLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
