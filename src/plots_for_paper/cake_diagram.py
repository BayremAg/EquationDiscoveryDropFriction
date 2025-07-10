import json

import numpy as np

from definitions import ROOT_DIR
import matplotlib.pyplot as plt
def autopct_filter(pct):
    return f"{pct:.1f}%" if pct > 3 else ""
save_path = ROOT_DIR / "results/SematicScholar/paper_data.json"
save_path.parent.mkdir(parents=True, exist_ok=True)
with open(save_path, "r") as f:
    fild_dict = json.load(f)
pass

# Extract labels and sizes
labels = list(fild_dict.keys())
sizes = [len(v) for v in fild_dict.values()]
sort_index = np.argsort(sizes)
sizes = np.array(sizes)[sort_index]
labels = np.array(labels)[sort_index]
# Plot the pie chart
fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
wedges, _, autotexts =ax.pie(
    sizes,
    #labels=labels,
    autopct=autopct_filter,
    startangle=180,
    labeldistance=1.1,  # Push labels outward
    pctdistance=0.85,  # Control where percentage appears
    wedgeprops=dict(width=0.3),  # Donut style (optional)
    textprops=dict(size=10),
)
ax.legend(
    reversed(wedges),
    reversed(labels),
    title="Fields",
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    fontsize=9
)

#plt.title("Distribution of Papers by Field", fontsize=14)
#fig.axis("equal")  # Keep it circular
fig.tight_layout()
fig.savefig(save_path.parent / "cake_diagram.pdf")
fig.show()
