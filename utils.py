import os

import os

#set correct path for repo, e.g. where you saved the ita-parl-health folder
BASE_PATH= "C:/insert repo path here"

def save_plot(fig, name, dpi=300):
   
    
    # Build the output directory
    OUTPUT_DIR = os.path.join(
        BASE_PATH,
        "numerical analysis",
        "results"
    )

    # Create folder if missing
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Full file path without extension
    base = os.path.join(OUTPUT_DIR, name)

    # Save both formats
    fig.savefig(f"{base}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")

    print(f"Saved to: {base}.png")
    print(f"Saved to: {base}.pdf")
