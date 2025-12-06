import os

def save_plot(fig, name, dpi=300):
    output_dir = os.path.join(os.getcwd(), "results", "graphs")
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.join(output_dir, name)

    fig.savefig(f"{base}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")

    print(f"Saved: {base}.png/.pdf")
