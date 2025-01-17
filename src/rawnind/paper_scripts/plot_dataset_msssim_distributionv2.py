import yaml
import numpy as np
import os


def read_yaml(file_path):
    """
    Reads a YAML file and returns the data.
    """
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data


def filter_scores(data):
    """
    Filters entries where 'f_fpath' != 'gt_fpath' and extracts 'rgb_msssim_score'.
    """
    scores = []
    for entry in data:
        if entry.get("f_fpath") != entry.get("gt_fpath"):
            score = entry.get("rgb_msssim_score")
            if score is not None:
                scores.append(score)
    return scores


def compute_histogram(scores, bin_start=0.40, bin_end=1.00, bin_width=0.05):
    """
    Computes histogram counts and bin centers.
    """
    bins = np.arange(bin_start, bin_end + bin_width, bin_width)
    counts, bin_edges = np.histogram(scores, bins=bins)
    bin_centers = bin_edges[:-1] + bin_width / 2  # Correctly calculate bin centers
    return counts, bin_centers, bins


def generate_tikz(bin_centers, counts, bins, output_path):
    """
    Generates a standalone TikZ figure using pgfplots.
    """
    # Define the LaTeX content
    tikz_content = r"""\documentclass{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17}
\usepackage{pgfplotstable}
\usepackage{siunitx}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
    width=3.5in,
    height=2.5in,
    ymode=log,
    ymin=1,
    xmin=0.40,
    xmax=1.00,
    xtick={0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00},
    xticklabels={0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00},
    xticklabel style={
        /pgf/number format/fixed,
        /pgf/number format/precision=2,
        font=\tiny, % Changed font size for x-axis tick labels to \tiny
        align=center
    },
    scaled y ticks=false, % Prevent scientific notation scaling
    ytick={1,10,100,1000,10000}, % Define explicit tick positions
    yticklabels={1,10,100,1000,10000}, % Define explicit tick labels
    yticklabel style={
        font=\tiny, % Set y-axis tick label font size to \tiny
    },
    xlabel={MS-SSIM score},
    ylabel={Number of images (log scale)},
    ymajorgrids=true,
    yminorgrids=true, % Enabled minor grids for y-axis
    xmajorgrids=true,
    minor tick num=1, % Number of minor ticks between major ticks
    bar width=0.035,
    enlarge x limits=0.025, % Slightly increased to accommodate bar widths
    tick label style={font=\tiny}, % Set all tick labels to \tiny
    label style={font=\small},
    tick align=outside,
    axis line style={line width=0.5pt},
    major grid style={line width=0.2pt, dashed},
    minor grid style={line width=0.1pt, dashed, gray!50},
]
\addplot+[ybar, fill=blue!60, draw=black] coordinates {
"""
    # Add the histogram data with 3 decimal places for bin centers (0.425, 0.475, etc.)
    for center, count in zip(bin_centers, counts):
        tikz_content += f"    ({center:.3f}, {count})\n"

    # Close the plot
    tikz_content += r"""};
\end{axis}
\end{tikzpicture}
\end{document}
"""

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Write to the output file
    with open(output_path, "w") as file:
        file.write(tikz_content)


def main():
    # Define the YAML file path
    yaml_file = "/orb/benoit_phd/datasets/RawNIND/RawNIND_masks_and_alignments.yaml"

    # Check if the YAML file exists
    if not os.path.isfile(yaml_file):
        print(f"Error: YAML file not found at {yaml_file}")
        return

    # Read YAML data
    data = read_yaml(yaml_file)

    # Filter and extract scores
    scores = filter_scores(data)

    if not scores:
        print(
            "No valid 'rgb_msssim_score' entries found where 'f_fpath' != 'gt_fpath'."
        )
        return

    # Compute histogram
    counts, bin_centers, bins = compute_histogram(scores)

    # Define output TikZ file path
    output_tikz = "/orb/benoit_phd/wiki/Papers/JDDC/journal_paper/figures/histogram.tex"

    # Generate TikZ figure
    generate_tikz(bin_centers, counts, bins, output_tikz)

    print(f"Histogram TikZ figure has been saved to {output_tikz}")


if __name__ == "__main__":
    main()
