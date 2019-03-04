import os
import csv
from tqdm import tqdm


if __name__ == "__main__":
    relpath = os.path.dirname(os.path.abspath(__file__))

    # Generate list of filenames
    filenames = set()
    with open(os.path.join(relpath, "analysis_dataset.csv"), "r") as inputcsv:
        reader = csv.reader(inputcsv)
        for row in reader:
            filename = row[0]
            filenames.add(filename)

    # Move filenames to new dir
    print(f"[INFO]: Moving {len(filenames)} files...")
    endpath = os.path.join(relpath, "analysis/")
    origpath = os.path.join(relpath, "train/")

    # Ensure all filenames are in the orig dir
    pdfpath = os.path.join(origpath, "pdf/")
    for filename in filenames:
        pdf = filename + ".pdf"
        if pdf not in os.listdir(pdfpath):
            raise ValueError(f"Filename {pdf} not in {pdfpath}")

    # Move filenames to new dir
    for filename in tqdm(filenames):
        pdf = filename + ".pdf"
        html = filename + ".html"
        os.rename(os.path.join(origpath, "pdf/" + pdf), os.path.join(endpath, "pdf/" + pdf))
        os.rename(os.path.join(origpath, "html/" + html), os.path.join(endpath, "html/" + html))
