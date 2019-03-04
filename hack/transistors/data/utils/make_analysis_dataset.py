import csv
import os

from tqdm import tqdm


"""
This script should read in the existing train/ set of documents
and pull out documents that we know that Digikey has and put them
into a separate dir for comparison purposes.
"""


def get_digikey_filenames(goldfile):
    digikey_filenames = set()
    with open(goldfile, "r") as gold:
        reader = csv.reader(gold)
        for line in reader:
            filename = line[0]
            digikey_filenames.add(filename.upper())
    if len(digikey_filenames) != 0:
        return digikey_filenames
    else:
        import pdb

        pdb.set_trace()


def get_valid_filenames(dirname, digikey_filenames):
    valid_filenames = set()
    for filename in tqdm(os.listdir(dirname)):
        if filename.endswith(".pdf"):
            if filename.upper().replace(".PDF", "") in digikey_filenames:
                valid_filenames.add(filename.replace(".pdf", ""))
    if len(valid_filenames) != 0:
        return valid_filenames
    else:
        import pdb

        pdb.set_trace()


def move_valid_files(pdf, html, valid_filenames, limit=100):
    endpath = os.path.join(os.path.dirname(__file__), "analysis/")
    if len(valid_filenames) < limit:
        raise ValueError(
            f"Valid filenames is not long enough to satisfy limit of {limit}"
        )
    elif len(valid_filenames) != limit:
        valid_filenames = set(list(valid_filenames)[:limit])

    for i, filename in enumerate(tqdm(valid_filenames)):
        if filename + ".pdf" not in os.listdir(pdf):
            raise ValueError(f"Filename {filename} not found in {pdf}")
        os.rename(
            os.path.join(pdf, filename + ".pdf"),
            os.path.join(endpath + "pdf/", filename + ".pdf"),
        )
        if filename + ".html" not in os.listdir(html):
            raise ValueError(f"Filename {filename} not found in {html}")
        os.rename(
            os.path.join(html, filename + ".html"),
            os.path.join(endpath + "html/", filename + ".html"),
        )
        if i >= limit:
            return


if __name__ == "__main__":
    # Run extraction on train set
    relpath = os.path.dirname(__file__)
    pdfdir = os.path.join(relpath, "train/pdf/")
    htmldir = os.path.join(relpath, "train/html")
    gold_files = [
        os.path.join(relpath, "standard_digikey_gold.csv"),
        os.path.join(relpath, "url_digikey_gold.csv"),
    ]

    files1 = get_digikey_filenames(gold_files[0])
    files2 = get_digikey_filenames(gold_files[1])
    digikey_filenames = files1.union(files2)

    filenames = get_valid_filenames(pdfdir, digikey_filenames)
    move_valid_files(pdfdir, htmldir, filenames)
