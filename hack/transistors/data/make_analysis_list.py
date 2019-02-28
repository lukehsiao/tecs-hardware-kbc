import os


if __name__ == "__main__":

    # Make a list of filenames to write
    filenames = set()
    for filename in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis/pdf/")):
        if not filename.endswith(".pdf"):
            raise ValueError(f"Invalid filename {filename}")
        filenames.add(filename.replace(".pdf", ""))
        print(f"[DEBUG]: Filename {filename} is valid")

    # Write filenames to CSV
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_dataset.csv"), "w") as outfile:
        for filename in filenames:
            outfile.write(str(filename) + "\n")
