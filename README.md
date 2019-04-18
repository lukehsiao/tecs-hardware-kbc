# HACK: Automating the generation of HArdware Component Knowledge Bases

## Dependencies

We use a few applications that you'll need to install and be sure are on your
PATH.

For OS X using [homebrew](https://brew.sh):

```bash
$ brew install poppler
$ brew install postgresql
$ brew install libpng freetype pkg-config
```

On Debian-based distros:

```bash
$ sudo apt install libxml2-dev libxslt-dev python3-dev
$ sudo apt build-dep python-matplotlib
$ sudo apt install poppler-utils
$ sudo apt install postgresql
```

We require `poppler-utils` to be version 0.36.0 or greater (which is already
the case for Ubuntu 18.04). If you do not meet this requirement, you can also
[install poppler manually](https://poppler.freedesktop.org/).

For the Python dependencies, we recommend using a
[virtualenv](https://virtualenv.pypa.io/en/stable/). Once you have cloned the
repository, change directories to the root of the repository and run

```bash
$ virtualenv -p python3 .venv
```

Once the virtual environment is created, activate it by running

```bash
$ source .venv/bin/activate
```

Any Python libraries installed will now be contained within this virtual
environment. To deactivate the environment, simply run `deactivate`.

Then, install our package, Fonduer, and any other Python dependencies by running:

```bash
$ make dev
```

## Downloading the Datasets

Each component has its own dataset which must be downloaded before running. To
do so, navigate to each component's directory and run the download data script.
Note that you must navigate to the directory before running the script, since
the script will automatically unpack into the `data` directory.

For example, to download the Op-Amp dataset:

```bash
$ cd hack/opamps/
$ ./download_data.sh
```

Each dataset is already divided into a training, development, and testing set.
Manually annotated gold labels are provided in CSV form for the development and
testing sets.

## Running End-to-end Knowledge Base Construction

After installing all the requirements, and ensuring the necessary databases
are created, you can run each individual hardware component script.

Note that in our paper, we used a server with 4x14-core CPUs, 1TB of memory, and
NVIDIA GPUs. With this server, a run with the full datasets takes 10s of hours
for each component. In order to support running our experiments on consumer
hardware, we provide instructions that do not use a GPU, and scale back the
number of documents significantly.

### Transistors

To run relation extraction on the transistor dataset, you can use the
`transistors` CLI provided by this repository. See `transistors -h` for a list
of commandline options. To run extraction from 500 train documents, and evaluate
the resulting score on the test set, you can run the following command.

```bash
$ createdb transistors
$ transistors --stg-temp-min --stg-temp-max --polarity --ce-v-max --parse --first-time --max-docs 500 --parallel 4 --conn-string="postgresql://<user>:<pw>@<host>:<port>/transistors"
```

### Op Amps

```bash
$ createdb opamps
$ opamps --gain --current --parse --first-time --max-docs 500 --parallel 4 --conn-string="postgresql://<user>:<pw>@<host>:<port>/opamps"
```

### Circular Connectors

```bash
$ createdb circular_connectors
$ circular_connectors --parse --first-time --max-docs 500 --parallel 4 --conn-string="postgresql://<user>:<pw>@<host>:<port>/circular_connectors"
```

For more detailed options, run `transistors -h`, `opamps -h`, or
`circular_connectors -h` to see a list of all possible options.

### Troubleshooting

If you get an `FATAL: role "<username>" does not exist.` error, or an
`fe_sendauth no password supplied` error, you will need to make sure you have a
PostgreSQL user set up, and that you either have a PostgreSQL password, or have
configured postgres to accept connections without a password.

See [Fonduer's FAQ](https://fonduer.readthedocs.io/en/latest/user/faqs.html#)
for additional instructions.


## Analysis
For our analysis, we create a set of entities from our generated knowledge bases
which are then scored against ground-truth gold labels. For a more direct
comparison, we only consider a subset of datasheets which we verify are
available on Digikey.

Each dataset contains its own `analysis.py` script, which will output three
scores: one for `test`, one for `dev`, and one for the `analysis` split of
documents (datasheets from `dev` and `test` that also occur in Digikey's data).

### Transistors
For our transistor analysis, we compare our automatically generated output with
Digikey using `ce_v_max` (collector emitter voltage max).

#### Generate Entity CSVs
After running `transistors` to generate a knowledge base, and a set of entity
probability CSVs (for convenience), run the analysis scripts to score those
entities against our ground truth gold labels and output a file of discrepancies
for further categorization:

```bash
$ python hack/transistors/analysis.py
```

This will produce a set of false positive and false negative entities which
are then written to `analysis_discrepancies.csv` for manual evaluation.

#### Use Existing Entities
For analysis purposes, you can use the included entity CSVs that were generated
from our knowledge bases. In order to exactly replicate our results, you can use
the already created entity CSVs found in `hack/transistors/analysis/` by
running:

```bash
$ python hack/transistors/analysis.py
```

This will also output an F1 Score and an `analysis_discrepancies.csv` file for
manual debugging.

### Scoring Digi-Key
To compare scores with Digikey, we grade Digikey's existing data with the same
ground truth labels and on the same datasheets used to score our automated
output:

```bash
$ python hack/transistors/digikey_analysis.py
```

This will output an F1 Score for Digikey and a `digikey_discrepancies.csv' file
for manual evaluation.

### Op Amps
For our opamp analysis, we evalutate our output against Digikey's knowledge base
using both relations: `typ_gbp` (typical gain bandwidth product) and
`typ_supply_current` (typical supply or quiescent current).

Running `analysis.py` will generate two sets of scores: one for `typ_gbp` and
one for `typ_supply_current`.

#### Generate Entity CSVs
After running `opamps` to generate a HACK knowledge base and a set of
entity probability CSVs, you can then run the analysis scripts to score those
entities against our ground truth gold labels and output a file of discrepancies
for further categorization:

```bash
$ python hack/opamps/analysis.py
```

Scoring will produce a set of false positive and false negative entities which
are then written to `analysis_discrepancies.csv` for manual evaluation.

#### Use Existing Entities
For analysis purposes, you can use the included entity CSVs that were generated
from our knowledge bases. In order to exactly replicate our results, you can use
the already created entity CSVs found in `hack/opamps/analysis/` by
running:

```bash
$ python hack/opamps/analysis.py
```

This will also ouput an F1 Score and a `analysis_discrepancies.csv` file for
manual debugging.

### Scoring Digi-Key
To compare scores with Digikey, we grade Digikey's existing data with the same
ground truth labels and on the same datasheets used to score our automated
output:

```bash
$ python hack/opamps/digikey_analysis.py
```

This will output an F1 Score for Digikey and a `digikey_discrepancies.csv' file
in the `analysis` directory for manual evaluation.

## Performance Experiments

In addition, we have a set of scripts in [`scripts/`](./scripts/) which show how
our scaling experiments were run. These can be modified to run on consumer
hardware by modifying the commandline arguments provided.
