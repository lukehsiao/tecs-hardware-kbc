# HACK: Automatically generating HArdware Component Knowledge Bases

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

Then, install Fonduer and any other Python dependencies by running:

```bash
$ make dev
```

## Downloading the Datasets

Each component has its own dataset which must be downloaded before running. To
do so, navigate to each component's directory and run the download data script.
Note that you must navigate to the directory before running the script, since
the script will automatically unpack into the `data` directory.

For example, to download the Op-Amp dataset:

```
$ cd hack/opamps/
$ ./download_data.sh
```

## Running

After installing all the requirements, and ensuring the necessary databases
are created, you can run each individual hardware component script.

### Transistors

```bash
$ createdb transistors
$ python hack/transistors/transistors.py
```

### Op Amps

```bash
$ createdb opamps
$ python hack/opamps/opamps.py
```

### Circular Connectors

```bash
$ createdb circular_connectors
$ python hack/circular_connectors/circular_connectors.py
```

## Analysis

After running transistors.py to create a HACK knowledge base, you can then run
the analysis scripts which will score that database against our gold labels.

```bash
$ python hack/transistors/get_entities.py
$ python hack/transistors/analysis.py
```
