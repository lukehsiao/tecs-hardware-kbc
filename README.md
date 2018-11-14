![fonduer-logo](https://raw.githubusercontent.com/HazyResearch/fonduer/master/docs/static/img/fonduer-logo.png)
# HACK: HArdware Component Knowledge Base

## Dependencies

We use a few applications that you'll need to install and be sure are on your
PATH.

For OS X using [homebrew](https://brew.sh):

```bash
brew install poppler
brew install postgresql
brew install libpng freetype pkg-config
```

On Debian-based distros:

```bash
sudo apt install libxml2-dev libxslt-dev python3-dev
sudo apt build-dep python-matplotlib
sudo apt install poppler-utils
sudo apt install postgresql
```

For the Python dependencies, we recommend using a
[virtualenv](https://virtualenv.pypa.io/en/stable/). Once you have cloned the
repository, change directories to the root of the repository and run

```bash
virtualenv -p python3 .venv
```

Once the virtual environment is created, activate it by running

```bash
source .venv/bin/activate
```

Any Python libraries installed will now be contained within this virtual
environment. To deactivate the environment, simply run `deactivate`.

Then, install Fonduer and any other python dependencies by running:

```bash
pip install -r requirements.txt
```

## Running

After installing all the requirements, just run:

```
jupyter notebook
```
