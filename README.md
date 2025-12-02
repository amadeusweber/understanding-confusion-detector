# understanding-confusion-detector
A study of automatic detection of understanding and confusion in learning
scenarios, assessing behavior under stress and relaxed conditions.

# Setup
This repository was developed using `Python 3.10`.
The dependencies were installed in a 
[Python Virtual Environment](https://docs.python.org/3/library/venv.html).

```bash
# create venv
python3 -m venv .venv
# activate venv
source .venv/bin/activate
```

All required packages are listed in the [requirements.txt](requirements.txt)

```bash
pip install -r requirements.txt
```

# Configuration
Default configuration is provided by [configuration/default.conf](configuration/default.conf). Default values are overwtitten by host machine specific configurations located at `configurations/[os.uname().nodename].conf`. Create the host machine specific file and only modify the sections and variables specific to the host machine.

**Example**
```conf
[Dataset]
path-root     = /vol/dataset/
...
```

**Patterns**
In the config file includes file naming patterns.
These are used to be filled with a specific value:
`file-pattern = data_%s.csv` has the placeholder `%s` variable formatted as string.
