# CORA Translation Project

Bachelor project: supervised automatic translation of the [CORA toolbox](https://github.com/TUMcps/CORA) from MATLAB to Python using Large Language Models (LLMs). The process is organized in [Cursor](https://www.cursor.com), with **Claude-4-Sonnet** as the primary model and **Cursor auto** as the backup. readme_florian2.md contains my instructions for the model.

## Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/Floriannue/Translate_Cora.git
cd Translate_Cora

# 2. Initialize and update the CORA submodule
# Only do so if you need CORA - the cora_python translation does not require it to run.
git submodule init
git submodule update
#Follow the install instructions in cora_matlab/README.md if you want to run cora_matlab
#The translation process can then run matlab tests

# 3. Install Python dependencies using Poetry (recommended)
#pip install poetry if not available
poetry install #(or pip install .)
```
