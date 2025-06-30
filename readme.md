# CORA Translation Project

Bachelor project to automatically translate the [CORA toolbox](https://github.com/TUMcps/CORA) from MATLAB to Python using Large Language Models (LLMs). The process is organized in [Cursor](https://www.cursor.com), with **Claude-4-Sonnet** as the primary model and **Gemini-2.5-Pro** as backup. readme_florian.md contains my instructions for the model.

## Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/Floriannue/Translate_Cora.git
cd Translate_Cora

# 2. Initialize and update the CORA submodule (MATLAB code)
git submodule init
git submodule update

# 3. Install Python dependencies using Poetry (recommended)
#pip install poetry if not available
poetry install #(or pip install .)
```
