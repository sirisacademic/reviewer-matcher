# ⚙️ Setup Your Working Environment

Follow this step-by-step guide to set up a working environment for developing and running the `expert-matcher` library.

> ⚠️ **Important**: Make sure you have access to all required dependencies and permissions if working with any protected resources.

**Table of contents**

- [Requirements](#requirements)
- [1. Create the environment variables file](#1-create-the-environment-variables-file)
- [2. Set up credentials (if required)](#2-set-up-credentials-if-required)
- [3. Install Python dependencies](#3-install-python-dependencies)

## Requirements

This project requires the following software components:

- Python 3.8+
- [Anaconda](https://docs.anaconda.com) or [Miniconda](https://docs.anaconda.com/miniconda/)
- (Optional) Google Cloud SDK, if using Google Cloud for certain integrations

Follow the **setup step-by-step guide** below for the one-time process needed to set up your working environment.

## 1. Create the environment variables file

The environment variables are stored in a `.env` file, which you need to create from the provided template. This file will help store any sensitive or configuration-specific settings.

1. Make a copy of the file `.env.sample`.
2. Rename the copied file to `.env` and store it at the root folder of this project.
3. Open the `.env` file and update the variables as necessary for your environment.

   - For example, if connecting to external services, update API keys or credentials in this file.

## 2. Set up credentials (if required)

If this project requires access to external APIs or Google Cloud resources, you’ll need to configure your credentials.

> TO SPECIFY

## 3. Install Python dependencies

We recommend creating a Python virtual environment to install the required dependencies for the project.

1. [Install Miniconda](https://docs.anaconda.com/miniconda/miniconda-install/) on your computer, if you don’t have it installed already.

2. Create and activate the conda environment:

   ```bash
   cd reviewer-matcher
   conda env create -f environment.yml
   # Or, if the environment is already created, update it with:
   # conda env update --file environment.yml --name reviewer-matcher_env
   conda activate expert_matcher_env