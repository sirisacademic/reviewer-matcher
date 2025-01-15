# Reviewer matcher 🤺

## Table of Contents
1. [Introduction](#introduction)
2. [Project Objectives](#project-objectives)
3. [Architecture Overview](#architecture-overview)
4. [Setup Guide](docs/setup.md)
5. [Class Reference](docs/classes.md)
6. [Contribution Guidelines](docs/contribute.md)
7. [Branching Model](docs/branching_model.md)

## Introduction

**Reviewer Matcher** is a Python-based software designed to automate the assignment of expert reviewers to project proposals.
Leveraging advanced NLP techniques, the tool ensures that reviewers' expertise aligns closely with the objectives and methodologies of the proposals.

### Project Objectives

The project aims to:

1. Facilitate the reviewer assignment process for project proposals.
2. Improve alignment between reviewer expertise and project requirements.
3. Provide a flexible, modular framework for managing data and assignments.

## Architecture Overview

The tool is modular, with components for data preprocessing, metadata enrichment, similarity calculation, and assignment optimization.
Refer to individual files for detailed documentation on each module.

## Scripts Overview

### `main.py`

The `main.py` script acts as the entry point for executing the Reviewer Matcher pipeline.

#### Key Features

1. **Pipeline Configuration and Initialization**:
   - Imports necessary modules and configurations.
   - Loads configuration dynamically at runtime.

2. **Component-Based Execution**:
   - Allows execution of specific components (e.g., similarity calculation).

3. **Testing and Debugging**:
   - Supports a test mode for smaller datasets or debugging.

   - Logs errors and debugging details.

#### Example Usage

1. **Run Full Pipeline**:
   ```bash
   python main.py --config config_general.py
   ```

2. **Run a Specific Component**:
   ```bash
   python main.py --component similarity_calculation --config config_similarity_scores.py
   ```

3. **Test Mode**:
   ```bash
   python main.py --test-mode --config config_general.py
   ```

## Preliminary code structure [to be updated]

In the image, we provide an overview of the core modules of the tool, highlighting its fundamental components and their primary functions. Each module represents a key area of functionality, illustrating how different parts of the tool work together to deliver a cohesive experience. This breakdown serves as an introduction to the essential building blocks of the tool, helping users understand its architecture and operational flow.

![image](https://github.com/user-attachments/assets/e47e2ad0-8946-4ad9-84ee-ecba3c8783f4)

## Data

The data shown for demonstartion is artificially generated to demonstrate the tool’s capabilities, as the actual training data is confidential and cannot be shared publicly.

https://docs.google.com/spreadsheets/d/1lUBxxTinGEsp1tmXDTsw29en6hKfYR798QdEFgQAPOU/edit?gid=0#gid=0

### 🛟 A note on language

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in [RFC 2119](https://www.ietf.org/rfc/rfc2119.txt).
