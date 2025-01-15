# Setup Guide

### Requirements
- Python 3.8+
- Virtual environment (e.g., Anaconda or Miniconda)
- Required Python packages (see `requirements.txt`)

### Steps to Set Up the Environment

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd reviewer-matcher
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   - Copy `.env.sample` to `.env` and update the required fields.
