# GPT Project

## Overview

This project is a lightweight implementation of a GPT-style transformer model for text generation. Originally developed over a year ago.

The project is designed to be modular, scalable, and production-ready, showcasing a well-structured codebase that separates configuration, data handling, model architecture, training, and execution.

## Project Structure

gpt_project/
├── config.py
├── data_loader.py
├── main.py
├── models
│   ├── __init__.py
│   └── gpt_model.py
└── trainer.py



## How to Run the Code

### Prerequisites

- **Python 3.8+**
- **PyTorch** (specified in `requirements.txt`)
- Optionally, **Docker** (recommended for an isolated, reproducible environment)

### Setting Up Without Docker

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/gpt_project.git
   cd gpt_project
   
## Install Dependencies:
pip install -r requirements.txt


## Run the Project:
python main.py

## Running with Docker
1: Build the Docker Image:
    docker build -t gpt_project .

2: Run the Docker Container:
docker run --rm gpt_project




