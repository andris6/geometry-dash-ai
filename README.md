# Geometry Dash AI

Simple NEAT AI for playing the game Geometry Dash  
**Inspired by [@CodeBullet](https://www.youtube.com/@CodeBullet)**

## How to actually try this out

_(On macOS/Linux/BSDs)_

### Setup

1. Install Python 3.11+ ([www.python.org](https://www.python.org)) and Git ([git-scm.com](https://git-scm.com/))
2. Clone this repo:
   ```
   git clone https://github.com/andris6/geometry-dash-ai.git && \
   cd geometry-dash-ai/
   ```
3. Create a virtual environment:
   ```
   python3 -m venv venv && \
   source venv/bin/activate
   ```
4. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

### Configure Geometry Dash

Before run, set these options in Geometry Dash:
- Windowed mode (Options / Graphics / Windowed)
- Low Detail Mode (Options / Graphics / Low Detail Mode)

### Start training

1. Open a random level in Geometry Dash (something easy, if you're running this the first time)
2. Then start training (still in the `venv`):
   ```
   python main.py train
   ```

You can always stop the training and restart it later (just press `Ctrl+C`)

