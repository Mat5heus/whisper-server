# Whisper Server

## Overview
A project for real-time audio transcription utilizing the [OpenAI Whisper](https://github.com/openai/whisper) library . The system is capable of capturing audio, identifying voice, recording, and processing it to transcribe.

## Setup & Usage

Clone this repository from GitHub to your local machine:
```bash
git clone https://github.com/Mat5heus/whisper-server.git
cd whisper-server
```

Ensure you have Python 3 installed and a virtual environment created if preferred:
```bash
python -m venv .venv

# On Unix or MacOS
source .venv/bin/activate

# On Windows
.\venv\Scripts\activate
```

### Prerequisites
Install dependencies from `requirements.txt` using pip:
```bash
pip install -r requirements.txt
```

Now you are ready to run the application. You can start a local server by executing:
```bash
python main.py
```

**Command Line Arguments (Optional):**
You can provide additional command-line arguments to customize the program's behavior. For instance, to set model size rate and language:
   ```bash
   python main.py -b small -l pt
   ```

## Argument Description

- `-r or --rate`: Sets the sampling rate in Hz. Default value is 16000.
- `-c or --chunk`: Defines the audio block size in bytes. Default value is 1024.
- `-b or --buffer`: Specifies the buffer duration in seconds. Default value is 5.
- `-m or --model`: Selects the Whisper model ('tiny', 'base', etc.). Default is 'tiny'.
- `-sd or --silence_threshold`: Sets the silence threshold for stopping transcription. Default value is 'auto'.
- `-st or --silence_timeout`: Defines the time in seconds to consider audio as silent before stopping transcription. Default is 3.
- `-v or --verbose`: Shows debugging messages. Default is false (off).
- `-n or --noise_reduction`: Specifies the noise reduction intensity, ranging from 0.0 to 1.0. Default is 0.75.
- `-l or --language`: Defines the language for transcription (e.g., 'pt' for Portuguese, 'en' for English). Default is 'auto'.
- `-p or --port`: Sets the HTTP API port number. Default value is 5000.


### Accessing and Configuring
Use `http://localhost` by default to access the web interface, allowing you to:
- Submit POST requests to `/config` to change transcription parameters.
- Fetch current status with `/status`.
- Control operations with `/control`, including `start/stop` actions for the Transcriber.

### Client Integration (optional)
For integrating client applications using SSE, connect them via a URL like:
```bash
http://localhost:5000/stream
```
Clients should implement Server-Sent Events parsing to receive transcriptions in real time.

This project aims at providing an efficient and lightweight audio transcription solution with flexibility for customizable configurations and seamless integration through web protocols.

# Structure Description

1. **Transcriber Class**: Handles all the main functionalities including audio capture, processing, recording, and communication with client servers through an event-driven API.

2. **Schemas.py**: Defines a Pydantic model for configuration of transcription settings.

3. **server.py**: A Flask web server that supports:
   - Real-time streaming of recognized speech via an event-stream protocol using Server-Sent Events (SSE).
   - Configuration retrieval and application through REST API endpoints.
   - Status checks on the current operational status, live recording state, transcription status, and control over starting/stopping the main transcriber.

4. **main.py**: The entry point to start the system with a specified configuration or load it from an environment variable.

5. **requirements.txt**: A list of all Python dependencies necessary for this project including audio processing libraries (SoundDevice), machine learning model libraries (TikToken, Transformers), web framework (Flask) and more.

6. **transcriber.py**: The primary class `Transcriber` that orchestrates the real-time transcription operations:
   - Audio capture using SoundDevice with configurable rate, buffer size, etc., while handling silence detection to optimize power consumption by switching off when no sound is detected for a specified duration.
   - Recording audio data in chunks when speech is detected and processing them in parallel or sequentially based on the `process_audio` method choice.
   - Processing of captured audio with noise reduction (if applicable), model transcription, and text delivery to connected clients using an event-driven approach for efficiency and low latency.


## Dependencies

The project relies on several libraries for execution:

1. **OpenAI's Whisper** ([Whisper GitHub Repository](https://github.com/openai/whisper)): A transcription library by OpenAI.
2. **Flask** ([Flask Documentation](https://flask.palletsprojects.com/en/2.0.x/)): A lightweight web framework for Python.
3. **Flask-CORS** ([Flask-CORS Documentation](https://flask-cors.readthedocs.io/en/latest/)): A Flask extension to enable cross-origin resource sharing from the frontend.
4. **Pydantic** ([Pydantic Documentation](https://pydantic-docs.helpmanual.io/)): A library for type validation and conversion.

## Contributing

Contributions are highly appreciated! If you encounter any issues or have suggestions, please don't hesitate to:

- Open an issue on the project's repository.
- Submit a pull request with your changes.



