# Whisper Server

## Overview
A project for real-time audio transcription utilizing the [OpenAI Whisper](https://github.com/openai/whisper) library . The system is capable of capturing audio, identifying voice, recording, and processing it to transcribe.

### Prerequisites
Ensure you have Python 3 installed and a virtual environment created if preferred:
```bash
python -m venv venv

# On Unix or MacOS
source venv/bin/activate

# On Windows
.\venv\Scripts\activate
```

## Setup & Usage
Clone this repository from GitHub to your local machine:
```bash
git clone https://github.com/Mat5heus/whisper-server.git
cd whisper-server
```

Install dependencies from `requirements.txt` using pip:
```bash
pip install -r requirements.txt
```

Now you are ready to run the application. You can start a local server by executing:
```bash
python main.py
```

 **Argumentos de Linha de Comando (opcional):**
   Você pode passar argumentos adicionais para personalizar o comportamento do programa. Por exemplo, para especificar a taxa de amostragem e o tamanho do bloco de áudio:
   ```bash
   python main.py -b small -l es
   ```

## Descrição dos Argumentos

- `-r --rate`: Define a taxa de amostragem (Hz). Default=16000.
- `-c --chunk`: Define o tamanho do bloco de áudio. Default=1024.
- `-b --buffer`: Define a duração do buffer em segundos. Default=5.
- `-m --model`: Define o modelo do Whisper ('tiny', 'base', etc.). Default=tiny'.
- `-sd --silence_threshold`: Define o limiar de silêncio. Default='auto'.
- `-st --silence_timeout`: Define o tempo de silêncio para parar a transcrição. Default=3.
- `-v --verbose`: Mostra mensagens de debugging. Default=false.
- `-n --noise_reduction`: Define a intensidade da redução de ruído (0.0-1.0). Default=0.75.
- `-l --language`: Define o idioma para transcrição (ex: 'pt', 'en'). Default='auto'.
- `-p --port`: Define a porta para a API HTTP. Default=5000.


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

## Dependências

O projeto utiliza várias bibliotecas para execução. Aqui estão algumas delas:

- [OpenAI's Whisper](https://github.com/openai/whisper) - Biblioteca de transcrição.
- [Flask](https://flask.palletsprojects.com/) - Framework web Python.
- [Flask-CORS](https://flask-cors.readthedocs.io/en/latest/) - Middleware para permitir o acesso ao servidor via frontend.
- [Pydantic](https://pydantic-docs.helpmanual.io/) - Biblioteca de validação e conversão de tipos.


## Contribuição

Contribuições são bem-vindas! Se você encontrar um problema ou tiver sugestões, sinta-se à vontade para abrir uma issue ou enviar um pull request.



