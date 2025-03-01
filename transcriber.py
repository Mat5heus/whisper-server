import logging
import threading
import time
from collections import deque
from queue import Queue, Full, Empty

import numpy as np
import sounddevice as sd
import torch
import whisper
import noisereduce as nr

# Adicione no início do arquivo
from tqdm import tqdm
import requests
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class AudioTranscriber:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(f"Transcriber.{self.__class__.__name__}")  # Namespace único
        self.running = False
        self.noise_profile = None
        
        self._init_components()
        self._configure_logging()  # Deve vir antes de apply_config
        self.apply_config(config)

    def _init_components(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.audio_queue = Queue()
        self.buffer_deque = deque()
        self.last_voice_time = time.time()
        self.last_voice_time_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.prealloc_buffer = np.empty((1,), dtype=np.float32)
        self.client_queues = []
        self.client_queues_lock = threading.Lock()
        self.has_transcribed = False

    def _configure_logging(self):
        """Configura o sistema de logging de forma hierárquica"""
        self.logger.propagate = False  # Impede a propagação para o root logger
        
        # Remove handlers existentes
        while self.logger.handlers:
            self.logger.removeHandler(self.logger.handlers[0])
        
        if self.config.verbose:
            self.logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(handler)
        else:
            self.logger.addHandler(logging.NullHandler())
            self.logger.setLevel(logging.WARNING)

    def apply_config(self, new_config):
        self.logger.debug("Aplicando nova configuração")
        
        with threading.Lock():
            self.config = new_config
            
            # Reinicializa componentes dependentes da configuração
            self.audio_queue = Queue(maxsize=int(1.5 * (self.config.rate * self.config.buffer / self.config.chunk)))
            self.buffer_deque = deque(maxlen=(self.config.rate * self.config.buffer) // self.config.chunk)
            self.prealloc_buffer = np.empty((self.config.chunk,), dtype=np.float32)
            
            if self.config.silence_threshold == 'auto':
                self.silence_threshold = self.calibrate_threshold()
            else:
                self.silence_threshold = self.config.silence_threshold

            if not hasattr(self, 'model') or self.config.model != self.model_name:
                self._load_model()

    def _load_model(self):
        model_name = self.config.model
        self.logger.info(f"Carregando modelo '{model_name}'...")

        try:
            # Tenta carregar o modelo sem forçar o progress_callback
            self.model = whisper.load_model(
                model_name,
                device=self.device,
                download_root=os.path.join(Path.home(), ".cache", "whisper")
            )
            self.logger.info(f"Modelo '{model_name}' carregado com sucesso!")
            
        except Exception as e:
            self.logger.error(f"Falha ao carregar o modelo: {str(e)}")
            raise
        
        self.model_name = model_name

    def calibrate_threshold(self, calibration_duration=1.0):
        logger.info("Calibrando limiar de silêncio...")
        rms_values = []
        chunks_needed = int(self.config.rate * calibration_duration / self.config.chunk)  # Alterado args para config

        noise_samples = []
        with sd.InputStream(
            channels=1,
            samplerate=self.config.rate,  
            blocksize=self.config.chunk,
            dtype='int16'
        ) as stream:
            for _ in range(chunks_needed):
                data, _ = stream.read(self.config.chunk)
                data = data.ravel().astype(np.float32) / 32768.0
                noise_samples.append(data)
                rms_values.append(np.sqrt(np.mean(data**2)))
                logger.debug(f"RMS durante calibração: {rms_values}")
        
        # Crie o perfil de ruído (usando os primeiros 100ms)
        self.noise_profile = np.concatenate(noise_samples[:int(0.1 * self.config.rate // self.config.chunk)])
        
        baseline = np.percentile(rms_values, 90)
        threshold = baseline * 1.5
        threshold = min(threshold, 0.3)  # Limite máximo
        logger.debug(f"Limiar automático: {threshold:.4f}")
        return threshold
    
    def audio_callback(self, indata, frames, time_info, status):
        # Em audio_callback (transcriber.py):
        np.clip(indata.ravel() / 32768.0, -1.0, 1.0, out=self.prealloc_buffer)
        data = self.prealloc_buffer.copy()

        if status:
            self.logger.debug(f"Status de áudio: {status}")
        
        try:
            self.audio_queue.put_nowait(data)
            self.update_voice_activity(data)
        except Full:
            pass

    def update_voice_activity(self, data):
        rms = np.sqrt(np.mean(data**2))
        if rms > self.silence_threshold:
            with self.last_voice_time_lock:
                self.last_voice_time = time.time()

    def process_audio_buffer(self):
        try:
            while True:
                data = self.audio_queue.get_nowait()
                self.buffer_deque.append(data)
        except Empty:
            pass
        
        if len(self.buffer_deque) == 0:
            return None
        
        audio = np.concatenate(self.buffer_deque)[:self.config.rate * self.config.buffer]

         # Aplique redução de ruído
        if self.noise_profile is not None:
            audio = nr.reduce_noise(
                y=audio,
                y_noise=self.noise_profile,
                sr=self.config.rate,
                stationary=True,
                prop_decrease=0.75  # Reduz 75% do ruído
            )

        return audio

    def transcribe_loop(self):
        min_transcription_length = self.config.rate * 1
        last_transcription_time = 0
        transcription_interval = 0.5
        
        while not self.stop_event.is_set():
            with self.last_voice_time_lock:
                silence_duration = time.time() - self.last_voice_time
                if self.has_transcribed and silence_duration > self.config.silence_timeout:
                    self.stop_event.set()
                    self.running = False
                    break

            if time.time() - last_transcription_time < transcription_interval:
                time.sleep(0.1)
                continue
                
            audio = self.process_audio_buffer()
            if audio is None or len(audio) < min_transcription_length:
                continue
            
            try:
                # Pré-processamento adicional
                audio = audio.astype(np.float32)  # Garante o tipo correto
                result = self.model.transcribe(audio, fp16=(self.device == 'cuda'))
                text = result['text'].strip()
                
                if text:
                    # Envia para todos os clientes conectados
                    cleaned_text = text.replace('\n', ' ')
                    with self.client_queues_lock:
                        for q in self.client_queues:
                            try:
                                q.put_nowait(cleaned_text)
                            except Full:
                                pass
                    self.has_transcribed = True
                    last_transcription_time = time.time()
                    
            except Exception as e:
                self.logger.error(f"Erro na transcrição: {str(e)}")

    def start(self):
        if self.running:
            self.logger.warning("Tentativa de iniciar transcriber já em execução")
            return
        
        # Reinicializa todos os estados críticos
        self.stop_event.clear()
        self.has_transcribed = False
        self.last_voice_time = time.time()
        self.buffer_deque.clear()

        # Esvazia a fila de áudio
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Empty:
                break
        
        self.logger.info("Iniciando transcriber...")
        self.running = True
        self.audio_thread = threading.Thread(target=self._run_audio, name="AudioThread")
        self.audio_thread.start()
        
        self.transcribe_thread = threading.Thread(target=self.transcribe_loop, name="TranscribeThread")
        self.transcribe_thread.start()
        self.logger.debug("Threads de áudio e transcrição iniciadas")


    def _run_audio(self):
        self.logger.info(f"Iniciando captura de áudio com configuração: {self.config}")
        try:
            with sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.config.rate,
                blocksize=self.config.chunk,
                dtype='int16'
            ):
                self.logger.info("Captura de áudio iniciada com sucesso")
                while self.running:
                    time.sleep(0.5)
        except Exception as e:
            self.logger.error(f"Erro na captura de áudio: {str(e)}", exc_info=True)
            self.stop()

    def stop(self):
        self.running = False
        self.stop_event.set()
        if self.audio_thread.is_alive():
            self.audio_thread.join()
        if self.transcribe_thread.is_alive():
            self.transcribe_thread.join()
        self.client_queues.clear()
