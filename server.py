from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from queue import Queue, Empty
import threading
from schemas import TranscriberConfig
import logging

def create_server(transcriber):
    app = Flask(__name__)
    CORS(app)
    logger = logging.getLogger(__name__)

    @app.route('/stream')
    def stream():
        logger.info("Nova conexão de streaming estabelecida")

        # Inicia o transcriber se não estiver rodando
        if not transcriber.running:
            transcriber.start()
            logger.debug("Transcriber iniciado via streaming")

        client_queue = Queue(maxsize=1)
        
        with transcriber.client_queues_lock:
            transcriber.client_queues.append(client_queue)
            logger.debug(f"Clientes conectados: {len(transcriber.client_queues)}")

        def generate():
            try:
                last_text = None
                while transcriber.running:
                    try:
                        text = client_queue.get(timeout=1)
                        
                        # Filtra repetições consecutivas
                        if text != last_text:
                            logger.debug(f"Enviando texto: {text}")
                            yield f"{text} "
                            last_text = text
                            
                    except Empty:
                        # Mantém a conexão ativa com keep-alive
                        pass
            except GeneratorExit:
                logger.info("Cliente desconectado")
            finally:
                yield "\n"
                with transcriber.client_queues_lock:
                    if client_queue in transcriber.client_queues:
                        transcriber.client_queues.remove(client_queue)
                        logger.debug(f"Clientes restantes: {len(transcriber.client_queues)}")

        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive'
            }
        )

    @app.route('/config', methods=['GET', 'POST'])
    def config():
        if request.method == 'POST':
            try:
                new_config = TranscriberConfig(**request.json)
                transcriber.apply_config(new_config)
                return jsonify({
                    "status": "success",
                    "config": transcriber.config.dict()
                })
            except Exception as e:
                return jsonify({
                    "status": "error",
                    "message": str(e)
                }), 400
        
        return jsonify(transcriber.config.dict())

    @app.route('/status')
    def status():
        return jsonify({
            "running": transcriber.running,
            "recording": transcriber.audio_thread.is_alive() if transcriber.audio_thread else False,
            "transcribing": transcriber.transcribe_thread.is_alive() if transcriber.transcribe_thread else False
        })

    @app.route('/control', methods=['POST'])
    def control():
        action = request.json.get('action')
        
        if action == 'start':
            transcriber.start()
            return jsonify({"status": "started"})
        elif action == 'stop':
            transcriber.stop()
            return jsonify({"status": "stopped"})
        else:
            return jsonify({"status": "error", "message": "Ação inválida"}), 400

    return app