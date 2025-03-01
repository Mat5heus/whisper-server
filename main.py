import argparse
from transcriber import AudioTranscriber
from server import create_server
from schemas import TranscriberConfig
import logging

def parse_args():
    parser = argparse.ArgumentParser(description="Transcrição de áudio em tempo real com Whisper")
    parser.add_argument("-r","--rate", type=int, default=16000, help="Taxa de amostragem (Hz). Default= 16000")
    parser.add_argument("-c","--chunk", type=int, default=1024, help="Tamanho do bloco de áudio. Default= 1024")
    parser.add_argument("-b","--buffer", type=int, default=3, help="Duração do buffer (segundos). Default= 5")
    parser.add_argument("-m","--model", type=str, default='tiny', help="Modelo do Whisper ('tiny', 'base', etc.). Default= 'tiny'")    
    parser.add_argument("--silence_threshold", type=lambda x: x if x == 'auto' else float(x), default='auto', help="Limiar de silêncio. Default= auto")    
    parser.add_argument("--silence_timeout", type=float, default=2, help="Tempo de silêncio para parar a transcrição. Default= 3")
    parser.add_argument("-v", "--verbose", action="store_true", help="Mostrar mensagens de debugging. Default= false")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
     # Configurar logging global primeiro
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    # Converter args para configuração validada
    config = TranscriberConfig(**vars(args))
    
    transcriber = AudioTranscriber(config)
    app = create_server(transcriber)
    
    logger = logging.getLogger(__name__)
    logger.debug("Iniciando servidor e transcriber...")
    
    # Iniciar em modo misto (API e transcrição)
    #transcriber.start()
    
    try:
        logger.debug("Servidor iniciado na porta 5000")
        app.run(port=5000, threaded=True, use_reloader=False, debug=args.verbose)
    except KeyboardInterrupt:
        logger.debug("Recebido KeyboardInterrupt, parando...")
        transcriber.stop()