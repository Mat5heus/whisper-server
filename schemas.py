from pydantic import BaseModel, validator, confloat, conint

class TranscriberConfig(BaseModel):
    rate: conint(gt=8000, lt=48000) = 16000
    chunk: conint(gt=256, lt=4096) = 1024
    buffer: conint(gt=1, lt=60) = 5
    model: str = 'tiny'
    silence_threshold: confloat(ge=0.0) | str = 'auto'
    silence_timeout: confloat(ge=0.5) = 3.0
    verbose: bool = False

    @validator('model')
    def validate_model(cls, v):
        valid_models = ['tiny', 'base', 'small', 'medium']
        if v not in valid_models:
            raise ValueError(f"Modelo inválido. Opções válidas: {valid_models}")
        return v

    @validator('silence_threshold')
    def validate_threshold(cls, v):
        if isinstance(v, str) and v != 'auto':
            raise ValueError("Valor inválido para silence_threshold. Use 'auto' ou um número.")
        return v
    