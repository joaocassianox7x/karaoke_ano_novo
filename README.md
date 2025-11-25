# Karaokê Ano Novo

Aplicativo Streamlit para baixar vídeos do YouTube, gerar legendas com Whisper e gravar/pontuar seu karaokê usando similaridade espectral (FFT).

## Requisitos
- Python 3.10+
- ffmpeg disponível no PATH para evitar avisos do pydub
- Dependências: `pip install -r requirements.txt`

## Como rodar
1. Ative o ambiente virtual (opcional, mas recomendado).
2. Instale dependências: `pip install -r requirements.txt`
3. Inicie o app: `streamlit run main.py`

## Testes
- Execute `pytest` na raiz do projeto.

## Bugs conhecidos
- O pydub emite aviso se o ffmpeg não estiver presente; a geração/filtragem de áudio pode falhar sem ele.
- Modelos de voz maiores (Whisper) podem demorar em CPU; ajuste na página de configurações conforme necessário.
