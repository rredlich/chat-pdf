# 💬 Chatea con un PDF
Script básico de Python que conecta gpt-3.5 con la información de un PDF a través de LangChain

# Requerimientos
* Python 3.7.1
* openai 0.27.0
* python-dotenv 0.21.1
* langchain
* chromadb
* pymupdf
* tiktoken


# Configuración
## Configura tu archvio env
1. Copia el archivo .eng-template y renómbralo .env

## Consigue tu API key de OpenAI
1. Créate una cuenta en OpenAI
2. En el menú lateral anda a User > API Keys (https://platform.openai.com/account/api-keys)
3. Haz click en Create new secret key
4. Copia tu API key secreta y pégala en la variable OPENAI_API_KEY en el archivo .env

## Instala las dependencias
```sh
pip3 install openai chromadb langchain pymupdf tiktoken
```

## Ejecuta el script de Python
```sh
python3 main.py
```