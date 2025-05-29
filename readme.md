# Instrucciones de Instalación

## 1. Clonar el repo
  - git clone [https://github.com/DiegoRdrz/Chatbot](https://github.com/DiegoRdrz/Chatbot)
  - cd repo clonado

## 2. Crear un entorno Voirtual 
  - python -m venv env
  - env\Scripts\activate

## 3. Instalar dependencias
  - pip install gpt4all gradio transformers nltk torch

## OPCIONAL SI TIENEN NVIDIA
  -  Primero Instalar Nvidia CUDA VERSION 11.7 En el PC

  -  Segundo Correr el siguente comando para que el modelo principal se ejectute con GPU
      - pip install llama-cpp-python --upgrade --extra-index-url https://jllllll.github.io/llama-cpp-python-cu117/
  -  Tercero correr el siguente comando para que los modelos secundarios se ejecuten con GPU
      - pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117


# Ejecutar el chatbot

## 1. Comando
  - python app.py
## 2. Navegador
  - http://127.0.0.1:7860
