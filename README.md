# **Local LLM Hub**

A locally running AI chatbot build using Python, Hugging Face Transformers, PyTorch, and Qwen2.5-1.5B-Instruct.

## **Features**

- Run pretrained language models locally.
- Generate AI responses using Hugging Face Transformers.
- GPU acceleration using NVIDIA CUDA.
- Interative terminal-based chatbot.
- Conversation history.
- Local model storage and offline interface after downloading.

## **Tech Stack**
- Python
- PyTorch
- Hugging Face Transformers
- Accelerate
- Qwen2.5-1.5B-Instruct
- NVIDIA CUDA

## **Project Structure**

local-model-hub/
├── chatbot.py
├── download_model.py
├── gpu_test.py
├── requirements.txt
├── README.md
└── .gitignore

## **Installation**

### **1. Clone the repository**

git clone REPOSITORY_URL
cd local-model-hub

### **2. Create a virtual environment**

python -m venv venv
venv\Scripts\activate

### **3. Install PyTorch**

Install a CUDA-enable PyTorch version compatible with your NVIDIA GPU using:
https://pytorch.org/get-started/locally/

### **4. Install dependencies**

pip install -r requirements.txt

### **5. Download the model**

python download_model.py

### **6. Run the chatbot**

python chatbot.py

## **Hardware**

Developed and tested on:
- RAM: 16 GB
- GPU: NVIDIA GeForce RTX 4050
- Dedicated VRAM: 6 GB

## **Model**

Qwen2.5-1.5B-Instruct

https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct

## **Future Improvements**

- Streaming AI responses
- Web-based chatbot interface
- 4-bit model quantization
- PDF question answering using RAG
- Support for multiple local models

## **Author**

Developed as a hands-on project to explore local LLM deployment and AI application development.