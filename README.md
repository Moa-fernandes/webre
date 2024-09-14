# WebRe - Reconhecimento Facial e Detecção de Emoções

**Desenvolvedor:** Moacir Fernandes

## Descrição do Projeto

WebRe é um sistema de reconhecimento facial e detecção de emoções que utiliza a webcam para identificar rostos e analisar suas emoções em tempo real. O projeto utiliza técnicas avançadas de visão computacional, aprendizado de máquina e fornece uma interface gráfica amigável desenvolvida com PyQt5.

## Funcionalidades

- **Reconhecimento Facial**: Identifica rostos previamente cadastrados e exibe seus nomes.
- **Detecção de Emoções**: Analisa as emoções dos rostos detectados (feliz, triste, surpreso, etc.) e exibe a emoção em inglês e português.
- **Interface Gráfica**: Interface amigável desenvolvida com PyQt5 para iniciar e parar o reconhecimento facial.
- **Visualização em Tempo Real**: Captura e exibe a saída da webcam em tempo real, com informações de reconhecimento facial e emoções.
- **Desempenho Otimizado**: Implementações otimizadas para funcionar em sistemas com recursos limitados.
## Tecnologias Utilizadas

- **Python**
- **OpenCV**
- **Dlib**
- **FER**
- **PyQt5**

## Pré-requisitos

- Python 3.x
- Pip (gerenciador de pacotes do Python)
- Webcam integrada ou externa

## Instalação

1. **Clone o repositório**:
   ```bash
   git clone https://github.com/moa-fernandes/webre.git
   ```
2. **Navegue até o diretório do projeto**:
   ```bash
   cd webre
   ```
3. **Crie um ambiente virtual**:
   ```bash
   python -m venv venv
   ```
4. **Ative o ambiente virtual**:
   - No Windows:
     ```bash
     venv\Scripts\activate
     ```
   - No Linux/Mac:
     ```bash
     source venv/bin/activate
     ```
5. **Instale as dependências**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuração

1. **Baixe os modelos necessários para o Dlib**:
   - `shape_predictor_68_face_landmarks.dat`
   - `dlib_face_recognition_resnet_model_v1.dat`
   
   Coloque esses arquivos na pasta `models/`.

2. **Treine o modelo com rostos conhecidos**:
   - Adicione imagens de referência para os rostos na pasta `datasets`.
   - Execute o script de treinamento:
     ```bash
     python src/train_model.py
     ```
   - Isso irá gerar o arquivo `faces_encodings.pkl` com as codificações faciais.

## Uso

1. **Inicie a interface gráfica**:
   ```bash
   python src/webre_gui.py
   ```
2. **Interaja com a interface**:
   - Clique em "Iniciar Reconhecimento" para começar a captura da webcam.
   - Clique em "Parar Reconhecimento" para finalizar a captura.
   
   A interface exibirá os rostos detectados e as emoções identificadas em inglês e português.

## Estrutura do Projeto

```
webre/
│
├── datasets/               # Imagens de referência dos rostos
├── models/                 # Modelos treinados e arquivos de suporte
│   ├── shape_predictor_68_face_landmarks.dat
│   ├── dlib_face_recognition_resnet_model_v1.dat
│   └── faces_encodings.pkl
│
├── src/                    # Código-fonte
│   ├── train_model.py      # Script para treinar o modelo
│   ├── webre_gui.py        # Script principal da interface gráfica
│   ├── capture_faces.py    # Script para capturar rostos
│
├── venv/                   # Ambiente virtual Python
├── requirements.txt        # Lista de dependências do Python
└── README.md               # Documentação do projeto
```

## Melhorias Futuras

- Aprimorar a precisão da detecção de emoções.
- Adicionar mais emoções reconhecíveis.
- Melhorar a detecção facial em ambientes de baixa iluminação.

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para fazer um fork deste repositório e enviar um pull request com melhorias.

## Licença

Este projeto está sob a licença MIT.

---

### README.md (English)

# WebRe - Facial Recognition and Emotion Detection

**Developer:** Moacir Fernandes

## Project Description

WebRe is a facial recognition and emotion detection system that uses a webcam to identify faces and analyze their emotions in real-time. The project utilizes advanced computer vision and machine learning techniques and provides a user-friendly graphical interface built with PyQt5.

## Features

- **Facial Recognition**: Identifies pre-registered faces and displays their names.
- **Emotion Detection**: Analyzes the emotions of detected faces (happy, sad, surprised, etc.) and displays the emotion in both English and Portuguese.
- **Graphical Interface**: User-friendly interface developed with PyQt5 to start and stop facial recognition.
- **Real-Time Visualization**: Captures and displays webcam output in real-time with facial recognition and emotion information.
- **Optimized Performance**: Tweaked for systems with limited resources.

## Technologies Used

- **Python**
- **OpenCV**
- **Dlib**
- **FER**
- **PyQt5**

## Prerequisites

- Python 3.x
- Pip (Python package manager)
- Integrated or external webcam

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/webre.git
   ```
2. **Navigate to the project directory**:
   ```bash
   cd webre
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```
4. **Activate the virtual environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On Linux/Mac:
     ```bash
     source venv/bin/activate
     ```
5. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Setup

1. **Download the necessary models for Dlib**:
   - `shape_predictor_68_face_landmarks.dat`
   - `dlib_face_recognition_resnet_model_v1.dat`
   
   Place these files in the `models/` directory.

2. **Train the model with known faces**:
   - Add reference images for faces in the `datasets` folder.
   - Run the training script:
     ```bash
     python src/train_model.py
     ```
   - This will generate the `faces_encodings.pkl` file with facial encodings.

## Usage

1. **Launch the graphical interface**:
   ```bash
   python src/webre_gui.py
   ```
2. **Interact with the interface**:
   - Click "Start Recognition" to begin webcam capture.
   - Click "Stop Recognition" to end the capture.
   
   The interface will display detected faces and identified emotions in both English and Portuguese.

## Project Structure

```
webre/
│
├── datasets/               # Reference images for faces
├── models/                 # Trained models and support files
│   ├── shape_predictor_68_face_landmarks.dat
│   ├── dlib_face_recognition_resnet_model_v1.dat
│   └── faces_encodings.pkl
│
├── src/                    # Source code
│   ├── train_model.py      # Script to train the model
│   ├── webre_gui.py        # Main script for the graphical interface
│   ├── capture_faces.py    # Script to capture faces
│
├── venv/                   # Python virtual environment
├── requirements.txt        # List of Python dependencies
└── README.md               # Project documentation
```

## Future Enhancements

- Improve emotion detection accuracy.
- Add more recognizable emotions.
- Enhance facial detection in low-light environments.

## Contribution

Contributions are welcome! Feel free to fork this repository and submit a pull request with improvements.

## License

This project is licensed under the MIT License.