import sys
import cv2
import dlib
import numpy as np
import pickle
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap, QPalette, QBrush, QFont
from PyQt5.QtCore import QTimer, Qt
from fer import FER
from collections import deque

class WebReGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        # Configurações da janela principal
        self.setWindowTitle(' - Reconhecimento Facial e Emoções')
        self.setGeometry(100, 100, 800, 600)

        # Configurar imagem de fundo centralizada
        self.set_background_image('2.png')

        # Configura o layout
        self.central_widget = QWidget(self)
        self.layout = QVBoxLayout(self.central_widget)
        self.setCentralWidget(self.central_widget)

        # Adiciona um rótulo para exibir o vídeo
        self.video_label = QLabel(self)
        self.layout.addWidget(self.video_label)

        # Adiciona botões para controle
        self.button_layout = QHBoxLayout()
        self.layout.addLayout(self.button_layout)

        # Estilizando os botões
        button_style = """
        QPushButton {
            background-color: #1E90FF;  /* Azul */
            color: white;               /* Texto branco */
            border-radius: 10px;        /* Bordas arredondadas */
            padding: 10px;              /* Espaçamento interno */
            font-size: 16px;            /* Tamanho da fonte */
        }
        QPushButton:hover {
            background-color: #1C86EE;  /* Azul escuro quando o botão é focado */
        }
        QPushButton:pressed {
            background-color: #104E8B;  /* Azul ainda mais escuro quando o botão é pressionado */
        }
        """

        self.start_button = QPushButton('Iniciar Reconhecimento', self)
        self.start_button.setStyleSheet(button_style)
        self.start_button.clicked.connect(self.start_recognition)
        self.button_layout.addWidget(self.start_button)

        self.stop_button = QPushButton('Parar Reconhecimento', self)
        self.stop_button.setStyleSheet(button_style)
        self.stop_button.clicked.connect(self.stop_recognition)
        self.button_layout.addWidget(self.stop_button)

        # Inicializa a captura de vídeo
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Carrega o modelo treinado para reconhecimento facial
        with open('models/faces_encodings.pkl', 'rb') as f:
            self.known_face_encodings, self.known_face_names = pickle.load(f)

        # Inicializa o detector e o predictor de marcos faciais do dlib
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
        self.face_encoder = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

        # Inicializa o detector de emoções
        self.emotion_detector = FER(mtcnn=True)

        # Controle de desempenho: Processa um em cada 3 frames
        self.frame_skip_counter = 0

        # Fila para suavizar emoções (pós-processamento)
        self.emotion_queue = deque(maxlen=10)

    def set_background_image(self, image_path):
        # Configura a imagem de fundo centralizada
        palette = QPalette()
        background_image = QPixmap(image_path)

        # Reduz o tamanho da imagem para ser menor
        smaller_image = background_image.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Define um brush para a imagem redimensionada
        brush = QBrush(smaller_image)

        # Preenche o plano de fundo com a imagem centralizada
        palette.setBrush(QPalette.Window, brush)
        self.setPalette(palette)

        # Define a posição da imagem
        self.setAutoFillBackground(True)

    def resizeEvent(self, event):
        # Redimensiona a imagem de fundo quando a janela é redimensionada
        self.set_background_image('2.png')
        super().resizeEvent(event)

    def start_recognition(self):
        # Inicializa a webcam
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)  # Atualiza a cada 30ms para um bom desempenho

    def stop_recognition(self):
        # Para a captura de vídeo
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        self.video_label.clear()

    def update_frame(self):
        # Controle de desempenho: Processa um em cada 3 frames
        self.frame_skip_counter += 1
        if self.frame_skip_counter % 3 != 0:
            return

        # Captura um frame da webcam
        ret, frame = self.cap.read()
        if not ret:
            return

        # Reduz o tamanho do frame para processamento mais rápido
        small_frame = cv2.resize(frame, (320, 240))  # Resolução menor para desempenho
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # Detecta rostos no frame
        face_locations = self.detector(gray)
        face_encodings = []

        for face in face_locations:
            # Obtenha os marcos faciais
            shape = self.predictor(gray, face)
            # Obtenha a codificação
            face_encoding = np.array(self.face_encoder.compute_face_descriptor(small_frame, shape))
            face_encodings.append(face_encoding)

        # Percorre os rostos detectados
        for face, face_encoding in zip(face_locations, face_encodings):
            # Compara o rosto atual com os rostos conhecidos
            matches = np.linalg.norm(self.known_face_encodings - face_encoding, axis=1) <= 0.6
            name = "Desconhecido"

            # Verifique se há algum rosto reconhecido
            if np.any(matches):
                # Encontra o índice do rosto reconhecido
                best_match_index = np.argmin(np.linalg.norm(self.known_face_encodings - face_encoding, axis=1))
                name = self.known_face_names[best_match_index]

            # Calcular a emoção no ROI
            (left, top, right, bottom) = (face.left(), face.top(), face.right(), face.bottom())
            left, top, right, bottom = [int(coord * 2) for coord in (left, top, right, bottom)]  # Escala para o frame original
            roi = frame[top:bottom, left:right]

            # Verifique se o ROI não está vazio e tem um tamanho mínimo
            if roi.size > 0 and right - left > 0 and bottom - top > 0:
                emotions = self.emotion_detector.detect_emotions(roi)
                if emotions:
                    top_emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
                    emotion_pt = self.translate_emotion(top_emotion)
                    # Adiciona a emoção à fila para pós-processamento
                    self.emotion_queue.append((top_emotion, emotion_pt))
                    # Define a emoção como a mais frequente na fila
                    most_common_emotion = max(set(self.emotion_queue), key=self.emotion_queue.count)
                    emotion, emotion_pt = most_common_emotion
                else:
                    emotion = "Indetectável"
                    emotion_pt = "Indetectável"
            else:
                emotion = "Indetectável"
                emotion_pt = "Indetectável"

            # Desenha uma elipse ao redor do rosto para uma detecção oval
            center = ((left + right) // 2, (top + bottom) // 2)
            axes = ((right - left) // 2, (bottom - top) // 2)
            cv2.ellipse(frame, center, axes, 0, 0, 360, (0, 0, 255), 2)

            # Exibe o nome e a emoção do rosto identificado em amarelo com fonte melhorada
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"{name}, {emotion} ({emotion_pt})"
            text_size = cv2.getTextSize(text, font, 0.6, 1)[0]
            text_x = left
            text_y = top - 10 if top - 10 > 10 else top + 10

            # Fundo do texto para maior contraste
            cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5), 
                          (text_x + text_size[0], text_y + 2), (0, 0, 0), cv2.FILLED)

            # Exibe o texto com suavidade
            cv2.putText(frame, text, (text_x, text_y), font, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

        # Exibe o frame na GUI
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(800, 600, aspectRatioMode=1)
        self.video_label.setPixmap(QPixmap.fromImage(p))

    def translate_emotion(self, emotion):
        # Traduz as emoções para o português
        translations = {
            'happy': 'Feliz',
            'sad': 'Triste',
            'angry': 'Zangado',
            'surprise': 'Surpreso',
            'fear': 'Medo',
            'neutral': 'Neutro',
            'disgust': 'Nojo',
            'boredom': 'Tédio',
            'indetectable': 'Indetectável'
        }
        return translations.get(emotion.lower(), 'Indetectável')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = WebReGUI()
    window.show()
    sys.exit(app.exec_())
