import cv2
import dlib
import numpy as np
import pickle
from fer import FER

def recognize_faces():
    # Carrega o modelo treinado
    with open('models/faces_encodings.pkl', 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)

    # Inicializa o detector e o predictor de marcos faciais do dlib
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')  # Certifique-se de ter este arquivo
    face_encoder = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')  # Certifique-se de ter este arquivo
    
    # Inicializa o detector de emoções
    emotion_detector = FER(mtcnn=True)

    # Inicializa a webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Reduz o tamanho do frame para processamento mais rápido
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Converte o frame para escala de cinza (recomendado para dlib)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # Detecta rostos no frame
        face_locations = detector(gray)

        # Extrai as características dos rostos
        face_encodings = []
        for face in face_locations:
            # Obtenha os marcos faciais
            shape = predictor(gray, face)
            # Obtenha a codificação
            face_encoding = np.array(face_encoder.compute_face_descriptor(small_frame, shape))
            face_encodings.append(face_encoding)

        # Percorre os rostos detectados
        for face, face_encoding in zip(face_locations, face_encodings):
            # Compara o rosto atual com os rostos conhecidos
            matches = np.linalg.norm(known_face_encodings - face_encoding, axis=1) <= 0.6
            name = "Desconhecido"

            # Verifique se há algum rosto reconhecido
            if np.any(matches):
                # Encontra o índice do rosto reconhecido
                best_match_index = np.argmin(np.linalg.norm(known_face_encodings - face_encoding, axis=1))
                name = known_face_names[best_match_index]

            # Calcular a emoção
            emotion, score = emotion_detector.top_emotion(frame)
            if emotion is None:
                emotion = "Indetectável"

            # Desenha um retângulo ao redor do rosto
            cv2.rectangle(frame, (face.left()*4, face.top()*4), (face.right()*4, face.bottom()*4), (0, 255, 0), 2)
            # Exibe o nome e a emoção do rosto identificado
            cv2.putText(frame, f"{name}, {emotion}", (face.left()*4, face.top()*4 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Exibe o frame com reconhecimento facial e emoções
        cv2.imshow('WebRe - Reconhecimento Facial e Emoções', frame)

        # Saída ao pressionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera a webcam e fecha as janelas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()
