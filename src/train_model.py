import face_recognition
import os
import cv2
import numpy as np
import pickle

def train_model():
    known_face_encodings = []
    known_face_names = []

    # Percorre o diretório datasets para encontrar as imagens
    for person_name in os.listdir('datasets'):
        person_dir = os.path.join('datasets', person_name)

        if not os.path.isdir(person_dir):
            continue

        # Processa todas as imagens de uma pessoa
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)

            # Carrega a imagem e extrai as características faciais
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)

            # Adiciona as características faciais à lista
            if len(face_encodings) > 0:
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(person_name)

    # Salva as características faciais e os nomes em um arquivo
    with open('models/faces_encodings.pkl', 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)

    print("Model trained and saved.")

if __name__ == "__main__":
    train_model()
