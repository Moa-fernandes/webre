import cv2
import os

def capture_images(name):
    # Define o diretório para salvar as imagens
    images_dir = f'datasets/{name}'
    os.makedirs(images_dir, exist_ok=True)
    
    # Inicializa a webcam
    cap = cv2.VideoCapture(0)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Exibe o frame capturado
        cv2.imshow('Capturing Images - Press q to Quit', frame)

        # Salva as imagens no diretório especificado
        img_path = os.path.join(images_dir, f'{name}_{count}.jpg')
        cv2.imwrite(img_path, frame)
        count += 1
        
        # Pressione 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 20:  # Captura 20 imagens
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f'{count} images saved to {images_dir}')

if __name__ == "__main__":
    # Insira o nome da pessoa a ser capturada
    person_name = input("Enter the name of the person to capture images: ")
    capture_images(person_name)
