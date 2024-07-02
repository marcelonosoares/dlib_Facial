import os
from imutils import face_utils
import dlib
import cv2
import face_recognition
#Esse modelo é capaz de ler fotos na pasta fotos
# Inicializar o detector de faces e o preditor de marcos faciais
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# Capturar vídeo da webcam
cap = cv2.VideoCapture(0)

# Carregar as imagens conhecidas e extrair as codificações faciais
known_face_encodings = []
known_face_names = []

known_faces_dir = "fotos"
for filename in os.listdir(known_faces_dir):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])

while True:
    # Ler imagem da webcam e convertê-la para escala de cinza
    ret, image = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detectar faces na imagem em escala de cinza
    rects = detector(gray, 0)

    # Para cada face detectada, prever marcos faciais
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Desenhar círculos em cada coordenada (x, y) dos marcos faciais
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        # Extraindo a região da face detectada
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face_image = image[y:y+h, x:x+w]
        rgb_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Codificar a face detectada
        face_encodings = face_recognition.face_encodings(rgb_face_image)

        name = "Desconhecido"
        if face_encodings:
            face_encoding = face_encodings[0]
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

        # Exibir o nome da pessoa na imagem
        cv2.putText(image, name, (rect.left(), rect.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.rectangle(image, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)

    # Mostrar a imagem com os pontos de interesse
    cv2.imshow("Output", image)

    # Esperar pela tecla ESC para sair
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

# Liberar os recursos
cv2.destroyAllWindows()
cap.release()
