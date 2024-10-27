
# Visión por Computador: Práctica 4. Detección y Reconocimiento de Objetos y Matrículas en Video

### Autores

- [@Mauro Gómez Guillén](https://github.com/MGGdesigns)
- [@Santiago Santana Martínez](https://github.com/Tiago1615)

## Tabla de Contenidos

- [Paquetes necesarios](#paquetes-necesarios)
- [Introducción](#introducción)
- [Tareas](#tareas)
  - [Tarea 1](#tarea-1)
    - [Entrenamiento del modelo](#entrenamiento-del-modelo)
    - [Procedimiento](#procedimiento)
    - [Extras](#extras)
- [Referencias Bibliográficas](#referencias-bibliográficas)

---

## Paquetes necesarios

- **Librerías**:
  - `opencv-python`
  - `math`
  - `ultralytics`
  - `collections`
  - `easyocr`
  - `re`
  - `csv`

```bash
pip install opencv-python
```

```bash
pip install ultralytics
```

```bash
pip install easyocr
```

## Introducción:
Este proyecto utiliza un modelo YOLO personalizado y EasyOCR para detectar y reconocer objetos como personas, vehículos, motocicletas y matrículas en un video dado. Los objetos detectados son rastreados en cuadros sucesivos, y las matrículas se procesan con OCR para obtener texto legible. Además, los resultados son exportados a un archivo CSV y el video procesado se guarda con los datos superpuestos.

---

## Tareas

### Tarea 1
Desarrollar un prototipo que procese uno o varios vídeos (incluyendo vídeos de cosecha propia), detectando las personas y vehículos presentes, las matrículas de dichos vehículos, que cuente el total de cada clase presente y que finalmente vuelque los resultados obtenidos a un archivo de vídeo y a un csv.

#### Entrenamiento del modelo

- El modelo ha sido entrenado con uno de los datasets de matrículas que se pueden encontrar en la web.
- Para llevar a cabo la anotación de las matrículas se ha empleado la herramienta [RoboFlow](https://roboflow.com/).
- La división entre entrenamiento, validación y tests ha sido realizada también con RoboFlow, quienes realizan esta división de forma automática, una vez finalizado el proceso de anotación. Esta división se puede ver en la imagen siguiente.

![image](https://github.com/user-attachments/assets/2c4f68e5-1f00-42a4-836f-76e4c3f84655)

- Una vez preparado el conjunto de datos y colocado en la carpeta *datasets*, empezamos el proceso de entrenamiento con el fragmento de código siguiente:

```py
model_car_plates = YOLO("yolo11n.pt")

#Entrenar el modelo
results = model_car_plates.train(data="datasets/data.yaml", epochs=40, imgsz=416, batch=4, device="cpu")

results = model_car_plates.val()
```

![image](https://github.com/user-attachments/assets/22094148-8c26-4990-bd4a-37f35921e1b1)

#### Procedimiento

Para cada fotograma, el código realiza las siguientes operaciones:

- Detección de Objetos
  - Los objetos se detectan en el fotograma actual utilizando el modelo por defecto. Para captar personas, coches, motos y guaguas.
  - Las coordenadas, clase y nivel de confianza de cada objeto detectado se obtienen y se anotan.
- Contar Objetos por Clase
  - Si el objeto detectado no ha sido contado antes, se agrega su ID a un conjunto para esa clase y se incrementa el contador total.
- Detección de Matrículas
  - Para los objetos de clase *car*, *motorbike* o *bus*, se utiliza el entrenado anteriormente para identificar matrículas en el área de interés ampliada.
  - Si se detecta una matrícula, se extraen las coordenadas de la placa con la mayor confianza.
- Reconocimiento de Matrículas (OCR)
  - Si la confianza de la detección de la matrícula supera un umbral mínimo, se aplica un preprocesamiento a la imagen de la matrícula para mejorar la precisión de OCR.
  - EasyOCR detecta el texto en la imagen de la placa y elimina caracteres no deseados.
  - La matrícula reconocida se escribe en el archivo CSV y se muestra en el fotograma del video.
- Anotación de Detecciones
  - Se dibujan cuadros delimitadores alrededor de cada objeto detectado y las matrículas reconocidas, incluyendo el ID de rastreo y la confianza.
  - Se muestra el conteo total de objetos detectados por clase en la parte superior del video.
- Recogida de los resultados
  - Cada fotograma procesado se escribe en el archivo de video de salida, se muestra en pantalla y se vuelcan sobre el archivo csv.

A continuación se muestra el enlace al vídeo obtenido con los resultados y una captura del archivo csv:

[Enlace al vídeo](https://youtu.be/oXd5fMVbrEc)

![image](https://github.com/user-attachments/assets/28cccadd-0fb0-4f08-b7b7-3a2e14b24213)

#### Extras

El procedimiento es el mismo de antes, pero ahora añaden los siguientes pasos:

- Detección de Personas y Pixelado
  - En cada cuadro, verifica si se detectan matrículas en los objetos de interés (automóviles, motocicletas, guaguas).
  - Si se detecta una matrícula con una confianza mínima, realiza un pixelado en esa región y registra el nivel de confianza.
- Detección de Matrículas y Pixelado
  - En cada cuadro, verifica si se detectan personas.
  - Si se detecta una persona, se calcula aproximadamente la región que ocuparía la cara y se aplica un pixelado.

A continuación se muestra el enlace al vídeo obtenido con los resultados y una captura del archivo csv:

[Enlace al vídeo](https://www.youtube.com/watch?v=04rSw_K7VE8)

![image](https://github.com/user-attachments/assets/f40f99a4-71cd-43c7-9ffc-6681b39743d9)

---

## Referencias Bibliográficas

- [Guión de la práctica](https://github.com/otsedom/otsedom.github.io/tree/main/VC/P4)
- [Documentación OpenCV](https://docs.opencv.org/4.x/)
- [Documentación YOLO](https://docs.ultralytics.com/)
- [Documentación EasyOCR](https://www.jaided.ai/easyocr/documentation/)

---
