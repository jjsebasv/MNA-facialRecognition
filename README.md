## MNA Face Recognition

Se debe tener Python 3 instalado.

Para instalar las dependencias de Python:

    $ pip3 install -r requirements.txt

Para ejecutar el proyecto:

    $ ./main.py

Argumentos disponibles:

    $ ./main.py -h
    usage: main.py [-h] [--method {pca,kpca}] [--imgdb {att_images,images}]
                   [--subjects SUBJECTS] [--img-per-subj IMG_PER_SUBJECT]
                   [--test-img-per-subj TEST_IMG_PER_SUBJECT] [--query QUERY] [-l]
                   [--webcams] [--webcam WEBCAM]

    optional arguments:
      -h, --help            show this help message and exit
      --method {pca,kpca}   Method used to perform the face recognition
      --imgdb {att_images,images}
                            Image database
      --subjects SUBJECTS   Number of subjects
      --img-per-subj IMG_PER_SUBJECT
                            Number of training images per subject
      --test-img-per-subj TEST_IMG_PER_SUBJECT
                            Number of test images per subject
      --query QUERY         Path of file to query
      -l, --live            Live subject recognition
      --webcams             Query video devices
      --webcam WEBCAM       Video device

## Uso de la webcam

Para usar la webcam, se debe primero consultar los dispositivos de captura disponibles.
Ello se puede hacer ejecutando:

    $ ./main.py --webcams
    Device found: /dev/video0

Luego, sabemos que el dispositivo `/dev/video0` puede ser usado. Procedemos a ejecutar:

    $ ./main.py --live --webcam=/dev/video0

Por defecto, el proyecto utilizara el dispositivo `/dev/video0`, por lo que no es necesario especificarlo. En caso de que su ruta sea diferente, será un requisito pasarlo como argumento.

## Papers

- http://ijarcet.org/wp-content/uploads/IJARCET-VOL-1-ISSUE-9-135-139.pdf
- https://www.bytefish.de/pdf/facerec_python.pdf
