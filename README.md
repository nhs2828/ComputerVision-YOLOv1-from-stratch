Hello, this is my implementation of YOLOv1 algorithm [[ref]](https://arxiv.org/abs/1506.02640) using Tensorflow.

### Dataset
I used [Traffic Signs Dataset](https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-dataset-in-yolo-format/data), which contains images of trafic signs and theirs annotations in YOLO format.

### Algorithm YOLO - You only look once
The ability to predict all objects in a single forward is the reason why it has this name `You only look once`. We divide the input images as grid of $\text{Cell} \times \text{Cell}$. Each cell will take responsibility to predict objects. The limitation is each cell can only predict 1 object of a time. Another important thing is the annotation of coordinates must be relative to the cell contains that object (but not weight and height since the box can be bigger than the cell).

![Capture d’écran 2023-10-11 à 11 10 50](https://github.com/nhs2828/ComputerVision-YOLOv1-from-stratch/assets/78078713/df9a3592-d3b0-45a0-9e27-01cf76fe91d8)



### Result
The prediction on train dataset is quite good

![Capture d’écran 2023-10-11 à 10 11 37](https://github.com/nhs2828/ComputerVision-YOLOv1-from-stratch/assets/78078713/f8774513-7e71-4e4a-9317-b01e76627d79)
![Capture d’écran 2023-10-11 à 10 12 47](https://github.com/nhs2828/ComputerVision-YOLOv1-from-stratch/assets/78078713/60bcb062-c754-42db-909c-325df5281d62)
![Capture d’écran 2023-10-11 à 10 19 03](https://github.com/nhs2828/ComputerVision-YOLOv1-from-stratch/assets/78078713/3a0bdf09-8c24-4b9b-8f75-cc6dd9e8fed7)
![Capture d’écran 2023-10-11 à 11 28 57](https://github.com/nhs2828/ComputerVision-YOLOv1-from-stratch/assets/78078713/c5607055-be0f-4a46-b1fb-5509b00fb7db)

On the other hand, the prediction on test dataset is not so perfect

![Capture d’écran 2023-10-11 à 11 08 21](https://github.com/nhs2828/ComputerVision-YOLOv1-from-stratch/assets/78078713/ea40b7a1-aa9c-4809-ac51-eb2fb585609b)
<img width="334" alt="Capture d’écran 2023-10-12 à 20 13 32" src="https://github.com/nhs2828/ComputerVision-YOLOv1-from-stratch/assets/78078713/a48ca711-7be2-4061-9dbb-ca15129e4e88">
