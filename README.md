Hello, this is my implementation of YOLOv1 algorithm [[ref]](https://arxiv.org/abs/1506.02640) using Tensorflow.

### Dataset
I used [Traffic Signs Dataset](https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-dataset-in-yolo-format/data), which contains images of trafic signs and theirs annotations in YOLO format.

### Algorithm YOLO - You only look once
The ability to predict all objects in a single forward is the reason why it has this name `You only look once`. The limitation is each cell can only predict 1 object of a time.
