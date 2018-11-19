# TSTutorial
TensorSpace.js tutorial

## Data
First you need to extract ```mnist_train.csv.zip``` in the ```data``` folder.

## Model
to train the just run 
```python main.py```

## Convertion
First you need to install TensorFlow.js with ```pip```:
```pip install tensorflowjs```

Then convert the model with the following command:
```tensorflowjs_converter --input_format=tf_saved_mode --output_node_names=$vs --saved_model_tags=serve Model TensorSpace/mnist/model```

## Run visualization
Run ```TensorSpace/mnist/index.html```
