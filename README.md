# tensorflow_practice

## 1. Exporting Model

- Export model

```
$ CHECK_POINT_PATH=mnist/model/mnist.dnn
$ EXPORT_MODEL_PATH=exported.model

$ python export_model.py \
     --input-checkpoint ${CHECK_POINT_PATH} \
     --saved-model-path ${EXPORT_MODEL_PATH}
```

- Infer by exported model

```
$ python infer_export_model.py
```