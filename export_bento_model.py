from bento_model import MnistTensorflow

bento_svc = MnistTensorflow()
bento_svc.pack("trackable", "exported.model/")
saved_path = bento_svc.save()