import paddle
import module
import dataio

model = module.MNIST()
params_file_path = 'mnist.pdparams'
img_path = './0.jpg'
param_dict = paddle.load(params_file_path)
model.load_dict(param_dict)
model.eval()
tensor_img = dataio.load_image(img_path)
result = model(paddle.to_tensor(tensor_img))
print('result',result)
print("本次预测的数字是", result.numpy().astype('int32'))
