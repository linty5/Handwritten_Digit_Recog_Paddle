import paddle
import module
import train

paddle.vision.set_image_backend('cv2')

model = module.MNIST()

train.train(model)
paddle.save(model.state_dict(), './mnist.pdparams')
