import paddle

def norm_img(img):
    assert len(img.shape) == 3
    batch_size, img_h, img_w = img.shape[0], img.shape[1], img.shape[2]
    img = img / 255
    img = paddle.reshape(img, [batch_size, img_h * img_w])

    return img