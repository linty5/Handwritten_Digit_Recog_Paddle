import paddle
import paddle.nn.functional as F
import utils
import dataset

def train(model):
    model.train()
    train_loader = dataset.train_dataset()
    opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
    EPOCH_NUM = 10
    for epoch in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            images = utils.norm_img(data[0]).astype('float32')
            labels = data[1].astype('float32')

            predicts = model(images)

            loss = F.square_error_cost(predicts, labels)
            avg_loss = paddle.mean(loss)

            if batch_id % 1000 == 0:
                print("epoch_id: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, avg_loss.numpy()))

            avg_loss.backward()
            opt.step()
            opt.clear_grad()
