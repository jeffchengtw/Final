class DefaultConfigs(object):
    train_data = "data/train/" # where is your train data
    test_data = "data/test/"   # your test data
    weights = "./checkpoints/"
    best_models = "./checkpoints/best_models/"
    submit = "./submit/"
    model_name = "effnetv2"
    num_classes = 28
    img_weight = 224
    img_height = 224
    channels = 4
    lr = 0.03
    batch_size = 4
    epochs = 50

config = DefaultConfigs()