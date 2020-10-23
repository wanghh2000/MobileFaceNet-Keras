from Model_Structures.MobileFaceNet import mobile_face_net
from Model_Structures.MobileFaceNet import mobile_face_net_train


def printModel():
    model = mobile_face_net()
    model.summary()


def printTrainModel():
    model = mobile_face_net_train(num_labels=67960, loss='softmax')
    model.summary()


if __name__ == '__main__':
    # printModel()
    printTrainModel()
