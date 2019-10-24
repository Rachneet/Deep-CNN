from torch.utils.data import DataLoader
from dataloader import *
from dpcnn_model import *
import bcolz
import torch.nn as nn
from torch.autograd import Variable
from evaluation import *
import pickle
torch.cuda.set_device(0)


def train(batch_size, num_epochs, learning_rate):

    #path = datapath
    training_params = {"batch_size": batch_size,
                       "shuffle": True,
                       "num_workers": 0}
    #test_params = {"batch_size": batch_size,
    #              "shuffle": False,
    #              "num_workers": 0}

    validation_params = {"batch_size": batch_size,
                         "shuffle": False,
                         "num_workers": 0}

    embedding_dim = 300

    training_set = MyDataset("train.csv")
    training_gen = DataLoader(training_set, **training_params)

    validation_set = MyDataset("validation.csv")
    validation_gen = DataLoader(validation_set, **validation_params)


    # target_vocab = training_set.vocabulary
    # target_vocab_len = len(training_set.vocabulary)
    # print(target_vocab_len)
    #
    # vectors = bcolz.open('fastext_vec.dat')[:]
    # words = pickle.load(open('ft_words.pkl', 'rb'))
    # word2idx = pickle.load(open('ft_w2idx.pkl', 'rb'))
    #
    # fastext = {w: vectors[word2idx[w]] for w in words}
    #
    # matrix_len = len(target_vocab)
    # weights_matrix = np.zeros((matrix_len, embedding_dim))
    # words_found = 0
    #
    # for i, word in enumerate(target_vocab):
    #     try:
    #         weights_matrix[i] = fastext[word]
    #         words_found += 1
    #     except KeyError:
    #         weights_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim, ))
    #
    # print(words_found)
    # print(weights_matrix.shape)
    # print(weights_matrix.size)
    # print(type(weights_matrix))
    # pickle.dump(open('embedding.pkl','w'),weights_matrix)

    # emb_layer = nn.Embedding(target_vocab_len, embedding_dim)
    weights_matrix = np.load('embedding.npy', 'r')
    #print(weights_matrix)

    model = DPCNN(weights_matrix=weights_matrix)

    model.cuda()
    # emb_layer.load_state_dict({'weight': weights_matrix})
    # emb_layer.cuda()


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate, momentum=0.9)

    model.train()

    best_accuracy =0
    num_iter_per_epoch = len(training_gen)
    output_file= open('logs_dpcnn.txt','w')

    for epoch in range(num_epochs):

        for iter, batch in enumerate(training_gen):
            true_label=[]

            _,n_true_label= batch
            batch= [Variable(record).cuda() for record in batch]
            t_data, t_label = batch

            prediction = model(t_data)

            n_prob_label = prediction.cpu().data.numpy()
            true_label.extend(n_true_label)

            loss = criterion(prediction,t_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print(n_true_label,n_prob_label)

            training_metrics = get_evaluation(np.array(true_label),n_prob_label,list_metrics =["accuracy"])

            #print(np.array(true_label),n_prob_label)
            print("Training Iteration: {}/{} Epoch: {}/{} Loss: {} Accuracy: {}".format(iter+1,num_iter_per_epoch,
                                                                                        epoch+1, num_epochs,
                                                                                        loss.item(),
                                                                                        training_metrics['accuracy']))

       	model.eval()

        with torch.no_grad():

            validation_true=[]
            validation_prob=[]

            for _, batch in enumerate(validation_gen):
                _,n_true_label=batch
                batch = [Variable(record).cuda() for record in batch]
                t_data, t_label = batch

                prediction=model(t_data)
                validation_prob.append(prediction)
                validation_true.extend(n_true_label)

            loss_val = criterion(prediction,t_label)
            validation_prob= torch.cat(validation_prob, 0)
            validation_prob= validation_prob.cpu().data.numpy()
            validation_true = np.array(validation_true)
            

        model.train()

        test_metrics = get_evaluation(validation_true, validation_prob,
                                         list_metrics=["accuracy", "confusion_matrix"])

        output_file.write(
                "Epoch: {}/{} \nTraining loss: {} Training accuracy: {} \nTest loss: {} Test accuracy: {} \nTest confusion matrix: \n{}\n\n".format(
                    epoch + 1, num_epochs,
                    loss.item(),
                    training_metrics["accuracy"],
                    loss_val.item(),
                    test_metrics["accuracy"],
                    test_metrics["confusion_matrix"]))
        print("\tTest:Epoch: {}/{} Loss: {} Accuracy: {}\r".format(epoch + 1, num_epochs, loss_val.item(),
                                                                       test_metrics["accuracy"]))

        if (num_epochs > 0 and num_epochs % 3 == 0):
                learning_rate = learning_rate / 2

            # saving the model with best accuracy
        if test_metrics["accuracy"] > best_accuracy:
                best_accuracy = test_metrics["accuracy"]
                torch.save(model,"trained_model_dpcnn")



if __name__ == "__main__" :
    #dp = ''
    train(5,5,0.01)
