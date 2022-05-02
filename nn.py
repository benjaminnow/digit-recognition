import numpy
import scipy.special
import scipy.misc
import gradio as gr
from PIL import Image
import cv2

class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        self.lr = learningrate

        # link weight matrices wih - input to hidden, and who - hidden to output
        # elem at index i,j in matrix is weight from node i in current layer to node j in next layer

        # number of rows is equal to number of nodes in next layer and number of columns is
        # equal to number of nodes in current layer
        # needs to be this way because of matrix multiplication where a row of weights is
        # multiplied by current layer outputs in a N by 1 vector

        # mathematicians found weights should be randomly chosen from norm dist with
        # mean of 0 and std of 1/sqrt(num incoming links)
        # last argument specifies that we want a matrix of those dimenstions returned of random vals

        # with standard deviation of the normal dist based on number of incoming links to hidden
        # and output layers
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # our sigmoid activation function as a lambda so can easily be changed elsewhere with
        # only one change here
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        # create col vectors from lists
        # reason for ndmin=2 is because python lists converting to numpy array
        # has undefined second dimension size like (784,) so this makes it
        # something like (784,1)
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_ouputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_ouputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs

        # who used to be dotting from hidden to outputs so cols had to match hidden and rows had to match output
        # now going backwards so transpose who to get errors for hidden layer
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update weights for links between hidden and output layers
        # hidden outputs becomes row of outputs from column of outputs
        # * operator is elementwise multiplication
        # this is dotting a modified output layer column vector with a hidden layer row
        # this creates a (output layer)x(hidden layer) matrix because each node in the output
        # layer is connected to each node in the hidden layer
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)), numpy.transpose(hidden_ouputs))

        # update weights for links between input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_ouputs * (1 - hidden_ouputs)), numpy.transpose(inputs))

    # takes input to NN and gives us back an output after running thru NN
    def query(self, inputs_list):
        # inputs_list needs to be converted to 2 dimensional array
        # T transposes array, so becomes column vector
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_ouputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_ouputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    # if already have pretrained weights and want to load these in so don't have
    # to go through time-consuming training
    def load(self):
        wih_bin_path = "savedNN/wih"
        who_bin_path = "savedNN/who"

        wih_bin_file = open(wih_bin_path, "rb")
        self.wih = numpy.load(wih_bin_file)
        wih_bin_file.close()

        who_bin_file = open(who_bin_path, "rb")
        self.who = numpy.load(who_bin_file)
        who_bin_file.close()


# setting dimensions for NN
input_nodes = 784 # 784 because there is a node for each pixel
# somewhat arbitrary amount, but since 200 < 784 NN has to summarize key features
# because the same data is trying to be expressed in shorter form
# if hidden nodes is too small, not enough to find key features so have to strike
# a balance
# found that 200 hidden nodes yielded best accuracy before diminishing returns
hidden_nodes = 400
output_nodes = 10 # the classification is 0-9 so 10 nodes output

# setting learning rate, should be some value < 1
learning_rate = 0.01

# creating NN object
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# csv is formatted as a row of data
# 1st number is number represented by image
# following 784 numbers are grayscale values (0-255) of each pixel in 28x28 image

# go through all the records in the training set and convert them to arrays
# each record is a line in the csv file
# then train the neural network

def trainNN(train_file, epochs):
    # loading training data file
    training_data_file = open(train_file, 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # can get better performance if we train on whole dataset mutliple times
    # each time is considered an "epoch"
    for e in range(epochs):
        for record in training_data_list:
            # split on commas since csv file
            all_values = record.split(',')

            # rescaling input color values from 0-255 to 0.01-1.00
            # since we don't want extreme values in sigmoid function (towards the tails)
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

            # create target output values, a 10x1 vector of what the output layer should
            # look like
            # since using sigmoid function can't get 0.00 or 1.00, so for target number
            # 0.99 is target and everything else is 0.01
            targets = numpy.zeros(output_nodes) + 0.01
            # all_values[0] is the target label for this record
            targets[int(all_values[0])] = 0.99

            # train the NN on each record!!
            n.train(inputs, targets)

def testNN(test_file):
    # loading test data from csv file
    test_data_file = open(test_file, 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # list of correct/incorrect classifications, just to give some kind of visual
    scorecard = []

    # test all the records in test data
    for record in test_data_list:
        all_values = record.split(',')
        # correct label stored at first index on csv row
        correct_label = int(all_values[0])
        # scale and shift inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # see what the NN output as answer
        outputs = n.query(inputs)

        # get prediction of NN
        # the "prediction" will be whatever index(node) had the highest output value
        # if an example is ambiguous several nodes could have a decently high output
        # but just choose highest
        label = numpy.argmax(outputs)
        # update scorecard on incorrect/correct classification
        scorecard.append(1) if label == correct_label else scorecard.append(0)

    # calculate performance (accuracy) of NN
    scorecard_array = numpy.asarray(scorecard)
    accuracy = scorecard_array.sum() / scorecard_array.size
    print("performance = {}".format(accuracy))

    # saving weights and lr of NN
    saveNN(n, accuracy)

# save the weights of the input to hidden and hidden to output layers so don't
# have to rerun training every time
# NN weights and lr saved as binary files
def saveNN(nn, accuracy):
    wih_bin_path = "savedNN/wih"
    who_bin_path = "savedNN/who"
    
    wih_bin = open(wih_bin_path, "wb")
    numpy.save(wih_bin, nn.wih)
    wih_bin.close()

    who_bin = open(who_bin_path, "wb")
    numpy.save(who_bin, nn.who)
    who_bin.close()

# return centered and scaled image array for digit handwritten input
def processImage(img):
    # 1) get col to col and row to row where image is filled in with black
    non_empty_columns = numpy.where(img.max(axis=0)>0)[0]
    non_empty_rows = numpy.where(img.max(axis=1)>0)[0]
    # 2) create a crop box based on last row/col img filled in
    cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
    # 3) make crop box square
    crop_width = max(non_empty_columns) - min(non_empty_columns)
    crop_height = max(non_empty_rows) - min(non_empty_rows)

    # padding for once image square for dimension that wasn't adjusted
    # NOTE PADDING 100 works well but have to make sure not to draw too BIG
    # otherwise error
    padding = 100
    # image taller than wide
    if crop_height > crop_width:
        diff = crop_height - crop_width
        # if diff not even, make even
        if diff % 2 != 0:
            diff += 1
        # add difference / 2 to width of image to make square
        # then add some padding to top and bottom, but add to width to keep square
        cropBox = (int(cropBox[0] - padding), 
                   int(cropBox[1] + padding), 
                   int(cropBox[2] - diff / 2.0 - padding), 
                   int(cropBox[3] + diff / 2.0 + padding))
    else: # image wider than tall
        diff = crop_width - crop_height
        if diff % 2 != 0:
            diff += 1
        # add difference / 2 to height of image to make square
        # also add padding
        cropBox = (int(cropBox[0] - diff / 2.0 - padding), 
                   int(cropBox[1] + diff / 2.0 + padding), 
                   int(cropBox[2] - padding), 
                   int(cropBox[3] + padding))

    # 4) resize to 28x28
    image_data_new = img[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1]
    image_data_scaled = cv2.resize(image_data_new, dsize=(28,28), interpolation=cv2.INTER_CUBIC)

    # testing to see what the image looked like and saving it for output
    new_image = Image.fromarray(image_data_scaled)
    new_image.save("digit_cropped.jpeg")
    
    return image_data_scaled
    

def predict(img):
    # img comes in and has to be processed to be 28x28
    # im = Image.fromarray(img)
    # im.save("test.jpeg")
    img = processImage(img)

    img = img.reshape(784).flatten()
    inputs = (img / 255.0 * 0.99) + 0.01
    # 10 elem array with probabilities input was number 0-9
    outputs = n.query(inputs)
    print(outputs)
    
    # get top 5 predictions
    # get their indices, aka the actual number prediction
    top_5_pred = numpy.argsort(outputs, axis=0)[-5:].flatten().tolist()
    print(top_5_pred)
    top_5_confidence = outputs[top_5_pred].flatten().tolist()

    # return dictionary where keys are predicted labels and values are
    # confidences, also return the cropped digit image so user can see what
    # they drew and what the NN had to look at

    # each return value is used for their respective output component in gradio
    prediction_dict = {label: conf for label, conf in zip(top_5_pred, top_5_confidence)}
    return prediction_dict, "digit_cropped.jpeg"


# commented out because already saved NN values in binary files
# accuracy of saved files is 0.9736
# trainNN("data/mnist_train.csv", epochs=10)

# load trained weights into NN
n.load()
# testNN("data/mnist_test.csv")

sketchpad_input = gr.inputs.Image(
    shape=None,
    image_mode="L",
    invert_colors=True,
    source="canvas",
    tool="select"
)

# output the digit the user wrote but scaled and what the NN would see
digit_image_output = gr.outputs.Image(
    type="file"
)

gr.Interface(fn=predict, 
             inputs=sketchpad_input,
             outputs=["label", digit_image_output],
             live=True, 
             ).launch(server_name="0.0.0.0")






