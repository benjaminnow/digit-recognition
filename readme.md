# Neural Network From Scratch

## Intro

This was a personal project for my honors college capstone class. It is the first step towards understanding machine learning from basic principles.

Program functionality:
- NN training
- Saving trained NN weights in binary files
- loading trained weights files
- front-end user interface to try your own handwriting using [gradio](https://gradio.app)

__TIPS__: Recognition works better if you write smaller and not close to the top of the input box.

Link to website: http://tinyurl.com/nndemo

If that link isn't working, use the direct ip: http://132.145.161.120:7860/

## Why

I eventually want to build human activity recognition algorithms and this is the first step towards understanding ML principles. It was an achievable project for the timespan of the class and the time I could dedicate towards it.

## How

I followed a book called [Make Your Own Neural Network](https://www.amazon.com/Make-Your-Own-Neural-Network/dp/1530826608) by Tariq Rashid. This is an amazing introduction to the basics of neural nets. It only assumes a high school level of math and no programming background. Because of that, it was very easy reading for a person like me and highly enjoyable throughout. I genuinely believe that someone who only has a high school math background could come away with a great understanding of NN after reading this book.

The book has two main parts: theory and practice. The theory part goes over how neural networks learn intuitively and then gets into some of the math behind them. The hardest math required is some basic derivative rules, partial differentiation, and some matrix multiplications. The practice section goes over how to convert this theory into code using Python and Numpy mainly. Tariq guides the reader on ways to improve the neural network and really builds the reader's intuition on how they learn and some of their shortcommings.

## What I Added

The neural network code provided in the book allows the user to train the NN on the MNIST Digit dataset and output the accuracy. In the extended section of the book, Tariq went over how someone could make their own test data using other software but it was a cumbersome process. I wanted a way for users to handwrite digits and get the classification in realtime.

Somehow I recently stumbled across gradio, which is a python library that makes it super easy to build nice GUIs for machine learning tasks. The GUIs are hosted on a website so it can work on pretty much any platform.

Once I got gradio running locally, I used the free tier of [Oracle Cloud](https://www.oracle.com/cloud/) (thanks!) to host the gradio website. Oracle is very generous in their always free offering because I was able to get access to a 4 core ARM based server with 18 gb of RAM :)))

### Problem I Ran Into and Solution

The NN that I trained locally had an accuracy of 97.36% on the MNIST Test Digit Dataset, but it was performing terribly in the beginning. My first hunch of the problem was that gradio wasn't doing a good job cropping the image. My NN expects to be fed in a 1x784 vector (28x28 image), and gradio has the ability to scale down images, but it just doesn't do it intelligently. It would crop out parts and sometimes the digit would not be centered.

This was a great learning moment on how fragile NNs are. A slight change in location or an edge being cropped out totally changes the classification. To solve this, I had to implement some smarter cropping and rescaling. Basically, I made a square bounding box around pixels that had values so nothing would be cropped out, and then I rescaled it to 28x28 using OpenCV. You can look at the code for more details.

And it worked! The problem was that the images being input weren't very similar to the MNIST data so the classification was totally off. My cropping is still not the smartest because writing too big or too close to the top of the input box will produce an error. However, I'm really happy that I was able to go through this process of making a hypothesis, testing the hypothesis, and implementing the fix.
