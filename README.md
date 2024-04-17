# Car Licence Number Plate Detection

<h2>ABSTRACT</h2>
<p>Automatic License Plate Recognition is essential for a variety of Intelligent Transportation System applications, ranging from access control to traffic monitoring.
Most existing systems are limited to a single setup (e.g., toll control) or a single license plate (LP) region. This project proposes a framework combining a fully convolutional network with koras for license plate recognition. The fully convolutional network is proposed for random-positioned object detection by the fusion of multi-scale and hierarchical features.A pre-trained Haar cascade classifier is employed to recognize an Indian license plate. 
For character recognition, removing contours that are surrounding to recognize letters and digits, respectively.For character segmentation we have used open CV’s morphological image processing operations.The compatibility and generality can be expected by applying the proposed method to Indian license plate.

</p>

<h2>OBJECTIVES</h2>
<p>
  Taking in the image of car and  detect an object(license plate) from an image 
Analyzing and performing some image processing on the License plate
Segmenting the alphanumeric characters from the license plate
Extracting the characters one by one, recognizing the characters, concatenating the results and giving out the plate number as a string
After having all the characters, passing the characters one by one into our trained model, and it should recognize the characters.
Then a building a web app using HTML, CSS, Java Script and Flask.
After uploading car image on the website, the car details of respective owners will be displayed on the website.
</p>

<h2>MODULES DESCRIPTION</h2>
<p>

It has significant effect on the LPR system. 
Traditional methods mainly depend on features such as rectangle, color, texture, morphological operations, and edge.
 Firstly, we introduce a cascaded classifier that uses the AdaBoost algorithm to detect a license plate accurately. 
Haar Cascade classifiers are an effective way for object detection. 
Haar Cascade is a machine learning-based approach where a lot of positive and negative images are used to train the classifier.
 We used an AdaBoost algorithm of selecting simple rectangle features to find license plate candidate regions. 


</p>
<h2>Character Segmentation </h2>
<p> 
Now we have to segment our plate number. The input is the image of the plate, we will have to be able to extract the uni character images. 
The result of this step, being used as input to the recognition phase, is of great importance. 
In a system of automatic reading of number plates. Segmentation is one of the most important processes for the automatic identification of license plates, because any other step is based on it. If the segmentation fails, recognition phase will not be correct.
 To ensure proper segmentation, preliminary processing will have to be performed. we should be ready to extract the characters from the plate, this can be done by thresholding, eroding, dilating and blurring the image skillfully such that at the end the image we have is almost noise-free and easy for further functions to work on.
 We now again use contour detection and some parameter tuning to extract the characters. 

</p>
<h2>Character Recognition </h2>
<p>
  

LPCR aims to recognize the license plate numbers. 
Using the extracted features for recognition is a mainstream method for LPCR. 
The method proposed extracted the salient features of training images to establish a feature database, and compared the input character’s salient feature with the database to determine its label. 
The algorithm used CNN for feature extraction and adopted a kernel-based ELM classifier for character classification. In addition, many recognition algorithms based on extracted features can be applied to LPCR, such as AlexNet.

</p>
<h2>  Creating a model</h2>
<p>

For modeling, we will be using a Convolutional Neural Network with AlexNet architecture
Containing 5 convolutional layers, 3 fully connected layers, 1 softmax layer.
To keep the model simple, we’ll start by creating a sequential object.
The first layer will be a convolutional layer with 32 output filters, a convolution window of size (5,5), and ‘Relu’ as activation function.
Next, we’ll be adding a max-pooling layer with a window size of (2,2). For reducing its dimensionality and allowing for assumptions to be made about features contained in the sub-regions binned.
Now, we will be adding some dropout rate to take care of overfitting. They are “dropped-out” randomly. We have chosen a dropout rate of 0.4 meaning 60% of the node will be retained.
Finally, we will be adding 2 dense layers, one with the dimensionality of the output space as 128, activation function='ReLU' and other, our final layer with 36 outputs for categorizing the 26 alphabets (A-Z) + 10 digits (0–9) and activation function= 'softmax'. 
</p>

<h2>Training our CNN model</h2>
<p>
The data we will be using contains images of alphabets (A-Z) and digits (0–9) of size 28x28, also the data is balanced so we won’t have to do any kind of data tuning here.
We created a folder that contains data as per the directory structure below, with a train test split of 80:20
We’ll be using ImageDataGenerator class available in keras to generate some more data using image augmentation techniques like width shift, height shift. To know more about ImageDataGenerator, please check out this nice blog.
It’s time to train our model now! we will use ‘categorical_crossentropy’ as loss function, ‘Adam’ as optimization function and ‘Accuracy’ as our error matrix.
After training for 16 epochs, the model achieved an accuracy of 98.14%.

</p>

<h2>Deploy Machine Learning Models Using Flask</h2>

<p>
  Installing Flask on our Machine
Using our Machine Learning Model  and integrating it with Flask
Creating Templates for website using Html,css
Connect the Webpage with the Model
We have successfully started the Flask server! Open the browser and go to this address – http://127.0.0.1:1234/.

</p>

<h2>Result</h2>
