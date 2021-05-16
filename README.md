# Real-Time-Gender-Detector
Can detect faces and gender in a live stream of web cam or a video.</br>
For gender classification, the model is trained on [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset. </br>
CelebA dataset has more than 200k celebrity images, each with 40 attributes. I used the __'Male'__ atrribute to train the model. </br>
Model used is a simple Convolutional Neural Networks architechure. It yields 98.53% accuracy on test dataset (although the notebook doesn't have test accuracy and loss plots, i lost them as i quit without the draft being saved and rebuilding the whole model will consume a lot of time). </br>
Model yields training accuracy of 95.99% after 20 epochs.</br>

I used **Harr Cascade**, which is a object detection algorithm used for detecting faces in an image or real time video. </br>
Ref: <https://ieeexplore.ieee.org/document/990517> </br>
I used OpenCV for harr cascade and camera capturing.</br>

Snapshot from my live web cam stream:

