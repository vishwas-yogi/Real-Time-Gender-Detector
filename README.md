# Real-Time-Gender-Detector
Can detect faces and gender in a live stream of web cam or a video.</br>
For gender classification, the model is trained on [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset. </br>
CelebA dataset has more than 200k celebrity images, each with 40 attributes. I used the __'Male'__ atrribute to train the model. </br>
Model used is a simple Convolutional Neural Networks architechure. It yields 98.53% accuracy on test dataset (although the notebook doesn't have test accuracy and loss plots, i lost them as i quit without the draft being saved and rebuilding the whole model will consume a lot of time). </br>
Model yields training accuracy of 95.99% after 20 epochs.</br>

I used **Harr Cascade**, which is a object detection algorithm used for detecting faces in an image or real time video. </br>
Ref: <https://ieeexplore.ieee.org/document/990517> </br>
I used OpenCV for harr cascade and camera capturing.</br>
<h2> Results: </h2>
<h3>Snapshot from my live web cam stream:</h3></br>
</br>
<img src= 'https://github.com/vishwas-yogi/Real-Time-Gender-Detector/blob/main/Images_for_readme/Screenshot%202021-05-27%20123520.png' alt = 'Web Cam Stream'>
</br>
</br>
<h3>Snapshot from a video stream:</h3></br>
</br>
<img src= 'https://github.com/vishwas-yogi/Real-Time-Gender-Detector/blob/main/Images_for_readme/Screenshot%202021-05-27%20124102.png' alt = 'Video Cam Stream'>
</br>
</br>
<h2>How to run script:</h2></br>
</br>
After downloading the python script can be run through terminal using:</br> </br>
python real_time_gender_detection.py </br>
This will run python script and will use the web cam stream to find faces and label them.</br> </br>
python real_time_gender_detection.py --video video_path </br>
This will run python script and will use the video stream to find faces and label them.</br> </br>
python real_time_gender_detection.py --model serialized_model_path </br>
This will use the serialized model provided by the path that will be used with harr cascade (which will just find face and label it).</br></br>
<h2>Future Work:</h2> </br></br>
Will be adding more features that will model deduce from the face like age, expressions etc.

