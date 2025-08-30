# detect-handwritten-digits
Recognising handwritten digits through KNN.

Features
- Uses OpenCV’s built-in `digits.png` dataset for training (5,000 samples).
- KNN classifier for digit recognition.
- Preprocessing: grayscale conversion, Gaussian blur, Canny edge detection.
- Contour-based digit segmentation.
- Draws bounding boxes and predicted digits on input images.

Results
- Accuracy on the `digits.png` dataset (70/30 split): **93.47%**
- On a custom test image (`num_test.jpeg`), the actual input was **2345** but the model predicted **2315** (misclassified the '4').
