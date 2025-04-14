## Prokudin-Gorskii Alignment

**This project aims to create color images using the Prokudin-Gorskii alignment strategy.**

Three images are present per uncolorized image with each one being used as a single color channel. 
Images are aligned using a brute-force approach that minimizes the SSD between channels.

#### Execution
Executing the evalToyAlignment.py file randomly shuffles the color channels of the images in data/sample-images/ then realigns them with the above appraoch.

Executing evalProkudinAlignment.py aligns and as a result colorizes the 'sets' of three images in data/prokudin-gorskii/
