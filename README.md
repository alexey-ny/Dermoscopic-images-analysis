# Dermoscopic images analysis
Here I attempt to analyze quality/acceptability of the images, so the PCP could retake an image immediately, thus reducing repeat patients' visits.

This is work in progress, therefore this repository contains a set of notebooks reflecting the progress.

There are a few common problems possible: 
1) blurriness - caused by out of focus, or movement
2) overexposure/underexposure - wrong exposure setting, error with skin darkness/lightness
3) color/hue shift  - error in hardware color calibration, wrong exposure setting
4) digital/sensor noise

The most popular approach to determine image blurriness is based on variance of laplacian operator over gray version of the image. In essence it allows to estimate how many edges there are in the images. So, the bigger the number the sharper the image - I'll call it Sharpness Index here. In general, this approach works pretty well for regular everyday images. One just needs to define a suitable threshold to make a decision if the image sharp or not. <br><br>
However with close-up images of the skin we have to deal with very smooth, mostly even surface leading to very low variance of the laplacian. In the <a href='https://github.com/alexey-ny/Desrmascopic-images-analysis/blob/main/eda-blur-rgb.ipynb'>first notebook</a> I analyze this approach and show that we can't rely on it, since there is no way to set a reasonable threshold for the decision-making.

In the <a href='https://github.com/alexey-ny/Desrmascopic-images-analysis/blob/main/eda-blur-rgb-hsv-ycrcb.ipynb'>second notebook</a> I introduce analysis of the image per each channel in 3 color spaces. Computing Sharpness Index for each channel gives us more features for each image, providing additional information for more confident conclusion on the image sharpness. 
Also I compute full contrast and clipped contrast (to remove over/underexposed spots, 1% outliers), and percentiles of intensities in each color space. This gives us data to decide if the image is suitable in terms of the exposure.

In the <a href='https://github.com/alexey-ny/Desrmascopic-images-analysis/blob/main/sharpness-contrast-intensities-mi-ie-maps.ipynb'>third notebook</a> I included  Melanin Index and Erythema Index ([1]). I believe we can use it as for image quality analysis and so for our future lesion classifier, either just by deriving features for deep net classifier or using as extra channels for CNN.

References:<br>
1) Face Detection by Color and Multilayer Feedforward Neural Network by Chiunhsiun Lin (Proceedings of the 2005 IEEE International Conference on Information Acquisition - June 27 - July 3, 2005, Hong Kong and Macau, China)<br>
2) Skin Detection - a Short Tutorial by Ahmed Elgammal, Crystal Muang and Dunxu Hu, Department of Computer Science, Rutgers University, Piscataway, NJ, 08902, USA<br>
3) Improving Dermoscopy Image Classification Using Color Constancy - Catarina Barata, M. Emre Celebi, Senior Member, IEEE, and Jorge S. Marques <br>
[1]: <https://rdcu.be/c3IJD><br>
[2]: <https://people.cs.rutgers.edu/~elgammal/pub/skin.pdf><br>
[3]: <https://faculty.uca.edu/ecelebi/documents/JBHI_2015.pdf><br>
