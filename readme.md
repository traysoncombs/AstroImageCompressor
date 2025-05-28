# AstroImageCompressor
This project was an attempt to better compress raw images produced for astrophotography.
When I initially tested the idea on some of my datasets I saw compression ratios that were 
10% better than those achieved by the RAW compression algorithm used by Nikon (I use a Nikon camera so all my images are in Nikon's NEF format).
Seeing the compression ratios, I decided to go ahead and create this project, however, eventually I realized I was incorrectly computing the ratios.
<br/>
<br/>
For the average dataset this will perform worse than your basic RAW compression algorithm. In some cases where the images are very well sampled 
and have minimal noise it can achieve up to 50% compression, but it's highly dependent on the data and is typically closer to 30%.
## How it works
Basically, the idea is to select a base image, subtract each pixel within the image being compressed from the base image, and then store this residual as a variable sized bit field.
There's some fancy logic that goes into denoting the size of each bit field in order to reduce overhead but I won't go through that here.

<br/>
<br/>
The code is very messy and there is currently no way to write decompressed files but if anyone wants to have a go at making this viable it might be possible with aligned/preprocessed images.
