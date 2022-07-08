# Seam carving

Content-aware image scaling as described by [[1]](https://doi.org/10.1145%2F1275808.1276390).

Usage example: `python carve.py broadway_tower.jpg output/broadway_tower --iteration_count=300`  

To see the carving animated, use the `make_gif.py` script.  
Usage example: `python make_gif.py output/broadway_tower`

Original image:  
![before](broadway_tower.jpg)

Energy function (e.g. gradient magnitude):  
![energy](broadway_tower_energy.png)

Lowest energy seam (e.g. using Dijkstra's algo):  
![during](broadway_tower_seamed.png)

Shrunk image (297 iterations):  
![shrunk](broadway_tower_shrunk.png)

## More examples

width=3440px  
![spring_lake](spring_lake.jpg)

width=2440px  
![spring_lake_small](spring_lake_small.png)

width=3440px  
![snowy_mountains_big](snowy_mountains.jpg)

width=2440px  
![snowy_mountains_small](snowy_mountains_small.png)

width=1440px  
![snowy_mountains_smaller](snowy_mountains_smaller.png)

Supported energy functions:

* gradient magnitude
* spectral residual saliency
* fine grained saliency
* spectral saliency + gradient magnitude
* entropy (3-channel or grayscale)

Entropy tends to be the slowest, spectral saliency is the fastest. Gradient usually perform the best.
Due to the overhead in conversion, 3-channel entropy is faster than grayscale.

\[1] Avidan, Shai; Shamir, Ariel (July 2007). "Seam carving for content-aware image resizing | ACM SIGGRAPH 2007 papers". Siggraph 2007: 10. doi:10.1145/1275808.1276390
