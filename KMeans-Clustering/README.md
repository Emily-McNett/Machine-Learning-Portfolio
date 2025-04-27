# KMeans Clustering for Yarn Selection

### Overview

##### Machine Learning Technique

KMeans clustering is an unsupervised learning algorithm that partitions a dataset into 'k' clusters. The algorithm begins by randomly selecting 'k' centroids (datapoints at the center of each cluster). Each data point left over is then assigned to the nearest centroid based on the euclidean distance. Centroids are then recalculated and changed to be the mean of all of the datapoints for each cluster. This process of calculation and reassignment is repeated until the position of the centroids no longer changes.


One technique, and the one I used, to find the optimal 'k' is to create a visualization called an elbow graph. In this graph, the 'elbow' refers to the point where an additional number of clusters no longer notably impacts the results. 

##### Program's Purpose

This program is used to assist in the selection of yarn and yarn colors from a photo for use in various fiber arts. After uploading a photo and deciding on the number of dominant colors in an image, the dominant colors of the image, found through KMeans clustering, will be matched to the closest yarn color in the dataset. Users will then be able to compare the ‘true’ colors to the selected yarn.

### Data

This project contains three example photos that I have taken. These photos are included for the following reasons:

(1) [Stout_Blanket](https://github.com/Emily-McNett/Machine-Learning-Portfolio/blob/main/KMeans-Clustering/Stout_Blanket.JPEG)

This photo inspired the creation of this program as I was interested in it enough to take a photograph but, when I returned to the store the blanket was sold out. I became interested in making something similar myself but didn't know where to start when purchasing the correct yarn colors.

(2) [Camping_Photo](https://github.com/Emily-McNett/Machine-Learning-Portfolio/blob/main/KMeans-Clustering/Camping_Photo.JPEG) 

This beautiful photo of my dad standing next to our tents on one of our latest bike-camping trips has a much wider variety of colors. For this image, I am less interested in making a photorealistic tapestry and more interested in capturing the general asthetic and key colors to create a personal and unique item. 

(3) [Nature_Photo](https://github.com/Emily-McNett/Machine-Learning-Portfolio/blob/main/KMeans-Clustering/Nature_Photo.JPEG) 

This image was added as a challenge to myself and the program. There are a lot of similar, earthy colors that can make clustering for dominant colors difficult. Again, however, I am not as interested in making a photorealistic item as I am with capturing the general asthetic of the colors.

This project also contains a dataset of yarn colors by brand, name, and hex code. These colors were pulled from the [Yarn Colorway Finder](https://temperature-blanket.com/). As this program was for more personal purposes, I only pulled yarn from brands that I was familiar with and would be likely to purchase from. However, all of their yarns can be found [here](https://github.com/jdvlpr/Temperature-Blanket-Web-App/tree/main/src/lib/yarns).

### Project Run Throughs 

##### Stout Blanket

<img src="https://github.com/Emily-McNett/Machine-Learning-Portfolio/blob/main/KMeans-Clustering/Stout_Blanket.JPEG" alt="Stout Blanket" width="300"/>

After loading in the Stout_Blanket image, the elbow graph below is displayed. For the purposes of this image, I found that <b>six</b> clusters created an appropriate color palette. From this palette, I was able to avoid the complications that come with shadows impacting RGB values while also gaining greater input on the final color choices.

<img src="https://github.com/user-attachments/assets/d27f99ab-5aa1-43db-ac48-20e193dfa463" alt="Stout Blanket Elbow Graph" width="300"/>

After altering the num_clusters value, the dominant colors are found via the KMeans algorithm. These values are then run through the find_closest_yarn function. This function uses euclidean distance between the clustered RGB value and the dataset RGB values to find the closest matching yarn. Once all closest yarns are calculated the below visualization is displayed to show the clustered colors on top and the matching yarn colors on bottom. Alongside this visualization, the yarn brand, color, and hex code are printed following the order of the visualization from left to right.

<img src="https://github.com/user-attachments/assets/0313c0ba-dab4-4dd3-a4b6-2ad131c2c86a" alt="Stout Blanket Final Yarn Colors" width="300"/>

<img src="https://github.com/user-attachments/assets/a639fcaa-5f1d-4cca-befb-12c71d532308" alt="Stout Blanket Final Yarn Color Names" width="300"/>

From this created palette, I chose the two middle colors of Off White and Denim Heather as the best options for attempting to recreate the initially provided blanket.

##### Camping Photo

<img src="https://github.com/Emily-McNett/Machine-Learning-Portfolio/blob/main/KMeans-Clustering/Camping_Photo.JPEG" alt="Camping Photo" width="300"/>

After loading in the Camping_Photo image, the elbow graph is displayed. From this graph, I decided that <b>six</b> clusters would once again be the best choice in creating a more thorough color palette. 

<img src="https://github.com/user-attachments/assets/a15610e7-e6ed-4c94-85e4-b6b0d72d4f55" alt="Camping Photo Elbow Graph" width="300"/>

Running KMeans and then the find_closest_yarn function produces the results below. 

<img src="https://github.com/user-attachments/assets/cd1580c1-b7d4-433a-be79-bbc459f87b73" alt="Camping Photo Final Yarn Color Names" width="300"/>

<img src="https://github.com/user-attachments/assets/172dc5b9-5e72-42d4-9455-dbc54109cdc7" alt="Camping Photo Final Yarn Colors" width="300"/>

From this created palette, the color choices may seem a bit strange. But, our bright blue tents are iconic to our trips and Bernat's Peacock appears to be an excellent choice for integrating these memories into a crochet project.

##### Nature Photo

<img src="https://github.com/Emily-McNett/Machine-Learning-Portfolio/blob/main/KMeans-Clustering/Nature_Photo.JPEG" alt="Nature Photo" width="300"/>

After loading in the Nature_Photo image, the elbow graph is displayed. From this graph, I decided that <b>ten</b> clusters would bring the variety of earth tones that are required to bring this image into the fiber arts world.

<img src="https://github.com/user-attachments/assets/fc51f53d-4579-42fd-aeb3-8581840e727f" alt="Nature Photo Elbow Graph" width="300"/>

Running KMeans and then the find_closest_yarn function produces the results below. 

<img src="https://github.com/user-attachments/assets/dabf628c-09f3-4733-b42d-d13bbcc5631c" alt="Nature Photo Final Yarn Colors" width="300"/>

<img src="https://github.com/user-attachments/assets/4a22bc37-5b0e-4755-98cf-ea760d4d65de" alt="Nature Photo Final Yarn Color Names" width="100"/>

The created yarn pallete brightens many of the colors. However, I believe including more high-end yarns in the dataset would provide the ability to see closer color options. With this in mind, the provided set is a good base when looking to capture the colors of the image.

### Reflection

KMeans clustering is an efficient way to gather the dominant colors of an image. The method is also quite flexable. An elbow graph helps guide the decision of how many clusters to create, but a greater palette can be created if desired. 

It is important to note that hex/RGB is not a perfect metric for matching the color of yarn. Outside factors such as shadows in an image or dye lots of the yarn can result in projects not being truly 'to image'. However, this program greatly assists in the selection of yarn when perfection is not necessary. In fact, in many fiber arts perfection is discouraged. 
