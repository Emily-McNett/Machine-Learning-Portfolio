from sklearn.cluster import KMeans
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt

# Finds the Closest Yarn via RGB Euclidean
# distance between a clustered image color and the yarn colors
def find_closest_yarn(rgb, yarn_db):
    closest_yarn = None
    min_distance = float('inf')

    for yarn in yarn_db:
        # Euclidean Distance between the RGB values
        # The distance is the square root of the squared sum of the
        # difference between the RGB values
        dist = np.sqrt(np.sum((np.array(rgb) - np.array(yarn['rgb']))**2))
        # To find the shortest Euclidean distance
        if dist < min_distance:
            min_distance = dist
            closest_yarn = yarn

    return closest_yarn

# Load in image
image = Image.open('Stout_Blanket.JPEG').resize((200, 200)) # 6 clusters
# image = Image.open('Camping_Photo.JPEG').resize((200,200)) #6
# image = Image.open('Nature_Photo.JPEG').resize((200,200)) #10
pixels = np.array(image).reshape(-1,3) # list of RGB pixels from the image

# Elbow Graph to Get Number of Color Clusters
inertia = np.zeros(14)
for i in range(1, 15):
    kmeans = KMeans(n_clusters=i).fit(pixels)
    inertia[i-1] = kmeans.inertia_

plt.plot(np.arange(1, 15), inertia)
plt.show()

# Cluster with KMeans clustering
num_clusters = 6
#Randomly picks num_cluster points, measures the distances and groups, takes
# a step to change the position of the centroids and then repeats.
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(pixels)
dominant_colors = kmeans.cluster_centers_

# Pull yarn colors from the yarn_colors.json file
# https://temperature-blanket.com/ through the Yarn Colorway Finder
# https://github.com/jdvlpr/Temperature-Blanket-Web-App/tree/main/src/lib/yarns
with open("yarn_colors.json", 'r') as f:
    yarn_colors = json.load(f)
#Convert the hex to rgb
for yarn in yarn_colors:
    hex_color = yarn['hex'].lstrip('#')
    yarn['rgb'] = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# Go through dominant colors to find the closest yarns by rgb using find_closest_yarn function
matched_yarns = []
for index, color in enumerate(dominant_colors):
    yarn = find_closest_yarn(tuple(color), yarn_colors)
    matched_yarns.append(yarn)
    print(f"Color {index+1}: RGB {color} -> {yarn['brand']} - {yarn['name']} ({yarn['hex']})")

# Go through the matched yarns to get the colors
matched_colors = []
for yarn in matched_yarns:
    matched_colors.append(yarn['rgb'])

# Display the Chosen Colors
figure, ax = plt.subplots(2, 1, figsize=(10,6))

ax[0].imshow([dominant_colors.astype(int)])
ax[0].axis("off")
ax[0].set_title("Image Colors")

ax[1].imshow([matched_colors])
ax[1].axis("off")
ax[1].set_title("Yarn Colors")

plt.show()

# Optionally save the color matches as a png image
# plt.savefig("yarn_color_matches.png")
