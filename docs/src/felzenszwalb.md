# Felzenszwalb

Graph-based Segmentation for Colored 3D Laser Point Clouds

```@example
using RGBDSegmentation, ImageSegmentation
dataset      = NYUDataset()
img          = get_image(dataset, 5)
segments     = felzenszwalb(img.image,10)
segments_rgb = felzenszwalb(color_image(img.image),10)
segmented_image     = color_image(map(i->segment_mean(segments,i), labels_map(segments)))
segmented_image_rgb = map(i->segment_mean(segments_rgb,i), labels_map(segments_rgb))

using FileIO # hide
save("segmented_image.png",segmented_image) # hide
save("segmented_image_rgb.png",segmented_image_rgb) # hide
```


```@raw html
<figure class="row_2">
 <figure>
  <img src="segmented_image_rgb.png" alt="Segmented image using felzenszwalb ing RGB image" height="300px;">
  <figcaption>Segmented image using felzenszwalb in RGB image</figcaption>
</figure>
<figure>
  <img src="segmented_image.png" alt="Simmilarity computed for 3 pixels" height="300px">
  <figcaption>Segmented image using the extended version of felzenszwalb</figcaption>
</figure>
</figure>
```
