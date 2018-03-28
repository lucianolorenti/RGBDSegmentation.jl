Graph-based Segmentation for Colored 3D Laser Point Clouds

```@example
using RGBDSegmentation, ImageSegmentation, Clustering, Colors
dataset      = NYUDataset()
img          = get_image(dataset, 5)
cfg          = RGBDSegmentation.GCF.Config(5, 3)
segments     = clusterize(cfg, img.image, 5)
segmented_image = reshape(assignments(segments),size(img.image))
segmented_image = Gray.(segmented_image./maximum(segmented_image))
Plots.plot(segmented_image)

```


