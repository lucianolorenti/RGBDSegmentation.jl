using RGBDSegmentation
using Base.Test
images = [rand(50,50,3) for i=1:3]
concat_images = cat(3,images...)
a = Matrix{RGBDN}(concat_images)
# write your own tests here
