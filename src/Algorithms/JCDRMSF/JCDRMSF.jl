"""
Joint Color and Depth Segmentation Based on Region Merging and Surface Fitting
Giampaolo Pagnutti and Pietro Zanuttigh
"""
module JCDRMSF

type Config
end
function clusterize(cfg::Config, RGB::Matrix, D::Matrix, N::Matrix)
    local LAB = rgb2lab(RGB)

end

end
