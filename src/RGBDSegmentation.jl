module RGBDSegmentation
export clusterize
abstract type RGBDSegmentationAlgorithm end
function clusterize() end
include("Utils.jl")
include("RGBDProcessing.jl")
include("Datasets/RGBDDataset.jl")
include("Algorithms/JCSA/DepthWMM.jl")
include("Algorithms/JCSA/JCSA.jl")
include("Algorithms/RGBDNGraph/RGBDNGraph.jl")
end
