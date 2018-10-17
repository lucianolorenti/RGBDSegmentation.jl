using RGBDSegmentation
using RGBDSegmentation.JCSA
using RGBDSegmentation.CDNGraph
using RGBDSegmentation.GCF
using ImageSegmentation
using Test
#@testset "Preprocess" begin
#    nyu = NYUDataset()
#    preprocess(nyu)
#    image = get_image(nyu, 1)
#end
#@testset "Datasets" begin
#    nyu = NYUDataset()
#    download(nyu)
#    image = get_image(nyu, 1)
#end
#@testset "JCA" begin
#    nyu = NYUDataset()
#    cfg = JCSA.Config()
#    image = get_image(nyu, 1)
#    labels = clusterize(cfg, image.image, 25)
#    ulabels = unique(labels)
#    @test length(ulabels) > 1
#end
#@testset "CDNGraph" begin
#    nyu = NYUDataset()
#    cfg = CDNGraph.Config(5)
#    image = get_image(nyu, 1)
#    img_clusterized = clusterize(cfg, image.image, 25)
#    labels = labels_map(img_clusterized)
#    ulabels = unique(labels)
#    @test length(ulabels) > 1 
#end
@testset "GCF" begin
    nyu = NYUDataset()
    lambdas = [0.0001, 0.001, 0.01, 0.1]
    cfg = GCF.Config(nev=5,
                     lambdas=lambdas,
                     proportion_of_pixels=0.01)
    img = get_image(nyu, 1)
    image = resize(img, 0.5)
    img_clusterized = clusterize(cfg, image.image, 25)
end
#@testset "Gaussian" begin
#    import RGBDSegmentation.JCSA: sufficient_statistic,
#                                  MvGaussianExponential
#    data = [ 1 0 1; 2 0 1; 2 1 2; 3 1 2;4 5 1]'
#    a = sufficient_statistic(MvGaussianExponential, data)
#    Σ = cov(data, dims=2)
#    μ = mean(data, dims=2)
    
#end;
