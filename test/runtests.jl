using RGBDSegmentation
using RGBDSegmentation.JCSA
using Test
@testset "Datasets" begin
    
    nyu = NYUDataset(
        joinpath(
            dirname(pathof(RGBDSegmentation)),
            "..",
            "test",
            "dataset"
        )
    )
#    preprocess(nyu, 1)
    image = get_image(nyu, 1)
    
end
@testset "JCA" begin

    using RGBDSegmentation
    using RGBDSegmentation.JCSA
    nyu = NYUDataset()
    download(nyu)
      cfg = JCSA.Config()
    image = get_image(nyu, 1)
    clusterize(cfg, image.image, 25)
end

@testset "Gaussian" begin
    import RGBDSegmentation.JCSA: sufficient_statistic,
                                  MvGaussianExponential
    data = [ 1 0 1; 2 0 1; 2 1 2; 3 1 2;4 5 1]'
    a = sufficient_statistic(MvGaussianExponential, data)
    Σ = cov(data, dims=2)
    μ = mean(data, dims=2)
    
end;
