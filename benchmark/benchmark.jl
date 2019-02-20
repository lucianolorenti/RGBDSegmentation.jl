using RGBDSegmentation
using RGBDSegmentation.JCSA
using RGBDSegmentation.CDNGraph
using RGBDSegmentation.GCF
using ImageSegmentation
using ImageSegmentationEvaluation
using LibPQ
using JSON
using DataStreams
using JLD
using Images
using FileIO
using ArgParse
using YAML
using Distributed
typedict(x) = Dict(fn=>getfield(x, fn)
                   for fn ∈ fieldnames(typeof(x)))

include("store.jl")

function unsupervised_metrics(img, segmented_image::SegmentedImage)
    default_params = Dict("ECW" => Dict(
                                   :threshold=>0.5),
                          )
    params = default_params

    metrics = Dict("ECW" => ECW(;params["ECW"]...),
                   "Zeboudj" => Zeboudj(),
                   "ValuesEntropy" => ValuesEntropy(),
                   "LiuYangF" =>  LiuYangF(),
                   "FPrime" => FPrime(),
                   "ErdemMethod" => ErdemMethod(5, 5),
                   "Q" => Q())
    result = Dict()
    for metric_name in sort(collect(keys(metrics)))
        result[metric_name] = evaluate(metrics[metric_name],
                                       img,
                                       segmented_image)
    end
    return result               
end

function dict_to_params(params::Dict)
    return Dict(Symbol(a.first)=>a.second for a in params)
end
function segment_image(dataset, i, algorithms, config)
    K = config["number_of_segments"]
    data_path = config["data_path"]
    image_scale = config["image_scale"]

    image = nothing
    for algo_name in keys(algorithms)
        algorithm_cfg = algorithms[algo_name]
        algorithm = get_or_insert(
            Algorithm(
                name=algo_name,
                params=typedict(algorithm_cfg)),
            conn)
        base_filename = filename(dataset, i)
        file_path = joinpath(data_path, algo_name)
        mkpath(file_path)
        file_path = joinpath(file_path, base_filename)
        segmented_image = SegmentedImage(
            dataset=name(dataset),
            image=i,
            file_path=file_path,
            algorithm=algorithm)
        if exists(segmented_image, conn, exclude=[:metrics])
            @info("Segmentation already stored")
            continue
        end
        if image == nothing
            @info "Loading image $i"
            image = dataset[i]
            @info("Resizing image to 50%")
            image = resize(image, image_scale)
        end
        @info("Segmenting image $i with algorithm $algorithm")
        time = @elapsed (
            segments = clusterize(
                algorithm_cfg,
                image.image,
                K)
        )
        segmented_image.metrics["elapsed"] = time
        save(segmented_image,
             image,
             segments,
             conn)
    end
        
end
function segment(config)
    dataset = NYUDataset()
    algorithms = Dict(
        "JCSA"=> (
            JCSA.Config(;
                dict_to_params(config["JCSA"])...)),
        "GCF" => GCF.Config(;
            dict_to_params(config["GCF"])...),
        "CDNGraph" =>  CDNGraph.Config(;
            dict_to_params(config["CDNGraph"])...)
    )
    asyncmap(i->segment_image(dataset, i, algorithms, config),
             1:length(dataset),
             ntasks=4)
end
function evaluate(config)
    dataset = NYUDataset()
    algorithms = Dict(
        "JCSA"=> (
            JCSA.Config(;
                dict_to_params(config["JCSA"])...)),
        "GCF" => GCF.Config(;
            dict_to_params(config["GCF"])...),
        "CDNGraph" =>  CDNGraph.Config(;
            dict_to_params(config["CDNGraph"])...)
    )
    K = config["number_of_segments"]
    data_path = config["data_path"]
    image_scale = config["image_scale"]
    for i=1:length(dataset)
        image = nothing
        for algo_name in keys(algorithms)
            algorithm_cfg = algorithms[algo_name]
            algorithm = get(
                Algorithm(
                    name=algo_name,
                    params=typedict(algorithm_cfg)),
                conn)

            base_filename = filename(dataset, i)
            file_path = joinpath(data_path, algo_name)
            mkpath(file_path)
            file_path = joinpath(file_path, base_filename)
            segmented_image = SegmentedImage(
                dataset=name(dataset),
                image=i,
                file_path=file_path,
                algorithm=algorithm)
            if image == nothing
                @info "Loading image $i"
                image = dataset[i]
                @info("Resizing image to 50%")
                image = resize(image, image_scale)
            end
            unsupervised_metrics(image, load(segmented_image))
        end
    end
end
function parse_commandline()
    s = ArgParseSettings()
    add_arg_table(
        s,
        "--config",
        Dict(:help=>"an option with an argument",
             :required=>true),
        "segment",
        Dict(:help =>"Segment images",
             :action=>:command),
        "evaluate",
        Dict(:help=>"Evaluate segmented image",
         :action=>:command)
    )
                  
    return parse_args(ARGS, s)
end

function main()
    args = parse_commandline()
    config = YAML.load(open(args["config"]))
    if args["%COMMAND%"] == "segment"
        segment(config)
    elseif args["%COMMAND%"] == "evaluate"
        evaluate(config)
    end
end
main()
