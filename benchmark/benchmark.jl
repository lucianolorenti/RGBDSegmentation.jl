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
using Statistics
using DataFrames
using Distributed
typedict(x) = Dict(fn=>getfield(x, fn)
                   for fn ∈ fieldnames(typeof(x)))

include("store.jl")

function supervised_metrics(img, segmented_image::ImageSegmentation.SegmentedImage)
    metrics = Dict(
        "FBoundary" => FBoundary(),
        "Precision" => Precision(),
        "FMeasure" => FMeasure(),
        "SegmentationCovering" => SegmentationCovering(),
        "VariationOfInformation" => VariationOfInformation(),
        "RandIndex" => RandIndex(),
        "FMeasureRegions" => FMeasureRegions(),
        "PRObjectsAndParts" => PRObjectsAndParts(),
        "BoundaryDisplacementError" =>  BoundaryDisplacementError()
    )
    result = Dict()
    for metric_name in sort(collect(keys(metrics)))
        @info("Computing $metric_name")
        result[metric_name] = ImageSegmentationEvaluation.evaluate(
            metrics[metric_name],
            segmented_image,
            img.ground_truth)
    end
    return result
end
function unsupervised_metrics(img, segmented_image::ImageSegmentation.SegmentedImage)
    default_params = Dict("ECW" => Dict(
                                   :threshold=>0.5),
                          )
    params = default_params

    RGB_img = colors(img.image)
    segmented_image_RGB = ImageSegmentation.SegmentedImage(
        segmented_image.image_indexmap,
        segmented_image.segment_labels,
        Dict(k => RGB(rm.color...) for (k, rm) in segmented_image.segment_means),
        segmented_image.segment_pixel_count)
    D   = distances(img.image)
    N   = normals(img.image)

    metrics_color = Dict(
        "ECW" => ECW(;params["ECW"]...),
        "Zeboudj" => Zeboudj(radius=5),
        "ValuesEntropy" => ValuesEntropy(),
        "LiuYangF" =>  LiuYangF(),
        "FPrime" => FPrime(),
        #"ErdemMethod" => ErdemMethod(5, 5),
        "Q" => Q())
    metrics_color_distance = Dict(
        "FRCRGBD" => FRCRGBD())


    result = Dict()
    for metric_name in sort(collect(keys(metrics_color)))
        @info("Computing $metric_name")
        result[metric_name] = ImageSegmentationEvaluation.evaluate(
            metrics_color[metric_name],
            RGB_img,
            segmented_image_RGB)
    end

    for metric_name in sort(collect(keys(metrics_color_distance)))
        @info("Computing $metric_name")
        result[metric_name] = ImageSegmentationEvaluation.evaluate(
            metrics_color_distance[metric_name],
            RGB_img,
            D[3, :, :],
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
        if !isfile(string(segmented_image.file_path, ".jld"))
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
                    K))
            segmented_image.metrics["elapsed"] = time
            save(segmented_image,
                 image,
                 segments,
                 conn)
        else
            @info("Inserting already segmented image")
            insert(segmented_image, conn)
        end
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
function evaluate_image(dataset, i, algorithms, config)
    image = nothing
    segmented_image = nothing
    K = config["number_of_segments"]
    data_path = config["data_path"]
    image_scale = config["image_scale"]

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
        segmented_image_db = get(
            SegmentedImage(
                dataset=name(dataset),
                image=i,
                file_path=file_path,
                algorithm=algorithm),
                conn,
            exclude=[:metrics])
        if (!isempty(segmented_image_db.metrics))
            continue
        end
        try

            if image == nothing
                @info "Loading image $i"
                image = dataset[i]
                @info("Resizing image to 50%")
                image = resize(image, image_scale)
            end
            segmented_image = load(segmented_image_db)
            results = unsupervised_metrics(image, segmented_image)
            merge!(results, supervised_metrics(image, segmented_image))
            segmented_image_db.metrics = results
            save(segmented_image_db, conn)
        catch e
            @warn(e)
        end

    end
    GC.gc()
end
function evaluate(config)
    dataset = NYUDataset()
    algorithms = Dict(
        "JCSA"=> (
            JCSA.Config(;
                dict_to_params(config["JCSA"])...)),
        #"GCF" => GCF.Config(;
        #    dict_to_params(config["GCF"])...),
        "CDNGraph" =>  CDNGraph.Config(;
            dict_to_params(config["CDNGraph"])...)
    )
    #for i=1:length(dataset)
    #    evaluate_image(dataset, i, algorithms, config)
    #end
    asyncmap(i->evaluate_image(dataset, i, algorithms, config),
             1:length(dataset),
             ntasks=4)

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
             :action=>:command),
        "summarize",
        Dict(:help=>"Summarize metrics",
            :action=>:command)
    )
    return parse_args(ARGS, s)
end
function summarize(config)
    dataset = NYUDataset()
    algorithms = Dict(
        "JCSA"=> (
            JCSA.Config(;
                dict_to_params(config["JCSA"])...)),
        #"GCF" => GCF.Config(;
        #    dict_to_params(config["GCF"])...),
        "CDNGraph" =>  CDNGraph.Config(;
            dict_to_params(config["CDNGraph"])...)
    )
    results = Dict()
    for i=1:length(dataset)
        image = nothing
        segmented_image = nothing
        K = config["number_of_segments"]
        data_path = config["data_path"]
        image_scale = config["image_scale"]
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
            filepath = joinpath(file_path, base_filename)
            segmented_image_db = get(
                SegmentedImage(
                    dataset=name(dataset),
                    image=i,
                    algorithm=algorithm),
                conn,
                exclude=[:metrics, :file_path])

            if !(haskey(results, algo_name))
                results[algo_name] = Dict()
            end
            for metric in keys(segmented_image_db.metrics)
                if !(haskey(results[algo_name], metric))
                    results[algo_name][metric] = []
                end
                if segmented_image_db.metrics[metric] != nothing
                    push!(results[algo_name][metric], first(segmented_image_db.metrics[metric]))
                end
            end
        end 
    end

    algorithm_names = sort(collect(keys(results)))
    for algo_name in algorithm_names
        for metric in sort(collect(keys(results[algo_name])))
            metrics = results[algo_name][metric]
            results[algo_name][metric] = mean(metrics)
        end
    end
    println(results)
end
function main()
    args = parse_commandline()
    config = YAML.load(open(args["config"]))
    if args["%COMMAND%"] == "segment"
        segment(config)
    elseif args["%COMMAND%"] == "evaluate"
        evaluate(config)
    elseif args["%COMMAND%"] == "summarize"
        summarize(config)
    end
end
main()
