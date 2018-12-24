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
typedict(x) = Dict(fn=>getfield(x, fn)
                   for fn ∈ fieldnames(typeof(x)))

include("store.jl")


function save_segmented_image(img, segmented_image, segments, conn)
    insert(segmented_image, conn)
    img = map(i->segment_mean(segments, i), labels_map(segments))
    save(string(segmented_image.file_path, ".jld"),
         "segments", segments)       
    save(string(segmented_image.file_path, ".png"),
         colorview(RGB, img))
 
end
function dict_to_params(params::Dict)
    return Dict(Symbol(a.first)=>a.second for a in params)
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
    K = config["number_of_segments"]
    data_path = config["data_path"]
    image_scale = config["image_scale"]
    for (i, image) in enumerate(dataset)
        @info("Resizing image to 50%")
 
        image = resize(image, image_scale)
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
            @info("Segmenting image $i with algorithm $algorithm")
            time = @elapsed (
            segments = clusterize(
                algorithm_cfg,
                image.image,
                K)
            )
            segmented_image.metrics["elapsed"] = time
            save_segmented_image(image,
                                 segmented_image,
                                 segments,
                                 conn)
            #unsupervised_metrics(image.image, img_clusterized)
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
    )

    add_arg_table(
        s["segment"],
        "--output-folder",
        Dict(:required=>true))

                  
    return parse_args(ARGS, s)
end

function main()
    args = parse_commandline()
    config = YAML.load(open(args["config"]))
    if args["%COMMAND%"] == "segment"
        segment(config)
    elseif args["%COMMAND%"] == "evaluate"
    end
end
main()
