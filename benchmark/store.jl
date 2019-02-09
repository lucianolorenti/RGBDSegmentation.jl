include("orm.jl")
mutable struct Algorithm <: DBTable
    id::Union{Integer, Nothing}
    name::String
    params::Dict
end
function Algorithm(;id::Union{Integer, Nothing}=nothing,
                   name::String="",
                   params::Dict=Dict())
    return Algorithm(id, name, params)
end

mutable struct SegmentedImage <: DBTable
    id::Union{Integer, Nothing}
    dataset::String
    image::Integer
    file_path::String
    algorithm::Union{Algorithm, Integer}
    metrics::Dict
end
function SegmentedImage(;
                        id::Union{Integer, Nothing}=nothing,
                        dataset::String="",
                        image::Integer=0,
                        file_path::String="",
                        algorithm::Union{Algorithm, Integer}=-1,
                        metrics::Dict=Dict())
    return SegmentedImage(
        nothing,
        dataset,
        image,
        file_path,
        algorithm,
        metrics)
end
mutable struct Result <: DBTable
    id::Union{Integer, Nothing}
    s::Union{SegmentedImage, Integer}
    results::Dict
end
function Result(;id::Union{Integer, Nothing}=nothing,
                s::Union{SegmentedImage, Integer}=-1,
                results::Dict=Dict())
    return Result(id, s, results)
end
    

  

function create_tables(con)
    execute(
        conn,
        create_table_sql(Algorithm))
    execute(
        conn,
        create_table_sql(SegmentedImage))
    execute(
        conn,
        create_table_sql(Result))
end


        

function sql_to_jsonb(d::Dict)
    return "'$(json(d))'::jsonb"
end

function filename(seg::SegmentedImage)
    base_filename = filename(seg.dataset, seg.image)
    return joinpath(seg.algorithm.name, base_filename)
end

function filename(dataset::RGBDDataset, image_number::Integer)
    return filename(name(dataset),image_number)
end
function filename(ds_name::String, image_number::Integer)
    return "$(ds_name)_$(image_number)_seg"
end

function exists(seg::SegmentedImage, conn; exclude::Array{Symbol}=[])
    do_exists = invoke(exists, Tuple{DBTable, Any}, seg, conn;exclude=exclude)
    if ! do_exists
        return do_exists
    else
        return isfile(string(seg.file_path, ".jld"))
    end
end
connection_params = Dict(:host=>"localhost",
                         :dbname=>"rgbd_segmentation_results",
                         :user=>"rgbd_segmentation",
                         :password=>"rgbd_seg")
connection = join([join(a,'=') for a in connection_params], ' ')
conn = LibPQ.Connection(connection)
create_tables(conn)

function tests()
    a = SegmentedImage(5,
                       "asa",
                       5,
                       "asdasd/aads",
                       Algorithm(15, "aaa", Dict("a"=>"b")))

    println(exists(a, conn))
    insert(a, conn)
    println(exists(a, conn))
end
