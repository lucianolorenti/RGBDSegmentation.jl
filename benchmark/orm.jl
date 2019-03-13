using LibPQ

abstract type DBTable end

psql_field_type(::Type{String}) = "VARCHAR"
psql_field_type(::Type{Dict}) = "JSONB"
psql_field_type(::Type{Integer}) = "INTEGER"
psql_field_type(::Type{Union{T, Integer}}) where T = "INTEGER"

psql_value(a::String) = "'$a'"
psql_value(a::T) where T<:Integer = string(a)
psql_value(a::DBTable) = a.id
psql_value(a::Dict) = "'$(JSON.json(a))'"


julia_value(a::String, b::Type{Dict}) = JSON.parse(a)
julia_value(a, b) = a

function table_name(d::Type)
    return lowercase(string(d))
end
function table_name(d)
    return table_name(typeof(d))
end
function get_id(d::T, conn; exclude::Array{Symbol}=Symbol[])  where T<:DBTable
    fields = [name for name in fieldnames(typeof(d))
              if (name != :id) && !(name in exclude)]
    values = [psql_value(getfield(d,field)) for field in fields]
    conditions = [join(field_value, " = ") for field_value in
                 zip(fields, values)]
    sql = """
        SELECT 
            id 
        FROM 
            $(table_name(d))
        WHERE
            $(join(conditions, " AND "))
    """
    @info sql
    data = fetch!(NamedTuple, execute(conn, sql)).id
    if length(data) == 0
        return nothing
    else
        return first(data)
    end
end
function exists(d::T, conn; exclude::Array{Symbol}=Symbol[]) where T <:DBTable
    return get_id(d, conn, exclude=exclude) != nothing
end

function create_table_sql(d)
    fields = []
    for name in fieldnames(d)
        ftype = fieldtype(d, name)
        if name == :id 
            push!(fields, "id SERIAL PRIMARY KEY")
        else
            push!(fields, "$name $(psql_field_type(ftype))")
        end
    end
    return """
           CREATE TABLE IF NOT EXISTS $(table_name(d)) (
           $(join(fields, " ,"))

    )
    """
end
function insert_sql(d)
    fields = [name for name in fieldnames(typeof(d))
              if name != :id]
    values = [psql_value(getfield(d,field)) for field in fields]
    sql = """
        INSERT INTO $(table_name(d))
           ($(join(fields, ", ")))
        VALUES 
              ($(join(values, ", ")))
        RETURNING id;
    """
end
function update(d::DBTable, conn)
    fields = [name for name in fieldnames(typeof(d))
              if name != :id]
    values = [psql_value(getfield(d,field)) for field in fields]
    kv_list = join([string(field, "=", value) for (field, value) in zip(fields, values)], ", ")
    sql = """
        UPDATE $(table_name(d))
        SET $(kv_list)
        WHERE
            id = $(d.id)
    """
    @info(sql)
    execute(conn, sql)
end
function insert(d::DBTable, conn)
    sql = insert_sql(d)
    @info sql
    result = execute(conn, sql)
    data = fetch!(NamedTuple, result)
    d.id = data.id[1]
    return d
end

function get(d::T, conn; exclude::Array{Symbol}=Symbol[])  where T<:DBTable
    fields = [name for name in fieldnames(typeof(d))
              if (name != :id) && !(name in exclude)]
    values = [psql_value(getfield(d,field)) for field in fields]
    conditions = [join(field_value, " = ") for field_value in
                 zip(fields, values)]
    sql = """
        SELECT 
            * 
        FROM 
            $(table_name(d))
        WHERE
            $(join(conditions, " AND "))
    """
    @info sql
    data = fetch!(NamedTuple, execute(conn, sql))
    if isempty(data[:id])
        throw("Element not found")
    end
    params = Dict(k=>julia_value(first(data[k]), fieldtype(T, k)) for k in keys(data))
    d = T(;params...)
    return d
end
function get_or_insert(d::DBTable, conn)
    id = get_id(d, conn)
    if id == nothing
        insert(d, conn)
        return d
    else
        return get(d, conn)
    end
end
