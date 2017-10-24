
function merge_data(idx1::Integer, idx2::Integer, a::Int64, b::Int64)
    return a*b
end
function merge_edges(idx1::Integer, idx2::Integer, d::Float64)
    return 1.0
end
    r = Graph{Integer, Float64}()
    for i = 1:8
        add_vertex!(r, i )
    end
    connect!(r, 1 ,2, 1.0)
    connect!(r, 1 ,3, 1.0)
    connect!(r, 2 ,3, 1.0)
    connect!(r, 1 ,4, 1.0)
    connect!(r, 1 ,5, 1.0)
    connect!(r, 1 ,8, 1.0)
    connect!(r, 4 ,6, 1.0)
    connect!(r, 5 ,7, 1.0)

    merge_vertices!(r, 4 ,3)

    [2;5;8;9] == sort([edge.node.index for edge in r.nodes[1].neighbors])
    [1;9] == sort([edge.node.index for edge in r.nodes[2].neighbors])
    [1;2;6]== sort([edge.node.index for edge in r.nodes[9].neighbors])
    [1;7]== sort([edge.node.index for edge in r.nodes[5].neighbors])
    [1]== sort([edge.node.index for edge in r.nodes[8].neighbors])
    [5]== sort([edge.node.index for edge in r.nodes[7].neighbors])
    [9]== sort([edge.node.index for edge in r.nodes[6].neighbors])
