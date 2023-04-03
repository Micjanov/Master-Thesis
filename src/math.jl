
"""
Transform numbers in an array or 2D matrix to fit in a given range.
"""
function scaler(row, lower, upper)
    minx = minimum(row)
    maxx = maximum(row)
    x = [lower + ((i - minx)*(upper-lower))/(maxx - minx) for i in row]
    return x
end

function mat_scaler(matrix, lower, upper, dim = 1)
    if dim == 2
        scaled = reduce(hcat, [scaler(matrix[:, i], lower, upper) for i in 1:size(matrix, 2)])
    elseif dim == 1
        scaled = permutedims(hcat([scaler(matrix[i, :], lower, upper) for i in 1:size(matrix, 1)]...))
    end
    return scaled
end
"""
Convert an array of arrays into a matrix, pd: transpose matrix
"""
function nested_arrays2mat(arrays, pd = false)
    if pd == false
        return reduce(hcat,arrays)
    else
        return permutedims(reduce(hcat,arrays))
    end
end