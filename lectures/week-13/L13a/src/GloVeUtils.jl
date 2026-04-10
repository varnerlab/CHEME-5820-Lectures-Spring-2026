"""
    download_glove(datapath; dim=100) -> String

Download the Stanford GloVe 6B embeddings (if not already cached) and return the path
to the text file for the requested dimension.
"""
function download_glove(datapath::String; dim::Int=100)::String

    txtfile = joinpath(datapath, "glove.6B.$(dim)d.txt")
    isfile(txtfile) && return txtfile

    zipfile = joinpath(datapath, "glove.6B.zip")
    expected_size = 862_182_613  # expected zip size in bytes

    # download if not cached, or if a previous download was truncated -
    if !isfile(zipfile) || filesize(zipfile) < expected_size
        isfile(zipfile) && rm(zipfile) # clean up truncated file
        @info "Downloading GloVe 6B embeddings (~862 MB) from Stanford NLP..."
        Downloads.download("https://nlp.stanford.edu/data/glove.6B.zip", zipfile;
            timeout=600.0)
        actual = filesize(zipfile)
        if actual < expected_size
            rm(zipfile)
            error("Download appears truncated ($actual bytes, expected $expected_size). Please retry.")
        end
        @info "Download complete ($(round(actual/1e6, digits=1)) MB)."
    end

    @info "Extracting glove.6B.$(dim)d.txt..."
    run(`unzip -o $zipfile glove.6B.$(dim)d.txt -d $datapath`)
    rm(zipfile) # clean up the zip to save disk space

    return txtfile
end

"""
    load_glove_embeddings(filepath) -> Dict{String, Vector{Float32}}

Parse a GloVe text file into a dictionary mapping words to embedding vectors.
"""
function load_glove_embeddings(filepath::String)::Dict{String, Vector{Float32}}

    embeddings = Dict{String, Vector{Float32}}()
    open(filepath) do f
        for line in eachline(f)
            parts = split(line)
            word = String(parts[1])
            vec = Float32.(parse.(Float64, @view parts[2:end]))
            embeddings[word] = vec
        end
    end

    return embeddings
end
