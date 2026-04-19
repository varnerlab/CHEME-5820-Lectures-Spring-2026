"""
    load_ohlc_dataset(path::AbstractString) -> Dict{String, DataFrame}

Load an OHLC jld2 file produced from the Polygon-sourced SP500 dataset.
The file is expected to contain a single top-level key `"dataset"` mapping
to a `Dict{String, DataFrame}` keyed by ticker.
"""
function load_ohlc_dataset(path::AbstractString)
    isfile(path) || error("dataset not found: $(path)")
    d = load(path)
    haskey(d, "dataset") || error("expected key 'dataset' in $(path)")
    return d["dataset"]::Dict{String, DataFrame}
end

"""
    save_model_checkpoint(path::AbstractString, model::MySisoLegSHippoModel)

Save a trained model to `path` as a jld2 file under the key `"model"`.
"""
function save_model_checkpoint(path::AbstractString, model::MySisoLegSHippoModel)
    jldsave(path; model = model)
    return path
end

"""
    load_model_checkpoint(path::AbstractString) -> MySisoLegSHippoModel

Load a trained model checkpoint previously written by `save_model_checkpoint`.
"""
function load_model_checkpoint(path::AbstractString)
    isfile(path) || error("checkpoint not found: $(path)")
    d = load(path)
    haskey(d, "model") || error("expected key 'model' in $(path)")
    return d["model"]::MySisoLegSHippoModel
end
