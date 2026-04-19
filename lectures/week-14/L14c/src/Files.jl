"""
    save_model_checkpoint(path::AbstractString, model::MyMimoLegSHippoModel)

Save a trained MIMO LegS model to `path` as a jld2 file under the key
`"model"`.
"""
function save_model_checkpoint(path::AbstractString, model::MyMimoLegSHippoModel)
    jldsave(path; model = model)
    return path
end

"""
    load_model_checkpoint(path::AbstractString) -> MyMimoLegSHippoModel

Load a MIMO LegS model checkpoint previously written by
`save_model_checkpoint`.
"""
function load_model_checkpoint(path::AbstractString)
    isfile(path) || error("checkpoint not found: $(path)")
    d = load(path)
    haskey(d, "model") || error("expected key 'model' in $(path)")
    return d["model"]::MyMimoLegSHippoModel
end
