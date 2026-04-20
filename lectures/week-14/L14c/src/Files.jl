"""
    save_model_checkpoint(path::AbstractString, model::MyMimoLegSHippoModel)

Save a trained MIMO LegS model to `path` as a jld2 file under the key
`"model"`. The function returns `path` so it can be threaded into longer
workflows.
"""
function save_model_checkpoint(path::AbstractString, model::MyMimoLegSHippoModel)
    # Keep the on-disk schema simple so loading is just a single keyed lookup.
    jldsave(path; model = model)
    return path
end

"""
    load_model_checkpoint(path::AbstractString) -> MyMimoLegSHippoModel

Load a MIMO LegS model checkpoint previously written by
`save_model_checkpoint`. The file must exist and contain a `"model"` entry with
type `MyMimoLegSHippoModel`.
"""
function load_model_checkpoint(path::AbstractString)
    isfile(path) || error("checkpoint not found: $(path)")
    d = load(path)
    # Check the expected key explicitly so malformed checkpoint files fail with
    # a clear message.
    haskey(d, "model") || error("expected key 'model' in $(path)")
    return d["model"]::MyMimoLegSHippoModel
end
