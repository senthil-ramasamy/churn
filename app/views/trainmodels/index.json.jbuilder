json.array!(@trainmodels) do |trainmodel|
  json.extract! trainmodel, :id, :modid, :moddesc, :modname, :trainfile, :testfile, :addtnl, :modcheck
  json.url trainmodel_url(trainmodel, format: :json)
end
