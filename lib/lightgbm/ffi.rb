module LightGBM
  module FFI
    extend ::FFI::Library

    begin
      ffi_lib LightGBM.ffi_lib
    rescue LoadError => e
      if ["/usr/local", "/opt/homebrew"].any? { |v| e.message.include?("Library not loaded: #{v}/opt/libomp/lib/libomp.dylib") } && e.message.include?("Reason: image not found")
        raise LoadError, "OpenMP not found. Run `brew install libomp`"
      else
        raise e
      end
    end

    # https://github.com/microsoft/LightGBM/blob/master/include/LightGBM/c_api.h
    # keep same order

    # error
    attach_function :LGBM_GetLastError, %i[], :string

    # dataset
    attach_function :LGBM_DatasetCreateFromFile, %i[string string pointer pointer], :int
    attach_function :LGBM_DatasetCreateFromMat, %i[pointer int int32 int32 int string pointer pointer], :int
    attach_function :LGBM_DatasetGetSubset, %i[pointer pointer int32 string pointer], :int
    attach_function :LGBM_DatasetSetFeatureNames, %i[pointer pointer int], :int
    attach_function :LGBM_DatasetGetFeatureNames, %i[pointer int pointer size_t pointer pointer], :int
    attach_function :LGBM_DatasetFree, %i[pointer], :int
    attach_function :LGBM_DatasetSaveBinary, %i[pointer string], :int
    attach_function :LGBM_DatasetDumpText, %i[pointer string], :int
    attach_function :LGBM_DatasetSetField, %i[pointer string pointer int int], :int
    attach_function :LGBM_DatasetGetField, %i[pointer string pointer pointer pointer], :int
    attach_function :LGBM_DatasetGetNumData, %i[pointer pointer], :int
    attach_function :LGBM_DatasetGetNumFeature, %i[pointer pointer], :int

    # booster
    attach_function :LGBM_BoosterCreate, %i[pointer string pointer], :int
    attach_function :LGBM_BoosterCreateFromModelfile, %i[string pointer pointer], :int
    attach_function :LGBM_BoosterLoadModelFromString, %i[string pointer pointer], :int
    attach_function :LGBM_BoosterFree, %i[pointer], :int
    attach_function :LGBM_BoosterAddValidData, %i[pointer pointer], :int
    attach_function :LGBM_BoosterGetNumClasses, %i[pointer pointer], :int
    attach_function :LGBM_BoosterUpdateOneIter, %i[pointer pointer], :int
    attach_function :LGBM_BoosterGetCurrentIteration, %i[pointer pointer], :int
    attach_function :LGBM_BoosterNumModelPerIteration, %i[pointer pointer], :int
    attach_function :LGBM_BoosterNumberOfTotalModel, %i[pointer pointer], :int
    attach_function :LGBM_BoosterGetEvalCounts, %i[pointer pointer], :int
    attach_function :LGBM_BoosterGetEvalNames, %i[pointer int pointer size_t pointer pointer], :int
    attach_function :LGBM_BoosterGetNumFeature, %i[pointer pointer], :int
    attach_function :LGBM_BoosterGetEval, %i[pointer int pointer pointer], :int
    attach_function :LGBM_BoosterPredictForMat, %i[pointer pointer int int32 int32 int int int int string pointer pointer], :int
    attach_function :LGBM_BoosterSaveModel, %i[pointer int int int string], :int
    attach_function :LGBM_BoosterSaveModelToString, %i[pointer int int int int64 pointer pointer], :int
    attach_function :LGBM_BoosterDumpModel, %i[pointer int int int int64 pointer pointer], :int
    attach_function :LGBM_BoosterFeatureImportance, %i[pointer int int pointer], :int
  end
end
