module LightGBM
  module FFI
    extend ::FFI::Library
    ffi_lib ["lightgbm", "lib_lightgbm.so"]

    # https://github.com/microsoft/LightGBM/blob/master/include/LightGBM/c_api.h
    attach_function :LGBM_GetLastError, %i[], :string
    attach_function :LGBM_BoosterCreate, %i[pointer string pointer], :int
    attach_function :LGBM_BoosterCreateFromModelfile, %i[string pointer pointer], :int
    attach_function :LGBM_BoosterLoadModelFromString, %i[string pointer pointer], :int
    attach_function :LGBM_BoosterFree, %i[pointer], :int
    attach_function :LGBM_DatasetFree, %i[pointer], :int
    attach_function :LGBM_BoosterUpdateOneIter, %i[pointer pointer], :int
    attach_function :LGBM_DatasetCreateFromMat, %i[pointer int int32 int32 int string pointer pointer], :int
    attach_function :LGBM_DatasetSetField, %i[pointer string pointer int int], :int
    attach_function :LGBM_BoosterPredictForMat, %i[pointer pointer int int32 int32 int int int string pointer pointer], :int
    attach_function :LGBM_BoosterSaveModel, %i[pointer int int string], :int
    attach_function :LGBM_BoosterFeatureImportance, %i[pointer int int pointer], :int
    attach_function :LGBM_BoosterGetNumFeature, %i[pointer pointer], :int
    attach_function :LGBM_BoosterDumpModel, %i[pointer int int int64 pointer pointer], :int
    attach_function :LGBM_BoosterAddValidData, %i[pointer pointer], :int
    attach_function :LGBM_DatasetGetNumData, %i[pointer pointer], :int
    attach_function :LGBM_DatasetGetNumFeature, %i[pointer pointer], :int
    attach_function :LGBM_DatasetSaveBinary, %i[pointer pointer], :int
    attach_function :LGBM_BoosterGetCurrentIteration, %i[pointer pointer], :int
  end
end
