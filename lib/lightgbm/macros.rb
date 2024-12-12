module LightGBM
  # Macro definition of prediction type in C API of LightGBM
  C_API_PREDICT_NORMAL = 0
  C_API_PREDICT_RAW_SCORE = 1
  C_API_PREDICT_LEAF_INDEX = 2
  C_API_PREDICT_CONTRIB = 3
end
