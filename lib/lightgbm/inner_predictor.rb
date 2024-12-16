module LightGBM
  class InnerPredictor
    include Utils

    MAX_INT32 = (1 << 31) - 1

    def initialize(booster, pred_parameter)
      @handle = booster.instance_variable_get(:@handle)
      @pred_parameter = params_str(pred_parameter)

      # keep booster for cached_feature_name
      @booster = booster
    end

    def self.from_booster(booster, pred_parameter)
      new(booster, pred_parameter)
    end

    def predict(data, start_iteration: 0, num_iteration: -1, raw_score: false, pred_leaf: false, pred_contrib: false)
      if data.is_a?(Dataset)
        raise TypeError, "Cannot use Dataset instance for prediction, please use raw data instead"
      end

      predict_type = FFI::C_API_PREDICT_NORMAL
      if raw_score
        predict_type = FFI::C_API_PREDICT_RAW_SCORE
      end
      if pred_leaf
        predict_type = FFI::C_API_PREDICT_LEAF_INDEX
      end
      if pred_contrib
        predict_type = FFI::C_API_PREDICT_CONTRIB
      end

      if daru?(data)
        data = data[*cached_feature_name].map_rows(&:to_a)
        singular = false
      elsif data.is_a?(Hash) # sort feature.values to match the order of model.feature_name
        data = [sorted_feature_values(data)]
        singular = true
      elsif data.is_a?(Array) && data.first.is_a?(Hash) # on multiple elems, if 1st is hash, assume they all are
        data = data.map(&method(:sorted_feature_values))
        singular = false
      elsif rover?(data)
        # TODO improve performance
        data = data[cached_feature_name].to_numo.to_a
        singular = false
      else
        data = data.to_a
        singular = !data.first.is_a?(Array)
        data = [data] if singular
      end

      preds, nrow =
        pred_for_data(
          data,
          start_iteration,
          num_iteration,
          predict_type
        )

      if pred_leaf
        preds = preds.map(&:to_i)
      end

      if preds.size != nrow
        if preds.size % nrow == 0
          preds = preds.each_slice(preds.size / nrow).to_a
        else
          raise Error, "Length of predict result (#{preds.size}) cannot be divide nrow (#{nrow})"
        end
      end

      singular ? preds.first : preds
    end

    private

    def pred_for_data(input, start_iteration, num_iteration, predict_type)
      nrow = input.count
      if nrow > MAX_INT32
        raise Error, "Not supported"
      end
      inner_predict_data(
        input,
        start_iteration,
        num_iteration,
        predict_type
      )
    end

    def inner_predict_data(input, start_iteration, num_iteration, predict_type)
      n_preds =
        num_preds(
          start_iteration,
          num_iteration,
          input.count,
          predict_type
        )

      flat_input = input.flatten
      handle_missing(flat_input)
      data = ::FFI::MemoryPointer.new(:double, input.count * input.first.count)
      data.write_array_of_double(flat_input)

      out_num_preds = ::FFI::MemoryPointer.new(:int64)
      out_result = ::FFI::MemoryPointer.new(:double, n_preds)
      safe_call FFI.LGBM_BoosterPredictForMat(@handle, data, 1, input.count, input.first.count, 1, predict_type, start_iteration, num_iteration, @pred_parameter, out_num_preds, out_result)
      if n_preds != out_num_preds.read_int64
        raise Error, "Wrong length for predict results"
      end
      preds = out_result.read_array_of_double(out_num_preds.read_int64)
      [preds, input.count]
    end

    def num_preds(start_iteration, num_iteration, nrow, predict_type)
      out = ::FFI::MemoryPointer.new(:int64)
      safe_call FFI.LGBM_BoosterCalcNumPredict(@handle, nrow, predict_type, start_iteration, num_iteration, out)
      out.read_int64
    end

    def sorted_feature_values(input_hash)
      input_hash.transform_keys(&:to_s).fetch_values(*cached_feature_name)
    end

    def cached_feature_name
      @booster.send(:cached_feature_name)
    end
  end
end
