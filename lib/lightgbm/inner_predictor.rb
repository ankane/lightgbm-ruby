module LightGBM
  class InnerPredictor
    def initialize(booster, pred_parameter)
      @booster = booster
      @pred_parameter = params_str(pred_parameter)
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

      preds, nrow, singular =
        preds_for_data(
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

    def handle
      @booster.send(:handle)
    end

    def preds_for_data(input, start_iteration, num_iteration, predict_type)
      input =
        if daru?(input)
          input[*cached_feature_name].map_rows(&:to_a)
        elsif input.is_a?(Hash) # sort feature.values to match the order of model.feature_name
          sorted_feature_values(input)
        elsif input.is_a?(Array) && input.first.is_a?(Hash) # on multiple elems, if 1st is hash, assume they all are
          input.map(&method(:sorted_feature_values))
        elsif rover?(input)
          # TODO improve performance
          input[cached_feature_name].to_numo.to_a
        else
          input.to_a
        end

      singular = !input.first.is_a?(Array)
      input = [input] if singular

      nrow = input.count
      n_preds =
        num_preds(
          start_iteration,
          num_iteration,
          nrow,
          predict_type
        )

      flat_input = input.flatten
      handle_missing(flat_input)
      data = ::FFI::MemoryPointer.new(:double, input.count * input.first.count)
      data.write_array_of_double(flat_input)

      out_len = ::FFI::MemoryPointer.new(:int64)
      out_result = ::FFI::MemoryPointer.new(:double, n_preds)
      check_result FFI.LGBM_BoosterPredictForMat(handle, data, 1, input.count, input.first.count, 1, predict_type, start_iteration, num_iteration, @pred_parameter, out_len, out_result)

      if n_preds != out_len.read_int64
        raise Error, "Wrong length for predict results"
      end

      preds = out_result.read_array_of_double(out_len.read_int64)

      [preds, nrow, singular]
    end

    def num_preds(start_iteration, num_iteration, nrow, predict_type)
      out = ::FFI::MemoryPointer.new(:int64)
      check_result FFI.LGBM_BoosterCalcNumPredict(handle, nrow, predict_type, start_iteration, num_iteration, out)
      out.read_int64
    end

    def sorted_feature_values(input_hash)
      input_hash.transform_keys(&:to_s).fetch_values(*cached_feature_name)
    end

    def cached_feature_name
      @booster.send(:cached_feature_name)
    end

    include Utils
  end
end
