module LightGBM
  class Booster
    attr_accessor :best_iteration, :train_data_name

    def initialize(params: nil, train_set: nil, model_file: nil, model_str: nil)
      if model_str
        model_from_string(model_str)
      elsif model_file
        out_num_iterations = ::FFI::MemoryPointer.new(:int)
        create_handle do |handle|
          check_result FFI.LGBM_BoosterCreateFromModelfile(model_file, out_num_iterations, handle)
        end
      else
        params ||= {}
        set_verbosity(params)
        create_handle do |handle|
          check_result FFI.LGBM_BoosterCreate(train_set.handle_pointer, params_str(params), handle)
        end
      end

      self.best_iteration = -1

      # TODO get names when loaded from file
      @name_valid_sets = []
    end

    def add_valid(data, name)
      check_result FFI.LGBM_BoosterAddValidData(handle_pointer, data.handle_pointer)
      @name_valid_sets << name
      self # consistent with Python API
    end

    def current_iteration
      out = ::FFI::MemoryPointer.new(:int)
      check_result FFI.LGBM_BoosterGetCurrentIteration(handle_pointer, out)
      out.read_int
    end

    def dump_model(num_iteration: nil, start_iteration: 0)
      num_iteration ||= best_iteration
      buffer_len = 1 << 20
      out_len = ::FFI::MemoryPointer.new(:int64)
      out_str = ::FFI::MemoryPointer.new(:char, buffer_len)
      feature_importance_type = 0 # TODO add option
      check_result FFI.LGBM_BoosterDumpModel(handle_pointer, start_iteration, num_iteration, feature_importance_type, buffer_len, out_len, out_str)
      actual_len = out_len.read_int64
      if actual_len > buffer_len
        out_str = ::FFI::MemoryPointer.new(:char, actual_len)
        check_result FFI.LGBM_BoosterDumpModel(handle_pointer, start_iteration, num_iteration, feature_importance_type, actual_len, out_len, out_str)
      end
      out_str.read_string
    end
    alias_method :to_json, :dump_model

    def eval_valid
      @name_valid_sets.each_with_index.flat_map { |n, i| inner_eval(n, i + 1) }
    end

    def eval_train
      inner_eval(train_data_name, 0)
    end

    def feature_importance(iteration: nil, importance_type: "split")
      iteration ||= best_iteration
      importance_type =
        case importance_type
        when "split"
          0
        when "gain"
          1
        else
          -1
        end

      num_feature = self.num_feature
      out_result = ::FFI::MemoryPointer.new(:double, num_feature)
      check_result FFI.LGBM_BoosterFeatureImportance(handle_pointer, iteration, importance_type, out_result)
      out_result.read_array_of_double(num_feature).map(&:to_i)
    end

    def feature_name
      len = self.num_feature
      out_len = ::FFI::MemoryPointer.new(:size_t)
      buffer_len = 255
      out_buffer_len = ::FFI::MemoryPointer.new(:size_t)
      out_strs = ::FFI::MemoryPointer.new(:pointer, num_feature)
      str_ptrs = len.times.map { ::FFI::MemoryPointer.new(:char, buffer_len) }
      out_strs.write_array_of_pointer(str_ptrs)
      check_result FFI.LGBM_BoosterGetFeatureNames(handle_pointer, len, out_len, buffer_len, out_buffer_len, out_strs)

      actual_len = out_buffer_len.read(:size_t)
      if actual_len > buffer_len
        str_ptrs = len.times.map { ::FFI::MemoryPointer.new(:char, actual_len) }
        out_strs.write_array_of_pointer(str_ptrs)
        check_result FFI.LGBM_BoosterGetFeatureNames(handle_pointer, len, out_len, actual_len, out_buffer_len, out_strs)
      end

      str_ptrs[0, out_len.read(:size_t)].map(&:read_string)
    end

    def model_from_string(model_str)
      out_num_iterations = ::FFI::MemoryPointer.new(:int)
      create_handle do |handle|
        check_result FFI.LGBM_BoosterLoadModelFromString(model_str, out_num_iterations, handle)
      end
      @cached_feature_name = nil
      self
    end

    def model_to_string(num_iteration: nil, start_iteration: 0)
      num_iteration ||= best_iteration
      buffer_len = 1 << 20
      out_len = ::FFI::MemoryPointer.new(:int64)
      out_str = ::FFI::MemoryPointer.new(:char, buffer_len)
      feature_importance_type = 0 # TODO add option
      check_result FFI.LGBM_BoosterSaveModelToString(handle_pointer, start_iteration, num_iteration, feature_importance_type, buffer_len, out_len, out_str)
      actual_len = out_len.read_int64
      if actual_len > buffer_len
        out_str = ::FFI::MemoryPointer.new(:char, actual_len)
        check_result FFI.LGBM_BoosterSaveModelToString(handle_pointer, start_iteration, num_iteration, feature_importance_type, actual_len, out_len, out_str)
      end
      out_str.read_string
    end

    def num_feature
      out = ::FFI::MemoryPointer.new(:int)
      check_result FFI.LGBM_BoosterGetNumFeature(handle_pointer, out)
      out.read_int
    end
    alias_method :num_features, :num_feature # legacy typo

    def num_model_per_iteration
      out = ::FFI::MemoryPointer.new(:int)
      check_result FFI.LGBM_BoosterNumModelPerIteration(handle_pointer, out)
      out.read_int
    end

    def num_trees
      out = ::FFI::MemoryPointer.new(:int)
      check_result FFI.LGBM_BoosterNumberOfTotalModel(handle_pointer, out)
      out.read_int
    end

    def predict(input, start_iteration: 0, num_iteration: nil, raw_score: false, pred_leaf: false, pred_contrib: false, **params)
      num_iteration ||= best_iteration

      if input.is_a?(Dataset)
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
      check_result FFI.LGBM_BoosterPredictForMat(handle_pointer, data, 1, input.count, input.first.count, 1, predict_type, start_iteration, num_iteration, params_str(params), out_len, out_result)

      if n_preds != out_len.read_int64
        raise Error, "Wrong length for predict results"
      end

      preds = out_result.read_array_of_double(out_len.read_int64)

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

    def save_model(filename, num_iteration: nil, start_iteration: 0)
      num_iteration ||= best_iteration
      feature_importance_type = 0 # TODO add
      check_result FFI.LGBM_BoosterSaveModel(handle_pointer, start_iteration, num_iteration, feature_importance_type, filename)
      self # consistent with Python API
    end

    def update
      finished = ::FFI::MemoryPointer.new(:int)
      check_result FFI.LGBM_BoosterUpdateOneIter(handle_pointer, finished)
      finished.read_int == 1
    end

    private

    def handle_pointer
      @handle
    end

    def create_handle
      ::FFI::MemoryPointer.new(:pointer) do |handle|
        yield handle
        @handle = ::FFI::AutoPointer.new(handle.read_pointer, FFI.method(:LGBM_BoosterFree))
      end
    end

    def eval_counts
      out = ::FFI::MemoryPointer.new(:int)
      check_result FFI.LGBM_BoosterGetEvalCounts(handle_pointer, out)
      out.read_int
    end

    def eval_names
      eval_counts = self.eval_counts
      out_len = ::FFI::MemoryPointer.new(:int)
      out_buffer_len = ::FFI::MemoryPointer.new(:size_t)
      out_strs = ::FFI::MemoryPointer.new(:pointer, eval_counts)
      buffer_len = 255
      str_ptrs = eval_counts.times.map { ::FFI::MemoryPointer.new(:char, buffer_len) }
      out_strs.write_array_of_pointer(str_ptrs)
      check_result FFI.LGBM_BoosterGetEvalNames(handle_pointer, eval_counts, out_len, buffer_len, out_buffer_len, out_strs)

      actual_len = out_buffer_len.read(:size_t)
      if actual_len > buffer_len
        str_ptrs = eval_counts.times.map { ::FFI::MemoryPointer.new(:char, actual_len) }
        out_strs.write_array_of_pointer(str_ptrs)
        check_result FFI.LGBM_BoosterGetEvalNames(handle_pointer, eval_counts, out_len, actual_len, out_buffer_len, out_strs)
      end

      str_ptrs.map(&:read_string)
    end

    def inner_eval(name, i)
      eval_names = self.eval_names

      out_len = ::FFI::MemoryPointer.new(:int)
      out_results = ::FFI::MemoryPointer.new(:double, eval_names.count)
      check_result FFI.LGBM_BoosterGetEval(handle_pointer, i, out_len, out_results)
      vals = out_results.read_array_of_double(out_len.read_int)

      eval_names.zip(vals).map do |eval_name, val|
        higher_better = ["auc", "ndcg@", "map@"].any? { |v| eval_name.start_with?(v) }
        [name, eval_name, val, higher_better]
      end
    end

    def num_class
      out = ::FFI::MemoryPointer.new(:int)
      check_result FFI.LGBM_BoosterGetNumClasses(handle_pointer, out)
      out.read_int
    end

    def num_preds(start_iteration, num_iteration, nrow, predict_type)
      out = ::FFI::MemoryPointer.new(:int64)
      check_result FFI.LGBM_BoosterCalcNumPredict(handle_pointer, nrow, predict_type, start_iteration, num_iteration, out)
      out.read_int64
    end

    def sorted_feature_values(input_hash)
      input_hash.transform_keys(&:to_s).fetch_values(*cached_feature_name)
    end

    def cached_feature_name
      @cached_feature_name ||= feature_name
    end

    include Utils
  end
end
