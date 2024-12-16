module LightGBM
  class Booster
    include Utils

    attr_accessor :best_iteration, :train_data_name, :params

    def initialize(params: nil, train_set: nil, model_file: nil, model_str: nil)
      if model_str
        model_from_string(model_str)
      elsif model_file
        out_num_iterations = ::FFI::MemoryPointer.new(:int)
        create_handle do |handle|
          safe_call FFI.LGBM_BoosterCreateFromModelfile(model_file, out_num_iterations, handle)
        end
        @pandas_categorical = load_pandas_categorical(file_name: model_file)
        if params
          warn "[lightgbm] Ignoring params argument, using parameters from model file."
        end
        @params = loaded_param
      else
        params ||= {}
        set_verbosity(params)
        create_handle do |handle|
          safe_call FFI.LGBM_BoosterCreate(train_set.handle, params_str(params), handle)
        end
      end

      self.best_iteration = -1

      # TODO get names when loaded from file
      @name_valid_sets = []
    end

    def add_valid(data, name)
      safe_call FFI.LGBM_BoosterAddValidData(@handle, data.handle)
      @name_valid_sets << name
      self # consistent with Python API
    end

    def current_iteration
      out = ::FFI::MemoryPointer.new(:int)
      safe_call FFI.LGBM_BoosterGetCurrentIteration(@handle, out)
      out.read_int
    end

    def dump_model(num_iteration: nil, start_iteration: 0, importance_type: "split")
      num_iteration ||= best_iteration
      importance_type_int = feature_importance_type_mapper(importance_type)
      buffer_len = 1 << 20
      out_len = ::FFI::MemoryPointer.new(:int64)
      out_str = ::FFI::MemoryPointer.new(:char, buffer_len)
      safe_call FFI.LGBM_BoosterDumpModel(@handle, start_iteration, num_iteration, importance_type_int, buffer_len, out_len, out_str)
      actual_len = out_len.read_int64
      if actual_len > buffer_len
        out_str = ::FFI::MemoryPointer.new(:char, actual_len)
        safe_call FFI.LGBM_BoosterDumpModel(@handle, start_iteration, num_iteration, importance_type_int, actual_len, out_len, out_str)
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
      importance_type_int = feature_importance_type_mapper(importance_type)
      num_feature = self.num_feature
      out_result = ::FFI::MemoryPointer.new(:double, num_feature)
      safe_call FFI.LGBM_BoosterFeatureImportance(@handle, iteration, importance_type_int, out_result)
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
      safe_call FFI.LGBM_BoosterGetFeatureNames(@handle, len, out_len, buffer_len, out_buffer_len, out_strs)

      actual_len = out_buffer_len.read(:size_t)
      if actual_len > buffer_len
        str_ptrs = len.times.map { ::FFI::MemoryPointer.new(:char, actual_len) }
        out_strs.write_array_of_pointer(str_ptrs)
        safe_call FFI.LGBM_BoosterGetFeatureNames(@handle, len, out_len, actual_len, out_buffer_len, out_strs)
      end

      str_ptrs[0, out_len.read(:size_t)].map(&:read_string)
    end

    def model_from_string(model_str)
      out_num_iterations = ::FFI::MemoryPointer.new(:int)
      create_handle do |handle|
        safe_call FFI.LGBM_BoosterLoadModelFromString(model_str, out_num_iterations, handle)
      end
      @pandas_categorical = load_pandas_categorical(model_str: model_str)
      @params = loaded_param
      @cached_feature_name = nil
      self
    end

    def model_to_string(num_iteration: nil, start_iteration: 0, importance_type: "split")
      num_iteration ||= best_iteration
      importance_type_int = feature_importance_type_mapper(importance_type)
      buffer_len = 1 << 20
      out_len = ::FFI::MemoryPointer.new(:int64)
      out_str = ::FFI::MemoryPointer.new(:char, buffer_len)
      safe_call FFI.LGBM_BoosterSaveModelToString(@handle, start_iteration, num_iteration, importance_type_int, buffer_len, out_len, out_str)
      actual_len = out_len.read_int64
      if actual_len > buffer_len
        out_str = ::FFI::MemoryPointer.new(:char, actual_len)
        safe_call FFI.LGBM_BoosterSaveModelToString(@handle, start_iteration, num_iteration, importance_type_int, actual_len, out_len, out_str)
      end
      out_str.read_string
    end

    def num_feature
      out = ::FFI::MemoryPointer.new(:int)
      safe_call FFI.LGBM_BoosterGetNumFeature(@handle, out)
      out.read_int
    end
    alias_method :num_features, :num_feature # legacy typo

    def num_model_per_iteration
      out = ::FFI::MemoryPointer.new(:int)
      safe_call FFI.LGBM_BoosterNumModelPerIteration(@handle, out)
      out.read_int
    end

    def num_trees
      out = ::FFI::MemoryPointer.new(:int)
      safe_call FFI.LGBM_BoosterNumberOfTotalModel(@handle, out)
      out.read_int
    end

    def predict(data, start_iteration: 0, num_iteration: nil, raw_score: false, pred_leaf: false, pred_contrib: false, **kwargs)
      predictor = InnerPredictor.from_booster(self, kwargs.transform_values(&:dup))
      if num_iteration.nil?
        if start_iteration <= 0
          num_iteration = best_iteration
        else
          num_iteration = -1
        end
      end
      predictor.predict(
        data,
        start_iteration: start_iteration,
        num_iteration: num_iteration,
        raw_score: raw_score,
        pred_leaf: pred_leaf,
        pred_contrib: pred_contrib
      )
    end

    def save_model(filename, num_iteration: nil, start_iteration: 0, importance_type: "split")
      num_iteration ||= best_iteration
      importance_type_int = feature_importance_type_mapper(importance_type)
      safe_call FFI.LGBM_BoosterSaveModel(@handle, start_iteration, num_iteration, importance_type_int, filename)
      self # consistent with Python API
    end

    def update
      finished = ::FFI::MemoryPointer.new(:int)
      safe_call FFI.LGBM_BoosterUpdateOneIter(@handle, finished)
      finished.read_int == 1
    end

    private

    def create_handle
      ::FFI::MemoryPointer.new(:pointer) do |handle|
        yield handle
        @handle = ::FFI::AutoPointer.new(handle.read_pointer, FFI.method(:LGBM_BoosterFree))
      end
    end

    def eval_counts
      out = ::FFI::MemoryPointer.new(:int)
      safe_call FFI.LGBM_BoosterGetEvalCounts(@handle, out)
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
      safe_call FFI.LGBM_BoosterGetEvalNames(@handle, eval_counts, out_len, buffer_len, out_buffer_len, out_strs)

      actual_len = out_buffer_len.read(:size_t)
      if actual_len > buffer_len
        str_ptrs = eval_counts.times.map { ::FFI::MemoryPointer.new(:char, actual_len) }
        out_strs.write_array_of_pointer(str_ptrs)
        safe_call FFI.LGBM_BoosterGetEvalNames(@handle, eval_counts, out_len, actual_len, out_buffer_len, out_strs)
      end

      str_ptrs.map(&:read_string)
    end

    def inner_eval(name, i)
      eval_names = self.eval_names

      out_len = ::FFI::MemoryPointer.new(:int)
      out_results = ::FFI::MemoryPointer.new(:double, eval_names.count)
      safe_call FFI.LGBM_BoosterGetEval(@handle, i, out_len, out_results)
      vals = out_results.read_array_of_double(out_len.read_int)

      eval_names.zip(vals).map do |eval_name, val|
        higher_better = ["auc", "ndcg@", "map@"].any? { |v| eval_name.start_with?(v) }
        [name, eval_name, val, higher_better]
      end
    end

    def num_class
      out = ::FFI::MemoryPointer.new(:int)
      safe_call FFI.LGBM_BoosterGetNumClasses(@handle, out)
      out.read_int
    end

    def cached_feature_name
      @cached_feature_name ||= feature_name
    end

    def feature_importance_type_mapper(importance_type)
      case importance_type
      when "split"
        FFI::C_API_FEATURE_IMPORTANCE_SPLIT
      when "gain"
        FFI::C_API_FEATURE_IMPORTANCE_GAIN
      else
        -1
      end
    end

    def load_pandas_categorical(file_name: nil, model_str: nil)
      pandas_key = "pandas_categorical:"
      offset = -pandas_key.length
      if !file_name.nil?
        max_offset = -File.size(file_name)
        lines = []
        File.open(file_name, "rb") do |f|
          loop do
            offset = [offset, max_offset].max
            f.seek(offset, IO::SEEK_END)
            lines = f.readlines
            if lines.length >= 2 || offset == max_offset
              break
            end
            offset *= 2
          end
        end
        last_line = lines[-1].strip
        if !last_line.start_with?(pandas_key)
          last_line = lines[-2].strip
        end
      elsif !model_str.nil?
        idx = model_str[..offset].rindex("\n")
        last_line = model_str[idx..].strip
      end
      if last_line.start_with?(pandas_key)
        JSON.parse(last_line[pandas_key.length..])
      end
    end

    def loaded_param
      buffer_len = 1 << 20
      out_len = ::FFI::MemoryPointer.new(:int64)
      out_str = ::FFI::MemoryPointer.new(:char, buffer_len)
      safe_call FFI.LGBM_BoosterGetLoadedParam(@handle, buffer_len, out_len, out_str)
      actual_len = out_len.read_int64
      if actual_len > buffer_len
        out_str = ::FFI::MemoryPointer.new(:char, actual_len)
        safe_call FFI.LGBM_BoosterGetLoadedParam(@handle, actual_len, out_len, out_str)
      end
      JSON.parse(out_str.read_string)
    end
  end
end
