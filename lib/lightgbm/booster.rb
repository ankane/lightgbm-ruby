module LightGBM
  class Booster
    def initialize(params: nil, train_set: nil, model_file: nil, model_str: nil)
      @handle = ::FFI::MemoryPointer.new(:pointer)
      if model_str
        out_num_iterations = ::FFI::MemoryPointer.new(:int)
        check_result FFI.LGBM_BoosterLoadModelFromString(model_str, out_num_iterations, @handle)
      elsif model_file
        out_num_iterations = ::FFI::MemoryPointer.new(:int)
        check_result FFI.LGBM_BoosterCreateFromModelfile(model_file, out_num_iterations, @handle)
      else
        check_result FFI.LGBM_BoosterCreate(train_set.handle_pointer, params_str(params), @handle)
      end
      # causes "Stack consistency error"
      # ObjectSpace.define_finalizer(self, self.class.finalize(handle_pointer))
    end

    def self.finalize(pointer)
      -> { FFI.LGBM_BoosterFree(pointer) }
    end

    # TODO handle name
    def add_valid(data, name)
      check_result FFI.LGBM_BoosterAddValidData(handle_pointer, data.handle_pointer)
      self # consistent with Python API
    end

    def predict(input)
      raise TypeError unless input.is_a?(Array)

      singular = input.first.is_a?(Array)
      input = [input] unless singular

      data = ::FFI::MemoryPointer.new(:float, input.count * input.first.count)
      data.put_array_of_float(0, input.flatten)

      out_len = ::FFI::MemoryPointer.new(:int64)
      out_result = ::FFI::MemoryPointer.new(:double, input.count)
      parameter = ""
      check_result FFI.LGBM_BoosterPredictForMat(handle_pointer, data, 0, input.count, input.first.count, 1, 0, 0, parameter, out_len, out_result)
      out = out_result.read_array_of_double(out_len.read_int64)

      singular ? out : out.first
    end

    def save_model(filename)
      check_result FFI.LGBM_BoosterSaveModel(handle_pointer, 0, 0, filename)
      self # consistent with Python API
    end

    def update
      finished = ::FFI::MemoryPointer.new(:int)
      check_result FFI.LGBM_BoosterUpdateOneIter(handle_pointer, finished)
      finished.read_int == 1
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

      num_features = self.num_features
      out_result = ::FFI::MemoryPointer.new(:double, num_features)
      check_result FFI.LGBM_BoosterFeatureImportance(handle_pointer, iteration, importance_type, out_result)
      out_result.read_array_of_double(num_features)
    end

    def num_features
      out = ::FFI::MemoryPointer.new(:int)
      check_result FFI.LGBM_BoosterGetNumFeature(handle_pointer, out)
      out.read_int
    end

    # TODO fix
    def best_iteration
      -1
    end

    def to_json(num_iteration: nil, start_iteration: 0)
      num_iteration ||= best_iteration
      buffer_len = 1 << 20
      out_len = ::FFI::MemoryPointer.new(:int64)
      out_str = ::FFI::MemoryPointer.new(:string, buffer_len)
      check_result FFI.LGBM_BoosterDumpModel(handle_pointer, start_iteration, num_iteration, buffer_len, out_len, out_str)
      actual_len = out_len.read_int64
      if actual_len > buffer_len
        out_str = ::FFI::MemoryPointer.new(:string, actual_len)
        check_result FFI.LGBM_BoosterDumpModel(handle_pointer, start_iteration, num_iteration, actual_len, out_len, out_str)
      end
      out_str.read_string
    end

    private

    def check_result(err)
      raise LightGBM::Error, FFI.LGBM_GetLastError if err != 0
    end

    def handle_pointer
      @handle.read_pointer
    end

    # remove spaces in keys and values to prevent injection
    def params_str(params)
      params.map { |k, v| [check_param(k.to_s), check_param(v.to_s)].join("=") }.join(" ")
    end

    def check_param(v)
      raise ArgumentError, "Invalid parameter" if /[[:space:]]/.match(v)
      v
    end
  end
end
