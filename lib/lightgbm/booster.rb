module LightGBM
  class Booster
    def initialize(params: nil, train_set: nil, model_file: nil, model_str: nil)
      @handle = ::FFI::MemoryPointer.new(:pointer)
      if model_str
        model_from_string(model_str)
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

    def save_model(filename, num_iteration: nil, start_iteration: 0)
      num_iteration ||= best_iteration
      check_result FFI.LGBM_BoosterSaveModel(handle_pointer, start_iteration, num_iteration, filename)
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

      num_feature = self.num_feature
      out_result = ::FFI::MemoryPointer.new(:double, num_feature)
      check_result FFI.LGBM_BoosterFeatureImportance(handle_pointer, iteration, importance_type, out_result)
      out_result.read_array_of_double(num_feature)
    end

    def num_feature
      out = ::FFI::MemoryPointer.new(:int)
      check_result FFI.LGBM_BoosterGetNumFeature(handle_pointer, out)
      out.read_int
    end
    alias_method :num_features, :num_feature # legacy typo

    def current_iteration
      out = ::FFI::MemoryPointer.new(:int)
      check_result FFI::LGBM_BoosterGetCurrentIteration(handle_pointer, out)
      out.read_int
    end

    def num_model_per_iteration
      out = ::FFI::MemoryPointer.new(:int)
      check_result FFI::LGBM_BoosterNumModelPerIteration(handle_pointer, out)
      out.read_int
    end

    def num_trees
      out = ::FFI::MemoryPointer.new(:int)
      check_result FFI::LGBM_BoosterNumberOfTotalModel(handle_pointer, out)
      out.read_int
    end

    # TODO fix
    def best_iteration
      -1
    end

    def model_from_string(model_str)
      out_num_iterations = ::FFI::MemoryPointer.new(:int)
      check_result FFI.LGBM_BoosterLoadModelFromString(model_str, out_num_iterations, @handle)
      self
    end

    def model_to_string(num_iteration: nil, start_iteration: 0)
      num_iteration ||= best_iteration
      buffer_len = 1 << 20
      out_len = ::FFI::MemoryPointer.new(:int64)
      out_str = ::FFI::MemoryPointer.new(:string, buffer_len)
      check_result FFI.LGBM_BoosterSaveModelToString(handle_pointer, start_iteration, num_iteration, buffer_len, out_len, out_str)
      actual_len = out_len.read_int64
      if actual_len > buffer_len
        out_str = ::FFI::MemoryPointer.new(:string, actual_len)
        check_result FFI.LGBM_BoosterSaveModelToString(handle_pointer, start_iteration, num_iteration, actual_len, out_len, out_str)
      end
      out_str.read_string
    end

    def dump_model(num_iteration: nil, start_iteration: 0)
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
    alias_method :to_json, :dump_model

    private

    def handle_pointer
      @handle.read_pointer
    end

    include Utils
  end
end
