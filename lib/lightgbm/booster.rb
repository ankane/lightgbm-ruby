module LightGBM
  class Booster
    def initialize(model_file: nil, params: nil, train_set: nil)
      @handle = ::FFI::MemoryPointer.new(:pointer)
      ObjectSpace.define_finalizer(self, self.class.finalize(handle_pointer))

      if model_file
        out_num_iterations = ::FFI::MemoryPointer.new(:int)
        check_result FFI.LGBM_BoosterCreateFromModelfile(model_file, out_num_iterations, @handle)
      else
        train_data = ::FFI::MemoryPointer.new(:pointer)

        # populate train_data: data
        input = train_set.data
        data = ::FFI::MemoryPointer.new(:float, input.count * input.first.count)
        data.put_array_of_float(0, input.flatten)
        check_result FFI.LGBM_DatasetCreateFromMat(data, 0, input.count, input.first.count, 1, "", nil, train_data)

        # populate train_data: label
        label = train_set.label
        label_data = ::FFI::MemoryPointer.new(:float, label.count)
        label_data.put_array_of_float(0, label)
        check_result FFI.LGBM_DatasetSetField(train_data.read_pointer, "label", label_data, label.count, 0)

        check_result FFI.LGBM_BoosterCreate(train_data.read_pointer, params_str(params), @handle)
      end
    end

    def self.finalize(pointer)
      -> { FFI.LGBM_BoosterFree(pointer) }
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
    end

    def update
      finished = ::FFI::MemoryPointer.new(:int)
      check_result FFI.LGBM_BoosterUpdateOneIter(handle_pointer, finished)
      finished.read_int == 1
    end

    private

    def check_result(err)
      if err != 0
        raise FFI.LGBM_GetLastError
      end
    end

    def handle_pointer
      @handle.read_pointer
    end

    # remove spaces in keys and values to prevent injection
    def params_str(params)
      params.map { |k, v| [k.to_s.gsub(/[[:space:]]/, ""), v.to_s.gsub(/[[:space:]]/, "")].join("=") }.join(" ")
    end
  end
end
