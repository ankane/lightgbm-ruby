module LightGBM
  class Dataset
    attr_reader :data, :label, :weight

    def initialize(data, label: nil, weight: nil)
      @data = data
      @label = label
      @weight = weight

      # prepare data
      c_data = ::FFI::MemoryPointer.new(:float, data.count * data.first.count)
      c_data.put_array_of_float(0, data.flatten)

      # create dataset
      @handle = ::FFI::MemoryPointer.new(:pointer)
      check_result FFI.LGBM_DatasetCreateFromMat(c_data, 0, data.count, data.first.count, 1, "", nil, @handle)
      # causes "Stack consistency error"
      # ObjectSpace.define_finalizer(self, self.class.finalize(handle_pointer))

      if label
        c_label = ::FFI::MemoryPointer.new(:float, label.count)
        c_label.put_array_of_float(0, label)
        check_result FFI.LGBM_DatasetSetField(handle_pointer, "label", c_label, label.count, 0)
      end

      if weight
        c_weight = ::FFI::MemoryPointer.new(:float, weight.count)
        c_weight.put_array_of_float(0, weight)
        check_result FFI.LGBM_DatasetSetField(handle_pointer, "weight", c_weight, weight.count, 0)
      end
    end

    def self.finalize(pointer)
      -> { FFI.LGBM_DatasetFree(pointer) }
    end

    def handle_pointer
      @handle.read_pointer
    end

    private

    def check_result(err)
      raise LightGBM::Error, FFI.LGBM_GetLastError if err != 0
    end
  end
end
