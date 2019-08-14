module LightGBM
  class Dataset
    attr_reader :data, :params

    def initialize(data, label: nil, weight: nil, params: nil)
      @data = data

      @handle = ::FFI::MemoryPointer.new(:pointer)
      if data.is_a?(String)
        check_result FFI.LGBM_DatasetCreateFromFile(data, params_str(params), nil, @handle)
      else
        c_data = ::FFI::MemoryPointer.new(:float, data.count * data.first.count)
        c_data.put_array_of_float(0, data.flatten)
        check_result FFI.LGBM_DatasetCreateFromMat(c_data, 0, data.count, data.first.count, 1, params_str(params), nil, @handle)
      end
      # causes "Stack consistency error"
      # ObjectSpace.define_finalizer(self, self.class.finalize(handle_pointer))

      set_field("label", label) if label
      set_field("weight", weight) if weight
    end

    def label
      field("label")
    end

    def weight
      field("weight")
    end

    def num_data
      out = ::FFI::MemoryPointer.new(:int)
      check_result FFI.LGBM_DatasetGetNumData(handle_pointer, out)
      out.read_int
    end

    def num_feature
      out = ::FFI::MemoryPointer.new(:int)
      check_result FFI.LGBM_DatasetGetNumFeature(handle_pointer, out)
      out.read_int
    end

    def save_binary(filename)
      check_result FFI.LGBM_DatasetSaveBinary(handle_pointer, filename)
    end

    def dump_text(filename)
      check_result FFI.LGBM_DatasetDumpText(handle_pointer, filename)
    end

    def self.finalize(pointer)
      -> { FFI.LGBM_DatasetFree(pointer) }
    end

    def handle_pointer
      @handle.read_pointer
    end

    private

    def field(field_name)
      num_data = self.num_data
      out_len = ::FFI::MemoryPointer.new(:int)
      out_ptr = ::FFI::MemoryPointer.new(:float, num_data)
      out_type = ::FFI::MemoryPointer.new(:int)
      check_result FFI.LGBM_DatasetGetField(handle_pointer, field_name, out_len, out_ptr, out_type)
      out_ptr.read_pointer.read_array_of_float(num_data)
    end

    def set_field(field_name, data)
      c_data = ::FFI::MemoryPointer.new(:float, data.count)
      c_data.put_array_of_float(0, data)
      check_result FFI.LGBM_DatasetSetField(handle_pointer, field_name, c_data, data.count, 0)
    end

    include Utils
  end
end
