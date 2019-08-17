module LightGBM
  class Dataset
    attr_reader :data, :params

    def initialize(data, label: nil, weight: nil, params: nil, reference: nil, used_indices: nil, categorical_feature: "auto")
      @data = data

      # TODO stringify params
      params ||= {}
      params["categorical_feature"] ||= categorical_feature.join(",") if categorical_feature != "auto"
      set_verbosity(params)

      @handle = ::FFI::MemoryPointer.new(:pointer)
      parameters = params_str(params)
      reference = reference.handle_pointer if reference
      if data.is_a?(String)
        check_result FFI.LGBM_DatasetCreateFromFile(data, parameters, reference, @handle)
      elsif used_indices
        used_row_indices = ::FFI::MemoryPointer.new(:int32, used_indices.count)
        used_row_indices.put_array_of_int32(0, used_indices)
        check_result FFI.LGBM_DatasetGetSubset(reference, used_row_indices, used_indices.count, parameters, @handle)
      else
        if matrix?(data)
          nrow = data.row_count
          ncol = data.column_count
          flat_data = data.to_a.flatten
        elsif daru?(data)
          nrow, ncol = data.shape
          flat_data = data.each_vector.map(&:to_a).flatten
        elsif narray?(data)
          nrow, ncol = data.shape
          flat_data = data.flatten.to_a
        else
          nrow = data.count
          ncol = data.first.count
          flat_data = data.flatten
        end

        c_data = ::FFI::MemoryPointer.new(:float, nrow * ncol)
        c_data.put_array_of_float(0, flat_data)
        check_result FFI.LGBM_DatasetCreateFromMat(c_data, 0, nrow, ncol, 1, parameters, reference, @handle)
      end
      ObjectSpace.define_finalizer(self, self.class.finalize(handle_pointer)) unless used_indices

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

    # not released yet
    # def dump_text(filename)
    #   check_result FFI.LGBM_DatasetDumpText(handle_pointer, filename)
    # end

    def subset(used_indices, params: nil)
      # categorical_feature passed via params
      params ||= self.params
      Dataset.new(nil,
        params: params,
        reference: self,
        used_indices: used_indices
      )
    end

    def handle_pointer
      @handle.read_pointer
    end

    def self.finalize(pointer)
      # must use proc instead of stabby lambda
      proc { FFI.LGBM_DatasetFree(pointer) }
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
      data = data.to_a unless data.is_a?(Array)
      c_data = ::FFI::MemoryPointer.new(:float, data.count)
      c_data.put_array_of_float(0, data)
      check_result FFI.LGBM_DatasetSetField(handle_pointer, field_name, c_data, data.count, 0)
    end

    def matrix?(data)
      defined?(Matrix) && data.is_a?(Matrix)
    end

    def daru?(data)
      defined?(Daru::DataFrame) && data.is_a?(Daru::DataFrame)
    end

    def narray?(data)
      defined?(Numo::NArray) && data.is_a?(Numo::NArray)
    end

    include Utils
  end
end
