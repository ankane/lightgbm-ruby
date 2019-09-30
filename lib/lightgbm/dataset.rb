module LightGBM
  class Dataset
    attr_reader :data, :params

    def initialize(data, label: nil, weight: nil, group: nil, params: nil, reference: nil, used_indices: nil, categorical_feature: "auto", feature_names: nil)
      @data = data

      # TODO stringify params
      params ||= {}
      if categorical_feature != "auto" && categorical_feature.any?
        params["categorical_feature"] ||= categorical_feature.join(",")
      end
      set_verbosity(params)

      @handle = ::FFI::MemoryPointer.new(:pointer)
      parameters = params_str(params)
      reference = reference.handle_pointer if reference
      if used_indices
        used_row_indices = ::FFI::MemoryPointer.new(:int32, used_indices.count)
        used_row_indices.put_array_of_int32(0, used_indices)
        check_result FFI.LGBM_DatasetGetSubset(reference, used_row_indices, used_indices.count, parameters, @handle)
      elsif data.is_a?(String)
        check_result FFI.LGBM_DatasetCreateFromFile(data, parameters, reference, @handle)
      else
        if matrix?(data)
          nrow = data.row_count
          ncol = data.column_count
          flat_data = data.to_a.flatten
        elsif daru?(data)
          nrow, ncol = data.shape
          flat_data = data.map_rows(&:to_a).flatten
        elsif narray?(data)
          nrow, ncol = data.shape
          flat_data = data.flatten.to_a
        else
          nrow = data.count
          ncol = data.first.count
          flat_data = data.flatten
        end

        handle_missing(flat_data)
        c_data = ::FFI::MemoryPointer.new(:float, nrow * ncol)
        c_data.put_array_of_float(0, flat_data)
        check_result FFI.LGBM_DatasetCreateFromMat(c_data, 0, nrow, ncol, 1, parameters, reference, @handle)
      end
      ObjectSpace.define_finalizer(self, self.class.finalize(handle_pointer)) unless used_indices

      self.label = label if label
      self.weight = weight if weight
      self.group = group if group
      self.feature_names = feature_names if feature_names
    end

    def label
      field("label")
    end

    def weight
      field("weight")
    end

    def label=(label)
      set_field("label", label)
    end

    def feature_names
      # must preallocate space
      num_feature_names = ::FFI::MemoryPointer.new(:int)
      out_strs = ::FFI::MemoryPointer.new(:pointer, 1000)
      str_ptrs = 1000.times.map { ::FFI::MemoryPointer.new(:string, 255) }
      out_strs.put_array_of_pointer(0, str_ptrs)
      check_result FFI.LGBM_DatasetGetFeatureNames(handle_pointer, out_strs, num_feature_names)
      str_ptrs[0, num_feature_names.read_int].map(&:read_string)
    end

    def weight=(weight)
      set_field("weight", weight)
    end

    def group=(group)
      set_field("group", group, type: :int32)
    end

    def feature_names=(feature_names)
      c_feature_names = ::FFI::MemoryPointer.new(:pointer, feature_names.size)
      c_feature_names.write_array_of_pointer(feature_names.map { |v| ::FFI::MemoryPointer.from_string(v) })
      check_result FFI.LGBM_DatasetSetFeatureNames(handle_pointer, c_feature_names, feature_names.size)
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

    def set_field(field_name, data, type: :float)
      data = data.to_a unless data.is_a?(Array)
      if type == :int32
        c_data = ::FFI::MemoryPointer.new(:int32, data.count)
        c_data.put_array_of_int32(0, data)
        check_result FFI.LGBM_DatasetSetField(handle_pointer, field_name, c_data, data.count, 2)
      else
        c_data = ::FFI::MemoryPointer.new(:float, data.count)
        c_data.put_array_of_float(0, data)
        check_result FFI.LGBM_DatasetSetField(handle_pointer, field_name, c_data, data.count, 0)
      end
    end

    include Utils
  end
end
