module LightGBM
  class Dataset
    include Utils

    attr_reader :data, :params

    def initialize(data, label: nil, weight: nil, group: nil, params: nil, reference: nil, used_indices: nil, categorical_feature: "auto", feature_name: nil, feature_names: nil)
      @data = data
      @label = label
      @weight = weight
      @group = group
      @params = params
      @reference = reference
      @used_indices = used_indices
      @categorical_feature = categorical_feature
      @feature_name = feature_name || feature_names || "auto"

      construct
    end

    def label
      field("label")
    end

    def weight
      field("weight")
    end

    def feature_name
      # must preallocate space
      num_feature_names = ::FFI::MemoryPointer.new(:int)
      out_buffer_len = ::FFI::MemoryPointer.new(:size_t)
      len = 1000
      out_strs = ::FFI::MemoryPointer.new(:pointer, len)
      buffer_len = 255
      str_ptrs = len.times.map { ::FFI::MemoryPointer.new(:char, buffer_len) }
      out_strs.write_array_of_pointer(str_ptrs)
      safe_call FFI.LGBM_DatasetGetFeatureNames(@handle, len, num_feature_names, buffer_len, out_buffer_len, out_strs)

      num_features = num_feature_names.read_int
      actual_len = out_buffer_len.read(:size_t)
      if num_features > len || actual_len > buffer_len
        out_strs = ::FFI::MemoryPointer.new(:pointer, num_features) if num_features > len
        str_ptrs = num_features.times.map { ::FFI::MemoryPointer.new(:char, actual_len) }
        out_strs.write_array_of_pointer(str_ptrs)
        safe_call FFI.LGBM_DatasetGetFeatureNames(@handle, num_features, num_feature_names, actual_len, out_buffer_len, out_strs)
      end

      # should be the same, but get number of features
      # from most recent call (instead of num_features)
      str_ptrs[0, num_feature_names.read_int].map(&:read_string)
    end
    alias_method :feature_names, :feature_name

    def label=(label)
      @label = label
      set_field("label", label)
    end

    def weight=(weight)
      @weight = weight
      set_field("weight", weight)
    end

    def group=(group)
      @group = group
      set_field("group", group, type: :int32)
    end

    def feature_name=(feature_names)
      feature_names = feature_names.map(&:to_s)
      @feature_names = feature_names
      c_feature_names = ::FFI::MemoryPointer.new(:pointer, feature_names.size)
      # keep reference to string pointers
      str_ptrs = feature_names.map { |v| ::FFI::MemoryPointer.from_string(v) }
      c_feature_names.write_array_of_pointer(str_ptrs)
      safe_call FFI.LGBM_DatasetSetFeatureNames(@handle, c_feature_names, feature_names.size)
    end
    alias_method :feature_names=, :feature_name=

    # TODO only update reference if not in chain
    def reference=(reference)
      if reference != @reference
        @reference = reference
        construct
      end
    end

    def num_data
      out = ::FFI::MemoryPointer.new(:int)
      safe_call FFI.LGBM_DatasetGetNumData(@handle, out)
      out.read_int
    end

    def num_feature
      out = ::FFI::MemoryPointer.new(:int)
      safe_call FFI.LGBM_DatasetGetNumFeature(@handle, out)
      out.read_int
    end

    def save_binary(filename)
      safe_call FFI.LGBM_DatasetSaveBinary(@handle, filename)
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

    def handle
      @handle
    end

    private

    def construct
      data = @data
      used_indices = @used_indices

      # TODO stringify params
      params = @params || {}
      if @categorical_feature != "auto" && @categorical_feature.any?
        params["categorical_feature"] ||= @categorical_feature.join(",")
      end
      set_verbosity(params)

      handle = ::FFI::MemoryPointer.new(:pointer)
      parameters = params_str(params)
      reference = @reference.handle if @reference
      if used_indices
        used_row_indices = ::FFI::MemoryPointer.new(:int32, used_indices.count)
        used_row_indices.write_array_of_int32(used_indices)
        safe_call FFI.LGBM_DatasetGetSubset(reference, used_row_indices, used_indices.count, parameters, handle)
      elsif data.is_a?(String)
        safe_call FFI.LGBM_DatasetCreateFromFile(data, parameters, reference, handle)
      else
        if matrix?(data)
          nrow = data.row_count
          ncol = data.column_count
          flat_data = data.to_a.flatten
        elsif daru?(data)
          if @feature_name == "auto"
            @feature_name = data.vectors.to_a
          end
          nrow, ncol = data.shape
          flat_data = data.map_rows(&:to_a).flatten
        elsif numo?(data)
          nrow, ncol = data.shape
        elsif rover?(data)
          if @feature_name == "auto"
            @feature_name = data.keys
          end
          data = data.to_numo
          nrow, ncol = data.shape
        elsif data.is_a?(Array) && data.first.is_a?(Hash)
          keys = data.first.keys
          if @feature_name == "auto"
            @feature_name = keys
          end
          nrow = data.count
          ncol = data.first.count
          flat_data = data.flat_map { |v| v.fetch_values(*keys) }
        else
          data = data.to_a
          check_2d_array(data)
          nrow = data.count
          ncol = data.first.count
          flat_data = data.flatten
        end

        c_data = ::FFI::MemoryPointer.new(:double, nrow * ncol)
        if numo?(data)
          c_data.write_bytes(data.cast_to(Numo::DFloat).to_string)
        else
          handle_missing(flat_data)
          c_data.write_array_of_double(flat_data)
        end

        safe_call FFI.LGBM_DatasetCreateFromMat(c_data, FFI::C_API_DTYPE_FLOAT64, nrow, ncol, 1, parameters, reference, handle)
      end
      if used_indices
        @handle = handle.read_pointer
      else
        @handle = ::FFI::AutoPointer.new(handle.read_pointer, FFI.method(:LGBM_DatasetFree))
      end

      self.label = @label if @label
      self.weight = @weight if @weight
      self.group = @group if @group
      self.feature_name = @feature_name if @feature_name && @feature_name != "auto"
    end

    def dump_text(filename)
      safe_call FFI.LGBM_DatasetDumpText(@handle, filename)
    end

    def field(field_name)
      num_data = self.num_data
      out_len = ::FFI::MemoryPointer.new(:int)
      out_ptr = ::FFI::MemoryPointer.new(:float, num_data)
      out_type = ::FFI::MemoryPointer.new(:int)
      safe_call FFI.LGBM_DatasetGetField(@handle, field_name, out_len, out_ptr, out_type)
      out_ptr.read_pointer.read_array_of_float(num_data)
    end

    def set_field(field_name, data, type: :float)
      data = data.to_a unless data.is_a?(Array)
      if type == :int32
        c_data = ::FFI::MemoryPointer.new(:int32, data.count)
        c_data.write_array_of_int32(data)
        safe_call FFI.LGBM_DatasetSetField(@handle, field_name, c_data, data.count, 2)
      else
        c_data = ::FFI::MemoryPointer.new(:float, data.count)
        c_data.write_array_of_float(data)
        safe_call FFI.LGBM_DatasetSetField(@handle, field_name, c_data, data.count, 0)
      end
    end
  end
end
