module LightGBM
  module FFI
    extend Fiddle::Importer

    libs = LightGBM.ffi_lib.dup
    begin
      dlload libs.shift
    rescue Fiddle::DLError => e
      retry if libs.any?
      raise e if ENV["LIGHTGBM_DEBUG"]
      raise LoadError, "Can't load LightGBM or OpenMP"
    end

    # https://github.com/microsoft/LightGBM/blob/master/include/LightGBM/c_api.h
    # keep same order

    typealias "int32_t", "int"

    typealias "DatasetHandle", "void*"
    typealias "BoosterHandle", "void*"

    extern "char* LGBM_GetLastError()"
    extern "int LGBM_DatasetCreateFromFile(char* filename, char* parameters, DatasetHandle reference, DatasetHandle* out)"
    extern "int LGBM_DatasetCreateFromSampledColumn(double** sample_data, int** sample_indices, int32_t ncol, int* num_per_col, int32_t num_sample_row, int32_t num_total_row, char* parameters, DatasetHandle* out)"
    extern "int LGBM_DatasetCreateByReference(DatasetHandle reference, int64_t num_total_row, DatasetHandle* out)"
    extern "int LGBM_DatasetPushRows(DatasetHandle dataset, void* data, int data_type, int32_t nrow, int32_t ncol, int32_t start_row)"
    extern "int LGBM_DatasetPushRowsByCSR(DatasetHandle dataset, void* indptr, int indptr_type, int32_t* indices, void* data, int data_type, int64_t nindptr, int64_t nelem, int64_t num_col, int64_t start_row)"
    extern "int LGBM_DatasetCreateFromCSR(void* indptr, int indptr_type, int32_t* indices, void* data, int data_type, int64_t nindptr, int64_t nelem, int64_t num_col, char* parameters, DatasetHandle reference, DatasetHandle* out)"
    extern "int LGBM_DatasetCreateFromCSRFunc(void* get_row_funptr, int num_rows, int64_t num_col, char* parameters, DatasetHandle reference, DatasetHandle* out)"
    extern "int LGBM_DatasetCreateFromCSC(void* col_ptr, int col_ptr_type, int32_t* indices, void* data, int data_type, int64_t ncol_ptr, int64_t nelem, int64_t num_row, char* parameters, DatasetHandle reference, DatasetHandle* out)"
    extern "int LGBM_DatasetCreateFromMat(void* data, int data_type, int32_t nrow, int32_t ncol, int is_row_major, char* parameters, DatasetHandle reference, DatasetHandle* out)"
    extern "int LGBM_DatasetCreateFromMats(int32_t nmat, void** data, int data_type, int32_t* nrow, int32_t ncol, int is_row_major, char* parameters, DatasetHandle reference, DatasetHandle* out)"
    extern "int LGBM_DatasetGetSubset(DatasetHandle handle, int32_t* used_row_indices, int32_t num_used_row_indices, char* parameters, DatasetHandle* out)"
    extern "int LGBM_DatasetSetFeatureNames(DatasetHandle handle, char** feature_names, int num_feature_names)"
    extern "int LGBM_DatasetGetFeatureNames(DatasetHandle handle, char** feature_names, int* num_feature_names)"
    extern "int LGBM_DatasetFree(DatasetHandle handle)"
    extern "int LGBM_DatasetSaveBinary(DatasetHandle handle, char* filename)"
    extern "int LGBM_DatasetDumpText(DatasetHandle handle, char* filename)"
    extern "int LGBM_DatasetSetField(DatasetHandle handle, char* field_name, void* field_data, int num_element, int type)"
    extern "int LGBM_DatasetGetField(DatasetHandle handle, char* field_name, int* out_len, void** out_ptr, int* out_type)"
    extern "int LGBM_DatasetUpdateParam(DatasetHandle handle, char* parameters)"
    extern "int LGBM_DatasetGetNumData(DatasetHandle handle, int* out)"
    extern "int LGBM_DatasetGetNumFeature(DatasetHandle handle, int* out)"
    extern "int LGBM_DatasetAddFeaturesFrom(DatasetHandle target, DatasetHandle source)"
    extern "int LGBM_BoosterCreate(DatasetHandle train_data, char* parameters, BoosterHandle* out)"
    extern "int LGBM_BoosterCreateFromModelfile(char* filename, int* out_num_iterations, BoosterHandle* out)"
    extern "int LGBM_BoosterLoadModelFromString(char* model_str, int* out_num_iterations, BoosterHandle* out)"
    extern "int LGBM_BoosterFree(BoosterHandle handle)"
    extern "int LGBM_BoosterShuffleModels(BoosterHandle handle, int start_iter, int end_iter)"
    extern "int LGBM_BoosterMerge(BoosterHandle handle, BoosterHandle other_handle)"
    extern "int LGBM_BoosterAddValidData(BoosterHandle handle, DatasetHandle valid_data)"
    extern "int LGBM_BoosterResetTrainingData(BoosterHandle handle, DatasetHandle train_data)"
    extern "int LGBM_BoosterResetParameter(BoosterHandle handle, char* parameters)"
    extern "int LGBM_BoosterGetNumClasses(BoosterHandle handle, int* out_len)"
    extern "int LGBM_BoosterUpdateOneIter(BoosterHandle handle, int* is_finished)"
    extern "int LGBM_BoosterRefit(BoosterHandle handle, int32_t* leaf_preds, int32_t nrow, int32_t ncol)"
    extern "int LGBM_BoosterUpdateOneIterCustom(BoosterHandle handle, float* grad, float* hess, int* is_finished)"
    extern "int LGBM_BoosterRollbackOneIter(BoosterHandle handle)"
    extern "int LGBM_BoosterGetCurrentIteration(BoosterHandle handle, int* out_iteration)"
    extern "int LGBM_BoosterNumModelPerIteration(BoosterHandle handle, int* out_tree_per_iteration)"
    extern "int LGBM_BoosterNumberOfTotalModel(BoosterHandle handle, int* out_models)"
    extern "int LGBM_BoosterGetEvalCounts(BoosterHandle handle, int* out_len)"
    extern "int LGBM_BoosterGetEvalNames(BoosterHandle handle, int* out_len, char** out_strs)"
    extern "int LGBM_BoosterGetFeatureNames(BoosterHandle handle, int* out_len, char** out_strs)"
    extern "int LGBM_BoosterGetNumFeature(BoosterHandle handle, int* out_len)"
    extern "int LGBM_BoosterGetEval(BoosterHandle handle, int data_idx, int* out_len, double* out_results)"
    extern "int LGBM_BoosterGetNumPredict(BoosterHandle handle, int data_idx, int64_t* out_len)"
    extern "int LGBM_BoosterGetPredict(BoosterHandle handle, int data_idx, int64_t* out_len, double* out_result)"
    extern "int LGBM_BoosterPredictForFile(BoosterHandle handle, char* data_filename, int data_has_header, int predict_type, int num_iteration, char* parameter, char* result_filename)"
    extern "int LGBM_BoosterCalcNumPredict(BoosterHandle handle, int num_row, int predict_type, int num_iteration, int64_t* out_len)"
    extern "int LGBM_BoosterPredictForCSR(BoosterHandle handle, void* indptr, int indptr_type, int32_t* indices, void* data, int data_type, int64_t nindptr, int64_t nelem, int64_t num_col, int predict_type, int num_iteration, char* parameter, int64_t* out_len, double* out_result)"
    extern "int LGBM_BoosterPredictForCSRSingleRow(BoosterHandle handle, void* indptr, int indptr_type, int32_t* indices, void* data, int data_type, int64_t nindptr, int64_t nelem, int64_t num_col, int predict_type, int num_iteration, char* parameter, int64_t* out_len, double* out_result)"
    extern "int LGBM_BoosterPredictForCSC(BoosterHandle handle, void* col_ptr, int col_ptr_type, int32_t* indices, void* data, int data_type, int64_t ncol_ptr, int64_t nelem, int64_t num_row, int predict_type, int num_iteration, char* parameter, int64_t* out_len, double* out_result)"
    extern "int LGBM_BoosterPredictForMat(BoosterHandle handle, void* data, int data_type, int32_t nrow, int32_t ncol, int is_row_major, int predict_type, int num_iteration, char* parameter, int64_t* out_len, double* out_result)"
    extern "int LGBM_BoosterPredictForMatSingleRow(BoosterHandle handle, void* data, int data_type, int ncol, int is_row_major, int predict_type, int num_iteration, char* parameter, int64_t* out_len, double* out_result)"
    extern "int LGBM_BoosterPredictForMats(BoosterHandle handle, void** data, int data_type, int32_t nrow, int32_t ncol, int predict_type, int num_iteration, char* parameter, int64_t* out_len, double* out_result)"
    extern "int LGBM_BoosterSaveModel(BoosterHandle handle, int start_iteration, int num_iteration, char* filename)"
    extern "int LGBM_BoosterSaveModelToString(BoosterHandle handle, int start_iteration, int num_iteration, int64_t buffer_len, int64_t* out_len, char* out_str)"
    extern "int LGBM_BoosterDumpModel(BoosterHandle handle, int start_iteration, int num_iteration, int64_t buffer_len, int64_t* out_len, char* out_str)"
    extern "int LGBM_BoosterGetLeafValue(BoosterHandle handle, int tree_idx, int leaf_idx, double* out_val)"
    extern "int LGBM_BoosterSetLeafValue(BoosterHandle handle, int tree_idx, int leaf_idx, double val)"
    extern "int LGBM_BoosterFeatureImportance(BoosterHandle handle, int num_iteration, int importance_type, double* out_results)"
    extern "int LGBM_NetworkInit(char* machines, int local_listen_port, int listen_time_out, int num_machines)"
    extern "int LGBM_NetworkFree()"
    extern "int LGBM_NetworkInitWithFunctions(int num_machines, int rank, void* reduce_scatter_ext_fun, void* allgather_ext_fun)"

    class Pointer
      attr_accessor :ptr

      def initialize(size = nil, count = 1)
        if size
          size =
            case size
            when :float
              Fiddle::SIZEOF_FLOAT
            when :double
              Fiddle::SIZEOF_DOUBLE
            when :int, :int32, :uint32
              Fiddle::SIZEOF_INT
            when :int64
              Fiddle::SIZEOF_LONG_LONG
            when :pointer
              Fiddle::SIZEOF_VOIDP
            when :char
              Fiddle::SIZEOF_CHAR
            else
              raise "Unknown size: #{size}"
            end

          @ptr = Fiddle::Pointer.malloc(size * count)
        end
      end

      def write_array_of_pointer(ary)
        ary.each_with_index do |a, i|
          @ptr[i * Fiddle::SIZEOF_VOIDP, Fiddle::SIZEOF_VOIDP] = a.ptr.ref
        end
      end

      def write_array_of_float(ary)
        write_array(ary, "f")
      end

      def write_array_of_int32(ary)
        write_array(ary, "i!")
      end

      def read_array_of_float(count)
        read_array(Fiddle::SIZEOF_FLOAT, "f", count)
      end

      def read_array_of_double(count)
        read_array(Fiddle::SIZEOF_DOUBLE, "d", count)
      end

      def read_string
        @ptr.to_s
      end

      def read_int
        unpack(Fiddle::SIZEOF_INT, "i!")
      end

      def read_int64
        unpack(Fiddle::SIZEOF_LONG_LONG, "q")
      end

      def read_pointer
        ptr = Pointer.new
        ptr.ptr = @ptr.ptr
        ptr
      end

      def to_i
        @ptr.to_i
      end

      def self.from_string(str)
        ptr = Pointer.new
        ptr.ptr = Fiddle::Pointer[str]
        ptr
      end

      private

      def read_array(size, fmt, count)
        @ptr[0, size * count].unpack("#{fmt}*")
      end

      def write_array(ary, fmt)
        str = ary.pack("#{fmt}*")
        @ptr[0, str.size] = str
      end

      def unpack(size, fmt)
        @ptr[0, size].unpack1(fmt)
      end
    end
  end
end
