module LightGBM
  module Utils
    private

    def check_result(err)
      raise LightGBM::Error, FFI.LGBM_GetLastError if err != 0
    end

    # remove spaces in keys and values to prevent injection
    def params_str(params)
      (params || {}).map { |k, v| [check_param(k.to_s), check_param(v.to_s)].join("=") }.join(" ")
    end

    def check_param(v)
      raise ArgumentError, "Invalid parameter" if /[[:space:]]/.match(v)
      v
    end
  end
end
