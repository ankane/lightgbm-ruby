module LightGBM
  module Utils
    private

    def check_result(err)
      raise LightGBM::Error, FFI.LGBM_GetLastError if err != 0
    end

    # remove spaces in keys and values to prevent injection
    def params_str(params)
      params.map { |k, v| [check_param(k.to_s), check_param(Array(v).join(",").to_s)].join("=") }.join(" ")
    end

    def check_param(v)
      raise ArgumentError, "Invalid parameter" if /[[:space:]]/.match?(v)
      v
    end

    # change default verbosity
    def set_verbosity(params)
      params_keys = params.keys.map(&:to_s)
      unless params_keys.include?("verbosity")
        params["verbosity"] = -1
      end
    end

    # for categorical, NaN and negative value are the same
    def handle_missing(data)
      data.map! { |v| v.nil? ? Float::NAN : v }
    end

    def matrix?(data)
      defined?(Matrix) && data.is_a?(Matrix)
    end

    def daru?(data)
      defined?(Daru::DataFrame) && data.is_a?(Daru::DataFrame)
    end

    def numo?(data)
      defined?(Numo::NArray) && data.is_a?(Numo::NArray)
    end

    def rover?(data)
      defined?(Rover::DataFrame) && data.is_a?(Rover::DataFrame)
    end
  end
end
