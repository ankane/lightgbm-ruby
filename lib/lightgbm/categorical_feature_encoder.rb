require "json"

module LightGBM
  # Converts LightGBM categorical featulres to Float, using label encoding.
  # The categorical and mappings are extracted from the LightGBM model file.
  class CategoricalFeatureEncoder
    # Initializes a new CategoricalFeatureEncoder instance.
    #
    # @param model_enumerable [Enumerable] Enumerable with each line of LightGBM model file.
    def initialize(model_enumerable)
      @categorical_feature = []
      @pandas_categorical = []

      load_categorical_features(model_enumerable)
    end

    # Returns a new array with categorical features converted to Float, using label encoding.
    def apply(feature_values)
      return feature_values if @categorical_feature.empty?

      transformed_features = feature_values.dup

      @categorical_feature.each_with_index do |feature_index, pandas_categorical_index|
        pandas_categorical_entry = @pandas_categorical[pandas_categorical_index]
        value = feature_values[feature_index]
        transformed_features[feature_index] = pandas_categorical_entry.fetch(value, Float::NAN).to_f
      end

      transformed_features
    end

    private

    def load_categorical_features(model_enumerable)
      categorical_found = false
      pandas_found = false

      model_enumerable.each_entry do |line|
        # Format: "[categorical_feature: 0,1,2,3,4,5]"
        if line.start_with?("[categorical_feature:")
          parts = line.split("categorical_feature:")
          last_part = parts.last
          next if last_part.nil?

          values = last_part.strip[0...-1]
          next if values.nil?

          @categorical_feature = values.split(",").map(&:to_i)
          categorical_found = true
        end

        # Format: "pandas_categorical:[[-1.0, 0.0, 1.0], ["", "a"], [false, true]]"
        if line.start_with?("pandas_categorical:")
          parts = line.split("pandas_categorical:")
          values = parts[1]
          next if values.nil?

          @pandas_categorical = JSON.parse(values).map do |array|
            array.each_with_index.to_h
          end
          pandas_found = true
        end

        # Break the loop if both lines are found
        break if categorical_found && pandas_found
      end

      if @categorical_feature.size != @pandas_categorical.size
        raise "categorical_feature and pandas_categorical mismatch"
      end
    end
  end
end
