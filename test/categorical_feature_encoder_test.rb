require_relative "test_helper"

class CategoricalFeatureEncoder < Minitest::Test
  def setup
    model = <<~MODEL
      [categorical_feature: 1,2,3]
      pandas_categorical:[[-1.0, 0.0, 1.0], ["red", "green", "blue"], [false, true]]
    MODEL

    @encoder = LightGBM::CategoricalFeatureEncoder.new(model.each_line)
  end

  def test_apply_with_categorical_features
    input = [42.0, 0.0, "green", true]
    expected = [42.0, 1.0, 1.0, 1.0]

    assert_equal(expected, @encoder.apply(input))
  end

  def test_apply_with_non_categorical_features
    input = [42.0, "non_categorical", 39.0, false]
    expected = [42.0, Float::NAN, Float::NAN, 0]

    assert_equal(expected, @encoder.apply(input))
  end

  def test_apply_with_missing_values
    input = [42.0, nil, "red", nil]
    expected = [42.0, Float::NAN, 0.0, Float::NAN]
    result = @encoder.apply(input)

    assert_equal(expected, result)
  end

  def test_apply_with_boolean_values
    input = [42.0, -1.0, "green", false]
    expected = [42.0, 0.0, 1.0, 0.0]

    assert_equal(expected, @encoder.apply(input))
  end
end
