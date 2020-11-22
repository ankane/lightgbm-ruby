require_relative "test_helper"

class BoosterTest < Minitest::Test
  def test_model_file
    skip

    x_test = [[3.7, 1.2, 7.2, 9.0], [7.5, 0.5, 7.9, 0.0]]
    booster = LightGBM::Booster.new(model_file: "test/support/model.txt")
    y_pred = booster.predict(x_test)
    assert_elements_in_delta [0.9823112229173586, 0.9583143724610858], y_pred.first(2)
  end

  def test_model_str
    skip

    x_test = [[3.7, 1.2, 7.2, 9.0], [7.5, 0.5, 7.9, 0.0]]
    booster = LightGBM::Booster.new(model_str: File.read("test/support/model.txt"))
    y_pred = booster.predict(x_test)
    assert_elements_in_delta [0.9823112229173586, 0.9583143724610858], y_pred.first(2)
  end

  def test_feature_importance
    expected = [280, 285, 335, 148]
    assert_equal expected, booster.feature_importance
  end

  def test_feature_importance_bad_importance_type
    error = assert_raises(LightGBM::Error) do
      booster.feature_importance(importance_type: "bad")
    end
    assert_includes error.message, "Unknown importance type"
  end

  def test_model_to_string
    assert booster.model_to_string
  end

  def test_to_json
    assert JSON.parse(booster.to_json)
  end

  def test_dump_model
    assert JSON.parse(booster.dump_model)
  end

  def test_current_iteration
    assert_equal 100, booster.current_iteration
  end

  def test_num_model_per_iteration
    assert_equal 1, booster.num_model_per_iteration
  end

  def test_num_trees
    assert_equal 100, booster.num_trees
  end

  private

  def booster
    @booster ||= LightGBM::Booster.new(model_file: "test/support/model.txt")
  end
end
