require_relative "test_helper"

class BoosterTest < Minitest::Test
  def test_model_file
    x_test = [[3.7, 1.2, 7.2, 9.0], [7.5, 0.5, 7.9, 0.0]]
    booster = LightGBM::Booster.new(model_file: "test/support/model.txt")
    y_pred = booster.predict(x_test)
    assert_elements_in_delta [0.9823112229173586, 0.9583143724610858], y_pred.first(2)
  end

  def test_model_str
    x_test = [[3.7, 1.2, 7.2, 9.0], [7.5, 0.5, 7.9, 0.0]]
    booster = LightGBM::Booster.new(model_str: File.read("test/support/model.txt"))
    y_pred = booster.predict(x_test)
    assert_elements_in_delta [0.9823112229173586, 0.9583143724610858], y_pred.first(2)
  end

  def test_model_from_string
    x_test = [[3.7, 1.2, 7.2, 9.0], [7.5, 0.5, 7.9, 0.0]]
    booster = LightGBM.train(binary_params, binary_train)
    booster.model_from_string(File.read("test/support/model.txt"))
    y_pred = booster.predict(x_test)
    assert_elements_in_delta [0.9823112229173586, 0.9583143724610858], y_pred.first(2)
  end

  def test_feature_importance
    assert_equal [280, 285, 335, 148], booster.feature_importance
  end

  def test_feature_name
    assert_equal ["x0", "x1", "x2", "x3"], booster.feature_name
  end

  def test_feature_importance_bad_importance_type
    error = assert_raises(LightGBM::Error) do
      booster.feature_importance(importance_type: "bad")
    end
    assert_includes error.message, "Unknown importance type"
  end

  def test_predict_hash
    pred = booster.predict({x0: 3.7, x1: 1.2, x2: 7.2, x3: 9.0})
    assert_in_delta 0.9823112229173586, pred

    pred = booster.predict({"x3" => 9.0, "x2" => 7.2, "x1" => 1.2, "x0" => 3.7})
    assert_in_delta 0.9823112229173586, pred

    pred =
      booster.predict([
        {"x3" => 9.0, "x2" => 7.2, "x1" => 1.2, "x0" => 3.7},
        {"x3" => 0.0, "x2" => 7.9, "x1" => 0.5, "x0" => 7.5}
      ])
    assert_elements_in_delta [0.9823112229173586, 0.9583143724610858], pred.first(2)

    assert_raises(KeyError) do
      booster.predict({"x0" => 3.7})
    end
  end

  def test_predict_daru
    x_test =
      Daru::DataFrame.new([
        {"x3" => 9.0, "x2" => 7.2, "x1" => 1.2, "x0" => 3.7},
        {"x3" => 0.0, "x2" => 7.9, "x1" => 0.5, "x0" => 7.5}
      ])
    pred = booster.predict(x_test)
    assert_elements_in_delta [0.9823112229173586, 0.9583143724610858], pred.first(2)

    assert_raises(IndexError) do
      booster.predict(Daru::DataFrame.new([{"x0" => 3.7}]))
    end
  end

  def test_predict_rover
    skip if jruby?

    require "rover"
    x_test =
      Rover::DataFrame.new([
        {"x3" => 9.0, "x2" => 7.2, "x1" => 1.2, "x0" => 3.7},
        {"x3" => 0.0, "x2" => 7.9, "x1" => 0.5, "x0" => 7.5}
      ])
    pred = booster.predict(x_test)
    assert_elements_in_delta [0.9823112229173586, 0.9583143724610858], pred.first(2)

    assert_raises(KeyError) do
      booster.predict(Rover::DataFrame.new([{"x0" => 3.7}]))
    end
  end

  def test_predict_raw_score
    x_test = [[3.7, 1.2, 7.2, 9.0], [7.5, 0.5, 7.9, 0.0]]
    expected = [0.9823112229173586, 0.9583143724610858]

    y_pred = booster.predict(x_test, raw_score: true)
    assert_elements_in_delta expected, y_pred.first(2)

    y_pred = booster.predict(x_test[0], raw_score: true)
    assert_in_delta expected[0], y_pred
  end

  def test_predict_pred_leaf
    x_test = [[3.7, 1.2, 7.2, 9.0], [7.5, 0.5, 7.9, 0.0]]
    expected = [[9, 8, 8, 11, 8, 6, 10, 12, 1, 10, 9, 10, 12, 5, 11, 9, 6, 4, 5, 12, 9, 11, 9, 11, 2, 10, 2, 10, 3, 5, 10, 6, 1, 5, 10, 10, 9, 4, 5, 4, 6, 5, 6, 6, 4, 6, 4, 10, 10, 3, 4, 4, 6, 3, 9, 11, 5, 4, 3, 6, 7, 3, 6, 7, 5, 10, 10, 6, 4, 5, 5, 9, 6, 6, 2, 2, 4, 9, 4, 3, 9, 4, 6, 11, 5, 5, 0, 9, 12, 10, 12, 4, 0, 8, 4, 8, 11, 0, 3, 10], [6, 1, 9, 7, 9, 8, 1, 7, 5, 1, 1, 1, 9, 10, 1, 1, 10, 9, 1, 11, 8, 2, 10, 3, 5, 10, 6, 0, 2, 5, 0, 0, 0, 0, 0, 0, 8, 1, 0, 0, 0, 10, 0, 0, 2, 0, 0, 9, 2, 9, 3, 1, 2, 2, 7, 9, 10, 1, 4, 4, 9, 10, 0, 1, 3, 11, 2, 5, 1, 1, 7, 8, 5, 1, 10, 10, 5, 4, 1, 10, 2, 1, 4, 2, 2, 2, 2, 10, 2, 9, 2, 11, 2, 5, 1, 11, 2, 9, 7, 7]]

    y_pred = booster.predict(x_test, pred_leaf: true)
    assert_equal expected, y_pred.first(2)

    y_pred = booster.predict(x_test[0], pred_leaf: true)
    assert_equal expected[0], y_pred
  end

  def test_predict_pred_contrib
    x_test = [[3.7, 1.2, 7.2, 9.0], [7.5, 0.5, 7.9, 0.0]]
    expected = [[-0.0733949225678886, -0.24289592050101766, 0.24183795683166504, 0.063430775771174, 0.9933333333834246], [0.1094902954684793, -0.2810485083947154, 0.26691627597706397, -0.13037702397316747, 0.9933333333834246]]

    y_pred = booster.predict(x_test, pred_contrib: true)
    assert_elements_in_delta expected[0], y_pred[0]
    assert_elements_in_delta expected[1], y_pred[1]

    y_pred = booster.predict(x_test[0], pred_contrib: true)
    assert_elements_in_delta expected[0], y_pred
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

  def test_copy
    booster.dup
    booster.clone
  end

  private

  def booster
    @booster ||= LightGBM::Booster.new(model_file: "test/support/model.txt")
  end
end
